// PagedKV-Fusion — Component B
// INT8 quantized paged-attention kernel (decode step: 1 query token/seq).
//
// KV layout (matches pagedkv_fusion.reference / quantize.py):
//   k_cache_q, v_cache_q : int8  [num_blocks, block_size, num_kv_heads, head_dim]
//   k_scale,  v_scale    : float [num_blocks, num_kv_heads]      (symmetric,
//                          per-(block, head) — coarse enough to sit in a
//                          register per page, fine enough to bound error)
//
// Algorithm: single-pass online softmax (flash-decoding style) over the
// sequence's pages, dequantizing K/V on the fly. INT8 KV halves the memory
// traffic of the dominant load (KV reads), which is where decode-attention
// time goes — that's the throughput thesis this kernel exists to measure.
//
// Parallelization:
//   grid  = (num_seqs, num_heads)      — one CTA per (sequence, query head)
//   block = BLOCK_THREADS (128)
// Each CTA:
//   1. loads its query vector into shared memory (fp32),
//   2. iterates the sequence's block table page by page,
//   3. per page: each warp takes a token, lanes stride head_dim to compute
//      the dequantized dot product q·k (coalesced int8 loads),
//   4. maintains running (max, sum, weighted-V accumulator) with the
//      standard online-softmax rescaling,
//   5. writes the normalized output vector.
//
// GQA is supported: kv_head = head / (num_heads / num_kv_heads).
//
// v1 constraints (checked in the launcher):
//   head_dim <= 256, block_size <= 32, head_dim % 32 == 0.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace pagedkv {

constexpr int ATTN_THREADS = 128;
constexpr int WARPS = ATTN_THREADS / 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_BLOCK_SIZE = 32;

__inline__ __device__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__global__ void quant_paged_attention_kernel(
    const float* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const int8_t* __restrict__ k_cache,    // [nb, bs, nkv, hd]
    const float* __restrict__ k_scale,     // [nb, nkv]
    const int8_t* __restrict__ v_cache,    // [nb, bs, nkv, hd]
    const float* __restrict__ v_scale,     // [nb, nkv]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    float* __restrict__ out,               // [num_seqs, num_heads, head_dim]
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const float sm_scale) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;

  const int seq_len = seq_lens[seq];
  if (seq_len == 0) return;
  const int n_pages = (seq_len + block_size - 1) / block_size;

  // --- shared memory ---------------------------------------------------
  __shared__ float s_q[MAX_HEAD_DIM];              // query vector (fp32)
  __shared__ float s_logits[MAX_BLOCK_SIZE];       // per-page logits
  __shared__ float s_acc[MAX_HEAD_DIM];            // running weighted-V sum
  __shared__ float s_m;                            // running max
  __shared__ float s_l;                            // running denom

  const float* q_ptr =
      q + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    s_q[d] = q_ptr[d];
    s_acc[d] = 0.0f;
  }
  if (tid == 0) {
    s_m = -INFINITY;
    s_l = 0.0f;
  }
  __syncthreads();

  const int64_t page_stride =
      static_cast<int64_t>(block_size) * num_kv_heads * head_dim;
  const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * head_dim;

  // --- main loop over pages ---------------------------------------------
  for (int p = 0; p < n_pages; ++p) {
    const int phys = block_tables[seq * max_blocks_per_seq + p];
    const int tokens_here =
        min(block_size, seq_len - p * block_size);  // partial last page
    const float ks = k_scale[phys * num_kv_heads + kv_head];
    const float vs = v_scale[phys * num_kv_heads + kv_head];
    const int8_t* k_page =
        k_cache + phys * page_stride + kv_head * head_dim;
    const int8_t* v_page =
        v_cache + phys * page_stride + kv_head * head_dim;

    // (1) logits: warp w handles tokens w, w+WARPS, ...; lanes stride hd.
    for (int t = wid; t < tokens_here; t += WARPS) {
      const int8_t* k_tok = k_page + t * token_stride;
      float dot = 0.0f;
      for (int d = lane; d < head_dim; d += 32) {
        dot += s_q[d] * static_cast<float>(k_tok[d]);
      }
      dot = warp_sum(dot);
      if (lane == 0) s_logits[t] = dot * ks * sm_scale;
    }
    __syncthreads();

    // (2) page max, online-softmax rescale of running state (thread 0
    //     scalar section: block_size <= 32, negligible next to KV loads).
    __shared__ float s_scale_old, s_pm;
    if (tid == 0) {
      float pm = -INFINITY;
      for (int t = 0; t < tokens_here; ++t) pm = fmaxf(pm, s_logits[t]);
      const float m_new = fmaxf(s_m, pm);
      s_scale_old = __expf(s_m - m_new);  // rescale factor for old state
      float l_page = 0.0f;
      for (int t = 0; t < tokens_here; ++t) {
        s_logits[t] = __expf(s_logits[t] - m_new);  // now probabilities*Z
        l_page += s_logits[t];
      }
      s_l = s_l * s_scale_old + l_page;
      s_m = m_new;
      s_pm = pm;  // (kept for debugability under ncu)
    }
    __syncthreads();

    // (3) rescale accumulator, then add this page's weighted V.
    //     Threads stride head_dim -> coalesced int8 V loads per token.
    for (int d = tid; d < head_dim; d += blockDim.x) {
      float acc = s_acc[d] * s_scale_old;
      for (int t = 0; t < tokens_here; ++t) {
        acc += s_logits[t] *
               static_cast<float>(v_page[t * token_stride + d]) * vs;
      }
      s_acc[d] = acc;
    }
    __syncthreads();
  }

  // --- normalize & write -------------------------------------------------
  float* out_ptr =
      out + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  const float inv_l = 1.0f / s_l;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out_ptr[d] = s_acc[d] * inv_l;
  }
}

}  // namespace pagedkv

// ---------------------------------------------------------------------------
// PyTorch-facing launcher
// ---------------------------------------------------------------------------

torch::Tensor quant_paged_attention_cuda(
    torch::Tensor q,
    torch::Tensor k_cache_q,
    torch::Tensor k_scale,
    torch::Tensor v_cache_q,
    torch::Tensor v_scale,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    double sm_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32 (v1)");
  TORCH_CHECK(k_cache_q.dtype() == torch::kInt8, "k_cache_q must be int8");
  TORCH_CHECK(block_tables.dtype() == torch::kInt32, "block_tables must be int32");

  const int num_seqs = q.size(0);
  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int block_size = k_cache_q.size(1);
  const int num_kv_heads = k_cache_q.size(2);
  const int max_blocks_per_seq = block_tables.size(1);

  TORCH_CHECK(head_dim <= pagedkv::MAX_HEAD_DIM && head_dim % 32 == 0,
              "v1 supports head_dim % 32 == 0, <= 256");
  TORCH_CHECK(block_size <= pagedkv::MAX_BLOCK_SIZE,
              "v1 supports block_size <= 32");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "GQA requires num_heads % num_kv_heads == 0");

  auto qc = q.contiguous();
  auto kc = k_cache_q.contiguous();
  auto ksc = k_scale.contiguous();
  auto vc = v_cache_q.contiguous();
  auto vsc = v_scale.contiguous();
  auto btc = block_tables.contiguous();
  auto slc = seq_lens.to(torch::kInt32).contiguous();

  auto out = torch::empty_like(qc);
  if (num_seqs == 0) return out;

  const dim3 grid(num_seqs, num_heads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pagedkv::quant_paged_attention_kernel<<<grid, pagedkv::ATTN_THREADS, 0, stream>>>(
      qc.data_ptr<float>(),
      kc.data_ptr<int8_t>(),
      ksc.data_ptr<float>(),
      vc.data_ptr<int8_t>(),
      vsc.data_ptr<float>(),
      btc.data_ptr<int>(),
      slc.data_ptr<int>(),
      out.data_ptr<float>(),
      num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq,
      static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
