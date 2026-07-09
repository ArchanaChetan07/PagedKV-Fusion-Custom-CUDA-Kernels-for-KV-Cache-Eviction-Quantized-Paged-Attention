// PagedKV-Fusion — Component A
// Fused KV-cache eviction-scoring kernel.
//
// Replaces the host-side (Python/NumPy) heuristic:
//     score[b] = w_r * recency[b] + w_f * frequency[b] + w_a * mean_attn[b]
// where mean_attn is the mean attention mass over *valid* tokens in block b.
//
// Design notes (see docs/PROFILING.md for the measured story):
//  * One thread block per KV block; BLOCK_THREADS threads cooperatively
//    reduce the attention row. block_size in vLLM is 16 or 32, so a single
//    warp usually suffices — we still use a two-level (warp shuffle +
//    shared memory) reduction so the kernel is correct for any block_size.
//  * All inputs stay GPU-resident: no host-device transfer per decision,
//    which is the entire latency win over the NumPy baseline (the math is
//    trivial; PCIe round-trips and Python overhead are what we're deleting).
//  * Reads of attn_weights are fully coalesced: thread t reads column
//    t, t+BT, t+2BT, ... of a contiguous row.
//  * Scores are written as float32; top-k candidate selection is done by a
//    device-side torch.topk on the score vector (already GPU-resident), so
//    the full evict decision never touches the host.

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace pagedkv {

constexpr int BLOCK_THREADS = 128;

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Two-level block reduction: warp shuffle then one warp over smem partials.
__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float smem[BLOCK_THREADS / 32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;  // valid in thread 0
}

// grid:  (num_blocks)
// block: (BLOCK_THREADS)
__global__ void eviction_score_kernel(
    const float* __restrict__ recency,       // [num_blocks]
    const float* __restrict__ frequency,     // [num_blocks]
    const float* __restrict__ attn_weights,  // [num_blocks, block_size]
    const bool* __restrict__ valid_mask,     // [num_blocks, block_size]
    float* __restrict__ scores,              // [num_blocks]
    const int block_size,
    const float w_recency,
    const float w_frequency,
    const float w_attention) {
  const int b = blockIdx.x;
  const int64_t row = static_cast<int64_t>(b) * block_size;

  // Cooperative masked sum + valid count over the block's token row.
  float attn_sum = 0.0f;
  float valid_cnt = 0.0f;
  for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
    const bool valid = valid_mask[row + t];
    attn_sum += valid ? attn_weights[row + t] : 0.0f;
    valid_cnt += valid ? 1.0f : 0.0f;
  }

  attn_sum = block_reduce_sum(attn_sum);
  __syncthreads();  // smem reuse barrier between the two reductions
  valid_cnt = block_reduce_sum(valid_cnt);

  if (threadIdx.x == 0) {
    const float mean_attn = valid_cnt > 0.0f ? attn_sum / valid_cnt : 0.0f;
    scores[b] = w_recency * recency[b] + w_frequency * frequency[b] +
                w_attention * mean_attn;
  }
}

}  // namespace pagedkv

// ---------------------------------------------------------------------------
// PyTorch-facing launcher
// ---------------------------------------------------------------------------

torch::Tensor eviction_scores_cuda(
    torch::Tensor recency,
    torch::Tensor frequency,
    torch::Tensor attn_weights,
    torch::Tensor valid_mask,
    double w_recency,
    double w_frequency,
    double w_attention) {
  TORCH_CHECK(recency.is_cuda() && frequency.is_cuda() &&
                  attn_weights.is_cuda() && valid_mask.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(recency.dtype() == torch::kFloat32, "recency must be float32");
  TORCH_CHECK(attn_weights.dim() == 2, "attn_weights must be [num_blocks, block_size]");
  TORCH_CHECK(valid_mask.dtype() == torch::kBool, "valid_mask must be bool");

  auto recency_c = recency.contiguous();
  auto frequency_c = frequency.contiguous();
  auto attn_c = attn_weights.contiguous();
  auto mask_c = valid_mask.contiguous();

  const int num_blocks = attn_c.size(0);
  const int block_size = attn_c.size(1);
  auto scores = torch::empty({num_blocks}, recency_c.options());

  if (num_blocks == 0) return scores;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pagedkv::eviction_score_kernel<<<num_blocks, pagedkv::BLOCK_THREADS, 0, stream>>>(
      recency_c.data_ptr<float>(),
      frequency_c.data_ptr<float>(),
      attn_c.data_ptr<float>(),
      mask_c.data_ptr<bool>(),
      scores.data_ptr<float>(),
      block_size,
      static_cast<float>(w_recency),
      static_cast<float>(w_frequency),
      static_cast<float>(w_attention));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}
