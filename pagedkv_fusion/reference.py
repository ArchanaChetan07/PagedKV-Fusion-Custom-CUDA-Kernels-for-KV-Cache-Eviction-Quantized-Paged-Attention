"""Reference implementations (NumPy) for PagedKV-Fusion kernels.

These are the *ground truth* for correctness testing. Every CUDA kernel in
``csrc/`` must produce output that matches these functions to within
documented tolerances. They are written for clarity, not speed — the whole
point of the project is that the CUDA versions beat these on latency while
matching them on output.

Conventions
-----------
- KV cache is paged: ``num_blocks`` physical blocks, each holding
  ``block_size`` tokens for ``num_kv_heads`` heads of ``head_dim`` channels.
- A sequence's logical layout is given by a ``block_table`` row of physical
  block indices; ``seq_len`` tokens are valid (last block may be partial).
- Eviction scoring follows the KV-Cache-Profiler heuristic:
  ``score = w_r * recency + w_f * frequency + w_a * mean_attention``
  where *lower* score means *better eviction candidate*.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "eviction_scores_ref",
    "select_eviction_candidates_ref",
    "quantize_kv_per_block_ref",
    "dequantize_kv_per_block_ref",
    "paged_attention_ref",
    "dense_attention_ref",
]


# ---------------------------------------------------------------------------
# Component A — eviction scoring
# ---------------------------------------------------------------------------

def eviction_scores_ref(
    recency: np.ndarray,        # [num_blocks] float32, normalized age (1 = just used)
    frequency: np.ndarray,      # [num_blocks] float32, normalized access count
    attn_weights: np.ndarray,   # [num_blocks, block_size] float32, per-token attention mass
    valid_mask: np.ndarray,     # [num_blocks, block_size] bool, token validity
    w_recency: float = 0.4,
    w_frequency: float = 0.3,
    w_attention: float = 0.3,
) -> np.ndarray:
    """Fused eviction score per KV block. Lower = evict first.

    The attention term is the mean attention mass over *valid* tokens in the
    block (empty blocks contribute 0 attention and are prime candidates).
    This mirrors the host-side heuristic in KV-Cache-Profiler, fused into a
    single pass so the CUDA version never materializes intermediates.
    """
    recency = np.asarray(recency, dtype=np.float32)
    frequency = np.asarray(frequency, dtype=np.float32)
    attn_weights = np.asarray(attn_weights, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_counts = valid_mask.sum(axis=1)                      # [num_blocks]
    attn_sum = np.where(valid_mask, attn_weights, 0.0).sum(axis=1)
    mean_attn = np.divide(
        attn_sum,
        valid_counts,
        out=np.zeros_like(attn_sum, dtype=np.float32),
        where=valid_counts > 0,
    )
    return (
        w_recency * recency
        + w_frequency * frequency
        + w_attention * mean_attn
    ).astype(np.float32)


def select_eviction_candidates_ref(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k lowest-scoring blocks, ascending by score.

    Ties broken by lower block index (deterministic, matches kernel's
    stable selection contract).
    """
    order = np.lexsort((np.arange(scores.shape[0]), scores))
    return order[:k].astype(np.int64)


# ---------------------------------------------------------------------------
# Component B — per-block INT8 quantization
# ---------------------------------------------------------------------------

def quantize_kv_per_block_ref(
    kv: np.ndarray,  # [num_blocks, block_size, num_kv_heads, head_dim] float32
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-(block, head) INT8 quantization.

    scale[b, h] = max(|kv[b, :, h, :]|) / 127   (0 -> scale 1 to avoid div-by-0)
    q = clip(round(kv / scale), -127, 127)

    Per-(block, head) granularity is the sweet spot used by the CUDA kernel:
    coarse enough that scales fit in registers/smem, fine enough to bound
    quantization error per attention head.
    """
    kv = np.asarray(kv, dtype=np.float32)
    absmax = np.abs(kv).max(axis=(1, 3))                       # [num_blocks, num_kv_heads]
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.round(kv / scale[:, None, :, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_kv_per_block_ref(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_kv_per_block_ref`."""
    return (q.astype(np.float32) * scale[:, None, :, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Component B — paged attention (decode step, single query token per seq)
# ---------------------------------------------------------------------------

def dense_attention_ref(
    q: np.ndarray,      # [num_heads, head_dim]
    k: np.ndarray,      # [seq_len, num_kv_heads, head_dim]
    v: np.ndarray,      # [seq_len, num_kv_heads, head_dim]
    scale: float,
) -> np.ndarray:
    """Textbook single-token attention over a dense KV, for cross-checking
    the paged reference. Supports grouped-query attention (num_heads a
    multiple of num_kv_heads)."""
    num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    out = np.empty((num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        kvh = h // group
        logits = (k[:, kvh, :] @ q[h]) * scale                 # [seq_len]
        logits = logits - logits.max()                         # stable softmax
        p = np.exp(logits)
        p /= p.sum()
        out[h] = p @ v[:, kvh, :]
    return out


def paged_attention_ref(
    q: np.ndarray,            # [num_seqs, num_heads, head_dim] float32
    k_cache_q: np.ndarray,    # [num_blocks, block_size, num_kv_heads, head_dim] int8
    k_scale: np.ndarray,      # [num_blocks, num_kv_heads] float32
    v_cache_q: np.ndarray,    # int8, same layout as k_cache_q
    v_scale: np.ndarray,      # [num_blocks, num_kv_heads] float32
    block_tables: np.ndarray, # [num_seqs, max_blocks_per_seq] int32 (-1 = unused)
    seq_lens: np.ndarray,     # [num_seqs] int32
    sm_scale: float,
) -> np.ndarray:
    """Decode-step paged attention over an INT8-quantized paged KV cache.

    Dequantizes per (block, head) with the stored scales, gathers each
    sequence's logical KV via its block table, then runs numerically stable
    softmax attention. This is exactly the computation the fused CUDA kernel
    performs in one pass with on-the-fly dequantization.
    """
    num_seqs, num_heads, head_dim = q.shape
    block_size = k_cache_q.shape[1]
    out = np.empty((num_seqs, num_heads, head_dim), dtype=np.float32)

    for s in range(num_seqs):
        L = int(seq_lens[s])
        n_blocks = (L + block_size - 1) // block_size
        k_parts, v_parts = [], []
        for i in range(n_blocks):
            b = int(block_tables[s, i])
            assert b >= 0, "block table underflow"
            k_parts.append(k_cache_q[b].astype(np.float32) * k_scale[b][None, :, None])
            v_parts.append(v_cache_q[b].astype(np.float32) * v_scale[b][None, :, None])
        k = np.concatenate(k_parts, axis=0)[:L]                # [L, num_kv_heads, head_dim]
        v = np.concatenate(v_parts, axis=0)[:L]
        out[s] = dense_attention_ref(q[s], k, v, sm_scale)
    return out
