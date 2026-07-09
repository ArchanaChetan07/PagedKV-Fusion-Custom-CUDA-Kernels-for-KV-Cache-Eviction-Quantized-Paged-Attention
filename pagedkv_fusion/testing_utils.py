"""Synthetic paged-KV problem generators and baseline computations shared by
the test suite and the benchmark scripts.

Kept in the library (not under tests/) so benchmarks don't import test code,
and so external users validating the kernels against their own data have a
documented reference for what a "problem" looks like.
"""

from __future__ import annotations

import numpy as np

from . import reference

__all__ = ["make_paged_kv_problem", "fp32_paged_attention"]


def make_paged_kv_problem(
    rng: np.random.Generator,
    num_seqs: int = 4,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    block_size: int = 16,
    max_seq_len: int = 200,
    kv_scale: float = 1.0,
):
    """Random decode-attention problem with a non-trivial paged layout.

    Physical blocks are allocated in a *shuffled* order so tests/benchmarks
    exercise real block-table indirection, not just contiguous layouts.
    """
    seq_lens = rng.integers(1, max_seq_len + 1, size=num_seqs).astype(np.int32)
    blocks_per_seq = (seq_lens + block_size - 1) // block_size
    total_blocks = int(blocks_per_seq.sum()) + 3  # a few unallocated blocks
    max_bps = int(blocks_per_seq.max())

    perm = rng.permutation(total_blocks)
    block_tables = np.full((num_seqs, max_bps), -1, dtype=np.int32)
    cursor = 0
    for s in range(num_seqs):
        n = int(blocks_per_seq[s])
        block_tables[s, :n] = perm[cursor:cursor + n]
        cursor += n

    q = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float32)
    k = (kv_scale * rng.standard_normal(
        (total_blocks, block_size, num_kv_heads, head_dim))).astype(np.float32)
    v = (kv_scale * rng.standard_normal(
        (total_blocks, block_size, num_kv_heads, head_dim))).astype(np.float32)
    sm_scale = 1.0 / np.sqrt(head_dim)
    return q, k, v, block_tables, seq_lens, sm_scale


def fp32_paged_attention(q, k, v, block_tables, seq_lens, sm_scale):
    """Full-precision paged-attention baseline (no quantization), built
    from the dense reference. Used as the "ground truth" quality baseline
    that INT8 output is compared against in tests and benchmarks."""
    block_size = k.shape[1]
    out = np.empty_like(q)
    for s in range(q.shape[0]):
        L = int(seq_lens[s])
        n = (L + block_size - 1) // block_size
        blocks = block_tables[s, :n]
        k_seq = k[blocks].reshape(-1, *k.shape[2:])[:L]
        v_seq = v[blocks].reshape(-1, *v.shape[2:])[:L]
        out[s] = reference.dense_attention_ref(q[s], k_seq, v_seq, sm_scale)
    return out
