"""Component B correctness: quantized paged-attention reference.

Strategy (each layer validates the one below it):
  1. paged gather + dense attention == direct dense attention (layout logic)
  2. INT8-quantized paged attention ~= FP32 paged attention (bounded error)
  3. output is a convex combination of V rows (softmax invariants)
"""

from __future__ import annotations

import numpy as np

from pagedkv_fusion import ops, reference
from pagedkv_fusion.testing_utils import fp32_paged_attention, make_paged_kv_problem


def test_paged_layout_matches_dense_gather(rng):
    """Shuffled block tables must reproduce exactly the same computation as
    manually gathering each sequence's KV and running dense attention."""
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(rng, kv_scale=0.5)
    kq, ks = reference.quantize_kv_per_block_ref(k)
    vq, vs = reference.quantize_kv_per_block_ref(v)
    out = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)

    block_size = k.shape[1]
    k_dq = reference.dequantize_kv_per_block_ref(kq, ks)
    v_dq = reference.dequantize_kv_per_block_ref(vq, vs)
    for s in range(q.shape[0]):
        L = int(seq_lens[s])
        n = (L + block_size - 1) // block_size
        blocks = bt[s, :n]
        k_seq = k_dq[blocks].reshape(-1, *k.shape[2:])[:L]
        v_seq = v_dq[blocks].reshape(-1, *v.shape[2:])[:L]
        expected = reference.dense_attention_ref(q[s], k_seq, v_seq, sm)
        np.testing.assert_allclose(out[s], expected, rtol=1e-4, atol=1e-5)


def test_int8_close_to_fp32(rng):
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(
        rng, num_seqs=6, max_seq_len=256)
    kq, ks = reference.quantize_kv_per_block_ref(k)
    vq, vs = reference.quantize_kv_per_block_ref(v)
    out_q = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)
    out_fp = fp32_paged_attention(q, k, v, bt, seq_lens, sm)

    rmse = float(np.sqrt(np.mean((out_q - out_fp) ** 2)))
    max_abs = float(np.abs(out_q - out_fp).max())
    assert rmse < 0.02, f"INT8 RMSE too high: {rmse}"
    assert max_abs < 0.15, f"INT8 max abs error too high: {max_abs}"


def test_output_in_v_convex_hull_per_head(rng):
    """Softmax attention output for each head lies within [min, max] of that
    head's V rows — a cheap invariant that catches normalization bugs."""
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(rng, num_seqs=3)
    kq, ks = reference.quantize_kv_per_block_ref(k)
    vq, vs = reference.quantize_kv_per_block_ref(v)
    out = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)
    v_dq = reference.dequantize_kv_per_block_ref(vq, vs)
    block_size = v.shape[1]
    num_kv_heads = v.shape[2]
    group = q.shape[1] // num_kv_heads

    for s in range(q.shape[0]):
        L = int(seq_lens[s])
        n = (L + block_size - 1) // block_size
        v_seq = v_dq[bt[s, :n]].reshape(-1, num_kv_heads, v.shape[3])[:L]
        for h in range(q.shape[1]):
            rows = v_seq[:, h // group, :]
            assert (out[s, h] <= rows.max(axis=0) + 1e-4).all()
            assert (out[s, h] >= rows.min(axis=0) - 1e-4).all()


def test_seq_len_one_and_partial_blocks(rng):
    """Edge cases: single-token sequence, and lengths straddling block
    boundaries (block_size-1, block_size, block_size+1)."""
    block_size = 16
    for L in (1, block_size - 1, block_size, block_size + 1):
        q, k, v, bt, seq_lens, sm = make_paged_kv_problem(
            rng, num_seqs=1, max_seq_len=L, block_size=block_size)
        seq_lens[:] = L
        kq, ks = reference.quantize_kv_per_block_ref(k)
        vq, vs = reference.quantize_kv_per_block_ref(v)
        out = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)
        out_fp = fp32_paged_attention(q, k, v, bt, seq_lens, sm)
        np.testing.assert_allclose(out, out_fp, atol=0.15)
        assert np.isfinite(out).all()


def test_ops_dispatch_matches_reference(rng):
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(rng)
    kq, ks = reference.quantize_kv_per_block_ref(k)
    vq, vs = reference.quantize_kv_per_block_ref(v)
    via_ops = ops.quant_paged_attention(q, kq, ks, vq, vs, bt, seq_lens, sm)
    via_ref = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)
    np.testing.assert_allclose(via_ops, via_ref, rtol=0, atol=0)
