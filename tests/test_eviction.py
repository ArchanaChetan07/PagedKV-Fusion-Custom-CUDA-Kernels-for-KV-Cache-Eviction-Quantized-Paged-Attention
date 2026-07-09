"""Component A correctness: eviction scoring reference + dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from pagedkv_fusion import ops, reference


def _random_problem(rng, num_blocks=64, block_size=16):
    recency = rng.random(num_blocks, dtype=np.float32)
    frequency = rng.random(num_blocks, dtype=np.float32)
    attn = rng.random((num_blocks, block_size), dtype=np.float32)
    mask = rng.random((num_blocks, block_size)) > 0.2
    return recency, frequency, attn, mask


def test_scores_match_manual_formula(rng):
    recency, frequency, attn, mask = _random_problem(rng)
    scores = reference.eviction_scores_ref(recency, frequency, attn, mask,
                                           0.4, 0.3, 0.3)
    # Independent, per-block recomputation.
    for b in range(len(scores)):
        valid = mask[b]
        mean_attn = attn[b][valid].mean() if valid.any() else 0.0
        expected = 0.4 * recency[b] + 0.3 * frequency[b] + 0.3 * mean_attn
        np.testing.assert_allclose(scores[b], expected, rtol=1e-5)


def test_empty_block_gets_zero_attention_term(rng):
    recency = np.array([0.5], dtype=np.float32)
    frequency = np.array([0.5], dtype=np.float32)
    attn = np.ones((1, 16), dtype=np.float32)
    mask = np.zeros((1, 16), dtype=bool)  # fully invalid block
    scores = reference.eviction_scores_ref(recency, frequency, attn, mask)
    np.testing.assert_allclose(scores[0], 0.4 * 0.5 + 0.3 * 0.5, rtol=1e-6)


def test_cold_unattended_blocks_score_lowest(rng):
    """Sanity: an old, rarely used, un-attended block must be the top
    eviction candidate — the property the whole heuristic exists for."""
    recency = np.array([0.05, 0.9, 0.8], dtype=np.float32)
    frequency = np.array([0.05, 0.8, 0.9], dtype=np.float32)
    attn = np.stack([
        np.full(16, 0.01), np.full(16, 0.5), np.full(16, 0.6),
    ]).astype(np.float32)
    mask = np.ones((3, 16), dtype=bool)
    scores = reference.eviction_scores_ref(recency, frequency, attn, mask)
    assert np.argmin(scores) == 0


def test_selection_is_deterministic_under_ties():
    scores = np.array([0.3, 0.1, 0.1, 0.2], dtype=np.float32)
    idx = reference.select_eviction_candidates_ref(scores, 3)
    np.testing.assert_array_equal(idx, [1, 2, 3])  # tie -> lower index first


def test_ops_dispatch_matches_reference_on_numpy(rng):
    recency, frequency, attn, mask = _random_problem(rng)
    via_ops = ops.eviction_scores(recency, frequency, attn, mask)
    via_ref = reference.eviction_scores_ref(recency, frequency, attn, mask)
    np.testing.assert_allclose(via_ops, via_ref, rtol=0, atol=0)


def test_ops_dispatch_roundtrips_torch_tensors(rng):
    torch = pytest.importorskip("torch")
    recency, frequency, attn, mask = _random_problem(rng)
    out = ops.eviction_scores(
        torch.from_numpy(recency), torch.from_numpy(frequency),
        torch.from_numpy(attn), torch.from_numpy(mask),
    )
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(
        out.numpy(),
        reference.eviction_scores_ref(recency, frequency, attn, mask),
        rtol=1e-6,
    )
