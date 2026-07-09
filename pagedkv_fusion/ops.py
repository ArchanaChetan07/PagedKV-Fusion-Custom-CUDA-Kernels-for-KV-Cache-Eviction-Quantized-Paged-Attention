"""Dispatch layer: CUDA kernels when available, reference path otherwise.

Resolution order for each op:
  1. Compiled extension (``pagedkv_fusion._C``) if importable AND all tensor
     inputs are CUDA tensors.
  2. Reference implementation (NumPy via :mod:`pagedkv_fusion.reference`,
     round-tripped through torch if inputs are torch tensors).

This keeps the package importable and fully testable on CPU-only machines
(laptops, CPU CI runners) while transparently using the fast path on GPU.
``backend_in_use()`` reports which path is live, and benchmarks assert on it
so we never accidentally benchmark the fallback and call it a kernel.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import reference

try:  # torch is an optional dependency for the reference-only install
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

_C = None
if _HAS_TORCH:
    try:
        from pagedkv_fusion import _C  # type: ignore[attr-defined, no-redef]
    except ImportError:
        _C = None


def backend_in_use() -> str:
    """'cuda' if the compiled extension is loaded, else 'reference'."""
    return "cuda" if _C is not None else "reference"


def _is_cuda(*tensors: Any) -> bool:
    return (
        _HAS_TORCH
        and all(isinstance(t, torch.Tensor) for t in tensors)
        and all(t.is_cuda for t in tensors)
    )


def _to_numpy(t: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Component A — eviction scoring
# ---------------------------------------------------------------------------

def eviction_scores(
    recency,
    frequency,
    attn_weights,
    valid_mask,
    w_recency: float = 0.4,
    w_frequency: float = 0.3,
    w_attention: float = 0.3,
):
    """Fused eviction score per KV block (lower = evict first).

    See :func:`pagedkv_fusion.reference.eviction_scores_ref` for semantics.
    """
    if _C is not None and _is_cuda(recency, frequency, attn_weights, valid_mask):
        return _C.eviction_scores(
            recency, frequency, attn_weights, valid_mask,
            w_recency, w_frequency, w_attention,
        )
    out = reference.eviction_scores_ref(
        _to_numpy(recency), _to_numpy(frequency),
        _to_numpy(attn_weights), _to_numpy(valid_mask),
        w_recency, w_frequency, w_attention,
    )
    if _HAS_TORCH and isinstance(recency, torch.Tensor):
        return torch.from_numpy(out).to(recency.device)
    return out


def select_eviction_candidates(scores, k: int):
    """Indices of the k lowest-scoring blocks.

    On GPU this is a device-side topk (no host transfer); on CPU it defers
    to the deterministic reference selection.
    """
    if _HAS_TORCH and isinstance(scores, torch.Tensor) and scores.is_cuda:
        return torch.topk(scores, k, largest=False, sorted=True).indices
    return reference.select_eviction_candidates_ref(_to_numpy(scores), k)


# ---------------------------------------------------------------------------
# Component B — quantized paged attention
# ---------------------------------------------------------------------------

def quant_paged_attention(
    q,
    k_cache_q,
    k_scale,
    v_cache_q,
    v_scale,
    block_tables,
    seq_lens,
    sm_scale: float,
):
    """Decode-step paged attention over an INT8 paged KV cache.

    See :func:`pagedkv_fusion.reference.paged_attention_ref` for semantics.
    """
    if _C is not None and _is_cuda(q, k_cache_q, k_scale, v_cache_q, v_scale,
                                   block_tables, seq_lens):
        return _C.quant_paged_attention(
            q, k_cache_q, k_scale, v_cache_q, v_scale,
            block_tables, seq_lens, sm_scale,
        )
    if _C is None and _is_cuda(q):  # pragma: no cover
        warnings.warn(
            "pagedkv_fusion._C not built; falling back to slow reference "
            "path on CPU. Build the extension for GPU execution.",
            stacklevel=2,
        )
    out = reference.paged_attention_ref(
        _to_numpy(q).astype(np.float32),
        _to_numpy(k_cache_q), _to_numpy(k_scale),
        _to_numpy(v_cache_q), _to_numpy(v_scale),
        _to_numpy(block_tables), _to_numpy(seq_lens),
        sm_scale,
    )
    if _HAS_TORCH and isinstance(q, torch.Tensor):
        return torch.from_numpy(out).to(q.device)
    return out
