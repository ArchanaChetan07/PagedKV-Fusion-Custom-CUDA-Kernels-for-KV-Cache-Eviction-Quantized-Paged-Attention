"""Per-(block, head) symmetric INT8 quantization for paged KV caches.

Torch mirror of :mod:`pagedkv_fusion.reference` quantization — used on the
hot path to quantize KV blocks as they are written, and by the vLLM backend.
Numerics are bit-identical to the NumPy reference (tested).
"""

from __future__ import annotations

import torch

__all__ = ["quantize_kv_per_block", "dequantize_kv_per_block"]


def quantize_kv_per_block(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a paged KV tensor to INT8 with per-(block, head) scales.

    Args:
        kv: float tensor ``[num_blocks, block_size, num_kv_heads, head_dim]``.

    Returns:
        ``(q, scale)`` where ``q`` is int8 with the same shape and
        ``scale`` is float32 ``[num_blocks, num_kv_heads]``.
    """
    if kv.dim() != 4:
        raise ValueError(f"expected 4D [nb, bs, nkv, hd], got {tuple(kv.shape)}")
    kv = kv.float()
    absmax = kv.abs().amax(dim=(1, 3))                      # [nb, nkv]
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(kv / scale[:, None, :, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.float()


def dequantize_kv_per_block(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_kv_per_block` (up to rounding error)."""
    return q.float() * scale[:, None, :, None].float()
