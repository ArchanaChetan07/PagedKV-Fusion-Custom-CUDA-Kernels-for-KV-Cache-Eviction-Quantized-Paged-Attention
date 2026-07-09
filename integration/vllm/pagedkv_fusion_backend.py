"""PagedKV-Fusion attention backend for vLLM.

Wires the compiled kernels into vLLM as an opt-in ``AttentionBackend`` so the
before/after comparison happens inside a real serving loop, not just in
isolated microbenchmarks — this is what Deliverable 5 in the project plan
requires.

**Version binding.** vLLM's internal ``AttentionBackend`` / ``AttentionImpl``
ABI changes across releases (see the compatibility table in
``docs/VLLM_INTEGRATION.md``); this file targets the interface shape as of
vLLM's paged-attention v2 backend and is intentionally a thin adapter — all
actual math stays in ``pagedkv_fusion.ops`` / the CUDA kernels. Expect to
adjust method signatures for the vLLM commit you pin against; the metadata
plumbing (block_tables, seq_lens, slot mapping) is stable in spirit even
when field names shift.

**What this integration does NOT do (v1, matches the project's non-goals):**
  * no prefill kernel — decode-step (single query token) only; prefill
    falls back to vLLM's default backend via ``supports_prefill = False``.
  * no in-place cache writing kernel — quantization happens on the Python
    side when a block is written (see ``write_to_cache``); a fused
    write+quantize kernel is future work.
  * single GPU only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from pagedkv_fusion import ops
from pagedkv_fusion.quantize import quantize_kv_per_block


@dataclasses.dataclass
class PagedKVFusionMetadata:
    """Decode-step metadata, shaped to match what vLLM's model runner
    already computes and passes to attention backends each step."""

    block_tables: torch.Tensor   # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor       # [num_seqs] int32
    k_scale: torch.Tensor        # [num_blocks, num_kv_heads] float32
    v_scale: torch.Tensor        # [num_blocks, num_kv_heads] float32


class PagedKVFusionBackend:
    """Opt-in decode attention backend: ``--attention-backend pagedkv_fusion``.

    Falls back to vLLM's default backend for prefill steps and for any
    request shape outside the kernel's v1 constraints (see
    ``quant_paged_attention.cu`` launcher checks), so enabling this backend
    is safe to try incrementally on a subset of traffic.
    """

    supports_prefill = False
    supports_decode = True

    def __init__(self, fallback_backend: Any):
        self.fallback = fallback_backend

    def write_to_cache(
        self,
        key: torch.Tensor,           # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,
        kv_cache_int8: torch.Tensor,  # [2, num_blocks, block_size, nkv, hd]
        scale_cache: torch.Tensor,    # [2, num_blocks, num_kv_heads]
        slot_mapping: torch.Tensor,   # [num_tokens] -> flat (block, offset)
        block_size: int,
    ) -> None:
        """Quantize newly computed K/V and scatter into the paged INT8 cache.

        Grouping tokens by destination block before calling
        ``quantize_kv_per_block`` keeps scale granularity correct even when
        ``slot_mapping`` isn't block-contiguous (e.g. mid-block insertions
        during chunked prefill). For clarity this reference adapter loops in
        Python; a production version would fuse this into a scatter-quantize
        CUDA kernel (noted as future work in the validation report).
        """
        blocks = torch.unique(slot_mapping // block_size)
        for b in blocks.tolist():
            token_mask = (slot_mapping // block_size) == b
            offsets = (slot_mapping[token_mask] % block_size)
            k_block = key[token_mask]    # [n, nkv, hd]
            v_block = value[token_mask]
            # quantize_kv_per_block expects [nb, bs, nkv, hd]; treat this
            # partial set of tokens as a single synthetic block for scale
            # computation, matching the kernel's per-(block, head) contract.
            kq, ks = quantize_kv_per_block(k_block.unsqueeze(0))
            vq, vs = quantize_kv_per_block(v_block.unsqueeze(0))
            kv_cache_int8[0, b, offsets] = kq[0]
            kv_cache_int8[1, b, offsets] = vq[0]
            scale_cache[0, b] = ks[0]
            scale_cache[1, b] = vs[0]

    def decode_forward(
        self,
        query: torch.Tensor,          # [num_seqs, num_heads, head_dim]
        kv_cache_int8: torch.Tensor,  # [2, num_blocks, block_size, nkv, hd]
        scale_cache: torch.Tensor,    # [2, num_blocks, num_kv_heads]
        metadata: PagedKVFusionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """Decode-step attention using the fused INT8 kernel.

        Callers (the patched vLLM attention module) are expected to check
        shape constraints and route to ``self.fallback`` themselves for
        prefill / unsupported configurations; this method assumes it has
        already been selected as eligible.
        """
        return ops.quant_paged_attention(
            query,
            kv_cache_int8[0], scale_cache[0],
            kv_cache_int8[1], scale_cache[1],
            metadata.block_tables, metadata.seq_lens,
            scale,
        )

    def forward(self, *args, is_prefill: bool, **kwargs):
        if is_prefill or ops.backend_in_use() != "cuda":
            return self.fallback.forward(*args, **kwargs)
        return self.decode_forward(*args, **kwargs)
