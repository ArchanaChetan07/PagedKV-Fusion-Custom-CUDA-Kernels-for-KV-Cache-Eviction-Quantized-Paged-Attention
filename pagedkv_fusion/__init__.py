"""PagedKV-Fusion: custom CUDA kernels for KV-cache eviction scoring and
INT8 quantized paged attention, with a pluggable vLLM backend.

Public API::

    from pagedkv_fusion import ops
    scores = ops.eviction_scores(recency, frequency, attn_weights, valid_mask)
    out = ops.quant_paged_attention(q, k_q, k_s, v_q, v_s, block_tables, seq_lens, sm_scale)

``ops`` dispatches to the compiled CUDA extension when available and inputs
are on GPU; otherwise it runs the (slow, correct) reference path so the
package is usable and testable on CPU-only machines and CI runners.
"""

__version__ = "0.1.0"

from . import reference  # noqa: F401

__all__ = ["reference", "__version__"]
