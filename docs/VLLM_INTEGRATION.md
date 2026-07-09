# vLLM Integration Notes

## Status

Adapter code (`integration/vllm/pagedkv_fusion_backend.py`,
`integration/vllm/patch_vllm.py`) is written and unit-testable in isolation
(the class methods call straight into `pagedkv_fusion.ops`, which is tested
on CPU). **It has not been run inside an actual vLLM process** — that
requires a GPU and a real vLLM checkout, neither available in the
environment this was built in. See `docs/VALIDATION_REPORT.md` §6 for the
concrete steps to close this gap.

## Why a patch script instead of a fork

vLLM's attention-backend selection internals move across releases. Rather
than committing to a pinned fork that goes stale, `patch_vllm.py` does the
minimal registration patch against a user-supplied checkout, and fails
loudly (not silently) if the expected hook has moved. If it fails on your
version, the manual steps are:

1. Find wherever your vLLM version dispatches on `--attention-backend` /
   `VLLM_ATTENTION_BACKEND` (commonly `vllm/attention/selector.py` or
   `vllm/attention/backends/selector.py`).
2. Import `PagedKVFusionBackend` from `pagedkv_fusion_backend.py` (put it on
   `PYTHONPATH` or `pip install -e` this repo inside the vLLM venv).
3. Register it as an option alongside the existing backends, wrapping
   whatever the default decode backend is as `fallback_backend=`.

## Scope of the v1 backend

- **Decode only.** `supports_prefill = False` — prefill requests fall
  through to the wrapped default backend. Fusing the eviction kernel and a
  prefill-capable INT8 attention kernel into one backend is future work.
- **No fused write-quantize kernel.** `write_to_cache` quantizes newly
  computed K/V in Python (via `pagedkv_fusion.quantize`) before scattering
  into the paged cache. This is correct but not fused — a production
  version would push the quantize-and-scatter into a single kernel launch.
  Flagged, not hidden: this is a real (unmeasured) latency cost of the v1
  integration that a full before/after serving benchmark would need to
  account for.
- **Single GPU.** No tensor/pipeline parallel awareness — matches the
  project plan's non-goals for v1.

## Version this was written against

Not yet pinned to a specific vLLM commit/tag — do this as the first step of
running §6 in the validation report, and record the exact commit here once
verified, along with any signature adjustments the patch script needed.
