# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Component A: fused CUDA eviction-scoring kernel (`csrc/eviction_score.cu`)
  + NumPy reference implementation + correctness tests.
- Component B: INT8 quantized paged-attention decode kernel
  (`csrc/quant_paged_attention.cu`), online-softmax, GQA support + reference
  implementation + correctness tests.
- Per-(block, head) symmetric INT8 quantization (`pagedkv_fusion/quantize.py`)
  with a bounded round-trip-error test.
- CPU/GPU dispatch layer (`pagedkv_fusion/ops.py`) — reference fallback
  when the compiled extension or a CUDA device isn't available.
- vLLM opt-in decode attention backend + patch script
  (`integration/vllm/`) — not yet exercised inside a real vLLM process.
- Benchmarks: eviction latency, quantization memory/quality,
  downstream-accuracy proxy (`benchmarks/`).
- End-to-end integration demo (`scripts/run_end_to_end_demo.py`) — runs the
  full eviction → quantize → attention pipeline as one composed system on
  the reference path.
- Nsight Compute/Systems profiling wrapper (`scripts/profile_kernels.py`).
- CI (CPU job active; GPU job templated for a self-hosted runner),
  Makefile, Dockerfile for reproducible CUDA builds.
- `docs/VALIDATION_REPORT.md` — every performance/quality claim labeled by
  what was actually measured vs. what's pending GPU access.

### Known gaps (tracked in docs/VALIDATION_REPORT.md)
- Kernels have never been compiled or run on a GPU in this development
  environment — `tests/test_kernels_gpu.py` and the GPU sections of the
  benchmarks are written but unexecuted.
- No Nsight profiles collected yet.
- vLLM integration untested inside an actual vLLM process; no vLLM
  version/commit pinned yet.
- Downstream-quality benchmark is a synthetic classification proxy, not
  real model perplexity (requires a GPU + model checkpoint download,
  neither available here).
