# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.1] — 2026-07-09

### Fixed
- CUDA build: add missing `ATen/cuda/CUDAContext.h` and `c10/cuda/CUDAException.h`
  includes so kernels compile against PyTorch 2.6.
- `setup.py`: Windows MSVC compile flags (`/O2`), skip hard CUDA errors during
  pip metadata phase, support `pip install --no-build-isolation` for CUDA builds.
- `Makefile`: `install-cuda` uses `--no-build-isolation` so torch is available
  at extension compile time.
- `bench_quant_attention.py`: progress output, `--speed-iters` flag, adaptive
  SDPA iteration cap when the fp16 baseline is pathologically slow on consumer GPUs.
- `profile_kernels.py`: Windows `ncu.exe`/`nsys.exe` discovery, correct
  `--force-overwrite` syntax, actionable error on `ERR_NVGPUCTRPERM`.
- `run_end_to_end_demo.py`: backend status message reflects CUDA when built.

### Verified (GPU — NVIDIA T1000 8GB, CUDA 12.5, PyTorch 2.6.0+cu124)
- `tests/test_kernels_gpu.py`: 10/10 kernel-vs-reference tests pass.
- Full test suite: 26/26 pass (CPU + GPU).
- End-to-end demo runs on CUDA backend.
- Eviction kernel: ~3× faster than host baseline at 16,384 blocks (p50 562 µs vs
  1,708 µs). See `results/eviction_bench_gpu_sections.json`.
- INT8 paged attention: 1.8–14 ms p50 vs 9–27,320 ms gathered SDPA baseline
  depending on batch/sequence length. See `results/quant_bench_gpu_sections.json`.
- Evidence and reproduction steps updated in `docs/VALIDATION_REPORT.md` §7.

### Known gaps (unchanged)
- Nsight Compute profiles not collected (`ERR_NVGPUCTRPERM` on development GPU).
- vLLM integration untested inside an actual vLLM process.
- Downstream-quality benchmark remains a synthetic proxy, not real perplexity.

## [0.1.0] — initial release

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
