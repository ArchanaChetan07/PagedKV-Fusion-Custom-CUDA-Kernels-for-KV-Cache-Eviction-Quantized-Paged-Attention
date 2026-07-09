# PagedKV-Fusion

Custom CUDA kernels for KV-cache eviction scoring and INT8 quantized paged
attention, integrated as a pluggable vLLM attention backend.

This project extends [KV-Cache-Profiler](#) from the Python/API layer down
into the CUDA/C++ kernel layer: replacing a host-side eviction heuristic
with a fused GPU kernel, and adding a quantized paged-attention kernel with
measured memory/throughput/quality tradeoffs.

**Honesty note up front:** reference-path correctness, quantization quality, and
memory arithmetic are fully tested on CPU. **CUDA kernels are now built,
correctness-verified, and benchmarked on real hardware** (NVIDIA T1000, CUDA
12.5 — see [`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) §7).
Nsight profiles and in-process vLLM integration remain pending. Every
performance number in this README links to a labeled source in the validation
report — read that before citing latency claims.

## What's here

| Component | Files | Status |
|---|---|---|
| **A — Eviction-scoring kernel** | `csrc/eviction_score.cu` | ✅ GPU correctness verified; ~3× faster than host at 16K blocks (T1000) |
| **B — INT8 paged-attention kernel** | `csrc/quant_paged_attention.cu` | ✅ GPU correctness verified; 5–1900× faster than gathered SDPA baseline |
| **C — vLLM integration** | `integration/vllm/` | Written; not yet run inside vLLM |
| Reference implementations | `pagedkv_fusion/reference.py`, `quantize.py` | ✅ Tested (CPU) |
| CPU/GPU dispatch layer | `pagedkv_fusion/ops.py` | ✅ Tested (CPU + CUDA dispatch) |
| Test suite | `tests/` | ✅ 26 passed (CPU + GPU kernel gates) |
| End-to-end pipeline demo | `scripts/run_end_to_end_demo.py` | ✅ Runs on CUDA backend end-to-end |
| Benchmarks (latency/memory/quality/downstream-proxy) | `benchmarks/` | ✅ CPU + GPU sections executed |
| Profiling wrapper | `scripts/profile_kernels.py` | ✅ CLI fixed; Nsight blocked on GPU counter permissions (see report) |
| CI | `.github/workflows/ci.yml` | CPU job runs on every PR; GPU job needs a self-hosted runner |
| Reproducible CUDA build | `docker/Dockerfile.cuda`, `Makefile` | ✅ Built on Windows + CUDA 12.5 |

## Quickstart

```bash
git clone <this-repo>
cd pagedkv-fusion
make install
make test
make demo         # runs eviction -> quantize -> attention as one real pipeline
```

This installs and tests the **reference path** on any machine. With a CUDA
toolchain, use `make install-cuda` and `make test-gpu` to build and verify
the kernels (see below).

### Building the CUDA kernels (requires an NVIDIA GPU + CUDA toolkit)

```bash
make install-cuda   # torch + build the extension
make test-gpu        # kernel-vs-reference correctness — the gate that matters for csrc/ changes
```

Or reproducibly, via Docker (see `docker/Dockerfile.cuda`):

```bash
make docker-build
make docker-test    # runs test-gpu inside the container with --gpus all
```

### Benchmarks

```bash
make bench           # eviction latency baseline + quantization memory/quality
make bench-quality   # downstream-decision proxy (see disclaimer in its output)
```

### Profiling (GPU + Nsight required)

```bash
make profile
```

## Repository layout

```
csrc/                       CUDA/C++ kernel sources + pybind11 bindings
pagedkv_fusion/             Python package: reference impls, dispatch, quantization, test helpers
tests/                      pytest suite (CPU-only + GPU-gated)
benchmarks/                 Latency/memory/quality/downstream-proxy measurement scripts
scripts/
  run_end_to_end_demo.py    Runs the full pipeline as one composed, executed system
  profile_kernels.py        Nsight Compute/Systems profiling wrapper
integration/vllm/           Opt-in vLLM attention backend + patch script
docker/Dockerfile.cuda      Reproducible CUDA build/test environment
docs/
  VALIDATION_REPORT.md      What's actually been verified, and how to reproduce it
  VLLM_INTEGRATION.md       Integration scope, version notes, manual patch steps
results/                    Tracked benchmark outputs backing the validation report
Makefile                    install / lint / test / demo / bench / profile / docker-*
.github/workflows/ci.yml    CPU tests on every push; GPU job on self-hosted runner
CONTRIBUTING.md             Dev loop, PR expectations (esp. re: reporting real GPU numbers)
CHANGELOG.md                What's done vs. known gaps, kept current per release
```

## Design summary

- **Eviction scoring** (`csrc/eviction_score.cu`): one CUDA block per KV
  page, two-level (warp-shuffle + shared-memory) reduction over the page's
  attention row, fused recency/frequency/attention scoring in a single
  kernel launch — no host round-trip per eviction decision.
- **Quantized paged attention** (`csrc/quant_paged_attention.cu`):
  online-softmax (flash-decoding style) decode kernel reading per-(block,
  head) INT8-quantized KV pages, dequantizing on the fly. Supports
  grouped-query attention.
- **Quantization** (`pagedkv_fusion/quantize.py`): symmetric per-(block,
  head) INT8, scale = max(|x|)/127. See the validation report for measured
  error on Gaussian vs. heavy-tailed synthetic data — including the honest
  negative result on outlier-heavy distributions.

## Non-goals (v1)

Matches the project plan: no full upstream vLLM merge, no AWQ/GPTQ, no
multi-GPU/multi-node, no fused prefill kernel, no fused write+quantize
kernel. See `docs/VLLM_INTEGRATION.md` for what the integration layer does
and doesn't cover.
