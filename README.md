# PagedKV-Fusion

Custom CUDA kernels for KV-cache eviction scoring and INT8 quantized paged
attention, integrated as a pluggable vLLM attention backend.

This project extends [KV-Cache-Profiler](#) from the Python/API layer down
into the CUDA/C++ kernel layer: replacing a host-side eviction heuristic
with a fused GPU kernel, and adding a quantized paged-attention kernel with
measured memory/throughput/quality tradeoffs.

**Honesty note up front:** this repo was built in a CPU-only environment.
Kernel *design*, *reference correctness*, and every CPU-measurable claim
(quantization error, memory savings) are done and tested. Kernel *execution
on real hardware* — latency numbers, Nsight profiles, and the vLLM
end-to-end run — is written and ready to go but has not yet been run on a
GPU. **[`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) states
exactly what's verified vs. pending, with every number labeled by source.**
Read that before trusting any performance claim about this project.

## What's here

| Component | Files | Status |
|---|---|---|
| **A — Eviction-scoring kernel** | `csrc/eviction_score.cu` | Written; reference-verified; GPU run pending |
| **B — INT8 paged-attention kernel** | `csrc/quant_paged_attention.cu` | Written; reference-verified; GPU run pending |
| **C — vLLM integration** | `integration/vllm/` | Written; not yet run inside vLLM |
| Reference implementations | `pagedkv_fusion/reference.py`, `quantize.py` | ✅ Tested (CPU) |
| CPU/GPU dispatch layer | `pagedkv_fusion/ops.py` | ✅ Tested (CPU); CUDA path pending GPU |
| Test suite | `tests/` | ✅ 14 passed, 3 skipped (torch/GPU-gated) |
| End-to-end pipeline demo | `scripts/run_end_to_end_demo.py` | ✅ Actually executed — proves the 3 components compose, not just unit-pass |
| Benchmarks (latency/memory/quality/downstream-proxy) | `benchmarks/` | ✅ CPU-runnable sections executed; GPU speed sections pending |
| Profiling wrapper | `scripts/profile_kernels.py` | Written; pending Nsight/GPU |
| CI | `.github/workflows/ci.yml` | CPU job runs on every PR; GPU job needs a self-hosted runner |
| Reproducible CUDA build | `docker/Dockerfile.cuda`, `Makefile` | Written; not built/run here (no Docker/GPU in this sandbox) |

## Quickstart

```bash
git clone <this-repo>
cd pagedkv-fusion
make install
make test
make demo         # runs eviction -> quantize -> attention as one real pipeline
```

This installs and tests the **reference path only** — correct on any
machine, including yours right now. You'll see `3 skipped`: those are the
torch- and CUDA-gated tests, skipping because torch/a GPU aren't present,
not because anything is broken. `make demo` is worth running: it's not a
unit test, it's the three components actually wired together and executed.

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
