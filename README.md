# PagedKV-Fusion

Custom CUDA kernels for KV-cache eviction scoring and INT8 quantized paged
attention, integrated as a pluggable vLLM attention backend.

**v0.1.1** — CUDA kernels built, correctness-verified, and benchmarked on real
hardware. See [`CHANGELOG.md`](CHANGELOG.md) and
[`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) for evidence.

This project extends KV-cache profiling from the Python/API layer down into
the CUDA/C++ kernel layer: replacing a host-side eviction heuristic with a
fused GPU kernel, and adding a quantized paged-attention kernel with measured
memory, throughput, and quality tradeoffs.

## Highlights

| Metric | Result | Source |
|---|---|---|
| Test suite | **26 / 26 passed** (CPU + GPU) | `pytest tests/` |
| Eviction kernel @ 16K blocks | **~3× faster** than host (p50 562 µs vs 1,708 µs) | `results/eviction_bench_gpu_sections.json` |
| INT8 attention @ 32 seqs × 1024 tokens | **~21× faster** than gathered fp16 SDPA (p50 3.3 ms vs 70 ms) | `results/quant_bench_gpu_sections.json` |
| KV memory (INT8 vs fp16) | **~50% savings** (exact arithmetic) | `results/quant_bench_cpu_sections.json` |
| End-to-end pipeline | Eviction → quantize → decode on **CUDA backend** | `scripts/run_end_to_end_demo.py` |

*Benchmarks measured on NVIDIA T1000 8GB, CUDA 12.5, PyTorch 2.6.0+cu124.
Datacenter GPU numbers (A100/L4) are recommended before citing production SLOs.*

## What's here

| Component | Files | Status |
|---|---|---|
| **A — Eviction-scoring kernel** | `csrc/eviction_score.cu` | ✅ GPU verified; ~3× faster than host at 16K blocks |
| **B — INT8 paged-attention kernel** | `csrc/quant_paged_attention.cu` | ✅ GPU verified; 5–1900× faster than gathered SDPA baseline |
| **C — vLLM integration** | `integration/vllm/` | Written; not yet run inside vLLM |
| Reference implementations | `pagedkv_fusion/reference.py`, `quantize.py` | ✅ Tested (CPU) |
| CPU/GPU dispatch layer | `pagedkv_fusion/ops.py` | ✅ Tested (CPU + CUDA dispatch) |
| Test suite | `tests/` | ✅ 26 passed |
| End-to-end pipeline demo | `scripts/run_end_to_end_demo.py` | ✅ Runs on CUDA backend |
| Benchmarks | `benchmarks/` | ✅ CPU + GPU sections executed |
| Profiling wrapper | `scripts/profile_kernels.py` | ✅ CLI fixed; Nsight needs GPU counter permissions |
| CI | `.github/workflows/ci.yml` | CPU on every PR; GPU job for self-hosted runner |
| Reproducible CUDA build | `docker/Dockerfile.cuda`, `Makefile` | ✅ Built on Windows + CUDA 12.5 |

## Quickstart

```bash
git clone https://github.com/ArchanaChetan07/PagedKV-Fusion-Custom-CUDA-Kernels-for-KV-Cache-Eviction-Quantized-Paged-Attention.git
cd PagedKV-Fusion-Custom-CUDA-Kernels-for-KV-Cache-Eviction-Quantized-Paged-Attention
make install
make test
make demo
```

This installs and tests the **reference path** on any machine. With an NVIDIA
GPU and CUDA toolkit:

```bash
make install-cuda    # builds pagedkv_fusion._C
make test-gpu        # kernel-vs-reference correctness gate
make demo            # backend should report: cuda
```

On Windows, if `make` is unavailable, use pip directly:

```powershell
pip install -e ".[dev]"
pytest tests/ -v
$env:PAGEDKV_FORCE_CUDA="1"
pip install -e ".[cuda,dev]" --no-build-isolation
pytest tests/test_kernels_gpu.py -v
python scripts/run_end_to_end_demo.py
```

### Benchmarks

```bash
make bench           # eviction latency + quantization memory/quality
make bench-quality   # downstream-decision proxy (synthetic, not perplexity)
```

Or individually:

```bash
python benchmarks/bench_eviction.py --num-blocks 1024 16384 --out results/eviction_bench.json
python benchmarks/bench_quant_attention.py --speed-iters 10 --out results/quant_bench.json
python benchmarks/bench_downstream_proxy.py --out results/downstream_proxy.json
```

### Profiling (GPU + Nsight required)

```bash
make profile
# or: python scripts/profile_kernels.py both --out results/profiles
```

Requires Nsight Compute on PATH. On Windows, run as Administrator if you
hit `ERR_NVGPUCTRPERM` (GPU performance counter permissions).

### Docker (reproducible CUDA environment)

```bash
make docker-build
make docker-test    # runs test-gpu inside the container with --gpus all
```

## Repository layout

```
csrc/                       CUDA/C++ kernel sources + pybind11 bindings
pagedkv_fusion/             Python package: reference impls, dispatch, quantization
tests/                      pytest suite (CPU + GPU-gated)
benchmarks/                 Latency / memory / quality / downstream-proxy scripts
scripts/
  run_end_to_end_demo.py    Full eviction → quantize → attention pipeline
  profile_kernels.py        Nsight Compute/Systems profiling wrapper
integration/vllm/           Opt-in vLLM attention backend + patch script
docker/Dockerfile.cuda      Reproducible CUDA build/test environment
docs/
  VALIDATION_REPORT.md      What's verified vs. pending, with labeled numbers
  VLLM_INTEGRATION.md       Integration scope and manual patch steps
results/                    Tracked benchmark outputs (CPU + GPU sections)
Makefile                    install / lint / test / demo / bench / profile
```

## Design summary

- **Eviction scoring** (`csrc/eviction_score.cu`): one CUDA block per KV page,
  warp-shuffle + shared-memory reduction, fused recency/frequency/attention
  scoring in a single kernel launch — no host round-trip per eviction decision.
- **Quantized paged attention** (`csrc/quant_paged_attention.cu`):
  online-softmax decode kernel over per-(block, head) INT8 KV pages with
  on-the-fly dequantization. Supports grouped-query attention (GQA).
- **Quantization** (`pagedkv_fusion/quantize.py`): symmetric per-(block, head)
  INT8, scale = max(|x|)/127. See the validation report for measured error on
  Gaussian vs. heavy-tailed distributions.

## Non-goals (v1)

No full upstream vLLM merge, no AWQ/GPTQ, no multi-GPU/multi-node, no fused
prefill kernel, no fused write+quantize kernel. See `docs/VLLM_INTEGRATION.md`.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Citation

If you use this work, please link to the repository and cite the validation
report for any performance numbers:

```
Archana Suresh Patil. PagedKV-Fusion: Custom CUDA Kernels for KV-Cache
Eviction and Quantized Paged Attention. https://github.com/ArchanaChetan07/
PagedKV-Fusion-Custom-CUDA-Kernels-for-KV-Cache-Eviction-Quantized-Paged-Attention
```
