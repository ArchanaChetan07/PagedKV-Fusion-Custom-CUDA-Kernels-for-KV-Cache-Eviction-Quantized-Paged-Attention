# results/

Curated benchmark outputs, tracked in git so the numbers in
`docs/VALIDATION_REPORT.md` are reproducible without re-running anything —
and so reviewers can diff future runs against a known-good baseline.

| File | Produced by | Contents |
|---|---|---|
| `eviction_bench_cpu_sections.json` | `benchmarks/bench_eviction.py` | Host-side (NumPy) eviction-scoring latency baseline |
| `eviction_bench_gpu_sections.json` | `benchmarks/bench_eviction.py` | CUDA fused kernel vs host latency (T1000, CUDA 12.5) |
| `quant_bench_cpu_sections.json` | `benchmarks/bench_quant_attention.py` | Memory footprint + quantization-quality-vs-FP32 sections |
| `quant_bench_gpu_sections.json` | `benchmarks/bench_quant_attention.py` | INT8 kernel vs gathered fp16 SDPA throughput |
| `downstream_proxy_cpu.json` | `benchmarks/bench_downstream_proxy.py` | Synthetic nearest-prototype classification proxy (NOT real perplexity — see disclaimer field in the file) |

All other files under `results/` (raw re-runs, GPU output, Nsight profiles)
are gitignored — regenerate them with the commands in the main README.
