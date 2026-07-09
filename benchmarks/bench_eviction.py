#!/usr/bin/env python3
"""Benchmark: fused CUDA eviction scoring vs host-side (NumPy) baseline.

Measures the *end-to-end eviction decision*, which is the metric that
matters to the scheduler: for the baseline that includes the device->host
copy of attention stats, the NumPy scoring pass, and the host-side argsort —
i.e. exactly what KV-Cache-Profiler does today. For the kernel it's a
device-side launch + device topk with no host round-trip.

Usage (GPU box):
    python benchmarks/bench_eviction.py --num-blocks 1024 4096 16384 65536 \
        --out results/eviction_bench.json

Every number in the validation report comes from this script's JSON output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pagedkv_fusion import ops, reference


def bench_host_baseline(num_blocks: int, block_size: int, iters: int,
                        torch=None) -> dict:
    """Host-side path: (optionally) D2H copy + NumPy score + argsort."""
    rng = np.random.default_rng(0)
    recency = rng.random(num_blocks, dtype=np.float32)
    frequency = rng.random(num_blocks, dtype=np.float32)
    mask = np.ones((num_blocks, block_size), dtype=bool)
    k = max(1, num_blocks // 20)

    if torch is not None and torch.cuda.is_available():
        attn_gpu = torch.rand(num_blocks, block_size, device="cuda")
        def step():
            attn = attn_gpu.cpu().numpy()          # the D2H transfer we're deleting
            s = reference.eviction_scores_ref(recency, frequency, attn, mask)
            return reference.select_eviction_candidates_ref(s, k)
    else:
        attn = rng.random((num_blocks, block_size), dtype=np.float32)
        def step():
            s = reference.eviction_scores_ref(recency, frequency, attn, mask)
            return reference.select_eviction_candidates_ref(s, k)

    times = _time(step, iters)
    return _summ("host_numpy", num_blocks, times)


def bench_cuda_kernel(num_blocks: int, block_size: int, iters: int) -> dict | None:
    try:
        import torch

        from pagedkv_fusion import _C  # noqa: F401
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    assert ops.backend_in_use() == "cuda", "refusing to benchmark the fallback"

    recency = torch.rand(num_blocks, device="cuda")
    frequency = torch.rand(num_blocks, device="cuda")
    attn = torch.rand(num_blocks, block_size, device="cuda")
    mask = torch.ones(num_blocks, block_size, dtype=torch.bool, device="cuda")
    k = max(1, num_blocks // 20)

    def step():
        s = ops.eviction_scores(recency, frequency, attn, mask)
        idx = ops.select_eviction_candidates(s, k)
        torch.cuda.synchronize()
        return idx

    step()  # warmup + JIT
    times = _time(step, iters)
    return _summ("cuda_fused", num_blocks, times)


def _time(fn, iters):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)  # µs
    return ts


def _summ(name, num_blocks, times_us):
    return {
        "impl": name,
        "num_blocks": num_blocks,
        "iters": len(times_us),
        "p50_us": statistics.median(times_us),
        "p90_us": statistics.quantiles(times_us, n=10)[8],
        "mean_us": statistics.fmean(times_us),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-blocks", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536])
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/eviction_bench.json"))
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    rows = []
    for nb in args.num_blocks:
        rows.append(bench_host_baseline(nb, args.block_size, args.iters, torch))
        cuda_row = bench_cuda_kernel(nb, args.block_size, args.iters)
        if cuda_row:
            rows.append(cuda_row)
        print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)},
                                    "results": rows}, indent=2))
    print(f"wrote {args.out}")
    if not any(r["impl"] == "cuda_fused" for r in rows):
        print("NOTE: CUDA path unavailable — only host baseline was measured.")


if __name__ == "__main__":
    main()
