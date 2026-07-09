#!/usr/bin/env python3
"""Benchmark: INT8 quantized paged attention — memory, throughput, accuracy.

Three measurements, matching the project plan's success metrics:
  1. memory  — KV-cache bytes INT8+scales vs FP16, per config (exact math,
               runs anywhere)
  2. speed   — decode-step latency of the INT8 kernel vs torch SDPA on the
               gathered fp16 KV (GPU only)
  3. quality — output RMSE / max-abs vs FP32 attention on random and
               heavy-tailed KV distributions (runs anywhere; model-level
               perplexity is measured separately by bench_perplexity.py)

Usage:
    python benchmarks/bench_quant_attention.py --out results/quant_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pagedkv_fusion import reference

try:
    import torch
except ImportError:  # benchmark still runs its CPU-only sections without torch
    torch = None  # type: ignore[assignment]


def kv_bytes_fp16(num_blocks, block_size, num_kv_heads, head_dim):
    return 2 * num_blocks * block_size * num_kv_heads * head_dim * 2  # K+V, fp16


def kv_bytes_int8(num_blocks, block_size, num_kv_heads, head_dim):
    data = 2 * num_blocks * block_size * num_kv_heads * head_dim  # int8
    scales = 2 * num_blocks * num_kv_heads * 4                    # fp32 scales
    return data + scales


def measure_memory(cfgs):
    rows = []
    for c in cfgs:
        fp16 = kv_bytes_fp16(**c)
        int8 = kv_bytes_int8(**c)
        rows.append(c | {
            "fp16_MiB": fp16 / 2**20,
            "int8_MiB": int8 / 2**20,
            "savings_pct": 100 * (1 - int8 / fp16),
        })
    return rows


def measure_quality(seed=0):
    """Output error of INT8 vs FP32 attention on two KV distributions."""
    from pagedkv_fusion.testing_utils import fp32_paged_attention, make_paged_kv_problem

    rows = []
    for name, kv_scale in [("gaussian_unit", 1.0), ("heavy_tail_x8", 8.0)]:
        rng = np.random.default_rng(seed)
        q, k, v, bt, seq_lens, sm = make_paged_kv_problem(
            rng, num_seqs=8, max_seq_len=512, kv_scale=kv_scale)
        if name == "heavy_tail_x8":
            # inject outliers — the known weakness of per-block symmetric quant
            spike = rng.random(k.shape) < 0.001
            k = np.where(spike, k * 10, k).astype(np.float32)
        kq, ks = reference.quantize_kv_per_block_ref(k)
        vq, vs = reference.quantize_kv_per_block_ref(v)
        out_q = reference.paged_attention_ref(q, kq, ks, vq, vs, bt, seq_lens, sm)
        out_fp = fp32_paged_attention(q, k, v, bt, seq_lens, sm)
        err = out_q - out_fp
        rows.append({
            "distribution": name,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs": float(np.abs(err).max()),
            "out_std": float(out_fp.std()),
        })
    return rows


def _sdpa_gathered(qf, kf, vf, num_seqs, seq_len, num_heads, num_kv_heads, head_dim):
    """fp16 torch SDPA over KV gathered into a dense contiguous layout.

    This is the "pay the gather, use a fast dense kernel" baseline that the
    INT8 paged kernel is compared against — an honest upper bound, since a
    real serving system pays the gather cost too if it isn't paged-native.
    """
    kg = kf.reshape(num_seqs, seq_len, num_kv_heads, head_dim)
    vg = vf.reshape(num_seqs, seq_len, num_kv_heads, head_dim)
    kg = kg.repeat_interleave(num_heads // num_kv_heads, dim=2)
    vg = vg.repeat_interleave(num_heads // num_kv_heads, dim=2)
    return torch.nn.functional.scaled_dot_product_attention(
        qf[:, :, None, :].transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2))


def measure_speed(iters=100):
    """INT8 kernel vs fp16 SDPA baseline. GPU only; returns [] elsewhere."""
    if torch is None:
        return []
    try:
        from pagedkv_fusion import (
            _C,  # noqa: F401
            ops,
        )
        from pagedkv_fusion.quantize import quantize_kv_per_block
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []

    rows = []
    for num_seqs, seq_len in [(8, 512), (32, 1024), (64, 2048)]:
        num_heads, num_kv_heads, head_dim, bs = 32, 8, 128, 16
        n_blk = num_seqs * ((seq_len + bs - 1) // bs)
        k = torch.randn(n_blk, bs, num_kv_heads, head_dim)
        v = torch.randn_like(k)
        kq, ksc = quantize_kv_per_block(k)
        vq, vsc = quantize_kv_per_block(v)
        bt = torch.arange(n_blk, dtype=torch.int32).reshape(num_seqs, -1).cuda()
        sl = torch.full((num_seqs,), seq_len, dtype=torch.int32).cuda()
        q = torch.randn(num_seqs, num_heads, head_dim, device="cuda")
        args = (q, kq.cuda(), ksc.cuda(), vq.cuda(), vsc.cuda(), bt, sl,
                head_dim ** -0.5)

        ops.quant_paged_attention(*args)
        torch.cuda.synchronize()  # warmup
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ops.quant_paged_attention(*args)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)

        # fp16 SDPA over gathered dense KV (upper-bound baseline: pays gather)
        kf, vf, qf = k.cuda().half(), v.cuda().half(), q.half()
        sdpa_args = (qf, kf, vf, num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
        _sdpa_gathered(*sdpa_args)
        torch.cuda.synchronize()
        ts_base = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _sdpa_gathered(*sdpa_args)
            torch.cuda.synchronize()
            ts_base.append((time.perf_counter() - t0) * 1e3)

        rows.append({
            "num_seqs": num_seqs, "seq_len": seq_len,
            "int8_kernel_p50_ms": statistics.median(ts),
            "fp16_sdpa_gathered_p50_ms": statistics.median(ts_base),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/quant_bench.json"))
    args = ap.parse_args()

    mem_cfgs = [
        dict(num_blocks=4096, block_size=16, num_kv_heads=8, head_dim=128),   # 7B-ish
        dict(num_blocks=16384, block_size=16, num_kv_heads=8, head_dim=128),
        dict(num_blocks=16384, block_size=16, num_kv_heads=40, head_dim=128), # 13B MHA
    ]
    report = {
        "memory": measure_memory(mem_cfgs),
        "quality": measure_quality(),
        "speed": measure_speed(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["speed"]:
        print("NOTE: speed section empty — CUDA path unavailable on this machine.")


if __name__ == "__main__":
    main()
