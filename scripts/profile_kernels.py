#!/usr/bin/env python3
"""Wrapper around Nsight Compute (ncu) and Nsight Systems (nsys) for the two
kernels. Produces the raw profiling artifacts that back every latency claim
in docs/VALIDATION_REPORT.md — this script IS the reproduction path for
those numbers, not just a demo.

Usage (on a CUDA box with Nsight tools installed):
    python scripts/profile_kernels.py eviction   --num-blocks 16384
    python scripts/profile_kernels.py attention  --num-seqs 32 --seq-len 2048
    python scripts/profile_kernels.py both --nsys   # timeline instead of ncu

Requires: nvcc-built extension installed (`pip install -e .` on a CUDA
machine), `ncu`/`nsys` on PATH (part of the CUDA Toolkit / Nsight installs).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_EVICTION_DRIVER = '''
import torch
from pagedkv_fusion import ops
n = {num_blocks}
bs = {block_size}
recency = torch.rand(n, device="cuda")
frequency = torch.rand(n, device="cuda")
attn = torch.rand(n, bs, device="cuda")
mask = torch.ones(n, bs, dtype=torch.bool, device="cuda")
for _ in range(5):
    ops.eviction_scores(recency, frequency, attn, mask)
torch.cuda.synchronize()
for _ in range(20):
    ops.eviction_scores(recency, frequency, attn, mask)
torch.cuda.synchronize()
'''

_ATTENTION_DRIVER = '''
import torch
from pagedkv_fusion import ops
from pagedkv_fusion.quantize import quantize_kv_per_block
num_seqs, seq_len = {num_seqs}, {seq_len}
num_heads, num_kv_heads, head_dim, bs = 32, 8, 128, 16
n_blk = num_seqs * ((seq_len + bs - 1) // bs)
k = torch.randn(n_blk, bs, num_kv_heads, head_dim)
v = torch.randn_like(k)
kq, ks = quantize_kv_per_block(k); vq, vs = quantize_kv_per_block(v)
bt = torch.arange(n_blk, dtype=torch.int32).reshape(num_seqs, -1).cuda()
sl = torch.full((num_seqs,), seq_len, dtype=torch.int32).cuda()
q = torch.randn(num_seqs, num_heads, head_dim, device="cuda")
args = (q, kq.cuda(), ks.cuda(), vq.cuda(), vs.cuda(), bt, sl, head_dim ** -0.5)
for _ in range(5):
    ops.quant_paged_attention(*args)
torch.cuda.synchronize()
for _ in range(20):
    ops.quant_paged_attention(*args)
torch.cuda.synchronize()
'''


def _require(tool: str):
    if shutil.which(tool) is None:
        sys.exit(f"ERROR: `{tool}` not found on PATH. Install the Nsight "
                  f"tools (bundled with the CUDA Toolkit) and re-run.")


def run(driver_src: str, name: str, out_dir: Path, use_nsys: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(tempfile.mkstemp(suffix=".py")[1])
    script.write_text(driver_src)

    if use_nsys:
        _require("nsys")
        out = out_dir / f"{name}.nsys-rep"
        cmd = ["nsys", "profile", "-o", str(out.with_suffix("")),
               "--force-overwrite", "true", sys.executable, str(script)]
    else:
        _require("ncu")
        out = out_dir / f"{name}.ncu-rep"
        cmd = ["ncu", "--set", "full", "-o", str(out.with_suffix("")),
               "--force-overwrite", sys.executable, str(script)]

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["eviction", "attention", "both"])
    ap.add_argument("--num-blocks", type=int, default=16384)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--nsys", action="store_true", help="use nsys instead of ncu")
    ap.add_argument("--out", type=Path, default=Path("results/profiles"))
    args = ap.parse_args()

    if args.target in ("eviction", "both"):
        src = _EVICTION_DRIVER.format(num_blocks=args.num_blocks,
                                       block_size=args.block_size)
        run(src, "eviction_score", args.out, args.nsys)
    if args.target in ("attention", "both"):
        src = _ATTENTION_DRIVER.format(num_seqs=args.num_seqs, seq_len=args.seq_len)
        run(src, "quant_paged_attention", args.out, args.nsys)


if __name__ == "__main__":
    main()
