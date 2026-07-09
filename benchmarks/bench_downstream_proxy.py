#!/usr/bin/env python3
"""Downstream-task quality proxy for INT8 vs FP32 paged attention.

The project plan calls for "quality delta (perplexity or downstream task
accuracy) versus FP16 baseline" (Deliverable 4). Real perplexity requires
running an actual language model end-to-end, which requires a GPU and a
downloaded checkpoint — neither available in this build environment (see
docs/VALIDATION_REPORT.md). Rather than skip the deliverable entirely or
fake a perplexity number, this script measures a **proxy task that
exercises the same failure mode perplexity would catch**: does INT8
quantization change which "concept" the attention output is closest to?

Task construction (fully synthetic, runs on CPU, no downloads):
  1. Build C random unit-norm "concept prototype" vectors in head_dim space
     (stand-ins for, e.g., next-token embedding directions).
  2. For each of N synthetic decode steps, construct a KV cache whose
     V rows are noisy draws around ONE prototype (the "ground truth
     concept" for that step) mixed with distractor rows from other
     prototypes — attention is expected to concentrate on the true
     concept's rows given a matching query.
  3. Run attention in FP32 and in INT8 (via the same reference kernels
     tested elsewhere in this repo). For each, classify the output by
     nearest prototype (cosine similarity).
  4. Report: FP32 accuracy, INT8 accuracy, and how often INT8's prediction
     *disagrees* with FP32's (the number that matters — it isolates
     quantization-caused decision flips from problems the task itself has).

This is honestly a proxy, not a substitute for real perplexity — it is
reported as such everywhere, including in the validation report. It DOES
give a real, reproducible signal on whether this quantization scheme
changes downstream decisions, which is the property perplexity would also
be sensitive to.

Usage:
    python benchmarks/bench_downstream_proxy.py --out results/downstream_proxy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pagedkv_fusion import reference


def _make_classification_problem(
    rng: np.random.Generator,
    num_trials: int,
    num_concepts: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    noise_std: float,
):
    """One synthetic 'decode step' per trial: query matches a random
    concept; V rows are drawn from all concepts (true concept dominant in
    count), K rows correlate with V's concept so attention can actually
    discriminate. Returns arrays shaped for paged_attention_ref with a
    single page per trial (seq_len <= block_size for simplicity)."""
    prototypes = rng.standard_normal((num_concepts, head_dim)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    labels = rng.integers(0, num_concepts, size=num_trials)
    q = np.empty((num_trials, num_heads, head_dim), dtype=np.float32)
    k = np.empty((num_trials, seq_len, 1, head_dim), dtype=np.float32)  # 1 kv head
    v = np.empty((num_trials, seq_len, 1, head_dim), dtype=np.float32)

    for t in range(num_trials):
        true_c = labels[t]
        # query aligned with the true concept (repeated across heads: MQA-style)
        q_vec = prototypes[true_c] + noise_std * rng.standard_normal(head_dim)
        q[t] = np.tile(q_vec, (num_heads, 1)).astype(np.float32)

        # majority of KV rows are the true concept; rest are distractors
        row_concepts = rng.integers(0, num_concepts, size=seq_len)
        n_true = max(1, seq_len // 2)
        row_concepts[:n_true] = true_c
        rng.shuffle(row_concepts)
        for i, c in enumerate(row_concepts):
            k[t, i, 0] = prototypes[c] + noise_std * rng.standard_normal(head_dim)
            v[t, i, 0] = prototypes[c] + noise_std * rng.standard_normal(head_dim)

    return q, k, v, prototypes, labels


def _classify(out_vec: np.ndarray, prototypes: np.ndarray) -> int:
    sims = prototypes @ (out_vec / (np.linalg.norm(out_vec) + 1e-8))
    return int(np.argmax(sims))


def run_proxy_task(rng, num_trials, num_concepts, num_heads, head_dim, seq_len,
                    noise_std):
    q, k, v, prototypes, labels = _make_classification_problem(
        rng, num_trials, num_concepts, num_heads, head_dim, seq_len, noise_std)

    block_tables = np.zeros((num_trials, 1), dtype=np.int32)
    for t in range(num_trials):
        block_tables[t, 0] = t  # one dedicated physical block per trial
    seq_lens = np.full(num_trials, seq_len, dtype=np.int32)
    sm_scale = 1.0 / np.sqrt(head_dim)

    # FP32 baseline via the "exact" quantization identity path: build a
    # block-per-trial cache and use fp32_paged_attention directly.
    from pagedkv_fusion.testing_utils import fp32_paged_attention
    out_fp32 = fp32_paged_attention(q, k, v, block_tables, seq_lens, sm_scale)

    kq, ks = reference.quantize_kv_per_block_ref(k)
    vq, vs = reference.quantize_kv_per_block_ref(v)
    out_int8 = reference.paged_attention_ref(
        q, kq, ks, vq, vs, block_tables, seq_lens, sm_scale)

    pred_fp32 = np.array([_classify(out_fp32[t, 0], prototypes) for t in range(num_trials)])
    pred_int8 = np.array([_classify(out_int8[t, 0], prototypes) for t in range(num_trials)])

    acc_fp32 = float((pred_fp32 == labels).mean())
    acc_int8 = float((pred_int8 == labels).mean())
    disagreement = float((pred_fp32 != pred_int8).mean())

    return {
        "num_trials": num_trials,
        "num_concepts": num_concepts,
        "seq_len": seq_len,
        "noise_std": noise_std,
        "fp32_accuracy": acc_fp32,
        "int8_accuracy": acc_int8,
        "accuracy_delta": acc_int8 - acc_fp32,
        "fp32_vs_int8_disagreement_rate": disagreement,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-trials", type=int, default=2000)
    ap.add_argument("--num-concepts", type=int, default=20)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/downstream_proxy.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    rows = []
    # sweep task difficulty (noise) and context length — the two axes that
    # plausibly interact with quantization error the most.
    for seq_len in (16, 64):
        for noise_std in (0.05, 0.3, 0.8):
            rows.append(run_proxy_task(
                rng, args.num_trials, args.num_concepts, args.num_heads,
                args.head_dim, seq_len, noise_std))
            print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": (
            "Synthetic nearest-prototype classification proxy, NOT real "
            "perplexity. Measures whether INT8 quantization flips the "
            "attention output's nearest-concept decision relative to FP32. "
            "Real perplexity requires a GPU + real model checkpoint; see "
            "docs/VALIDATION_REPORT.md for status."
        ),
        "results": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
