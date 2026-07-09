# Contributing

## Development loop

```bash
make install    # reference-path package, no GPU required
make lint
make test       # 14 passed, 3 skipped (torch/GPU-gated) is the expected baseline
make demo       # sanity-check the full pipeline composes correctly
```

If you have a CUDA GPU:

```bash
make install-cuda
make test-gpu   # this is the correctness gate that actually matters for csrc/ changes
```

## Before opening a PR

- `make lint test` must pass. If you touched `csrc/`, `make test-gpu` must
  also pass on real hardware — CPU-only review can't catch kernel bugs.
- New kernel behavior needs a matching reference implementation in
  `pagedkv_fusion/reference.py` (or `quantize.py`) *and* a test in `tests/`
  that pins the numeric contract. Kernels are validated against these
  references, not against "it compiled."
- If a change affects a number quoted in `docs/VALIDATION_REPORT.md`,
  regenerate that section (`make bench`, `make bench-quality`, or the
  relevant GPU target) and update the report in the same PR — stale
  performance claims are worse than none.
- Don't report GPU numbers you didn't measure. If you can't run the GPU
  suite, say so in the PR description and let a maintainer with hardware
  run it before merge.

## Style

- `ruff` config lives in `pyproject.toml`; `make lint` is the source of
  truth, not any editor plugin's defaults.
- CUDA/C++ in `csrc/` follows the commenting style already there: explain
  the *why* (memory access pattern, reduction strategy, numerical
  approach) next to the code, not just the *what*.
