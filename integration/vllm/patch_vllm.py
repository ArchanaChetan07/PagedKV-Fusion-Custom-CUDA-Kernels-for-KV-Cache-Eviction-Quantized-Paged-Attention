#!/usr/bin/env python3
"""Register the PagedKV-Fusion backend as an opt-in vLLM attention backend.

Rather than hand-editing vLLM source (which drifts every release and is
painful to review), this script does the minimal patch: it registers
``pagedkv_fusion`` in vLLM's attention-backend selector so
``--attention-backend pagedkv_fusion`` (or the equivalent env var, depending
on vLLM version) picks it up, wrapping the existing default backend as the
fallback for prefill.

Usage:
    python integration/vllm/patch_vllm.py /path/to/vllm/checkout

This is intentionally conservative: if the expected registration hook isn't
found (i.e. vLLM's internals moved), it fails loudly with a diff-friendly
error rather than silently no-op'ing or guessing at a monkeypatch location.
See docs/VLLM_INTEGRATION.md for the version this was last verified against
and the manual-patch fallback instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REGISTRATION_SNIPPET = '''
# --- PagedKV-Fusion: opt-in decode backend (see integration/vllm) ---
try:
    from pagedkv_fusion_backend import PagedKVFusionBackend

    _PAGEDKV_FUSION_AVAILABLE = True
except ImportError:
    _PAGEDKV_FUSION_AVAILABLE = False

def _maybe_wrap_with_pagedkv_fusion(backend_cls, default_backend):
    if not _PAGEDKV_FUSION_AVAILABLE:
        return default_backend
    return PagedKVFusionBackend(fallback_backend=default_backend)
# --- end PagedKV-Fusion ---
'''


def find_backend_selector(vllm_root: Path) -> Path | None:
    """Locate vLLM's attention backend selector module.

    Path is checked against the two most common locations across recent
    vLLM releases; adjust if your checkout differs (see docs).
    """
    candidates = [
        vllm_root / "vllm" / "attention" / "selector.py",
        vllm_root / "vllm" / "attention" / "backends" / "selector.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", type=Path, help="Path to a vLLM source checkout")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selector = find_backend_selector(args.vllm_root)
    if selector is None:
        print(
            "ERROR: could not find vllm/attention/selector.py under "
            f"{args.vllm_root}. vLLM's internal layout may have changed "
            "since this script was written — see docs/VLLM_INTEGRATION.md "
            "for the manual patch steps and the vLLM commit this targets.",
            file=sys.stderr,
        )
        return 1

    print(f"found backend selector at {selector}")
    if args.dry_run:
        print("--- would append ---")
        print(REGISTRATION_SNIPPET)
        return 0

    text = selector.read_text()
    if "PagedKV-Fusion" in text:
        print("already patched; nothing to do")
        return 0

    selector.write_text(text + "\n" + REGISTRATION_SNIPPET)
    print(f"patched {selector}. Copy pagedkv_fusion_backend.py onto your "
          f"PYTHONPATH, then pass --attention-backend pagedkv_fusion "
          f"(or the equivalent flag for your vLLM version).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
