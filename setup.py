"""Build script for PagedKV-Fusion.

Builds the CUDA extension (`pagedkv_fusion._C`) when torch + nvcc are
available; otherwise installs the pure-Python package (reference path only)
so CPU-only development and CI keep working. `PAGEDKV_FORCE_CUDA=1` turns a
missing toolchain into a hard error (used by the gpu CI job so a broken
toolchain can never silently ship a reference-only wheel).
"""

import os
import sys

from setuptools import setup

ext_modules = []
cmdclass = {}


def _compile_args() -> dict[str, list[str]]:
    if sys.platform == "win32":
        return {
            "cxx": ["/O2"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        }
    return {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
    }


def _try_cuda_ext():
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        return None, None
    if not torch.cuda.is_available() and os.environ.get("PAGEDKV_FORCE_CUDA") != "1":
        # Allow cross-compile on GPU-less build boxes only when forced with
        # an explicit TORCH_CUDA_ARCH_LIST.
        return None, None
    ext = CUDAExtension(
        name="pagedkv_fusion._C",
        sources=[
            "csrc/bindings.cpp",
            "csrc/eviction_score.cu",
            "csrc/quant_paged_attention.cu",
        ],
        extra_compile_args=_compile_args(),
    )
    return [ext], {"build_ext": BuildExtension}


def _is_metadata_only_invocation() -> bool:
    # pip/setuptools may import setup.py before torch is installed.
    return any(
        token in sys.argv
        for token in (
            "egg_info",
            "dist_info",
            "get_requires_for_build_wheel",
            "get_requires_for_build_editable",
        )
    )


exts, cc = _try_cuda_ext()
if exts:
    ext_modules, cmdclass = exts, cc
elif os.environ.get("PAGEDKV_FORCE_CUDA") == "1" and not _is_metadata_only_invocation():
    raise RuntimeError("PAGEDKV_FORCE_CUDA=1 but torch/CUDA toolchain unavailable")
elif not _is_metadata_only_invocation():
    print("[pagedkv-fusion] torch+CUDA not found: installing reference-only package")

setup(ext_modules=ext_modules, cmdclass=cmdclass)
