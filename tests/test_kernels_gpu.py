"""CUDA kernel vs reference — requires a GPU and the built extension.

These are the tests that actually validate the kernels. They skip cleanly on
CPU-only machines (laptop, CPU CI) and run in the `gpu` CI job / on the
cloud profiling box. Kernel output must match the NumPy reference to fp32
accumulation tolerance — same math, different silicon.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

try:
    from pagedkv_fusion import _C  # noqa: F401
except ImportError:
    pytest.skip("pagedkv_fusion._C not built (pip install -e . on a CUDA box)",
                allow_module_level=True)

from pagedkv_fusion import ops, reference  # noqa: E402
from pagedkv_fusion.quantize import quantize_kv_per_block  # noqa: E402
from pagedkv_fusion.testing_utils import make_paged_kv_problem  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize("num_blocks,block_size", [(1, 16), (64, 16), (517, 32), (2048, 16)])
def test_eviction_kernel_matches_reference(rng, num_blocks, block_size):
    recency = rng.random(num_blocks, dtype=np.float32)
    frequency = rng.random(num_blocks, dtype=np.float32)
    attn = rng.random((num_blocks, block_size), dtype=np.float32)
    mask = rng.random((num_blocks, block_size)) > 0.2

    expected = reference.eviction_scores_ref(recency, frequency, attn, mask)
    got = ops.eviction_scores(
        torch.from_numpy(recency).cuda(),
        torch.from_numpy(frequency).cuda(),
        torch.from_numpy(attn).cuda(),
        torch.from_numpy(mask).cuda(),
    )
    assert got.is_cuda and ops.backend_in_use() == "cuda"
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-5, atol=1e-6)


def test_eviction_kernel_handles_fully_invalid_blocks(rng):
    mask = np.zeros((8, 16), dtype=bool)
    got = ops.eviction_scores(
        torch.rand(8).cuda(), torch.rand(8).cuda(),
        torch.rand(8, 16).cuda(), torch.from_numpy(mask).cuda(),
    )
    assert torch.isfinite(got).all()


@pytest.mark.parametrize(
    "num_seqs,num_heads,num_kv_heads,head_dim,block_size,max_seq_len",
    [
        (1, 1, 1, 32, 16, 1),        # degenerate
        (4, 8, 8, 64, 16, 200),      # MHA
        (4, 8, 2, 128, 16, 300),     # GQA
        (16, 32, 8, 128, 32, 1024),  # llama-ish decode batch
    ],
)
def test_attention_kernel_matches_reference(
        rng, num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_seq_len):
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(
        rng, num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_seq_len)
    kq_t, ks_t = quantize_kv_per_block(torch.from_numpy(k))
    vq_t, vs_t = quantize_kv_per_block(torch.from_numpy(v))

    expected = reference.paged_attention_ref(
        q, kq_t.numpy(), ks_t.numpy(), vq_t.numpy(), vs_t.numpy(),
        bt, seq_lens, sm)

    got = ops.quant_paged_attention(
        torch.from_numpy(q).cuda(),
        kq_t.cuda(), ks_t.cuda(), vq_t.cuda(), vs_t.cuda(),
        torch.from_numpy(bt).cuda(), torch.from_numpy(seq_lens).cuda(), sm)
    # fp32 accumulation, different reduction order -> small tolerance
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=2e-4, atol=2e-4)


def test_attention_kernel_deterministic(rng):
    q, k, v, bt, seq_lens, sm = make_paged_kv_problem(rng)
    kq_t, ks_t = quantize_kv_per_block(torch.from_numpy(k))
    vq_t, vs_t = quantize_kv_per_block(torch.from_numpy(v))
    args = (torch.from_numpy(q).cuda(), kq_t.cuda(), ks_t.cuda(),
            vq_t.cuda(), vs_t.cuda(), torch.from_numpy(bt).cuda(),
            torch.from_numpy(seq_lens).cuda(), sm)
    a = ops.quant_paged_attention(*args)
    b = ops.quant_paged_attention(*args)
    assert torch.equal(a, b)
