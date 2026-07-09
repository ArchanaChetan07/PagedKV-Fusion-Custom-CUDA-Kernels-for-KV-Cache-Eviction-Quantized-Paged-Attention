"""Component B correctness: per-(block, head) INT8 quantization."""

from __future__ import annotations

import numpy as np
import pytest

from pagedkv_fusion import reference


def test_roundtrip_error_bounded_by_half_step(rng):
    """|x - dq(q(x))| <= scale/2 elementwise — the symmetric-quantization
    guarantee everything downstream (attention error bounds) rests on."""
    kv = rng.standard_normal((8, 16, 4, 64)).astype(np.float32) * 3.0
    q, scale = reference.quantize_kv_per_block_ref(kv)
    dq = reference.dequantize_kv_per_block_ref(q, scale)
    err = np.abs(kv - dq)
    bound = scale[:, None, :, None] / 2.0 + 1e-7
    assert (err <= bound).all(), f"max violation {(err - bound).max()}"


def test_scales_are_per_block_and_head(rng):
    kv = rng.standard_normal((4, 16, 2, 32)).astype(np.float32)
    kv[2, :, 1, :] *= 100.0  # blow up one (block, head) pair
    _, scale = reference.quantize_kv_per_block_ref(kv)
    assert scale.shape == (4, 2)
    assert scale[2, 1] > 10 * scale[2, 0]
    assert scale[2, 1] > 10 * scale[1, 1]


def test_zero_block_is_safe():
    kv = np.zeros((2, 16, 2, 32), dtype=np.float32)
    q, scale = reference.quantize_kv_per_block_ref(kv)
    assert (q == 0).all() and (scale == 1.0).all()
    dq = reference.dequantize_kv_per_block_ref(q, scale)
    assert (dq == 0).all()


def test_int8_range_saturates_at_127(rng):
    kv = rng.standard_normal((2, 16, 2, 32)).astype(np.float32) * 50
    q, _ = reference.quantize_kv_per_block_ref(kv)
    assert q.max() <= 127 and q.min() >= -127
    assert q.max() == 127 or q.min() == -127  # absmax hits the rail


def test_torch_mirror_bit_identical_to_numpy(rng):
    torch = pytest.importorskip("torch")
    from pagedkv_fusion.quantize import (
        dequantize_kv_per_block,
        quantize_kv_per_block,
    )

    kv = rng.standard_normal((6, 16, 4, 64)).astype(np.float32)
    q_np, s_np = reference.quantize_kv_per_block_ref(kv)
    q_t, s_t = quantize_kv_per_block(torch.from_numpy(kv))
    np.testing.assert_array_equal(q_t.numpy(), q_np)
    np.testing.assert_allclose(s_t.numpy(), s_np, rtol=1e-7)
    np.testing.assert_allclose(
        dequantize_kv_per_block(q_t, s_t).numpy(),
        reference.dequantize_kv_per_block_ref(q_np, s_np),
        rtol=1e-6,
    )
