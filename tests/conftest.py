"""Shared test fixtures for PagedKV-Fusion.

Problem-generation helpers live in ``pagedkv_fusion.testing_utils`` (shared
with the benchmark scripts); this file just wires up pytest fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from pagedkv_fusion.testing_utils import (  # noqa: F401  (re-exported for tests/*)
    fp32_paged_attention,
    make_paged_kv_problem,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=1234)
