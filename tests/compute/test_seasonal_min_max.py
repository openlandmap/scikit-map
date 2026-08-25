"""Equivalence tests for the seasonal_min_max backend op.

Runs on real toy NDVI data (``toy_arr`` fixture: (1024, 24) with time on the
last axis, matching the seasonal_min_max contract).
"""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


@pytest.mark.parametrize("min_max", ["min", "max"])
def test_numpy_seasonal_min_max(toy_arr, min_max):
    out = NumpyBackend().seasonal_min_max(toy_arr, 4, min_max)
    assert out.shape == (1024, 6)  # 24 / 4 = 6 seasons


def test_numpy_nan_propagates(toy_arr):
    out = NumpyBackend().seasonal_min_max(toy_arr, 4, "max")
    # if any pixel has a NaN in a season chunk, that season must be NaN
    nan_mask = np.isnan(toy_arr)
    if nan_mask.any():
        # find a pixel+season with NaN and verify propagation
        px = np.argwhere(nan_mask.any(axis=1))[0, 0]
        t = np.argwhere(nan_mask[px])[0, 0]
        season = t // 4
        assert np.isnan(out[px, season])


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
@pytest.mark.parametrize("min_max", ["min", "max"])
def test_seasonal_min_max_backends_match(toy_arr, Backend, min_max):
    ref = NumpyBackend().seasonal_min_max(toy_arr, 4, min_max)
    out = Backend().seasonal_min_max(toy_arr, 4, min_max)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_seasonal_min_max_scaling(toy_arr):
    ref = NumpyBackend().seasonal_min_max(toy_arr, 4, "max", scaling=2.0)
    base = NumpyBackend().seasonal_min_max(toy_arr, 4, "max")
    np.testing.assert_allclose(ref, base * 2.0, rtol=1e-6)