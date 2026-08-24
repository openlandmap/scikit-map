"""Equivalence tests for the seasonal_min_max backend op."""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


@pytest.fixture
def arr3d():
    rng = np.random.default_rng(5)
    a = rng.random((4, 5, 24)).astype(np.float32) * 100
    a[0, 0, 0] = np.nan
    a[1, 2, 7] = np.nan
    return a


@pytest.mark.parametrize("min_max", ["min", "max"])
def test_numpy_seasonal_min_max(arr3d, min_max):
    out = NumpyBackend().seasonal_min_max(arr3d, 4, min_max)
    assert out.shape == (4, 5, 6)


def test_numpy_nan_propagates(arr3d):
    out = NumpyBackend().seasonal_min_max(arr3d, 4, "max")
    assert np.isnan(out[0, 0, 0])
    assert np.isnan(out[1, 2, 1])


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
@pytest.mark.parametrize("min_max", ["min", "max"])
def test_seasonal_min_max_backends_match(arr3d, Backend, min_max):
    ref = NumpyBackend().seasonal_min_max(arr3d, 4, min_max)
    out = Backend().seasonal_min_max(arr3d, 4, min_max)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_seasonal_min_max_scaling(arr3d):
    ref = NumpyBackend().seasonal_min_max(arr3d, 4, "max", scaling=2.0)
    base = NumpyBackend().seasonal_min_max(arr3d, 4, "max")
    np.testing.assert_allclose(ref, base * 2.0, rtol=1e-6)
