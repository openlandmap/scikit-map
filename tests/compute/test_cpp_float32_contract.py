"""Tests for the CppBackend opt-in float32 contract.

float32-path tests use real toy NDVI data (``toy_arr`` fixture). float64
fallback tests stay synthetic — they need a deliberately float64 array to
verify the cast contract.
"""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumpyBackend


@pytest.fixture
def cbe():
    return CppBackend()


def test_float32_input_uses_cpp(toy_arr, cbe):
    cbe.fallbacks.clear()
    out = cbe.nanmean(toy_arr, axis=-1)
    assert out.shape == (1024,)
    assert cbe.fallbacks == []  # no fallback recorded


def test_float64_input_falls_back_without_cast(cbe):
    # synthetic: needs a deliberately float64 array
    arr = np.random.rand(10, 12).astype(np.float64)
    cbe.fallbacks.clear()
    out = cbe.nanmean(arr, axis=-1)
    ref = NumpyBackend().nanmean(arr, axis=-1)
    np.testing.assert_allclose(out, ref)
    assert any(op == "nanmean" for op, _ in cbe.fallbacks)


def test_float64_input_casts_with_allow_cast(cbe):
    # synthetic: needs a deliberately float64 array
    arr = np.random.rand(10, 12).astype(np.float64)
    cbe.fallbacks.clear()
    out = cbe.nanmean(arr, axis=-1, allow_cast=True)
    assert out.shape == (10,)
    assert cbe.fallbacks == []  # cast allowed, no fallback


def test_scale_offset_falls_back_for_float64(cbe):
    # synthetic: needs a deliberately float64 array
    arr = np.random.rand(10, 12).astype(np.float64)
    cbe.fallbacks.clear()
    out = cbe.scale_offset(arr, 2.0, 1.0)
    ref = NumpyBackend().scale_offset(arr, 2.0, 1.0)
    np.testing.assert_allclose(out, ref)
    assert any(op == "scale_offset" for op, _ in cbe.fallbacks)


def test_mask_nan_falls_back_for_float64(cbe):
    # synthetic: needs a deliberately float64 array with NaN
    arr = np.random.rand(10, 12).astype(np.float64)
    arr[0, 0] = np.nan
    cbe.fallbacks.clear()
    out = cbe.mask_nan(arr, -999.0)
    ref = NumpyBackend().mask_nan(arr, -999.0)
    np.testing.assert_allclose(out, ref)
    assert any(op == "mask_nan" for op, _ in cbe.fallbacks)


def test_tsirf_falls_back_for_float64(cbe):
    # synthetic: needs a deliberately float64 array
    data = np.random.rand(24, 40).astype(np.float64)
    data[3, 5] = np.nan
    cp = np.ones(24, dtype=np.float64)
    cf = np.ones(24, dtype=np.float64)
    cbe.fallbacks.clear()
    out = cbe.tsirf(data, cp, cf)
    ref = NumpyBackend().tsirf(data, cp, cf)
    np.testing.assert_allclose(out, ref)
    assert any(op == "tsirf" for op, _ in cbe.fallbacks)


def test_tsirf_float32_uses_cpp(toy_arr, cbe):
    data = toy_arr.T  # (24, 1024) — tsirf expects time on axis 0
    cp = np.ones(24, dtype=np.float32)
    cf = np.ones(24, dtype=np.float32)
    cbe.fallbacks.clear()
    out = cbe.tsirf(data, cp, cf)
    assert out.shape == (24, 1024)
    assert cbe.fallbacks == []