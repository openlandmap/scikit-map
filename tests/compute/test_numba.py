"""Equivalence tests: NumbaBackend must match NumpyBackend to tolerance.

Equivalence assertions run on real toy NDVI data (``toy_arr`` fixture).
Edge-case tests that need constructed inputs stay synthetic.
"""

import numpy as np
import pytest

from skmap.compute import NumpyBackend, NumbaBackend, get_backend

RTOL = 1e-4
ATOL = 1e-3


@pytest.fixture(scope="module")
def nbe():
    return NumbaBackend()


@pytest.fixture
def ref():
    return NumpyBackend()


@pytest.mark.parametrize(
    "op",
    ["nanmean", "nanstd", "nanmin", "nanmax", "nansum", "nanmedian"],
)
def test_reduction_equivalence(op, toy_arr, nbe, ref):
    fn = getattr(nbe, op)
    refn = getattr(ref, op)
    out = fn(toy_arr, axis=-1)
    expected = refn(toy_arr, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_nanpercentile_equivalence(toy_arr, nbe, ref):
    q = [25, 50, 75]
    out = nbe.nanpercentile(toy_arr, q, axis=-1)
    expected = ref.nanpercentile(toy_arr, q, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_evaluate_equivalence(toy_arr, nbe, ref):
    a = toy_arr
    b = toy_arr + 1
    out = nbe.evaluate("a * 2 + b", {"a": a, "b": b})
    expected = ref.evaluate("a * 2 + b", {"a": a, "b": b})
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_scale_offset_equivalence(toy_arr, nbe, ref):
    out = nbe.scale_offset(toy_arr, 2.0, 1.0)
    expected = ref.scale_offset(toy_arr, 2.0, 1.0)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_convolve1d_equivalence(toy_arr_filled, nbe, ref):
    w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = nbe.convolve1d(toy_arr_filled, w, axis=-1)
    expected = ref.convolve1d(toy_arr_filled, w, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_toeplitz_matmul_equivalence(nbe, ref):
    # synthetic: specific Toeplitz matrix structure
    c = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 4.0, 5.0])
    data = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = nbe.toeplitz_matmul(c, r, data)
    expected = ref.toeplitz_matmul(c, r, data)
    np.testing.assert_allclose(out, expected)


def test_apply_along_axis_equivalence(toy_arr, nbe, ref):
    out = nbe.apply_along_axis(np.nansum, -1, toy_arr)
    expected = ref.apply_along_axis(np.nansum, -1, toy_arr)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_mask_nan_equivalence(toy_arr, nbe, ref):
    out = nbe.mask_nan(toy_arr, -999.0)
    expected = ref.mask_nan(toy_arr, -999.0)
    np.testing.assert_allclose(out, expected)
    assert not np.isnan(out).any()


def test_numba_backend_name():
    assert NumbaBackend().name == "numba"


def test_get_backend_numba():
    assert isinstance(get_backend("numba"), NumbaBackend)


def test_all_nan_pixel_returns_nan(toy_arr, nbe):
    # synthetic: needs a constructed all-NaN row to verify NaN propagation
    a = toy_arr.copy()
    a[0, :] = np.nan
    out = nbe.nanmean(a, axis=-1)
    assert np.isnan(out[0])