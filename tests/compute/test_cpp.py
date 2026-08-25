"""Equivalence tests: CppBackend must match NumpyBackend to tolerance.

Equivalence assertions run on real toy NDVI data (``toy_arr`` fixture: a
(1024, 24) gappy float32 slice with real NaN gaps). Edge-case tests that need
constructed inputs (all-NaN rows, float64 fallback, specific Toeplitz
matrices) stay synthetic — marked with ``# synthetic:`` comments.
"""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumpyBackend, get_backend

RTOL = 1e-4
ATOL = 1e-3


@pytest.fixture(scope="module")
def cbe():
    return CppBackend()


@pytest.fixture
def ref():
    return NumpyBackend()


def test_cpp_backend_name():
    assert CppBackend().name == "cpp"


def test_get_backend_cpp():
    assert isinstance(get_backend("cpp"), CppBackend)


def test_nanmean_equivalence(toy_arr, cbe, ref):
    out = cbe.nanmean(toy_arr, axis=-1)
    expected = ref.nanmean(toy_arr, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_nanpercentile_equivalence(toy_arr, cbe, ref):
    q = [25, 50, 75]
    out = cbe.nanpercentile(toy_arr, q, axis=-1)
    expected = ref.nanpercentile(toy_arr, q, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_scale_offset_equivalence(toy_arr, cbe, ref):
    out = cbe.scale_offset(toy_arr, 2.0, 1.0)
    expected = ref.scale_offset(toy_arr, 2.0, 1.0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_mask_nan_equivalence(toy_arr, cbe, ref):
    out = cbe.mask_nan(toy_arr, -999.0)
    expected = ref.mask_nan(toy_arr, -999.0)
    np.testing.assert_allclose(out, expected)
    assert not np.isnan(out).any()


def test_nanmean_axis0(toy_arr, cbe, ref):
    out = cbe.nanmean(toy_arr, axis=0)
    expected = ref.nanmean(toy_arr, axis=0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == (24,)


def test_nanmean_all_nan_row(toy_arr, cbe):
    # synthetic: needs a constructed all-NaN row to verify NaN propagation
    a = toy_arr.copy()
    a[0, :] = np.nan
    out = cbe.nanmean(a, axis=-1)
    assert np.isnan(out[0])


def test_fallback_ops_match(toy_arr, cbe, ref):
    """Operations without a C++ kernel must fall back to the numpy backend."""
    for op in ("nanstd", "nanmin", "nanmax", "nansum", "nanmedian"):
        out = getattr(cbe, op)(toy_arr, axis=-1)
        exp = getattr(ref, op)(toy_arr, axis=-1)
        np.testing.assert_allclose(out, exp, rtol=RTOL, atol=ATOL)


def test_evaluate_fallback(toy_arr, cbe, ref):
    a = toy_arr
    b = toy_arr + 1
    out = cbe.evaluate("a * 2 + b", {"a": a, "b": b})
    expected = ref.evaluate("a * 2 + b", {"a": a, "b": b})
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_convolve1d_fallback(toy_arr_filled, cbe, ref):
    w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = cbe.convolve1d(toy_arr_filled, w, axis=-1)
    expected = ref.convolve1d(toy_arr_filled, w, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_toeplitz_matmul_fallback(cbe, ref):
    # synthetic: specific Toeplitz matrix structure
    c = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 4.0, 5.0])
    data = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = cbe.toeplitz_matmul(c, r, data)
    expected = ref.toeplitz_matmul(c, r, data)
    np.testing.assert_allclose(out, expected)


def test_apply_along_axis_fallback(toy_arr, cbe, ref):
    out = cbe.apply_along_axis(np.nansum, -1, toy_arr)
    expected = ref.apply_along_axis(np.nansum, -1, toy_arr)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_float64_falls_back_without_warning(cbe):
    # synthetic: needs a deliberately float64 array to test the cast contract
    arr = np.random.rand(10, 12).astype(np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = cbe.nanmean(arr, axis=-1)
        assert not any("float32" in str(x.message) for x in w)
    # result matches numpy (no precision loss)
    ref = NumpyBackend().nanmean(arr, axis=-1)
    np.testing.assert_allclose(out, ref)