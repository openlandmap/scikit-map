"""Equivalence tests: CppBackend must match NumpyBackend to tolerance."""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumpyBackend, get_backend

RTOL = 1e-4
ATOL = 1e-3


@pytest.fixture
def arr3d_f32():
    rng = np.random.default_rng(42)
    a = rng.random((4, 5, 6), dtype=np.float32) * 100
    a[0, 0, 0] = np.nan
    a[1, 2, 3] = np.nan
    a[3, 4, 5] = np.nan
    return a


@pytest.fixture
def arr2d_f32():
    rng = np.random.default_rng(7)
    return (rng.random((10, 12), dtype=np.float32) * 50).astype(np.float32)


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


def test_nanmean_equivalence(arr3d_f32, cbe, ref):
    out = cbe.nanmean(arr3d_f32, axis=-1)
    expected = ref.nanmean(arr3d_f32, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_nanpercentile_equivalence(arr3d_f32, cbe, ref):
    q = [25, 50, 75]
    out = cbe.nanpercentile(arr3d_f32, q, axis=-1)
    expected = ref.nanpercentile(arr3d_f32, q, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_scale_offset_equivalence(arr2d_f32, cbe, ref):
    out = cbe.scale_offset(arr2d_f32, 2.0, 1.0)
    expected = ref.scale_offset(arr2d_f32, 2.0, 1.0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_mask_nan_equivalence(arr3d_f32, cbe, ref):
    out = cbe.mask_nan(arr3d_f32, -999.0)
    expected = ref.mask_nan(arr3d_f32, -999.0)
    np.testing.assert_allclose(out, expected)
    assert not np.isnan(out).any()


def test_nanmean_axis0(arr3d_f32, cbe, ref):
    out = cbe.nanmean(arr3d_f32, axis=0)
    expected = ref.nanmean(arr3d_f32, axis=0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == (5, 6)


def test_nanmean_all_nan_row(arr3d_f32, cbe):
    a = arr3d_f32.copy()
    a[0, 0, :] = np.nan
    out = cbe.nanmean(a, axis=-1)
    assert np.isnan(out[0, 0])


def test_fallback_ops_match(arr3d_f32, cbe, ref):
    """Operations without a C++ kernel must fall back to the numpy backend."""
    for op in ("nanstd", "nanmin", "nanmax", "nansum", "nanmedian"):
        out = getattr(cbe, op)(arr3d_f32, axis=-1)
        exp = getattr(ref, op)(arr3d_f32, axis=-1)
        np.testing.assert_allclose(out, exp, rtol=RTOL, atol=ATOL)


def test_evaluate_fallback(arr2d_f32, cbe, ref):
    a = arr2d_f32
    b = arr2d_f32 + 1
    out = cbe.evaluate("a * 2 + b", {"a": a, "b": b})
    expected = ref.evaluate("a * 2 + b", {"a": a, "b": b})
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_convolve1d_fallback(arr2d_f32, cbe, ref):
    w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = cbe.convolve1d(arr2d_f32, w, axis=0)
    expected = ref.convolve1d(arr2d_f32, w, axis=0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_toeplitz_matmul_fallback(cbe, ref):
    c = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 4.0, 5.0])
    data = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = cbe.toeplitz_matmul(c, r, data)
    expected = ref.toeplitz_matmul(c, r, data)
    np.testing.assert_allclose(out, expected)


def test_apply_along_axis_fallback(arr3d_f32, cbe, ref):
    out = cbe.apply_along_axis(np.nansum, 2, arr3d_f32)
    expected = ref.apply_along_axis(np.nansum, 2, arr3d_f32)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_float64_falls_back_without_warning(cbe):
    """float64 input now falls back to numpy silently (no lossy cast)."""
    arr = np.random.rand(10, 12).astype(np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = cbe.nanmean(arr, axis=-1)
        assert not any("float32" in str(x.message) for x in w)
    # result matches numpy (no precision loss)
    ref = NumpyBackend().nanmean(arr, axis=-1)
    np.testing.assert_allclose(out, ref)