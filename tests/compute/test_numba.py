"""Equivalence tests: NumbaBackend must match NumpyBackend to tolerance."""

import numpy as np
import pytest

from skmap.compute import NumpyBackend, NumbaBackend, get_backend

RTOL = 1e-4
ATOL = 1e-3


@pytest.fixture
def arr3d():
    rng = np.random.default_rng(42)
    a = rng.random((4, 5, 6), dtype=np.float32) * 100
    a[0, 0, 0] = np.nan
    a[1, 2, 3] = np.nan
    a[3, 4, 5] = np.nan
    return a


@pytest.fixture
def arr2d():
    rng = np.random.default_rng(7)
    return (rng.random((10, 12), dtype=np.float32) * 50).astype(np.float32)


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
def test_reduction_equivalence(op, arr3d, nbe, ref):
    fn = getattr(nbe, op)
    refn = getattr(ref, op)
    out = fn(arr3d, axis=-1)
    expected = refn(arr3d, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_nanpercentile_equivalence(arr3d, nbe, ref):
    q = [25, 50, 75]
    out = nbe.nanpercentile(arr3d, q, axis=-1)
    expected = ref.nanpercentile(arr3d, q, axis=-1)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
    assert out.shape == expected.shape


def test_evaluate_equivalence(arr2d, nbe, ref):
    a = arr2d
    b = arr2d + 1
    out = nbe.evaluate("a * 2 + b", {"a": a, "b": b})
    expected = ref.evaluate("a * 2 + b", {"a": a, "b": b})
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_scale_offset_equivalence(arr2d, nbe, ref):
    out = nbe.scale_offset(arr2d, 2.0, 1.0)
    expected = ref.scale_offset(arr2d, 2.0, 1.0)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_convolve1d_equivalence(arr2d, nbe, ref):
    w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = nbe.convolve1d(arr2d, w, axis=0)
    expected = ref.convolve1d(arr2d, w, axis=0)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_toeplitz_matmul_equivalence(nbe, ref):
    c = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 4.0, 5.0])
    data = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = nbe.toeplitz_matmul(c, r, data)
    expected = ref.toeplitz_matmul(c, r, data)
    np.testing.assert_allclose(out, expected)


def test_apply_along_axis_equivalence(arr3d, nbe, ref):
    out = nbe.apply_along_axis(np.nansum, 2, arr3d)
    expected = ref.apply_along_axis(np.nansum, 2, arr3d)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_mask_nan_equivalence(arr3d, nbe, ref):
    out = nbe.mask_nan(arr3d, -999.0)
    expected = ref.mask_nan(arr3d, -999.0)
    np.testing.assert_allclose(out, expected)
    assert not np.isnan(out).any()


def test_numba_backend_name():
    assert NumbaBackend().name == "numba"


def test_get_backend_numba():
    assert isinstance(get_backend("numba"), NumbaBackend)


def test_all_nan_pixel_returns_nan(arr3d, nbe):
    a = arr3d.copy()
    a[0, 0, :] = np.nan
    out = nbe.nanmean(a, axis=-1)
    assert np.isnan(out[0, 0])