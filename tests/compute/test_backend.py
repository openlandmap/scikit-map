"""Unit tests for the compute-backend interface and the NumpyBackend reference."""

import numpy as np
import pytest

from skmap.compute import ComputeBackend, NumpyBackend, get_backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arr3d():
    """A small (4, 5, 6) array with some NaNs, float32."""
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


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestInterface:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ComputeBackend()

    def test_numpy_backend_name(self):
        assert NumpyBackend().name == "numpy"

    def test_get_backend_by_name(self):
        assert isinstance(get_backend("numpy"), NumpyBackend)

    def test_get_backend_passthrough(self):
        b = NumpyBackend()
        assert get_backend(b) is b

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError):
            get_backend("wat")


# ---------------------------------------------------------------------------
# NumpyBackend reference behaviour
# ---------------------------------------------------------------------------

class TestNumpyBackend:
    be = NumpyBackend()

    def test_nanmean(self, arr3d):
        out = self.be.nanmean(arr3d, axis=-1)
        assert out.shape == (4, 5)
        ref = np.nanmean(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanstd(self, arr3d):
        out = self.be.nanstd(arr3d, axis=-1)
        assert out.shape == (4, 5)
        ref = np.nanstd(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-4)

    def test_nanmin(self, arr3d):
        out = self.be.nanmin(arr3d, axis=-1)
        ref = np.nanmin(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_nanmax(self, arr3d):
        out = self.be.nanmax(arr3d, axis=-1)
        ref = np.nanmax(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_nansum(self, arr3d):
        out = self.be.nansum(arr3d, axis=-1)
        ref = np.nansum(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanmedian(self, arr3d):
        out = self.be.nanmedian(arr3d, axis=-1)
        ref = np.nanmedian(arr3d, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanpercentile(self, arr3d):
        q = [25, 50, 75]
        out = self.be.nanpercentile(arr3d, q, axis=-1)
        assert out.shape == (4, 5, 3)
        # numpy puts the percentile axis first; the backend returns it last
        # (matching the (H, W, n_percs) layout used by TimeAggregate).
        ref = np.nanpercentile(arr3d, q, axis=-1).transpose(1, 2, 0)
        np.testing.assert_allclose(out, ref, rtol=1e-4)

    def test_evaluate(self, arr2d):
        a = arr2d
        b = arr2d + 1
        out = self.be.evaluate("a * 2 + b", {"a": a, "b": b})
        np.testing.assert_allclose(out, a * 2 + b, rtol=1e-6)

    def test_scale_offset(self, arr2d):
        out = self.be.scale_offset(arr2d, 2.0, 1.0)
        np.testing.assert_allclose(out, arr2d * 2.0 + 1.0, rtol=1e-6)

    def test_convolve1d(self, arr2d):
        w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        out = self.be.convolve1d(arr2d, w, axis=0)
        from scipy.ndimage import convolve1d

        # The backend default is mode='constant', cval=0 (as used by process.py)
        ref = convolve1d(arr2d, w, axis=0, mode="constant", cval=0.0)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_toeplitz_matmul(self):
        c = np.array([1.0, 2.0, 3.0])
        r = np.array([1.0, 4.0, 5.0])
        data = np.arange(6, dtype=np.float64).reshape(3, 2)
        out = self.be.toeplitz_matmul(c, r, data)
        from scipy.linalg import matmul_toeplitz

        ref = matmul_toeplitz((c, r), data, check_finite=False, workers=None)
        np.testing.assert_allclose(out, ref)

    def test_apply_along_axis(self, arr3d):
        out = self.be.apply_along_axis(np.nansum, 2, arr3d)
        assert out.shape == (4, 5)
        ref = np.nansum(arr3d, axis=2)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_mask_nan(self, arr3d):
        out = self.be.mask_nan(arr3d, -999.0)
        assert not np.isnan(out).any()
        assert out[0, 0, 0] == -999.0
        # original untouched
        assert np.isnan(arr3d[0, 0, 0])

    def test_sparse_solve(self):
        from scipy.sparse import csc_matrix

        A = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        y = np.array([1.0, 2.0])
        x = self.be.sparse_solve(A, y)
        np.testing.assert_allclose(A @ x, y, atol=1e-10)

    def test_find_peaks(self):
        data = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        peaks, _ = self.be.find_peaks(data, height=0.5)
        assert list(peaks) == [1, 3]

    def test_theilslopes(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.arange(4)
        slope, *_ = self.be.theilslopes(data, x)
        np.testing.assert_allclose(slope, 1.0)

    def test_stl_decompose(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(24) + np.arange(24) * 0.5
        res = self.be.stl_decompose(data, period=4, seasonal=5)
        assert hasattr(res, "trend")
        assert res.trend.shape == (24,)

    def test_ols(self):
        import statsmodels.api as sm

        y = np.array([1.0, 3.0, 2.0, 5.0])
        X = sm.add_constant(np.arange(4, dtype=float))
        res = self.be.ols(y, X)
        assert hasattr(res, "params")
        assert hasattr(res, "rsquared")