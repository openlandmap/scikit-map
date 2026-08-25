"""Unit tests for the compute-backend interface and the NumpyBackend reference.

Reduction / elementwise / convolution tests run on real toy NDVI data
(``toy_arr`` / ``toy_arr_filled`` fixtures). Tests that need constructed
signals (find_peaks, theilslopes, STL, OLS, sparse_solve, Toeplitz) stay
synthetic — marked with ``# synthetic:`` comments.
"""

import numpy as np
import pytest

from skmap.compute import ComputeBackend, NumpyBackend, get_backend


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

    def test_nanmean(self, toy_arr):
        out = self.be.nanmean(toy_arr, axis=-1)
        assert out.shape == (1024,)
        ref = np.nanmean(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanstd(self, toy_arr):
        out = self.be.nanstd(toy_arr, axis=-1)
        assert out.shape == (1024,)
        ref = np.nanstd(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-4)

    def test_nanmin(self, toy_arr):
        out = self.be.nanmin(toy_arr, axis=-1)
        ref = np.nanmin(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_nanmax(self, toy_arr):
        out = self.be.nanmax(toy_arr, axis=-1)
        ref = np.nanmax(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_nansum(self, toy_arr):
        out = self.be.nansum(toy_arr, axis=-1)
        ref = np.nansum(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanmedian(self, toy_arr):
        out = self.be.nanmedian(toy_arr, axis=-1)
        ref = np.nanmedian(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_nanpercentile(self, toy_arr):
        q = [25, 50, 75]
        out = self.be.nanpercentile(toy_arr, q, axis=-1)
        assert out.shape == (1024, 3)
        # numpy puts the percentile axis first; the backend returns it last
        ref = np.nanpercentile(toy_arr, q, axis=-1).T
        np.testing.assert_allclose(out, ref, rtol=1e-4)

    def test_evaluate(self, toy_arr_filled):
        a = toy_arr_filled
        b = toy_arr_filled + 1
        out = self.be.evaluate("a * 2 + b", {"a": a, "b": b})
        np.testing.assert_allclose(out, a * 2 + b, rtol=1e-6)

    def test_scale_offset(self, toy_arr_filled):
        out = self.be.scale_offset(toy_arr_filled, 2.0, 1.0)
        np.testing.assert_allclose(out, toy_arr_filled * 2.0 + 1.0, rtol=1e-6)

    def test_convolve1d(self, toy_arr_filled):
        w = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        out = self.be.convolve1d(toy_arr_filled, w, axis=0)
        from scipy.ndimage import convolve1d

        ref = convolve1d(toy_arr_filled, w, axis=0, mode="constant", cval=0.0)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_toeplitz_matmul(self):
        # synthetic: specific Toeplitz matrix structure
        c = np.array([1.0, 2.0, 3.0])
        r = np.array([1.0, 4.0, 5.0])
        data = np.arange(6, dtype=np.float64).reshape(3, 2)
        out = self.be.toeplitz_matmul(c, r, data)
        from scipy.linalg import matmul_toeplitz

        ref = matmul_toeplitz((c, r), data, check_finite=False, workers=None)
        np.testing.assert_allclose(out, ref)

    def test_apply_along_axis(self, toy_arr):
        out = self.be.apply_along_axis(np.nansum, -1, toy_arr)
        assert out.shape == (1024,)
        ref = np.nansum(toy_arr, axis=-1)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_mask_nan(self, toy_arr):
        out = self.be.mask_nan(toy_arr, -999.0)
        assert not np.isnan(out).any()
        # a position that was NaN in the input must be -999 in the output
        nan_pos = np.argwhere(np.isnan(toy_arr))[0]
        assert out[nan_pos[0], nan_pos[1]] == -999.0
        # original untouched
        assert np.isnan(toy_arr[nan_pos[0], nan_pos[1]])

    def test_sparse_solve(self):
        # synthetic: specific sparse system
        from scipy.sparse import csc_matrix

        A = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        y = np.array([1.0, 2.0])
        x = self.be.sparse_solve(A, y)
        np.testing.assert_allclose(A @ x, y, atol=1e-10)

    def test_find_peaks(self):
        # synthetic: specific peak signal
        data = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        peaks, _ = self.be.find_peaks(data, height=0.5)
        assert list(peaks) == [1, 3]

    def test_theilslopes(self):
        # synthetic: specific trend
        data = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.arange(4)
        slope, *_ = self.be.theilslopes(data, x)
        np.testing.assert_allclose(slope, 1.0)

    def test_stl_decompose(self):
        # synthetic: constructed seasonal+trend signal
        rng = np.random.default_rng(0)
        data = rng.standard_normal(24) + np.arange(24) * 0.5
        res = self.be.stl_decompose(data, period=4, seasonal=5)
        assert hasattr(res, "trend")
        assert res.trend.shape == (24,)

    def test_ols(self):
        # synthetic: specific regression
        import statsmodels.api as sm

        y = np.array([1.0, 3.0, 2.0, 5.0])
        X = sm.add_constant(np.arange(4, dtype=float))
        res = self.be.ols(y, X)
        assert hasattr(res, "params")
        assert hasattr(res, "rsquared")