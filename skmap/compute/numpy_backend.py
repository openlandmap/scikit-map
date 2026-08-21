"""Reference NumPy backend (NumPy + Bottleneck + NumExpr + SciPy).

This backend reproduces the exact behaviour of the original ``skmap.io.process``
runners and serves as the correctness baseline for the Numba and C++ backends.
"""

import numpy as np

from .base import ComputeBackend


class NumpyBackend(ComputeBackend):
    """Pure-Python backend using NumPy, Bottleneck, NumExpr and SciPy."""

    name = "numpy"

    # --- reductions ----------------------------------------------------

    def nanmean(self, arr, axis):
        import bottleneck as bn

        return bn.nanmean(arr, axis=axis)

    def nanstd(self, arr, axis):
        import bottleneck as bn

        return bn.nanstd(arr, axis=axis)

    def nanmin(self, arr, axis):
        import bottleneck as bn

        return bn.nanmin(arr, axis=axis)

    def nanmax(self, arr, axis):
        import bottleneck as bn

        return bn.nanmax(arr, axis=axis)

    def nansum(self, arr, axis):
        import bottleneck as bn

        return bn.nansum(arr, axis=axis)

    def nanmedian(self, arr, axis):
        import bottleneck as bn

        return bn.nanmedian(arr, axis=axis)

    def nanpercentile(self, arr, q, axis):
        from skmap.misc import nan_percentile

        # nan_percentile reduces along axis 0 and mutates the input in
        # place, so transpose the target axis to position 0 and pass a copy.
        if arr.ndim == 3:
            moved = np.moveaxis(arr, axis, 0)
            return nan_percentile(moved.copy(), q=q).transpose((1, 2, 0))
        return np.nanpercentile(arr, q, axis=axis)

    # --- elementwise ---------------------------------------------------

    def evaluate(self, expr, local_dict):
        import numexpr as ne

        return ne.evaluate(expr, local_dict=local_dict)

    def scale_offset(self, arr, scale, offset):
        return arr * scale + offset

    # --- convolution ---------------------------------------------------

    def convolve1d(self, arr, weights, axis, mode="constant", cval=0.0):
        from scipy.ndimage import convolve1d as _convolve1d

        return _convolve1d(arr, weights, axis=axis, mode=mode, cval=cval)

    def toeplitz_matmul(self, c, r, data):
        from scipy.linalg import matmul_toeplitz

        return matmul_toeplitz((c, r), data, check_finite=False, workers=None)

    # --- per-pixel apply ----------------------------------------------

    def apply_along_axis(self, func, axis, arr, *args, n_jobs=1, **kwargs):
        if n_jobs and n_jobs > 1:
            from skmap import parallel

            return parallel.apply_along_axis(
                func, axis, arr, n_jobs, *args, **kwargs
            )
        return np.apply_along_axis(func, axis, arr, *args, **kwargs)

    # --- nan handling --------------------------------------------------

    def mask_nan(self, arr, replace_value):
        out = np.array(arr, dtype=arr.dtype, copy=True)
        out[np.isnan(out)] = replace_value
        return out