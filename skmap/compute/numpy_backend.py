"""Reference NumPy backend (NumPy + Bottleneck + NumExpr + SciPy).

This backend reproduces the exact behaviour of the original ``skmap.io.process``
runners and serves as the correctness baseline for the Numba and C++ backends.
"""

import numpy as np

from .base import ComputeBackend


class NumpyBackend(ComputeBackend):
    """Pure-Python backend using NumPy, Bottleneck, NumExpr and SciPy."""

    name = "numpy"

    def __init__(self, n_threads: int = 0):
        import os

        self.n_threads = n_threads if n_threads > 0 else (os.cpu_count() or 1)

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
        # Use np.nanpercentile for consistent semantics across all backends.
        # (The legacy skmap.misc.nan_percentile NaNs single-observation pixels
        # except the median; np.nanpercentile returns the single value, which
        # is the mathematically well-defined behaviour shared by Numba/Cpp.)
        # numpy puts the percentile axis at position 0; move it to the original
        # axis position so the result shape matches the other backends.
        result = np.nanpercentile(arr, q, axis=axis)
        return np.moveaxis(result, 0, axis)

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
    def tsirf(self, data, conv_vect_past, conv_vect_future, keep_original_values=True):
        """TSIRF gap-fill via Toeplitz convolution + mask normalisation."""
        from scipy.linalg import matmul_toeplitz

        data = np.asarray(data, dtype=np.float64)
        valid_mask = ~np.isnan(data)
        masked = data.copy()
        masked[~valid_mask] = 0.0

        norm_vec = matmul_toeplitz(
            (conv_vect_past, conv_vect_future),
            np.ones((data.shape[0], 1)), check_finite=False, workers=None,
        )
        filled = matmul_toeplitz(
            (conv_vect_past, conv_vect_future),
            masked, check_finite=False, workers=None,
        )
        filled_qa = matmul_toeplitz(
            (conv_vect_past, conv_vect_future),
            valid_mask.astype(float), check_finite=False, workers=None,
        )
        conv_vec = np.concatenate((conv_vect_past, conv_vect_future[-1:0:-1]))
        nz = conv_vec[conv_vec > 0]
        min_conv_val = np.min(nz) if nz.size else 0.0
        filled = filled / filled_qa
        no_fill = filled_qa < min_conv_val
        filled[no_fill] = np.nan
        if keep_original_values:
            filled[valid_mask] = masked[valid_mask]
        return filled

    def fft_convolve(self, data, kernel, n_s):
        # Per-row FFT convolution: ifft(fft(kernel) * fft(row))[:n_s]
        data = np.asarray(data, dtype=np.float64)
        kernel = np.asarray(kernel, dtype=np.float64)
        n_e = data.shape[1]
        W = np.fft.rfft(kernel, n_e)
        V = np.fft.rfft(data, n_e, axis=1)
        out = np.fft.irfft(V * W, n_e, axis=1)[:, :n_s]
        return out

    def seasonal_min_max(self, arr, season_size, min_max, scaling=1.0):
        arr = np.asarray(arr)
        moved = np.moveaxis(arr, -1, -1)  # time already last
        flat = moved.reshape(-1, moved.shape[-1])
        n_seasons = flat.shape[1] // season_size
        out = np.empty((flat.shape[0], n_seasons), dtype=np.float64)
        # np.min/np.max propagate NaN (a single NaN in a chunk -> NaN result),
        # matching the FindMinMax contract.
        reduce = np.min if min_max == "min" else np.max
        for s in range(n_seasons):
            chunk = flat[:, s * season_size : (s + 1) * season_size]
            out[:, s] = reduce(chunk, axis=-1) * scaling
        return out.reshape(moved.shape[:-1] + (n_seasons,))
