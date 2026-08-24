"""Numba-backed compute: jitted reductions and per-pixel apply.

The core reductions (nanmean/std/min/max/sum/median/percentile) and
``apply_along_axis`` are compiled with Numba.  Elementwise expression
evaluation falls back to NumExpr (Numba cannot parse arbitrary expression
strings); convolution uses a jitted FIR loop; the specialised statistics
inherit the SciPy implementation from the base class.
"""

import numpy as np

from .base import ComputeBackend


# ---------------------------------------------------------------------------
# Numba jitted kernels -- module-level so Numba caches the compilation.
# ---------------------------------------------------------------------------

def _import_numba():
    import numba

    return numba


def _make_kernels():
    """Build (and cache) the jitted reduction kernels."""

    numba = _import_numba()
    njit = numba.njit(cache=True)
    # NOTE: fastmath is NOT used -- it makes np.isnan() return False for NaNs.

    @njit
    def _nanmean_1d(row):
        s = 0.0
        n = 0
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v):
                s += v
                n += 1
        if n == 0:
            return np.nan
        return s / n

    @njit
    def _nanstd_1d(row):
        s = 0.0
        sq = 0.0
        n = 0
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v):
                s += v
                sq += v * v
                n += 1
        if n == 0:
            return np.nan
        mean = s / n
        var = (sq - mean * s) / n
        if var < 0:
            var = 0.0
        return np.sqrt(var)

    @njit
    def _nanmin_1d(row):
        m = np.inf
        found = False
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v) and v < m:
                m = v
                found = True
        if not found:
            return np.nan
        return m

    @njit
    def _nanmax_1d(row):
        m = -np.inf
        found = False
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v) and v > m:
                m = v
                found = True
        if not found:
            return np.nan
        return m

    @njit
    def _nansum_1d(row):
        s = 0.0
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v):
                s += v
        return s

    @njit
    def _nanmedian_1d(row):
        # collect non-nan values
        tmp = np.empty(row.shape[0], dtype=np.float64)
        n = 0
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v):
                tmp[n] = v
                n += 1
        if n == 0:
            return np.nan
        # insertion sort (small per-pixel series)
        for i in range(1, n):
            key = tmp[i]
            j = i - 1
            while j >= 0 and tmp[j] > key:
                tmp[j + 1] = tmp[j]
                j -= 1
            tmp[j + 1] = key
        mid = n // 2
        if n % 2 == 1:
            return tmp[mid]
        return (tmp[mid - 1] + tmp[mid]) / 2.0

    @njit
    def _nanpercentile_1d(row, qs):
        tmp = np.empty(row.shape[0], dtype=np.float64)
        n = 0
        for i in range(row.shape[0]):
            v = row[i]
            if not np.isnan(v):
                tmp[n] = v
                n += 1
        out = np.empty(qs.shape[0], dtype=np.float64)
        if n == 0:
            for k in range(qs.shape[0]):
                out[k] = np.nan
            return out
        for i in range(1, n):
            key = tmp[i]
            j = i - 1
            while j >= 0 and tmp[j] > key:
                tmp[j + 1] = tmp[j]
                j -= 1
            tmp[j + 1] = key
        for k in range(qs.shape[0]):
            pos = (n - 1) * (qs[k] / 100.0)
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            if lo == hi:
                out[k] = tmp[lo]
            else:
                out[k] = tmp[lo] + (tmp[hi] - tmp[lo]) * (pos - lo)
        return out

    @njit
    def _convolve1d_1d(row, w):
        n = row.shape[0]
        nw = w.shape[0]
        half = nw // 2
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(nw):
                ii = i + j - half
                if ii < 0 or ii >= n:
                    s += 0.0  # constant cval=0
                else:
                    v = row[ii]
                    if np.isnan(v):
                        v = 0.0
                    # flip the kernel to match scipy.ndimage.convolve1d (true convolution)
                    s += w[nw - 1 - j] * v
            out[i] = s
        return out

    @njit
    def _tsirf_1d(row, c_past, c_future):
        # TSIRF for a single pixel time series (1-D, length n).
        # c_past/c_future are the half-vectors (center at index 0).
        # out[j] = (sum_k w[j-k]*row[k]) / (sum_k w[j-k]*valid[k])
        n = row.shape[0]
        n_p = c_past.shape[0]
        n_f = c_future.shape[0]
        out = np.empty(n, dtype=np.float64)
        # min nonzero weight for the no-fill threshold
        min_w = np.inf
        for i in range(n_p):
            if c_past[i] > 0 and c_past[i] < min_w:
                min_w = c_past[i]
        for i in range(n_f):
            if c_future[i] > 0 and c_future[i] < min_w:
                min_w = c_future[i]
        for j in range(n):
            num = 0.0
            den = 0.0
            # center
            v = row[j]
            if np.isnan(v):
                num += c_past[0] * 0.0
            else:
                num += c_past[0] * v
                den += c_past[0]
            # past taps: c_past[m] * row[j-m], m=1..n_p-1
            for m in range(1, n_p):
                k = j - m
                if k >= 0:
                    vv = row[k]
                    if np.isnan(vv):
                        num += c_past[m] * 0.0
                    else:
                        num += c_past[m] * vv
                        den += c_past[m]
            # future taps: c_future[m] * row[j+m], m=1..n_f-1
            for m in range(1, n_f):
                k = j + m
                if k < n:
                    vv = row[k]
                    if np.isnan(vv):
                        num += c_future[m] * 0.0
                    else:
                        num += c_future[m] * vv
                        den += c_future[m]
            if den < min_w:
                out[j] = np.nan
            else:
                out[j] = num / den
        return out

    @njit
    def _seasonal_min_max_1d(row, season_size, is_max, scaling):
        n = row.shape[0]
        n_seasons = n // season_size
        out = np.empty(n_seasons, dtype=np.float64)
        for s in range(n_seasons):
            i0 = s * season_size
            i1 = i0 + season_size
            has_nan = False
            if is_max:
                m = -np.inf
            else:
                m = np.inf
            for k in range(i0, i1):
                v = row[k]
                if np.isnan(v):
                    has_nan = True
                    break
                if is_max:
                    if v > m:
                        m = v
                else:
                    if v < m:
                        m = v
            if has_nan:
                out[s] = np.nan
            else:
                out[s] = m * scaling
        return out

    return {
        "nanmean": _nanmean_1d,
        "nanstd": _nanstd_1d,
        "nanmin": _nanmin_1d,
        "nanmax": _nanmax_1d,
        "nansum": _nansum_1d,
        "nanmedian": _nanmedian_1d,
        "nanpercentile": _nanpercentile_1d,
        "convolve1d": _convolve1d_1d,
        "tsirf": _tsirf_1d,
        "seasonal_min_max": _seasonal_min_max_1d,
    }


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _make_kernels()
    return _KERNELS


def _apply_reduction(kernel, arr, axis):
    """Apply a 1-D jitted kernel along ``axis`` of a 2/3-D array via gufunc-style loop."""

    arr64 = np.asarray(arr, dtype=np.float64)
    moved = np.moveaxis(arr64, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = kernel(flat[i])
    return out.reshape(moved.shape[:-1])


def _apply_multi(kernel, arr, axis, qs):
    """Apply a 1-D kernel returning multiple values (percentiles) along ``axis``."""

    arr64 = np.asarray(arr, dtype=np.float64)
    moved = np.moveaxis(arr64, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty((flat.shape[0], qs.shape[0]), dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = kernel(flat[i], qs)
    shape = moved.shape[:-1] + (qs.shape[0],)
    # Move the q-axis back to the original axis position (numpy semantics).
    return np.moveaxis(out.reshape(shape), -1, axis)


class NumbaBackend(ComputeBackend):
    """Compute backend using Numba-jitted per-pixel kernels."""

    name = "numba"

    def __init__(self, n_threads: int = 0):
        self.n_threads = n_threads if n_threads > 0 else _cpu_count()
        try:
            import numba

            numba.set_num_threads(self.n_threads)
        except Exception:
            pass
        _kernels()  # trigger compilation eagerly

    # --- reductions ----------------------------------------------------

    def nanmean(self, arr, axis):
        return _apply_reduction(_kernels()["nanmean"], arr, axis)

    def nanstd(self, arr, axis):
        return _apply_reduction(_kernels()["nanstd"], arr, axis)

    def nanmin(self, arr, axis):
        return _apply_reduction(_kernels()["nanmin"], arr, axis)

    def nanmax(self, arr, axis):
        return _apply_reduction(_kernels()["nanmax"], arr, axis)

    def nansum(self, arr, axis):
        return _apply_reduction(_kernels()["nansum"], arr, axis)

    def nanmedian(self, arr, axis):
        return _apply_reduction(_kernels()["nanmedian"], arr, axis)

    def nanpercentile(self, arr, q, axis):
        qs = np.asarray(q, dtype=np.float64)
        return _apply_multi(_kernels()["nanpercentile"], arr, axis, qs)

    # --- elementwise ---------------------------------------------------

    def evaluate(self, expr, local_dict):
        # Numba cannot parse arbitrary expression strings -- fall back to
        # NumExpr (same as the NumpyBackend) for correctness.
        self._record_fallback("evaluate", "numba cannot parse expression strings")
        import numexpr as ne

        return ne.evaluate(expr, local_dict=local_dict)

    def scale_offset(self, arr, scale, offset):
        return arr * scale + offset

    # --- convolution ---------------------------------------------------

    def convolve1d(self, arr, weights, axis, mode="constant", cval=0.0):
        # The jitted kernel implements mode='constant', cval=0.
        if mode != "constant" or cval != 0.0:
            self._record_fallback("convolve1d", "non-constant mode")
            from scipy.ndimage import convolve1d as _convolve1d

            return _convolve1d(arr, weights, axis=axis, mode=mode, cval=cval)
        w = np.asarray(weights, dtype=np.float64)
        arr64 = np.asarray(arr, dtype=np.float64)
        moved = np.moveaxis(arr64, axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            out[i] = _kernels()["convolve1d"](flat[i], w)
        out = out.reshape(moved.shape)
        # move the convolved axis back to its original position
        out = np.moveaxis(out, -1, axis)
        return out.astype(arr.dtype, copy=False)

    def toeplitz_matmul(self, c, r, data):
        # Fall back to SciPy -- the Toeplitz product is already FFT-based.
        self._record_fallback("toeplitz_matmul", "no numba Toeplitz kernel")
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
        cp = np.asarray(conv_vect_past, dtype=np.float64)
        cf = np.asarray(conv_vect_future, dtype=np.float64)
        data64 = np.asarray(data, dtype=np.float64)
        # data is (n_imag, n_pixels); iterate over pixels (columns)
        n_imag, n_pix = data64.shape
        out = np.empty((n_imag, n_pix), dtype=np.float64)
        kernel = _kernels()["tsirf"]
        for p in range(n_pix):
            out[:, p] = kernel(data64[:, p], cp, cf)
        if keep_original_values:
            valid = ~np.isnan(data64)
            out[valid] = data64[valid]
        return out

    def fft_convolve(self, data, kernel, n_s):
        # Use the vectorised numpy FFT -- numba's numpy.fft support is limited
        # and per-row jitting loses to the vectorised BLAS path here.
        self._record_fallback("fft_convolve", "no numba FFT kernel")
        data = np.asarray(data, dtype=np.float64)
        kernel = np.asarray(kernel, dtype=np.float64)
        n_e = data.shape[1]
        W = np.fft.rfft(kernel, n_e)
        V = np.fft.rfft(data, n_e, axis=1)
        return np.fft.irfft(V * W, n_e, axis=1)[:, :n_s]

    def seasonal_min_max(self, arr, season_size, min_max, scaling=1.0):
        arr64 = np.asarray(arr, dtype=np.float64)
        moved = np.moveaxis(arr64, -1, -1)  # already last
        flat = moved.reshape(-1, moved.shape[-1])
        n_seasons = flat.shape[1] // season_size
        out = np.empty((flat.shape[0], n_seasons), dtype=np.float64)
        is_max = 1 if min_max == "max" else 0
        kernel = _kernels()["seasonal_min_max"]
        for i in range(flat.shape[0]):
            out[i] = kernel(flat[i], season_size, is_max, scaling)
        return out.reshape(moved.shape[:-1] + (n_seasons,))


def _cpu_count():
    import os

    return os.cpu_count() or 1
