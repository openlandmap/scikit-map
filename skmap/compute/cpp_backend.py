"""C++ compute backend dispatching to the compiled ``skmap_bindings`` kernels.

The C++ extension exposes a handful of array kernels (``nanMean``,
``computePercentiles``, ``offsetAndScale``, ``maskNan``) that operate on
2-D ``float32`` C-contiguous arrays, reducing along the column axis.  This
backend dispatches to those kernels where available and falls back to the
:class:`NumpyBackend` for every other operation (reductions without a C++
kernel, expression evaluation, convolution, Toeplitz matmul, per-pixel
apply and the specialised statistics).

The float32-only contract of the bindings is honoured transparently: a
non-float32 or non-contiguous input is cast/copied and a warning is emitted
so callers know a conversion happened.
"""

import warnings

import numpy as np

from .base import ComputeBackend
from .numpy_backend import NumpyBackend


def _import_bindings():
    import skmap_bindings as sb

    return sb


class CppBackend(ComputeBackend):
    """Compute backend using the compiled ``skmap_bindings`` kernels."""

    name = "cpp"

    def __init__(self, n_threads: int = 0):
        self._sb = _import_bindings()
        self._n_threads = n_threads if n_threads > 0 else _cpu_count()
        # Fallback backend for operations without a C++ kernel.
        self._fallback = NumpyBackend()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_f32_2d(self, arr, axis):
        """Return a float32 C-contiguous 2-D view with the reduction axis as cols.

        Returns ``(matrix_2d, original_shape, axis_moved_to_last)`` so the caller
        can reshape the result back.
        """

        moved = np.moveaxis(arr, axis, -1)
        shape = moved.shape
        flat = np.ascontiguousarray(moved.reshape(-1, shape[-1]))
        if flat.dtype != np.float32:
            warnings.warn(
                "CppBackend requires float32; casting %s -> float32 (lossy)"
                % flat.dtype,
                stacklevel=3,
            )
            flat = flat.astype(np.float32)
        return flat, shape

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def nanmean(self, arr, axis):
        flat, shape = self._to_f32_2d(arr, axis)
        out = np.empty(flat.shape[0], dtype=np.float32)
        self._sb.nanMean(flat, self._n_threads, out)
        return out.reshape(shape[:-1])

    def nanstd(self, arr, axis):
        return self._fallback.nanstd(arr, axis)

    def nanmin(self, arr, axis):
        return self._fallback.nanmin(arr, axis)

    def nanmax(self, arr, axis):
        return self._fallback.nanmax(arr, axis)

    def nansum(self, arr, axis):
        return self._fallback.nansum(arr, axis)

    def nanmedian(self, arr, axis):
        return self._fallback.nanmedian(arr, axis)

    def nanpercentile(self, arr, q, axis):
        qs = [float(x) for x in (q if hasattr(q, "__iter__") else [q])]
        flat, shape = self._to_f32_2d(arr, axis)
        n_cols = flat.shape[1]
        out = np.empty((flat.shape[0], len(qs)), dtype=np.float32)
        self._sb.computePercentiles(
            flat,
            self._n_threads,
            list(range(n_cols)),
            out,
            list(range(len(qs))),
            qs,
        )
        return out.reshape(shape[:-1] + (len(qs),))

    # ------------------------------------------------------------------
    # Elementwise
    # ------------------------------------------------------------------

    def evaluate(self, expr, local_dict):
        return self._fallback.evaluate(expr, local_dict)

    def scale_offset(self, arr, scale, offset):
        # scaleAndOffset mutates in place: data = data * scaling + offset
        out = np.array(arr, dtype=np.float32, copy=True, order="C")
        if arr.dtype != np.float32:
            warnings.warn(
                "CppBackend requires float32; casting %s -> float32" % arr.dtype,
                stacklevel=2,
            )
        self._sb.scaleAndOffset(out, self._n_threads, float(offset), float(scale))
        return out

    # ------------------------------------------------------------------
    # Convolution -- fall back (the C++ convolveRows is TSIRF-specific)
    # ------------------------------------------------------------------

    def convolve1d(self, arr, weights, axis, mode="constant", cval=0.0):
        return self._fallback.convolve1d(arr, weights, axis, mode, cval)

    def toeplitz_matmul(self, c, r, data):
        return self._fallback.toeplitz_matmul(c, r, data)

    # ------------------------------------------------------------------
    # Per-pixel apply -- fall back
    # ------------------------------------------------------------------

    def apply_along_axis(self, func, axis, arr, *args, n_jobs=1, **kwargs):
        return self._fallback.apply_along_axis(
            func, axis, arr, *args, n_jobs=n_jobs, **kwargs
        )

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    def mask_nan(self, arr, replace_value):
        out = np.array(arr, dtype=np.float32, copy=True, order="C")
        if arr.dtype != np.float32:
            warnings.warn(
                "CppBackend requires float32; casting %s -> float32" % arr.dtype,
                stacklevel=2,
            )
        flat = out.reshape(-1, out.shape[-1] if out.ndim > 1 else 1)
        self._sb.maskNan(
            flat, self._n_threads, list(range(flat.shape[0])), float(replace_value)
        )
        return out

    def seasonal_min_max(self, arr, season_size, min_max, scaling=1.0):
        # No dedicated C++ kernel; fall back to the numpy vectorised path.
        return self._fallback.seasonal_min_max(arr, season_size, min_max, scaling)

    def fft_convolve(self, data, kernel, n_s):
        # No C++ FFT kernel; fall back to the numpy vectorised FFT.
        return self._fallback.fft_convolve(data, kernel, n_s)

    def tsirf(self, data, conv_vect_past, conv_vect_future, keep_original_values=True):
        # C++ applyTsirf operates on (rows=pixels, cols=time) float32 and uses
        # w_0 (center) + w_p (past taps, applied reversed) + w_f (future taps).
        # SeasConvFill passes (time, pixels) with conv_vect_{past,future} half-vectors.
        cp = np.asarray(conv_vect_past, dtype=np.float32)
        cf = np.asarray(conv_vect_future, dtype=np.float32)
        w_0 = float(cp[0])
        w_p = cp[:0:-1].copy()          # reversed past, excluding center
        w_f = cf[1:].copy()             # future, excluding center
        # transpose (time, pixels) -> (pixels, time), float32 C-contiguous
        pix_time = np.ascontiguousarray(np.asarray(data, dtype=np.float32).T)
        out = np.empty_like(pix_time)
        self._sb.applyTsirf(
            pix_time, self._n_threads, out, 0,
            w_0, w_p, w_f, bool(keep_original_values), "default", "default",
        )
        return out.T.astype(np.float64, copy=False)


def _cpu_count():
    import os

    return os.cpu_count() or 1
