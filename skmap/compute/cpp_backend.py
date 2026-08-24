"""C++ compute backend dispatching to the compiled ``skmap_bindings`` kernels.

The C++ extension exposes a handful of array kernels (``nanMean``,
``computePercentiles``, ``scaleAndOffset``, ``maskNan``, ``applyTsirf``) that
operate on 2-D ``float32`` C-contiguous arrays, reducing along the column axis.
This backend dispatches to those kernels where available and falls back to the
:class:`NumpyBackend` for every other operation (reductions without a C++
kernel, expression evaluation, convolution, Toeplitz matmul, per-pixel apply
and the specialised statistics).

The float32-only contract of the bindings is honoured *explicitly*: the
float32 ops accept an ``allow_cast`` flag (default ``False``).  A non-float32
input without ``allow_cast`` falls back to the numpy implementation (no silent
precision loss) and the fallback is recorded in ``self.fallbacks`` so callers
can report what actually ran.
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
        self.n_threads = n_threads if n_threads > 0 else _cpu_count()
        # Fallback backend for operations without a C++ kernel.
        self._fallback = NumpyBackend()
        # Records (op_name, reason) for every fallback since the last reset.
        self.fallbacks = []

    def _record_fallback(self, op, reason):
        self.fallbacks.append((op, reason))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_f32_2d(self, arr, axis, allow_cast=False):
        """Return a float32 C-contiguous 2-D view with the reduction axis as cols.

        Returns ``(matrix_2d, shape)`` or ``None`` if the input is not float32
        and ``allow_cast`` is False (caller should fall back to numpy).
        """

        moved = np.moveaxis(arr, axis, -1)
        shape = moved.shape
        flat = moved.reshape(-1, shape[-1])
        if flat.dtype != np.float32:
            if not allow_cast:
                return None
            warnings.warn(
                "CppBackend requires float32; casting %s -> float32 (lossy)"
                % flat.dtype,
                stacklevel=3,
            )
            flat = flat.astype(np.float32)
        flat = np.ascontiguousarray(flat)
        return flat, shape

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def nanmean(self, arr, axis, allow_cast=False):
        res = self._to_f32_2d(arr, axis, allow_cast)
        if res is None:
            self._record_fallback("nanmean", "non-float32 input")
            return self._fallback.nanmean(arr, axis)
        flat, shape = res
        out = np.empty(flat.shape[0], dtype=np.float32)
        self._sb.nanMean(flat, self.n_threads, out)
        return out.reshape(shape[:-1])

    def nanstd(self, arr, axis):
        self._record_fallback("nanstd", "no C++ kernel")
        return self._fallback.nanstd(arr, axis)

    def nanmin(self, arr, axis):
        self._record_fallback("nanmin", "no C++ kernel")
        return self._fallback.nanmin(arr, axis)

    def nanmax(self, arr, axis):
        self._record_fallback("nanmax", "no C++ kernel")
        return self._fallback.nanmax(arr, axis)

    def nansum(self, arr, axis):
        self._record_fallback("nansum", "no C++ kernel")
        return self._fallback.nansum(arr, axis)

    def nanmedian(self, arr, axis):
        self._record_fallback("nanmedian", "no C++ kernel")
        return self._fallback.nanmedian(arr, axis)

    def nanpercentile(self, arr, q, axis, allow_cast=False):
        res = self._to_f32_2d(arr, axis, allow_cast)
        if res is None:
            self._record_fallback("nanpercentile", "non-float32 input")
            return self._fallback.nanpercentile(arr, q, axis)
        flat, shape = res
        qs = [float(x) for x in (q if hasattr(q, "__iter__") else [q])]
        n_cols = flat.shape[1]
        out = np.empty((flat.shape[0], len(qs)), dtype=np.float32)
        self._sb.computePercentiles(
            flat,
            self.n_threads,
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
        self._record_fallback("evaluate", "no C++ expression VM")
        return self._fallback.evaluate(expr, local_dict)

    def scale_offset(self, arr, scale, offset, allow_cast=False):
        if arr.dtype != np.float32 and not allow_cast:
            self._record_fallback("scale_offset", "non-float32 input")
            return self._fallback.scale_offset(arr, scale, offset)
        # scaleAndOffset mutates in place: data = data * scaling + offset
        out = np.array(arr, dtype=np.float32, copy=True, order="C")
        self._sb.scaleAndOffset(out, self.n_threads, float(offset), float(scale))
        return out

    # ------------------------------------------------------------------
    # Convolution -- fall back (the C++ convolveRows is TSIRF-specific)
    # ------------------------------------------------------------------

    def convolve1d(self, arr, weights, axis, mode="constant", cval=0.0):
        self._record_fallback("convolve1d", "no generic C++ FIR kernel")
        return self._fallback.convolve1d(arr, weights, axis, mode, cval)

    def toeplitz_matmul(self, c, r, data):
        self._record_fallback("toeplitz_matmul", "no C++ Toeplitz kernel")
        return self._fallback.toeplitz_matmul(c, r, data)

    # ------------------------------------------------------------------
    # Per-pixel apply -- fall back
    # ------------------------------------------------------------------

    def apply_along_axis(self, func, axis, arr, *args, n_jobs=1, **kwargs):
        self._record_fallback("apply_along_axis", "no C++ per-pixel apply")
        return self._fallback.apply_along_axis(
            func, axis, arr, *args, n_jobs=n_jobs, **kwargs
        )

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    def mask_nan(self, arr, replace_value, allow_cast=False):
        if arr.dtype != np.float32 and not allow_cast:
            self._record_fallback("mask_nan", "non-float32 input")
            return self._fallback.mask_nan(arr, replace_value)
        out = np.array(arr, dtype=np.float32, copy=True, order="C")
        flat = out.reshape(-1, out.shape[-1] if out.ndim > 1 else 1)
        self._sb.maskNan(
            flat, self.n_threads, list(range(flat.shape[0])), float(replace_value)
        )
        return out

    def seasonal_min_max(self, arr, season_size, min_max, scaling=1.0):
        # No dedicated C++ kernel; fall back to the numpy vectorised path.
        self._record_fallback("seasonal_min_max", "no C++ kernel")
        return self._fallback.seasonal_min_max(arr, season_size, min_max, scaling)

    def fft_convolve(self, data, kernel, n_s):
        # No C++ FFT kernel; fall back to the numpy vectorised FFT.
        self._record_fallback("fft_convolve", "no C++ FFT kernel")
        return self._fallback.fft_convolve(data, kernel, n_s)

    def tsirf(self, data, conv_vect_past, conv_vect_future, keep_original_values=True, allow_cast=False):
        if data.dtype != np.float32 and not allow_cast:
            self._record_fallback("tsirf", "non-float32 input")
            return self._fallback.tsirf(
                data, conv_vect_past, conv_vect_future, keep_original_values
            )
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
            pix_time, self.n_threads, out, 0,
            w_0, w_p, w_f, bool(keep_original_values), "default", "default",
        )
        return out.T.astype(np.float64, copy=False)


def _cpu_count():
    import os

    return os.cpu_count() or 1
