"""Abstract compute backend + shared SciPy/statsmodels implementations."""

from abc import ABC, abstractmethod

import numpy as np


class ComputeBackend(ABC):
    """Interface for the array-computation backends used by scikit-map runners.

    Core array operations (reductions, elementwise, convolution, per-pixel
    apply, nan masking) are abstract and implemented per backend.  The
    specialised statistical operations below have a single SciPy /
    statsmodels implementation inherited by every backend -- they are part
    of the interface so runner code is backend-agnostic, but they are not
    performance-critical enough to warrant three implementations.
    """

    name: str = "abstract"

    # ------------------------------------------------------------------
    # Core array operations -- overridden by each concrete backend.
    # ------------------------------------------------------------------

    @abstractmethod
    def nanmean(self, arr, axis):
        """NaN-aware mean along ``axis``."""

    @abstractmethod
    def nanstd(self, arr, axis):
        """NaN-aware standard deviation along ``axis``."""

    @abstractmethod
    def nanmin(self, arr, axis):
        """NaN-aware minimum along ``axis``."""

    @abstractmethod
    def nanmax(self, arr, axis):
        """NaN-aware maximum along ``axis``."""

    @abstractmethod
    def nansum(self, arr, axis):
        """NaN-aware sum along ``axis``."""

    @abstractmethod
    def nanmedian(self, arr, axis):
        """NaN-aware median along ``axis``."""

    @abstractmethod
    def nanpercentile(self, arr, q, axis):
        """NaN-aware percentiles. ``q`` is a sequence of percentile values."""

    @abstractmethod
    def evaluate(self, expr, local_dict):
        """Evaluate a NumExpr-style string expression elementwise."""

    @abstractmethod
    def scale_offset(self, arr, scale, offset):
        """Return ``arr * scale + offset`` elementwise."""

    @abstractmethod
    def convolve1d(self, arr, weights, axis, mode="constant", cval=0.0):
        """1-D FIR convolution of ``arr`` along ``axis``."""

    @abstractmethod
    def toeplitz_matmul(self, c, r, data):
        """Toeplitz matrix-vector product: first column ``c``, first row ``r``."""

    @abstractmethod
    def fft_convolve(self, data, kernel, n_s):
        """Per-row FFT convolution of ``data`` (n_rows, n_e) with 1-D ``kernel``
        (length n_e); return the first ``n_s`` columns of each row's result."""

    @abstractmethod
    def apply_along_axis(self, func, axis, arr, *args, n_jobs=1, **kwargs):
        """Apply ``func`` along ``axis`` (optionally parallel)."""

    @abstractmethod
    def mask_nan(self, arr, replace_value):
        """Replace NaNs in ``arr`` with ``replace_value``."""

    @abstractmethod
    def tsirf(self, data, conv_vect_past, conv_vect_future, keep_original_values=True):
        """TSIRF gap-fill: weighted FIR convolution of each time series,
        normalised by the convolved validity mask.

        ``data`` is ``(n_imag, n_pixels)`` (time axis 0), NaN marks gaps.
        ``conv_vect_past`` / ``conv_vect_future`` are the 1-D half-vectors
        (center at index 0).  Returns the filled ``(n_imag, n_pixels)`` array,
        NaN where no fill is possible.
        """

    # ------------------------------------------------------------------
    # Specialised statistics -- single SciPy/statsmodels implementation,
    # inherited unchanged by every backend.
    # ------------------------------------------------------------------

    def sparse_solve(self, coefmat, y):
        """Solve ``coefmat @ x = y`` for sparse ``coefmat`` (SciPy SPLU)."""

        from scipy.sparse.linalg import splu

        return splu(coefmat).solve(y)

    def find_peaks(self, data, **kwargs):
        """Find local peaks in a 1-D signal (SciPy)."""

        from scipy.signal import find_peaks

        return find_peaks(data, **kwargs)

    def theilslopes(self, data, x):
        """Theil-Sen slope + intercept (SciPy)."""

        from scipy.stats import theilslopes

        return theilslopes(data, x)

    def stl_decompose(self, data, **kwargs):
        """Seasonal-Trend decomposition (statsmodels STL)."""

        from statsmodels.tsa.seasonal import STL

        return STL(data, **kwargs).fit()

    def ols(self, y, X):
        """Ordinary least squares (statsmodels)."""

        import statsmodels.api as sm

        return sm.OLS(y, X).fit()