"""Compute-backend abstraction for scikit-map array operations.

A :class:`ComputeBackend` exposes the array operations used by the
``skmap.io.process`` runners (reductions, elementwise expressions,
convolution, per-pixel apply, specialised statistics) behind a single
interface so that the same runner code can execute on top of:

* :class:`NumpyBackend`  -- NumPy + Bottleneck + NumExpr + SciPy (reference)
* :class:`NumbaBackend` -- Numba-jitted reductions and per-pixel apply
* :class:`CppBackend`   -- the compiled ``skmap_bindings`` kernels

The specialised statistical operations (STL, find_peaks, theilslopes,
sparse solve, OLS) have a single SciPy/statsmodels implementation shared
by every backend -- they are declared on the abstract base so a runner can
ask for them on any backend, but only the *core* array ops are overridden
per backend.
"""

from .base import ComputeBackend
from .numpy_backend import NumpyBackend
from .numba_backend import NumbaBackend
from .cpp_backend import CppBackend

__all__ = ["ComputeBackend", "NumpyBackend", "NumbaBackend", "CppBackend"]


def get_backend(name_or_obj):
    """Resolve a backend from a name (``"numpy"``/``"numba"``/``"cpp"``) or instance."""

    if isinstance(name_or_obj, ComputeBackend):
        return name_or_obj
    name = str(name_or_obj).lower()
    if name == "numpy":
        return NumpyBackend()
    if name == "numba":
        from .numba_backend import NumbaBackend

        return NumbaBackend()
    if name == "cpp":
        from .cpp_backend import CppBackend

        return CppBackend()
    raise ValueError(
        f"Unknown compute backend {name_or_obj!r}; use 'numpy', 'numba' or 'cpp'"
    )