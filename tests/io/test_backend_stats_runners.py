"""Cross-backend equivalence tests for WhittakerSmooth, FindMinMax,
SlopeAnalysis, PeakAnalysis and TrendAnalysis (toy data).

The specialised statistics (splu, find_peaks, theilslopes, STL, OLS) fall
back to SciPy/statsmodels on every backend, so results match to float
precision; the only backend-dispatched op is ``evaluate`` (scale_expr).
"""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io.process import (
    FindMinMax,
    PeakAnalysis,
    SlopeAnalysis,
    TrendAnalysis,
    WhittakerSmooth,
)


RTOL = 1e-4
ATOL = 1e-3


def _make_rdata(backend, gappy=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = toy.ndvi_rdata(gappy=gappy, verbose=False)
    r.backend = backend
    return r


def _new_bands(rdata, n_orig=24):
    return rdata.array[n_orig:, :]


# ---------------------------------------------------------------------------
# WhittakerSmooth (uses sparse_solve + apply_along_axis)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_whittaker():
    r = _make_rdata(NumpyBackend())
    r.run(WhittakerSmooth(lmbd=10, d=2, verbose=False))
    return _new_bands(r).copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_whittaker_backends_match(ref_whittaker, backend):
    r = _make_rdata(backend())
    r.run(WhittakerSmooth(lmbd=10, d=2, verbose=False))
    np.testing.assert_allclose(_new_bands(r), ref_whittaker, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# FindMinMax (uses evaluate + apply_along_axis)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_findminmax():
    r = _make_rdata(NumpyBackend())
    r.run(FindMinMax(season_size=4, min_max="max", verbose=False))
    return _new_bands(r).copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_findminmax_backends_match(ref_findminmax, backend):
    r = _make_rdata(backend())
    r.run(FindMinMax(season_size=4, min_max="max", verbose=False))
    np.testing.assert_allclose(_new_bands(r), ref_findminmax, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# SlopeAnalysis (uses evaluate + theilslopes + apply_along_axis)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_slope():
    r = _make_rdata(NumpyBackend())
    r.run(SlopeAnalysis(verbose=False))
    return _new_bands(r).copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_slope_backends_match(ref_slope, backend):
    r = _make_rdata(backend())
    r.run(SlopeAnalysis(verbose=False))
    np.testing.assert_allclose(_new_bands(r), ref_slope, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# PeakAnalysis (uses evaluate + find_peaks + apply_along_axis)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_peak():
    r = _make_rdata(NumpyBackend())
    r.run(PeakAnalysis(season_size=4, verbose=False))
    return _new_bands(r).copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_peak_backends_match(ref_peak, backend):
    r = _make_rdata(backend())
    r.run(PeakAnalysis(season_size=4, verbose=False))
    np.testing.assert_allclose(_new_bands(r), ref_peak, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# TrendAnalysis (uses stl_decompose + ols + apply_along_axis)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_trend():
    r = _make_rdata(NumpyBackend())
    r.run(TrendAnalysis(season_size=4, verbose=False))
    return _new_bands(r).copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_trend_backends_match(ref_trend, backend):
    r = _make_rdata(backend())
    r.run(TrendAnalysis(season_size=4, verbose=False))
    np.testing.assert_allclose(_new_bands(r), ref_trend, rtol=RTOL, atol=ATOL)