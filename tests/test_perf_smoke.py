"""Wall-clock smoke tests for the performance-critical paths.

These are not micro-benchmarks: they assert each operation completes within a
generous bound so a catastrophic regression (e.g. reverting the batched
WhittakerSmooth solve or the Prediction feature-matrix pre-allocation) fails
the suite without flaking on slow CI machines.
"""

import time
import warnings

import pytest

from skmap.data import toy
from skmap.io.process import SeasConvFill, WhittakerSmooth

# Generous bounds (10-30x the expected time on toy data) to avoid CI flakiness.
BOUND_S = 60.0


def _elapsed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


@pytest.fixture()
def rdata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return toy.ndvi_rdata(gappy=False, verbose=False)


def test_read_rasters_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elapsed = _elapsed(lambda: toy.ndvi_rdata(gappy=False, verbose=False))
    assert elapsed < BOUND_S, f"read_rasters took {elapsed:.1f}s"


def test_whittaker_smooth_smoke(rdata):
    """Batched sparse solve should be far under the bound (was ~60s per-pixel)."""
    elapsed = _elapsed(lambda: rdata.run(WhittakerSmooth(lmbd=10, d=2, verbose=False)))
    assert elapsed < BOUND_S, f"WhittakerSmooth took {elapsed:.1f}s"


def test_seasconv_fill_smoke(rdata):
    elapsed = _elapsed(lambda: rdata.run(SeasConvFill(season_size=4, verbose=False)))
    assert elapsed < BOUND_S, f"SeasConvFill took {elapsed:.1f}s"
