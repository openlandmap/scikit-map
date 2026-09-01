"""End-to-end integration: full toy-data pipeline on each backend.

Asserts (a) correctness -- outputs match across backends -- and (b) the
expected fallback set for a main-process runner.
"""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io.process import SeasConvFill, TimeAggregate, TimeEnum, WhittakerSmooth


RTOL = 1e-3
ATOL = 1e-2


def _make_rdata(backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = toy.ndvi_rdata(gappy=True, verbose=False)
    r.backend = backend
    return r


def _run_pipeline(backend):
    """SeasConvFill + TimeAggregate, returning the appended bands."""
    r = _make_rdata(backend)
    r.run(SeasConvFill(season_size=4, verbose=False))
    r.run(
        TimeAggregate(
            time=[TimeEnum.YEARLY],
            operations=["p25", "p50", "p75", "std", "mean"],
            verbose=False,
        )
    )
    return r.array.get()[24:, :].copy()


@pytest.fixture(scope="module")
def ref_pipeline():
    return _run_pipeline(NumpyBackend())


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_full_pipeline_matches_numpy(ref_pipeline, backend):
    out = _run_pipeline(backend())
    np.testing.assert_allclose(out, ref_pipeline, rtol=RTOL, atol=ATOL)


def test_seasconvfill_cpp_uses_cpp_kernel():
    """SeasConvFill on float32 toy data must NOT fall back for tsirf."""
    r = _make_rdata(CppBackend())
    runner = SeasConvFill(season_size=4, verbose=False)
    r.run(runner)
    ops = {op for op, _ in runner.backend.fallbacks}
    assert "tsirf" not in ops


def test_whittaker_cpp_no_apply_fallback():
    """WhittakerSmooth batches the solve via inherited sparse_solve, so it
    no longer dispatches apply_along_axis (and records no fallback)."""
    r = _make_rdata(CppBackend())
    runner = WhittakerSmooth(lmbd=10, d=2, verbose=False)
    r.run(runner)
    ops = {op for op, _ in runner.backend.fallbacks}
    assert "apply_along_axis" not in ops


def test_numpy_backend_no_fallbacks():
    r = _make_rdata(NumpyBackend())
    runner = SeasConvFill(season_size=4, verbose=False)
    r.run(runner)
    assert runner.backend.fallbacks == []
