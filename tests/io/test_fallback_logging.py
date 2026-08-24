"""Tests for compute-backend fallback recording and reporting."""

import warnings

import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io.process import TimeAggregate, TimeEnum, TrendAnalysis


def _make_rdata(backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = toy.ndvi_rdata(gappy=False, verbose=False)
    r.backend = backend
    return r


def test_cpp_whittaker_records_apply_fallback():
    """WhittakerSmooth runs in the main process, so its fallback is captured."""
    from skmap.io.process import WhittakerSmooth

    r = _make_rdata(CppBackend())
    runner = WhittakerSmooth(lmbd=10, d=2, verbose=False)
    r.run(runner)
    ops = {op for op, _ in runner.backend.fallbacks}
    assert "apply_along_axis" in ops


def test_cpp_trendanalysis_records_apply_fallback():
    r = _make_rdata(CppBackend())
    runner = TrendAnalysis(season_size=4, verbose=False)
    r.run(runner)
    ops = {op for op, _ in runner.backend.fallbacks}
    assert "apply_along_axis" in ops


def test_numpy_backend_has_no_fallbacks():
    r = _make_rdata(NumpyBackend())
    runner = TimeAggregate(time=[TimeEnum.YEARLY], operations=["mean"], verbose=False)
    r.run(runner)
    assert runner.backend.fallbacks == []


def test_reset_fallbacks_clears_log():
    be = CppBackend()
    be._record_fallback("x", "test")
    assert be.fallbacks
    be.reset_fallbacks()
    assert be.fallbacks == []


def test_report_fallbacks_prints(capsys):
    r = _make_rdata(CppBackend())
    r.verbose = True
    runner = TrendAnalysis(season_size=4, verbose=False)
    r.run(runner)
    out = capsys.readouterr().out
    assert "fell back" in out
