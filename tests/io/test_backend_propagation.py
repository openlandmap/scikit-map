"""Tests for the backend attribute on RasterData and its propagation to runners."""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io import RasterData
from skmap.io.process import Calc, SeasConvFill


@pytest.fixture
def rdata_numpy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return toy.ndvi_rdata(gappy=True, verbose=False)


class TestBackendAttribute:
    def test_default_backend_is_numpy(self, rdata_numpy):
        assert isinstance(rdata_numpy.backend, NumpyBackend)

    def test_backend_by_name(self):
        r = RasterData({"g": ["x.tif"]}, backend="numba")
        assert isinstance(r.backend, NumbaBackend)

    def test_backend_by_instance(self):
        be = CppBackend()
        r = RasterData({"g": ["x.tif"]}, backend=be)
        assert r.backend is be

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            RasterData({"g": ["x.tif"]}, backend="wat")

    def test_backend_propagates_to_runner(self, rdata_numpy):
        rdata_numpy.backend = NumbaBackend()
        runner = SeasConvFill(season_size=4, verbose=False)
        rdata_numpy.run(runner, drop_input=False)
        assert runner.backend is rdata_numpy.backend

    def test_backend_propagates_to_calc(self, rdata_numpy):
        rdata_numpy.backend = CppBackend()
        calc = Calc({"ndvi.seasconv": "ndvi * 2"}, verbose=False)
        # run a SeasConvFill first so the 'ndvi.seasconv' group exists
        rdata_numpy.run(SeasConvFill(season_size=4, verbose=False))
        rdata_numpy.run(calc)
        assert calc.backend is rdata_numpy.backend

    def test_select_preserves_backend(self, rdata_numpy):
        rdata_numpy.backend = NumbaBackend()
        sub = rdata_numpy.filter("index < 3")
        assert sub.backend is rdata_numpy.backend or isinstance(
            sub.backend, NumbaBackend
        )

    def test_runner_default_backend_is_numpy(self):
        r = SeasConvFill(season_size=4, verbose=False)
        assert isinstance(r.backend, NumpyBackend)

def test_per_run_backend_override(rdata_numpy):
    """run(process, backend=...) overrides for this run only."""
    from skmap.compute import CppBackend

    rdata_numpy.backend = NumpyBackend()
    runner = SeasConvFill(season_size=4, verbose=False)
    rdata_numpy.run(runner, backend="cpp")
    assert isinstance(runner.backend, CppBackend)
    # the RasterData backend is unchanged
    assert isinstance(rdata_numpy.backend, NumpyBackend)


def test_per_run_backend_override_by_instance(rdata_numpy):
    be = NumbaBackend()
    runner = SeasConvFill(season_size=4, verbose=False)
    rdata_numpy.run(runner, backend=be)
    assert runner.backend is be
