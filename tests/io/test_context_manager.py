"""Tests for the RasterData context manager and silent cleanup."""

import io
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest

from skmap.data import toy
from skmap.io import RasterData
from skmap.misc import is_memmap


@pytest.fixture
def rdata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return toy.ndvi_rdata(gappy=True, verbose=False)


def test_context_manager_returns_self(rdata):
    with rdata as r:
        assert r is rdata


def test_context_manager_cleans_memmap(rdata):
    from skmap.parallel import SharedArray

    assert isinstance(rdata.array, SharedArray)
    with rdata as r:
        assert isinstance(r.array, SharedArray)
    # after exit the ObjectRef is dropped (Ray GCs the object-store array)
    assert not hasattr(r, "array")


def test_del_does_not_print(rdata):
    """__del__ must not print 'Deleting' to stdout (regression)."""
    f = io.StringIO()
    with redirect_stdout(f):
        rdata.__del__()
    assert "Deleting" not in f.getvalue()


def test_del_safe_without_array():
    """A RasterData that was never read must not crash on __del__."""
    r = RasterData({"g": ["x.tif"]})
    # should not raise
    r.__del__()


def test_exit_signature_accepts_exc_args(rdata):
    # __exit__ should accept the standard (exc_type, exc_val, exc_tb) args
    rdata.__exit__(None, None, None)