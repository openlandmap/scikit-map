"""Cross-backend equivalence tests for SeasConvFill and SircleTransformer."""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io.process import SeasConvFill, SircleTransformer


RTOL = 1e-3
ATOL = 1e-2


def _make_rdata(backend, gappy=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = toy.ndvi_rdata(gappy=gappy, verbose=False)
    r.backend = backend
    return r


@pytest.fixture(scope="module")
def ref_seasconv():
    r = _make_rdata(NumpyBackend())
    r.run(SeasConvFill(season_size=4, verbose=False))
    return r.array[:, :, 24:].copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_seasconvfill_backends_match(ref_seasconv, backend):
    r = _make_rdata(backend())
    r.run(SeasConvFill(season_size=4, verbose=False))
    np.testing.assert_allclose(
        r.array[:, :, 24:], ref_seasconv, rtol=RTOL, atol=ATOL
    )


def test_seasconvfill_fills_gaps():
    """The gap-filled output should have fewer NaNs than the input."""
    r = _make_rdata(NumpyBackend(), gappy=True)
    before = np.isnan(r.array).sum()
    r.run(SeasConvFill(season_size=4, verbose=False))
    after = np.isnan(r.array[:, :, 24:]).sum()
    assert after < before


@pytest.mark.parametrize("conv_backend", ["dense", "sparse", "FFT"])
def test_sircletransformer_backends_match(conv_backend):
    """SircleTransformer conv_backend across compute backends."""
    wv_0 = 1.0
    wv_p = np.array([0.5, 0.25], dtype=np.float64)
    wv_f = np.array([0.5, 0.25], dtype=np.float64)

    results = {}
    for Backend in [NumpyBackend, NumbaBackend, CppBackend]:
        r = _make_rdata(Backend(), gappy=False)
        r.run(
            SircleTransformer(
                wv_0=wv_0,
                wv_f=wv_f,
                wv_p=wv_p,
                conv_backend=conv_backend,
                verbose=False,
            )
        )
        results[Backend] = r.array[:, :, 24:].copy()

    np.testing.assert_allclose(
        results[NumpyBackend], results[NumbaBackend], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        results[NumpyBackend], results[CppBackend], rtol=RTOL, atol=ATOL
    )