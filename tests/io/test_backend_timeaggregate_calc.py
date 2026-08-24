"""Cross-backend equivalence tests for TimeAggregate and Calc (toy data)."""

import warnings

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend
from skmap.data import toy
from skmap.io.process import Calc, SeasConvFill, TimeAggregate, TimeEnum


RTOL = 1e-3
ATOL = 1e-2


def _make_rdata(backend, gappy=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = toy.ndvi_rdata(gappy=gappy, verbose=False)
    r.backend = backend
    return r


@pytest.fixture(scope="module")
def ref_aggregate():
    """Run TimeAggregate on the numpy backend to get the reference output."""
    r = _make_rdata(NumpyBackend())
    r.run(
        TimeAggregate(
            time=[TimeEnum.YEARLY],
            operations=["p25", "p50", "p75", "std", "mean"],
            verbose=False,
        )
    )
    return r.array.get().copy(), r.info.copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_timeaggregate_backends_match(ref_aggregate, backend):
    r = _make_rdata(backend())
    r.run(
        TimeAggregate(
            time=[TimeEnum.YEARLY],
            operations=["p25", "p50", "p75", "std", "mean"],
            verbose=False,
        )
    )
    ref_arr, ref_info = ref_aggregate
    # the new aggregate bands are appended after the original ndvi bands
    n_orig = 24
    new_cols = slice(n_orig, r.array.shape[0])
    np.testing.assert_allclose(
        r.array.get()[new_cols, :],
        ref_arr[new_cols, :],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.fixture(scope="module")
def ref_calc():
    """Run Calc on the numpy backend using the non-dotted 'ndvi' input group."""
    r = _make_rdata(NumpyBackend(), gappy=True)
    r.run(Calc({"ndvi_x2": "ndvi * 2"}, verbose=False))
    return r.array.get().copy()


@pytest.mark.parametrize("backend", [NumbaBackend, CppBackend])
def test_calc_backends_match(ref_calc, backend):
    r = _make_rdata(backend(), gappy=True)
    r.run(Calc({"ndvi_x2": "ndvi * 2"}, verbose=False))
    np.testing.assert_allclose(r.array.get(), ref_calc, rtol=RTOL, atol=ATOL)


def test_timeaggregate_post_expression_backends():
    """post_expression should evaluate identically across backends."""
    results = {}
    for Backend in [NumpyBackend, NumbaBackend, CppBackend]:
        r = _make_rdata(Backend())
        r.run(
            TimeAggregate(
                time=[TimeEnum.YEARLY],
                operations=["mean"],
                post_expression="new_array * 100",
                verbose=False,
            )
        )
        results[Backend] = r.array.get()[24:, :].copy()
    np.testing.assert_allclose(
        results[NumpyBackend], results[NumbaBackend], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        results[NumpyBackend], results[CppBackend], rtol=RTOL, atol=ATOL
    )