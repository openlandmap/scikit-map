"""Unit + equivalence tests for the tsirf backend op.

The equivalence test runs on real gappy toy NDVI data (``toy_arr`` fixture,
transposed to (n_imag, n_pixels) = (24, 1024) as tsirf expects). The all-NaN
edge case stays synthetic. ``_conv_vectors`` uses a known convolution
structure (synthetic by design).
"""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


def _conv_vectors(n_imag, season_size=4):
    """Replicate SeasConvFill._compute_conv_mat_row (all-positive envelope)."""
    att_seas, att_env = 60.0, 20.0
    base_func = np.zeros(season_size)
    period_y = season_size / 2.0
    slope_y = att_seas / 10 / period_y
    for i in range(season_size):
        base_func[i] = -slope_y * i if i <= period_y else slope_y * (i - period_y) - att_seas / 10
    env_func = np.zeros(n_imag)
    slope_e = att_env / 10 / n_imag
    for i in range(n_imag):
        env_func[i] = -slope_e * i
    return 10.0 ** (np.resize(base_func, n_imag) + env_func)


@pytest.fixture
def convs():
    return _conv_vectors(24), _conv_vectors(24)


def test_numpy_tsirf_fills_gaps(toy_arr, convs):
    cp, cf = convs
    data = toy_arr.T  # tsirf expects (n_imag, n_pixels) = (24, 1024)
    out = NumpyBackend().tsirf(data, cp, cf)
    assert out.shape == data.shape
    nan_mask = np.isnan(data)
    if nan_mask.any():
        filled = ~np.isnan(out)
        assert filled[nan_mask].any(), "some gaps should be filled"


def test_numpy_tsirf_keeps_originals(toy_arr, convs):
    cp, cf = convs
    data = toy_arr.T
    out = NumpyBackend().tsirf(data, cp, cf, keep_original_values=True)
    valid = ~np.isnan(data)
    np.testing.assert_allclose(out[valid], data[valid])


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
def test_tsirf_backends_match_numpy(toy_arr, convs, Backend):
    cp, cf = convs
    data = toy_arr.T
    ref = NumpyBackend().tsirf(data, cp, cf)
    out = Backend().tsirf(data, cp, cf)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)


def test_tsirf_no_fill_when_all_nan(convs):
    # synthetic: needs a constructed all-NaN input to verify NaN propagation
    cp, cf = convs
    data = np.full((24, 5), np.nan)
    out = NumpyBackend().tsirf(data, cp, cf)
    assert np.isnan(out).all()