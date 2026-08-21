"""Unit + equivalence tests for the tsirf backend op."""

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
def ts_data():
    rng = np.random.default_rng(1)
    data = rng.random((24, 40)) * 100  # (n_imag, n_pixels)
    # punch some gaps
    data[3, 5] = np.nan
    data[10, 20] = np.nan
    data[0, 0] = np.nan
    return data


@pytest.fixture
def convs():
    return _conv_vectors(24), _conv_vectors(24)


def test_numpy_tsirf_fills_gaps(ts_data, convs):
    cp, cf = convs
    out = NumpyBackend().tsirf(ts_data, cp, cf)
    assert out.shape == ts_data.shape
    # gaps at interior positions should be filled (not NaN)
    assert not np.isnan(out[3, 5])
    assert not np.isnan(out[10, 20])


def test_numpy_tsirf_keeps_originals(ts_data, convs):
    cp, cf = convs
    out = NumpyBackend().tsirf(ts_data, cp, cf, keep_original_values=True)
    # original valid values preserved exactly
    valid = ~np.isnan(ts_data)
    np.testing.assert_allclose(out[valid], ts_data[valid])


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
def test_tsirf_backends_match_numpy(ts_data, convs, Backend):
    cp, cf = convs
    ref = NumpyBackend().tsirf(ts_data, cp, cf)
    out = Backend().tsirf(ts_data, cp, cf)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)


def test_tsirf_no_fill_when_all_nan(convs):
    cp, cf = convs
    data = np.full((24, 5), np.nan)
    out = NumpyBackend().tsirf(data, cp, cf)
    assert np.isnan(out).all()
