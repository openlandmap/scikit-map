"""Equivalence tests for the fft_convolve backend op.

Runs on real toy NDVI data (``toy_arr_filled`` fixture, transposed to
(n_pixels, n_imag) = (1024, 24) as fft_convolve expects rows of time series).
"""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


def test_numpy_fft_convolve(toy_arr_filled):
    data = toy_arr_filled  # (1024, 24) — rows of time series
    kernel = np.zeros(24)
    kernel[0] = 0.5
    kernel[1:3] = [0.25, 0.1]
    out = NumpyBackend().fft_convolve(data, kernel, 20)
    assert out.shape == (1024, 20)


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
def test_fft_convolve_backends_match(toy_arr_filled, Backend):
    data = toy_arr_filled
    kernel = np.zeros(24)
    kernel[0] = 0.5
    kernel[1:4] = [0.3, 0.2, 0.1]
    ref = NumpyBackend().fft_convolve(data, kernel, 20)
    out = Backend().fft_convolve(data, kernel, 20)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6)