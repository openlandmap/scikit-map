"""Equivalence tests for the fft_convolve backend op."""

import numpy as np
import pytest

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    return rng.random((8, 30)) * 50


def test_numpy_fft_convolve(data):
    kernel = np.zeros(30)
    kernel[0] = 0.5
    kernel[1:3] = [0.25, 0.1]
    out = NumpyBackend().fft_convolve(data, kernel, 20)
    assert out.shape == (8, 20)


@pytest.mark.parametrize("Backend", [NumbaBackend, CppBackend])
def test_fft_convolve_backends_match(data, Backend):
    kernel = np.zeros(30)
    kernel[0] = 0.5
    kernel[1:4] = [0.3, 0.2, 0.1]
    ref = NumpyBackend().fft_convolve(data, kernel, 20)
    out = Backend().fft_convolve(data, kernel, 20)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6)
