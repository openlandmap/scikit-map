"""Tests for backend thread control (n_threads)."""

from skmap.compute import CppBackend, NumbaBackend, NumpyBackend


def test_numpy_backend_n_threads():
    b = NumpyBackend(n_threads=3)
    assert b.n_threads == 3


def test_numba_backend_n_threads():
    b = NumbaBackend(n_threads=2)
    assert b.n_threads == 2


def test_cpp_backend_n_threads():
    b = CppBackend(n_threads=2)
    assert b.n_threads == 2


def test_default_n_threads_positive():
    for Backend in (NumpyBackend, NumbaBackend, CppBackend):
        b = Backend()
        assert b.n_threads >= 1
