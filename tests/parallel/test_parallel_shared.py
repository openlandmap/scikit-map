"""Tests for the Ray object-store sharing helpers (SharedArray).

SharedArray mechanics; controlled arrays required, toy data adds no value.
"""

import numpy as np

from skmap import parallel


def test_put_get_roundtrip():
    arr = np.arange(24, dtype=np.float32).reshape(3, 8)
    sa = parallel.put_shared(arr)
    assert sa.shape == (3, 8)
    assert sa.dtype == np.float32
    assert sa.ndim == 2
    np.testing.assert_array_equal(parallel.get_shared(sa), arr)


def test_get_shared_accepts_raw_ref():
    arr = np.ones((2, 5), dtype=np.float32)
    sa = parallel.put_shared(arr)
    np.testing.assert_array_equal(parallel.get_shared(sa.ref), arr)


def test_stack_bands_worker():
    bands = [np.full((4,), i, dtype=np.float32) for i in range(3)]
    refs = [parallel.put_shared(b).ref for b in bands]
    out_ref = parallel._remote(parallel._stack_bands, refs, (3, 4))
    out = parallel.get_shared(out_ref)
    np.testing.assert_array_equal(out, np.stack(bands, axis=0).reshape(3, 4))


def test_assemble_worker():
    arr_in = np.arange(12, dtype=np.float32).reshape(3, 4)
    ref_in = parallel.put_shared(arr_in).ref
    # two output slices at band indices 3 and 4
    s0 = np.full((1, 4), 100.0, dtype=np.float32)
    s1 = np.full((1, 4), 200.0, dtype=np.float32)
    specs = [([3], 0, 4, s0), ([4], 0, 4, s1)]
    out_ref = parallel._remote(parallel._assemble, [ref_in], (5, 4), specs, 3)
    out = parallel.get_shared(out_ref)
    assert out.shape == (5, 4)
    np.testing.assert_array_equal(out[:3], arr_in)
    np.testing.assert_array_equal(out[3], s0[0])
    np.testing.assert_array_equal(out[4], s1[0])


def test_assemble_worker_partial_slice():
    arr_in = np.zeros((2, 8), dtype=np.float32)
    ref_in = parallel.put_shared(arr_in).ref
    s = np.full((1, 4), 7.0, dtype=np.float32)
    specs = [([2], 2, 6, s)]
    out_ref = parallel._remote(parallel._assemble, [ref_in], (3, 8), specs, 2)
    out = parallel.get_shared(out_ref)
    assert out.shape == (3, 8)
    np.testing.assert_array_equal(out[2, 2:6], s[0])
    np.testing.assert_array_equal(out[:2], arr_in)


def test_select_bands_worker():
    arr = np.arange(20, dtype=np.float32).reshape(5, 4)
    ref = parallel.put_shared(arr).ref
    out_ref = parallel._remote(parallel._select_bands, [ref], [0, 2, 4], (3, 4))
    out = parallel.get_shared(out_ref)
    np.testing.assert_array_equal(out, arr[[0, 2, 4], :])


def test_concat_worker():
    a = np.zeros((2, 4), dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    refs = [parallel.put_shared(a).ref, parallel.put_shared(b).ref]
    out_ref = parallel._remote(parallel._concat, refs, [(2, 4), (3, 4)])
    out = parallel.get_shared(out_ref)
    assert out.shape == (5, 4)
    np.testing.assert_array_equal(out[:2], a)
    np.testing.assert_array_equal(out[2:], b)
