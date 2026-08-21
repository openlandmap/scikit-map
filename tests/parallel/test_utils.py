import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import ray
from numpy.typing import NDArray

import skmap.parallel as parallel
from skmap.misc import ttprint
from skmap.parallel import TaskSequencer, TilingProcessing, job


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
    ray.shutdown()


def rnd_data(const: Any, size: int) -> Tuple[Any, NDArray[np.float32]]:  # noqa: ANN401
    data = np.random.rand(size, size, size)
    time.sleep(0.002)
    return (const, data)


def max_value(const: Any, data: NDArray[np.float32]) -> NDArray[np.float32]:  # noqa: ANN401
    ttprint(f"Calculating the max value over {data.shape}")
    time.sleep(0.008)
    result = np.max(data + const)
    return result


class TestJob:
    def test_basic(self) -> None:
        def square(x: int) -> int:
            return x * x

        assert list(job(square, [(0,), (1,), (2,), (3,)])) == [0, 1, 4, 9]

    def test_empty(self) -> None:
        def square(x: int) -> int:
            return x * x

        assert list(job(square, [])) == []

    def test_submission_order(self) -> None:
        def slow_square(i: int) -> int:
            # later tasks finish first, but results must stay in submission order
            time.sleep(0.05 * (10 - i))
            return i * i

        assert list(job(slow_square, [(i,) for i in range(10)])) == [
            i * i for i in range(10)
        ]

    def test_n_jobs_1(self) -> None:
        def square(x: int) -> int:
            return x * x

        assert list(job(square, [(i,) for i in range(5)], n_jobs=1)) == [
            0,
            1,
            4,
            9,
            16,
        ]

    def test_exception_propagates(self) -> None:
        def bad(x: int) -> int:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            list(job(bad, [(1,)]))

    def test_joblib_args_ignored(self) -> None:
        def square(x: int) -> int:
            return x * x

        # joblib_args is accepted for backward compatibility and ignored
        assert list(
            job(square, [(0,), (1,)], joblib_args={"backend": "loky"})
        ) == [0, 1]

    def test_numpy_return(self) -> None:
        def make_array(i: int) -> NDArray[np.float32]:
            return np.full((2, 2), i, dtype=np.float32)

        res = list(job(make_array, [(0,), (1,)]))
        assert np.allclose(res[0], np.zeros((2, 2)))
        assert np.allclose(res[1], np.ones((2, 2)))


class TestApplyAlongAxis:
    def test_matches_numpy(self) -> None:
        def fn(arr: NDArray[np.float32], const: int) -> float:
            return np.sum(arr) + const

        arr = np.ones((4, 4, 4), dtype=np.float32)
        out = parallel.apply_along_axis(fn, 0, arr, 2, 1)
        expected = np.apply_along_axis(fn, 0, arr, 1)
        assert np.allclose(out, expected)


class TestTaskSequencer:
    def test__doctest_run(self) -> None:
        taskSeq = TaskSequencer(tasks=[rnd_data, (max_value, 2)], verbose=True)
        res = taskSeq.run(input_data=[(const, 10) for const in range(0, 3)])
        assert np.allclose(sorted(res), [1, 2, 3], rtol=1e-2)
        res = taskSeq.run(input_data=[(const, 20) for const in range(3, 6)])
        assert np.allclose(sorted(res), [1, 2, 3, 4, 5, 6], rtol=1e-2)


class TestTilingProcessing:
    REPO_ROOT = Path(__file__).parent.parent.parent
    ELEV_FILE = (
        REPO_ROOT
        / "skmap/data/toy/static/elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    )

    def test_process_multiple(self, tmp_path) -> None:
        tiles = TilingProcessing.generate_tiles(
            1000, (4020600, 3210130, 4028280, 3217810), "epsg:3035"
        )
        tiling_fn = tmp_path / "tiles.gpkg"
        tiles.to_file(tiling_fn, driver="GPKG")

        tp = TilingProcessing(
            tiling_system_fn=str(tiling_fn), base_raster_fn=str(self.ELEV_FILE)
        )

        def run(idx: int, tile: Any, window: Any) -> int:
            return idx

        res = tp.process_multiple([0, 1, 2], run, max_workers=2)
        assert sorted(res) == [0, 1, 2]


class TestDeadCodeRemoved:
    def test_lazy_generators_removed(self) -> None:
        assert not hasattr(parallel, "ThreadGeneratorLazy")
        assert not hasattr(parallel, "ProcessGeneratorLazy")
        assert not hasattr(parallel, "ProcessGeneratorLazy2")

    def test_global_executor_removed(self) -> None:
        assert not hasattr(parallel, "executor")
