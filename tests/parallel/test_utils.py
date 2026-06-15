import time
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from skmap.misc import ttprint
from skmap.parallel import TaskSequencer


def rnd_data(const: Any, size: int) -> Tuple[Any, NDArray[np.float32]]:  # noqa: ANN401
    data = np.random.rand(size, size, size)
    time.sleep(0.002)
    return (const, data)


def max_value(const: Any, data: NDArray[np.float32]) -> NDArray[np.float32]:  # noqa: ANN401
    ttprint(f"Calculating the max value over {data.shape}")
    time.sleep(0.008)
    result = np.max(data + const)
    return result


class TestTaskSequencer:
    def test__doctest_run(self) -> None:
        taskSeq = TaskSequencer(tasks=[rnd_data, (max_value, 2)], verbose=True)
        res = taskSeq.run(input_data=[(const, 10) for const in range(0, 3)])
        assert np.allclose(sorted(res), [1, 2, 3], rtol=1e-2)
        res = taskSeq.run(input_data=[(const, 20) for const in range(3, 6)])
        assert np.allclose(sorted(res), [1, 2, 3, 4, 5, 6], rtol=1e-2)
