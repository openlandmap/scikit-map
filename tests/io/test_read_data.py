import skmap_bindings as sb
import numpy as np

from pathlib import Path


def test_read_data_multiple_files_smoketest():
    repo_root = Path(__file__).resolve().parents[2]
    swir1_dir = repo_root / "skmap" / "data" / "toy" / "swir1"
    assert swir1_dir.is_dir()

    paths = sorted(str(p) for p in swir1_dir.glob("*.tif"))
    assert len(paths) > 0

    n_layers = len(paths)
    width = 256
    height = 256
    res = np.empty((n_layers, width * height), dtype=np.float32)
    sb.readData(
        res,
        8,
        paths,
        list(range(0, n_layers)),
        0,
        0,
        width,
        height,
        [1 for _ in paths],
        {},
        None,
        np.nan,
    )

    n_files = len(paths)

    assert res.shape == (n_files, width * height)
    assert res.dtype == np.float32

    # smoke checks (not exact correctness)
    assert np.isfinite(res).all()
    assert not np.all(res == 0)
