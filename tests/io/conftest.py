import pytest
import numpy as np
import rasterio
from pathlib import Path
from rasterio.windows import Window


@pytest.fixture
def temp_raster_file(tmp_path):
    """Create a temporary single-band raster file for testing."""
    file_path = tmp_path / "test_raster.tif"
    data = np.random.rand(100, 100).astype(np.float32)
    transform = rasterio.transform.from_origin(0, 0, 1, 1)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return data, file_path


@pytest.fixture
def temp_multi_raster_files(tmp_path):
    """Create multiple temporary raster files for testing."""
    file_paths = []
    datas = []
    for i in range(3):
        file_path = tmp_path / f"test_raster_{i}.tif"
        data = np.random.rand(100, 100).astype(np.float32)
        transform = rasterio.transform.from_origin(i * 100, 0, 1, 1)

        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        file_paths.append(file_path)
        datas.append(data)

    return datas, file_paths
