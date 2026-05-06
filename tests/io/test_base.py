from skmap.io import save_rasters, read_rasters
from numpy.testing import assert_array_equal
import rasterio
import numpy as np
from pathlib import Path
import pytest

# Import your _new_raster function
from skmap.io.base import _new_raster


def test_with_statement_closes_file(tmp_path):
    """
    This test exists because the _new_raster internal function had a somewhat weird return structure:
    It returned a contextmanager, because of rasterio.open being the last value
    but it was not annotated with @contextmanager, so ty didn't understand
    """
    # Create a temporary base raster file
    base_file = tmp_path / "base.tif"
    data = np.random.rand(1, 10, 10).astype(np.float32)
    with rasterio.open(
        base_file,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)

    # Create a temporary output file
    output_file = tmp_path / "output.tif"
    test_data = np.random.rand(5, 5, 1).astype(np.float32)

    # Use _new_raster in a with statement
    with _new_raster(base_file, output_file, test_data) as new_raster:
        # Check that the file exists and is open
        assert output_file.exists()
        # Check that the dataset is open (e.g., by trying to read it)
        assert new_raster.shape == (5, 5)

    # After the with block, check if the file is closed
    # Try to open the file in write mode (will fail if the file is still open)
    try:
        with rasterio.open(output_file, "r+") as src:  # Open in read-write mode
            pass
        # If we get here, the file was closed properly
        assert True
    except rasterio.errors.RasterioError as e:
        # If we get an error, the file was not closed
        pytest.fail(f"File was not closed: {e}")


def test_new_raster_shape_roundtrip(tmp_path):
    """Verify that _new_raster preserves shape for non-square arrays.

    Non-square arrays are critical to test because x_size/y_size swaps
    are silent for square arrays — a 100x200 array written as 200x100
    will load back with swapped dimensions and be obviously wrong.
    """
    # Deliberately non-square: 30 rows (height) x 70 cols (width)
    original = np.random.rand(30, 70).astype(np.float32)
    base_raster_path = tmp_path / "base.tif"
    output_path = tmp_path / "output.tif"

    # Create a minimal base raster to satisfy _new_raster's base_raster param
    with rasterio.open(
        base_raster_path,
        "w",
        driver="GTiff",
        height=30,
        width=70,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(original, 1)

    # Write via _new_raster
    with _new_raster(base_raster_path, output_path, original) as ds:
        ds.write(original, 1)

    # Read back and compare
    with rasterio.open(output_path) as src:
        loaded = src.read(1)

    assert loaded.shape == original.shape, (
        f"Shape mismatch: wrote {original.shape}, got {loaded.shape}"
    )
    assert_array_equal(original, loaded)


def test_save_read_rasters_roundtrip(tmp_path):
    """Round-trip test: save_rasters -> read_rasters on a non-square array.

    Uses a non-square array (30 rows x 70 cols) with 3 bands to catch
    any height/width axis swaps — these are silent for square arrays.

    Conventions:
      - save_rasters expects: (height, width, n_files) = (30, 70, 3)
      - read_rasters returns: (height, width, n_files) = (30, 70, 3)
    No reshape or transpose should be needed.
    """
    H, W, N = 30, 70, 3
    memmap_path = tmp_path / "original.npy"
    original = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(H, W, N))
    original[:] = np.random.rand(H, W, N).astype(np.float32)

    base_raster_path = str(tmp_path / "base.tif")
    output_paths = [str(tmp_path / f"out_{i}.tif") for i in range(N)]

    # Create base raster
    with rasterio.open(
        base_raster_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(original[:, :, 0], 1)

    save_rasters(base_raster_path, output_paths, original, n_jobs=1)

    loaded = read_rasters(raster_files=output_paths, n_jobs=1)

    assert loaded.shape == original.shape, (
        f"Shape mismatch: saved {original.shape}, loaded {loaded.shape}"
    )
    assert_array_equal(original, loaded)
