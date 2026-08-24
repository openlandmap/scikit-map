from skmap.io import save_rasters, read_rasters
from numpy.testing import assert_array_equal
import rasterio
import numpy as np
from pathlib import Path
import pytest

# Import your _new_raster function
from skmap.io.base import _new_raster

def test_with_statement_closes_file(tmp_path):
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
        base_raster_path, "w", driver="GTiff",
        height=30, width=70, count=1, dtype="float32",
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

    assert loaded.shape == original.shape, \
        f"Shape mismatch: wrote {original.shape}, got {loaded.shape}"
    assert_array_equal(original, loaded)

def test_save_read_rasters_roundtrip(tmp_path):
    """Round-trip test: save_rasters -> read_rasters on a non-square array.

    Uses a non-square array (30 rows x 70 cols) with 3 bands to catch
    any height/width axis swaps — these are silent for square arrays.

    Conventions:
      - save_rasters expects: (n_files, height * width) = (3, 2100)
      - read_rasters returns: (n_files, height * width) = (3, 2100)
    """
    H, W, N = 30, 70, 3
    memmap_path = tmp_path / "original.npy"
    flat = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(N, H * W))
    flat[:] = np.random.rand(H, W, N).astype(np.float32).transpose(2, 0, 1).reshape(N, -1)

    base_raster_path = str(tmp_path / "base.tif")
    output_paths = [str(tmp_path / f"out_{i}.tif") for i in range(N)]

    # Create base raster
    with rasterio.open(
        base_raster_path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(flat[0].reshape(H, W), 1)

    save_rasters(base_raster_path, output_paths, flat, n_jobs=1)

    loaded = read_rasters(raster_files=output_paths, backend="python", n_jobs=1)

    assert loaded.shape == (N, H * W), \
        f"Shape mismatch: saved {flat.shape}, loaded {loaded.shape}"
    assert_array_equal(loaded, flat)
