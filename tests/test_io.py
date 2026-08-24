from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from skmap import io
from skmap.data import toy
from skmap.misc import make_tempdir


class TestReadRaster:
    def test_001(self) -> None:
        out_data = io.read_rasters(str(toy._static_raster()[0]), backend="cpp")
        assert out_data.shape == (1, 65536)

    def test_002(self) -> None:
        out_data = io.read_rasters(toy._static_raster(), backend="cpp")
        assert out_data.shape == (2, 65536)

    def test_003(self) -> None:
        out_data = io.read_rasters(toy._static_raster(), backend="cpp")
        assert np.nanmax(out_data) == 523.0

    def test_005(self) -> None:
        out_data = io.read_rasters(
            toy._static_raster(), window=Window(100, 100, 28, 28), backend="cpp"
        )
        assert out_data.shape == (2, 784)


class TestSaveRaster:
    def test_001(self) -> None:
        out_data = io.read_rasters(toy._static_raster(), backend="cpp")
        base_raster = str(toy._static_raster()[0])
        out_files = io.save_rasters_cpp(
            base_raster, out_data, "test", str(make_tempdir())
        )

        ds = rasterio.open(out_files[0])
        dtype = ds.dtypes[0]
        Path(out_files[0]).unlink()

        assert dtype == "int16"
