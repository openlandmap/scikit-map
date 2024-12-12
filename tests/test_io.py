import pytest

from rasterio.windows import Window
import numpy as np

from skmap.data import toy
from skmap import io

class TestReadRaster:

  def test_001(self):
    out_data = io.read_rasters_cpp(str(toy._static_raster()[0]))
    assert (out_data.shape == (1, 65536))

  def test_002(self):
    out_data = io.read_rasters_cpp(toy._static_raster())
    assert (out_data.shape == (2, 65536))

  def test_003(self):
    out_data = np.empty((2, 65536), dtype='float32')
    out_data = io.read_rasters_cpp(toy._static_raster(), out_data=out_data)
    assert (np.max(out_data) == 523.0)

  def test_004(self):
    out_data = np.empty((4, 65536), dtype='float32')
    out_data = io.read_rasters_cpp(toy._static_raster(), out_data=out_data, out_idx=[2,3])
    assert(np.min(out_data[2:3,:]) == 17.0)

  def test_005(self):
    out_data = io.read_rasters_cpp(toy._static_raster(), window=Window(100,100,28,28))
    assert(out_data.shape == (2, 784))
  
  def test_006(self):
    out_data = np.empty((4, 784), dtype='float32')
    out_data = io.read_rasters_cpp(toy._static_raster(), out_data=out_data, out_idx=[2,3], window=Window(100,100,28,28))
    assert(np.max(out_data[2:3,:]) == 106.0)