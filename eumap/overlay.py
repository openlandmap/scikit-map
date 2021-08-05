'''
Point overlay considering multiple raster layers (space and spacetime)
'''
from typing import List
import rasterio
import rasterio.windows
from pathlib import Path
import numpy as np
import os
import pandas as pd
import geopandas as gpd
import concurrent.futures
import multiprocessing
import traceback
import re
import math

from . import parallel
from .misc import ttprint, find_files

import warnings
warnings.filterwarnings('ignore')

#%%

class _ParallelOverlay:
    # optimized for up to 200 points and about 50 layers
    # sampling only first band in every layer
    # assumption is that all layers have same blocks

    def __init__(self, 
      points_x: np.ndarray, 
      points_y:np.ndarray, 
      fn_layers:List[str], 
      max_workers:int = multiprocessing.cpu_count(), 
      verbose:bool = True
    ):
        
      self.error_val = -32768
      self.points_x = points_x
      self.points_y = points_y
      self.points_len = len(points_x)
      self.fn_layers = fn_layers
      self.max_workers = max_workers
      self.verbose = verbose

      self.layer_names = [fn_layer.with_suffix('').name for fn_layer in fn_layers]
      sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
      self.dimensions = [self._get_dimension(src) for src in sources]

      self.points_blocks = self.find_blocks()
      self.result = None

    @staticmethod
    def _get_dimension(src):
      return (src.height, src.width, *src.block_shapes[0], *src.transform.to_gdal())

    def _find_blocks_for_src_mt(self, ij, block, src, ptsx, ptsy):
      left, bottom, right, top = rasterio.windows.bounds(block, src.transform)
      ind = (ptsx>=left) & (ptsx<right) & (ptsy>bottom) & (ptsy<=top)

      if ind.any():
        inv_block_transform = ~rasterio.windows.transform(block, src.transform)
        col, row = inv_block_transform * (ptsx[ind], ptsy[ind])
        result = [block, np.nonzero(ind)[0], col.astype(int), row.astype(int)]
        return ij, result
      else:
        return None, None

    def _find_blocks_for_src(self, src, ptsx, ptsy):
      # find blocks for every point in given source
      blocks = {}

      args = src.block_windows(1)
      fixed_args = (src, ptsx, ptsy)

      chunk_size = math.ceil(len(list(src.block_windows(1))) / self.max_workers)

      for ij, result in parallel.ThreadGeneratorLazy(self._find_blocks_for_src_mt, args,
                                      self.max_workers, chunk=chunk_size, fixed_args = fixed_args):
        if ij is not None:
          blocks[ij] = result

      return blocks

    def find_blocks(self):
      # for every type of dimension find block for each point

      points_blocks = {}
      dimensions_set = set(self.dimensions)
      sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
      for dim in dimensions_set:
          src = sources[self.dimensions.index(dim)]
          points_blocks[dim] = self._find_blocks_for_src(src, self.points_x, self.points_y)
      return points_blocks

    def _sample_one_layer_sp(self, fn_layer):
      with rasterio.open(fn_layer) as src:
        #src=rasterio.open(fn_layer)
        dim = self._get_dimension(src)
        # out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])
        out_sample = np.full((self.points_len,), np.nan, np.float32)

        blocks = self.points_blocks[dim]
        for ij in blocks:
          # ij=next(iter(blocks)); (window, ind, col, row) = blocks[ij]
          (window, ind, col, row) = blocks[ij]
          try:
            data = src.read(1, window=window)
            mask = src.read_masks(1, window=window)
            sample = data[row,col].astype(np.float32)
            sample_mask = mask[row,col].astype(bool)
            sample[~sample_mask] = np.nan
            #sample = data[row.astype(int),col.astype(int)]
          except Exception as exception:
            traceback.print_exc()
            sample = self.error_val

          out_sample[ind] = sample
    
      return out_sample, fn_layer

    def _sample_one_block(self, args):
      out_sample, fn_layer, window, ind, col, row = args
      with rasterio.open(fn_layer) as src:
        
        try:
          data = src.read(1, window=window)
          sample = data[row,col]
        except Exception as exception:
          traceback.print_exc()
          sample = self.error_val
            
        out_sample[ind] = sample
          #return sample, ind

    def _sample_one_layer_mt(self, fn_layer):
      with rasterio.open(fn_layer) as src:
          dim = self._get_dimension(src)
          out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

      blocks = self.points_blocks[dim]
      args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)
      
      with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        for _ in executor.map(self._sample_one_block, args, chunksize=1000):
          pass #out_sample[ind] = sample
      
      return out_sample

    def _sample_one_layer_mp(self, fn_layer):
      with rasterio.open(fn_layer) as src:
          dim = self._get_dimension(src)
          out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

      blocks = self.points_blocks[dim]
      args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)
      
      with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        for _ in executor.map(self._sample_one_block, args, chunksize=1000):
            pass #out_sample[ind] = sample

      return out_sample

    def sample_v1(self):
      '''
      Serial sampling, block by block
      '''
      res={}
      n_layers = len(self.fn_layers)
      
      for i_layer, fn_layer in enumerate(self.fn_layers):
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
        # fn_layer=self.fn_layers[0]
        col = Path(fn_layer).with_suffix('').name
        sample,_ = self._sample_one_layer_sp(fn_layer)
        res[col]=sample

      return res

    def sample_v2(self):
      '''
      Serial layers, parallel blocks in layer
      '''
      
      res={}
      n_layers = len(self.fn_layers)
      
      for i_layer, fn_layer in enumerate(self.fn_layers):
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
        # fn_layer=self.fn_layers[0]
        col = Path(fn_layer).with_suffix('').name
        sample = self._sample_one_layer_mt(fn_layer)
        res[col]=sample

      return res

    def sample_v3(self):
      '''
      Parallel layers in threads, serial blocks in layer
      '''
      res={}
      i_layer=1
      n_layers=len(self.fn_layers)
      args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)
      
      for sample, fn_layer in parallel.ThreadGeneratorLazy(self._sample_one_layer_sp, args,
                          self.max_workers, self.max_workers*2):
        col = Path(fn_layer).with_suffix('').name
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {col}')
        res[col] = sample
        i_layer += 1

      return res

    def sample_v4(self):
      '''
      Parallel layers in processes, serial blocks in layer
      '''
      res={}
      i_layer=1
      n_layers=len(self.fn_layers)
      args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)
      #with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
          #for sample, fn_layer in executor.map(self._sample_one_layer_sp, args, chunksize=n_layers//self.max_workers):
      
      for sample, fn_layer in parallel.ProcessGeneratorLazy(self._sample_one_layer_sp, args, self.max_workers, 1):
        col = Path(fn_layer).with_suffix('').name
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {col}')
        res[col] = sample
        i_layer += 1

      return res

    def run(self):
      # For now sample_v3 is the best method, because is parallel and use less memory than sample_v4
      if self.result == None:
        self.result = self.sample_v3()
      else:
        if self.verbose:
          ttprint('You already did run the overlay. Geting the cached result')

      return self.result

class SpaceOverlay():
  """
  Overlay a set of points over multiple raster files.
  The retrieved pixel values are organized in columns 
  according to the filenames.

  :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
    the points.
  :param fn_layers: A list with the raster file paths. If it's not provided 
    the ``dir_layers`` and ``regex_layers`` are used to retrieve the raster files.
  :param dir_layers: A list of folders where raster files are located. The raster 
    are selected according to the pattern specified in ``regex_layers``.
  :param regex_layers: Pattern to select the raster files in ``dir_layers``.
    By default all GeoTIFF files are selected.
  :param max_workers: Number of CPU cores to be used in parallel. By default all cores
    are used.
  :param verbose: Use ``True`` to print the overlay progress.

  >>> from eumap.overlay import SpaceOverlay
  >>> 
  >>> spc_overlay = SpaceOverlay('./my_points.gpkg', ['./raster_dir_1', './raster_dir_2'])
  >>> result = spc_overlay.run()
  >>> 
  >>> print(result.shape)

  """

  def __init__(self, 
    points, 
    fn_layers:List[str] = [], 
    dir_layers:List[str] = [], 
    regex_layers = '*.tif', 
    max_workers:int = parallel.CPU_COUNT,
    verbose:bool = True
  ):
      
    if len(fn_layers) == 0:
      fn_layers = find_files(dir_layers, regex_layers)

    self.fn_layers = [ Path(l) for l in fn_layers ]
    
    if not isinstance(points, gpd.GeoDataFrame):
      points = gpd.read_file(points)

    self.pts = points
    self.pts['overlay_id'] = range(1,len(self.pts)+1)

    self.parallelOverlay = _ParallelOverlay(self.pts.geometry.x.values, self.pts.geometry.y.values, 
      fn_layers, max_workers, verbose)

  def run(self, 
    dict_newnames:set = {}
  ):
    """
    Execute the space overlay.

    :param dict_newnames: A dictionary used to update the column names after the overlay. 
      The ``key`` is the new name and the ``value`` is the raster file name (without extension).

    :returns: Data frame with the original columns plus the overlay result (one new 
      column per raster).
    :rtype: geopandas.GeoDataFrame
    """
    result = self.parallelOverlay.run()

    for col in result:
      new_col = col
      for newname in dict_newnames.keys():
        new_col = re.sub(dict_newnames[newname], newname, new_col)
      self.pts.loc[:,new_col] = result[col]

    return self.pts

class SpaceTimeOverlay():
  """
  Overlay a set of points over multiple raster considering the year information.
  The retrieved pixel values are organized in columns according to the filenames.

  :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
    the points.
  :param col_date: Date column to retrieve the year information.
  :param fn_layers: A list with the raster file paths. The file path placeholders 
    ``{year}``, ``{year_minus_1}``, ``{year_plus_1}`` are replaced considering the 
    year information of each point.
  :param max_workers: Number of CPU cores to be used in parallel. By default all cores
    are used.
  :param verbose: Use ``True`` to print the overlay progress.

  >>> from eumap.overlay import SpaceTimeOverlay
  >>> 
  >>> fn_layers = [ 'raster_{year_minus_1}1202..{year}0320.tif' ] # raster_20101202..20110320.tif, ...
  >>> spt_overlay = SpaceTimeOverlay('./my_points.gpkg', 'survey_date' fn_layers)
  >>> result = spt_overlay.run()
  >>> 
  >>> print(result.shape) 

  """

  def __init__(self, 
    points, 
    col_date:str, 
    fn_layers:List[str] = [], 
    max_workers:int = parallel.CPU_COUNT,
    verbose:bool = True
  ):

    if not isinstance(points, gpd.GeoDataFrame):
      points = gpd.read_file(points)

    self.pts = points
    self.col_date = col_date
    self.overlay_objs = {}
    self.verbose = verbose
    self.year_placeholder = '{year}'

    self.pts.loc[:,self.col_date] = pd.to_datetime(self.pts[self.col_date])
    self.uniq_years = self.pts[self.col_date].dt.year.unique()
    
    self.fn_layers = [ Path(l) for l in fn_layers ]

    for year in self.uniq_years:
      
      year = int(year)
      year_points = self.pts[self.pts[self.col_date].dt.year == year]
      
      fn_layers_year = []
      for fn_layer in self.fn_layers:
        fn_layers_year.append(Path(self._replace_year(fn_layer, year)))

      if self.verbose:
        ttprint(f'Overlay {len(year_points)} points from {year} in {len(fn_layers_year)} raster layers')
      
      self.overlay_objs[year] = SpaceOverlay(points=year_points, fn_layers=fn_layers_year, 
        max_workers=max_workers, verbose=verbose)

  def _replace_year(self, fn_layer, year = None):
    y, y_m1, y_p1 = ('', '', '')
    if year != None:
      y, y_m1, y_p1 = str(year), str((year - 1)), str((year + 1))

    fn_layer = str(fn_layer)

    return fn_layer \
        .replace('{year}', y) \
        .replace('{year_minus_1}', y_m1) \
        .replace('{year_plus_1}', y_p1) \

  def run(self):
    """
    Execute the spacetime overlay. It removes the year part from the column names. 
    For example, the raster ``raster_20101202..20110320.tif`` results in the column
    name ``raster_1202..0320``.

    :returns: Data frame with the original columns plus the overlay result (one new 
      column per raster).
    :rtype: geopandas.GeoDataFrame
    """
    self.result = None

    for year in self.uniq_years:
      
      year_newnames = {}      
      for fn_layer in self.fn_layers:
        
        name = str(fn_layer.stem)
        curname = self._replace_year(name, year)
        newname = self._replace_year(name)
        
        year_newnames[newname] = curname

      if self.verbose:
        ttprint(f'Running the overlay for {year}')
      year_result = self.overlay_objs[year].run(year_newnames)

      if self.result is None:
        self.result = year_result
      else:
        self.result = self.result.append(year_result)

    return self.result