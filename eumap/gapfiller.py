'''
Gapfilling approaches using temporal and spatial neighbor pixels
'''
from .misc import ttprint
from .datasets import DATA_ROOT_NAME
import multiprocessing
import math
from itertools import cycle, islice
import math
import traceback
import rasterio
import threading
import time
from scipy import ndimage
from pathlib import Path
from rasterio.windows import Window
from pyts.decomposition import SingularSpectrumAnalysis

from . import parallel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from typing import List, Dict, Union

from itertools import chain
import numpy as np
from osgeo import gdal
from osgeo import osr
import cv2 as cv
import os

from abc import ABC, abstractmethod
import bottleneck as bc
from .misc import ttprint
from .raster import read_rasters, write_new_raster

CPU_COUNT = multiprocessing.cpu_count()
_OUT_DIR = os.path.join(os.getcwd(), 'gapfilled')

class ImageGapfill(ABC):
  """
  Abstract class responsable for read/save the raster files used/produced 
  by all implemented gapfilling methods. 
    
  :param fn_files: Raster file paths to be read and gapfilled.
  :param data: 3D array where the last dimension is the time.
  :param verbose: Use ``True`` to print the progress of the gapfilled.
  """
  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    drop_outliers:bool = False,
    th_outliers:bool = [2,98],
    verbose:bool = True,
  ):

    if data is None and fn_files is None:
      raise Exception(f'You should provide at least one of these: data or fn_files')

    n_jobs_read = CPU_COUNT
    if n_jobs_read > 10:
      n_jobs_read = 10

    if data is None:
      self.fn_files = fn_files
      self.data, _ = read_rasters(raster_files=fn_files, verbose=verbose, n_jobs=n_jobs_read)
    else:
      self.data = data

    self.drop_outliers = drop_outliers
    self.th_outliers = th_outliers

    self.verbose = verbose

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)

  def _n_gaps(self, data = None):
    if data is None:
      data = self.data

    return np.sum(np.isnan(data).astype('int'))

  def _remove_outliers(self, ts_data):
    lower_pct, higher_pct = np.nanpercentile(ts_data, q = self.th_outliers)
    outlier_mask = np.logical_or(ts_data <= lower_pct, ts_data >= higher_pct)
    ts_data[outlier_mask] = np.nan
    return ts_data

  def _perc_gaps(self, gapfilled_data):
    data_nan = np.isnan(self.data)
    n_nan = np.sum(data_nan.astype('int'))
    n_gapfilled = np.sum(data_nan[~np.isnan(gapfilled_data)].astype('int'))

    return n_gapfilled / n_nan

  def run(self):
    """
    Execute the gapfilling approach.
    
    """
    self._verbose(f'There are {self._n_gaps()} gaps in {self.data.shape}')

    start = time.time()

    if self.drop_outliers:
      self._verbose(f'Removing temporal outliers considering {self.th_outliers} quantiles.')
      self.data = parallel.apply_along_axis(self._remove_outliers, 2, self.data)

    self.gapfilled_data, self.gapfilled_data_flag = self._gapfill()

    gaps_perc = self._perc_gaps(self.gapfilled_data)
    self._verbose(f'{gaps_perc*100:.2f}% of the gaps filled in {(time.time() - start):.2f} segs')

    if gaps_perc < 1:
      self._verbose(f'Remained gaps: {self._n_gaps(self.gapfilled_data)}')

    return self.gapfilled_data

  @abstractmethod
  def _gapfill(self):
    pass

  def save_rasters(self, 
      out_dir, 
      data_type:str = None, 
      out_mantain_subdirs:bool = True,
      root_dir_name:str = DATA_ROOT_NAME, 
      fn_files:List = None, 
      spatial_win:Window = None
    ):
    """
    Save the result in raster files maintaining the same filenames
    of the read rasters.
    
    :param out_dir: Folder path to save the files.
    :param data_type: Convert the rasters for the specified Numpy ``data_type`` before save.
    :param out_mantain_subdirs: Keep the full folder hierarchy of the read raster in the ``out_dir``.
    :param root_dir_name: Keep the relative folder hierarchy of the read raster in the ``out_dir`` 
                          considering the ``root_dir_name``.
    :param fn_files: Raster file paths to retreive the filenames. Use this parameter in
                     situations where the ``data`` parameter is informed in the class constructor.
    :param spatial_win: Save the files considering the specified spatial window.
    """

    if fn_files is not None:
      self.fn_files = fn_files

    if self.fn_files is None:
      raise Exception(f'To use save_rasters you should provide a fn_files list.')

    base_img_i = 0
    fn_base_img = self.fn_files[base_img_i]
    while not fn_base_img.is_file():
      base_img_i += 1
      fn_base_img = self.fn_files[base_img_i]

    n_files = len(self.fn_files)
    n_data = self.gapfilled_data.shape[2]

    if n_files != n_data:
      raise Exception(f'The fn_files incompatible with gapfilled_data shape ({self.gapfilled_data.shape})')

    out_dir = Path(out_dir)
    fn_result = []

    for i in range(0,n_files):

      src_fn = Path(self.fn_files[i])

      if out_mantain_subdirs:
        cur_out_dir = out_dir.joinpath(str(src_fn.parent).split(root_dir_name)[-1][1:])
        fn_gapfilled_data = cur_out_dir.joinpath('%s.tif' % src_fn.stem)
        fn_gapfilled_data_flag = cur_out_dir.joinpath('%s_flag.tif' % src_fn.stem)
      else:
        fn_gapfilled_data = out_dir.joinpath('%s.tif' % src_fn.stem)
        fn_gapfilled_data_flag = out_dir.joinpath('%s_flag.tif' % src_fn.stem)

      fn_gapfilled_data.parent.mkdir(parents=True, exist_ok=True)

      write_new_raster(fn_base_img, fn_gapfilled_data, self.gapfilled_data[:,:,i], data_type = data_type, spatial_win = spatial_win)
      write_new_raster(fn_base_img, fn_gapfilled_data_flag, self.gapfilled_data_flag[:,:,i], data_type = data_type, spatial_win = spatial_win)

      fn_result.append(fn_gapfilled_data)
      fn_result.append(fn_gapfilled_data_flag)

    self._verbose(f'Number of files saved in {out_dir}: {len(fn_result)}')
    return fn_result

class _TMWMData():
  def __init__(self, time_order: List, time_data, time_win_size = 5,
    cpu_max_workers:int = multiprocessing.cpu_count(),
    engine='CPU',
    gpu_tile_size:int = 250):

    self.time_win_size = time_win_size
    self.time_order = time_order
    self.time_data = time_data
    self.cpu_max_workers = cpu_max_workers
    self.gpu_tile_size = gpu_tile_size
    self.engine = engine

    self.gapfilled_data = {}

  def _get_time_window(self, t, max_win_size, win_size):
    return(
      (t - win_size) if (t - win_size) > 0 else 0,
      (t + win_size) if (t + win_size) < (max_win_size-1) else (max_win_size-1),
    )

  def _key_from_layer_pos(self, time, layer_pos, win_size):
    _, _, n_layers = self.time_data[time].shape
    t1, t2 = self._get_time_window(layer_pos, n_layers, win_size)
    return self._key_from_time(time, t1, t2)

  def _key_from_time(self, time, t1, t2):
    return f'{time}_{t1}_{t2}'

  def _calc_nanmedian(self, time, t1, t2):
    #return self._key_from_time(time, t1, t2), np.nanmedian(self.time_data[time][:,:,t1:t2], axis=2)
    return self._key_from_time(time, t1, t2), bc.nanmedian(self.time_data[time][:,:,t1:t2], axis=2)
    #med = nanPercentile(np.transpose(self.time_data[time][:,:,t1:t2], axes=[2, 0, 1]), [50])
    #return self._key_from_time(time, t1, t2), med[0]

  def _cpu_processing(self, args):
    for key, data in parallel.ThreadGeneratorLazy(self._calc_nanmedian, iter(args), max_workers=self.cpu_max_workers, chunk=self.cpu_max_workers):
      self.gapfilled_data[key] = data

  def _gpu_processing(self, args):

    import cupy as cp

    x_size, y_size, _ = self.time_data[self.time_order[0]].shape

    for x1 in chain(range(0, x_size, self.gpu_tile_size), [self.gpu_tile_size]):
      for y1 in chain(range(0, x_size, self.gpu_tile_size), [self.gpu_tile_size]):

        gpu_data = {}

        x2 = ( (x_size-1) if (x1 + self.gpu_tile_size) >= x_size else (x1 + self.gpu_tile_size))
        y2 = ( (y_size-1) if (y1 + self.gpu_tile_size) >= y_size else (y1 + self.gpu_tile_size))

        for time in self.time_order:
          gpu_data[time] = cp.asarray(self.time_data[time][x1:x2,y1:y2,:])

        for time, t1, t2 in iter(args):
          key = self._key_from_time(time, t1, t2)
          if key not in self.gapfilled_data:
            self.gapfilled_data[key] = np.zeros((x_size, y_size), dtype='float32')

          gpu_median = cp.nanmedian(gpu_data[time][:,:,t1:t2], axis=2)
          self.gapfilled_data[key][x1:x2,y1:y2] = cp.asnumpy(gpu_median)
          gpu_median = None

        gpu_data = None
  

  def run(self):
    """Perform the implemented gapfilling operation

    """
    self.gapfilled_data = {}
    self.available_windows = {}

    args = set()

    for time in self.time_order:

      _, _, n_layers = self.time_data[time].shape
      self.available_windows[time] = []

      for layer_pos in range(0, n_layers):

        self.available_windows[time].append([])

        for w in range(self.time_win_size, n_layers, self.time_win_size):
          t1, t2 = self._get_time_window(layer_pos, n_layers, w)
          self.available_windows[time][layer_pos].append(w)
          args.add((time, t1, t2))

    ttprint(f'Calculating {len(args)} gap filling possibilities')

    if self.engine == 'CPU':
      ttprint(f'Using cpu engine')
      self._cpu_processing(args)
    elif  self.engine == 'GPU':
      ttprint(f'Using GPU engine')
      self._gpu_processing(args)

    ttprint(f'Possibilities calculated')

  def get(self, time, layer_pos):

    time_list = (time if type(time) == list else [time])
    result = {}

    for t in time_list:
      try:
        for win_size in (self.available_windows[t][layer_pos]):
          result[win_size] = ( [] if t not in result else result[win_size])
          key = self._key_from_layer_pos(t, layer_pos, win_size)
          result[win_size].append(self.gapfilled_data[key])
      except:
        traceback.format_exc()
        continue

    for win_size in list(result.keys()):
      result[win_size] = bc.nanmean(np.stack(result[win_size], axis=2), axis=2)

    return result

class TMWM(ImageGapfill):
  """
  Temporal Moving Window Median able to gapfill the missing pixels using the temporal neighborhood
  by a growing window to calculate several median possibilities. The approach prioritizes
  values derived for **1–the same day/month/season**, **2–neighboring days/months/seasons**, 
  **and 3–all the year**. 

  For example, in the best case scenario a missing pixel on Jan-2005 is filled using a median
  value derived from January of other years, and in the worst case scenario it uses a value derived
  of all the months and available years. If a pixel remains with a missing value, it is because
  there is **no valid data on the entire time series**.

  :param fn_files: Raster file paths to be read and gapfilled.
  :param data: 3D array where the last dimension is the time.
  :param yearly_temporal_resolution: Season size of a year (For monthly time series it is equal ``12``).
  :param time_win_size: Size of the temporal window used to calculate the median value possibilities.
  :param cpu_max_workers: Number of CPU cores to be used in parallel.
  :param engine: Execute in ``CPU`` [1] or ``GPU`` [2].
  :param gpu_tile_size: Tile size used to split the processing in the GPU.
  :param verbose: Use ``True`` to print the progress of the gapfilled.

  >>> # For a 4-season time series
  >>> tmwm = gapfiller.TMWM(fn_files=fn_rasters, yearly_temporal_resolution=4, time_win_size=4)
  >>> data_tmwm = tmwm.run()
  >>> fn_rasters_tmwm = tmwm.save_rasters('./gapfilled_tmwm')

  [1] `Bootleneck nanmedian <https://kwgoodman.github.io/bottleneck-doc/reference.html#bottleneck.nanmedian>`_

  [2] `CuPY nanmedian <https://docs.cupy.dev/en/stable/reference/generated/cupy.nanmedian.html>`_
  
  """

  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    yearly_temporal_resolution = None,
    time_win_size: int=8,
    cpu_max_workers:int = multiprocessing.cpu_count(),
    engine='CPU',
    gpu_tile_size:int = 250,
    verbose = True,
  ):

    self.cpu_max_workers = cpu_max_workers
    self.time_win_size = time_win_size
    self.engine = engine
    self.gpu_tile_size = gpu_tile_size
    self.yearly_temporal_resolution = yearly_temporal_resolution

    super().__init__(fn_files=fn_files, data=data, verbose=verbose)
    self._do_time_data()

    if self.time_win_size > math.floor(self.n_years/2):
      raise Exception(f'The time_win_size can not bigger than {math.floor(self.n_years/2)}')

  def _do_time_data(self):
    total_times = self.data.shape[2]

    self.time_order = [ str(time) for time in range(0, self.yearly_temporal_resolution) ]
    self.time_data = {}
    self.n_years = 0

    for time in self.time_order:
      idx = list(range(int(time), total_times, self.yearly_temporal_resolution))

      if len(idx) > self.n_years:
        self.n_years = len(idx)

      self.time_data[time] = self.data[:,:,idx]
      self._verbose(f"Data {self.time_data[time].shape} organized in time={time}")

  def _get_neib_times(self, time):

    total_times = len(self.time_order)
    i = self.time_order.index(time)

    time_order_rev = self.time_order.copy()
    time_order_rev.reverse()

    neib_times = []

    for j in range(1, math.ceil(total_times / 2) + 1):

      # Setup slicer in the right positions
      after = islice(cycle(self.time_order), i, None)
      before = islice(cycle(time_order_rev), total_times-i, None)
      next(after)

      before_tim = [next(before) for t in range(0,j)]
      after_tim = [next(after) for t in range(0,j)]

      neib_times.append((before_tim, after_tim))

    return neib_times

  def _fill_gaps(self, time, layer_pos, newdata_dict, verbose_suffix='', gapflag_offset = 0):

    end_msg = None

    for i in newdata_dict.keys():

      gaps_mask = np.isnan(self.time_data[time][:,:,layer_pos])
      newdata = newdata_dict[i][gaps_mask]

      gaps_pct = np.count_nonzero(~np.isnan(newdata))
      newdata_pct = np.count_nonzero(gaps_mask.flatten())

      if newdata_pct != 0:
        gapfilled_pct =  gaps_pct / newdata_pct

        self.time_data[time][:,:,layer_pos][gaps_mask] = newdata
        self.time_data_gaps[time][:,:,layer_pos][gaps_mask] = int(i) + gapflag_offset

        if gapfilled_pct == 1:
          end_msg = f'Layer {layer_pos}: reached {100*gapfilled_pct:.2f}% with {i}-year window from {verbose_suffix})'
          break

    return end_msg

  def _fill_gaps_all_times(self, time, layer_pos):

    _, _, n_layers = self.time_data[time].shape

    end_msg = None
    all_data = self.tmwm_data.get(self.time_order, layer_pos)
    end_msg = self._fill_gaps(time, layer_pos, all_data, verbose_suffix='all seasons', gapflag_offset = n_layers*2)

    return end_msg

  def _fill_gaps_neib_time(self, time, layer_pos):

    newdata_dict = {}
    end_msg = None
    _, _, n_layers = self.time_data[time].shape

    for before_times, after_times in self._get_neib_times(time):

      before_data = self.tmwm_data.get(before_times, layer_pos)
      after_data = self.tmwm_data.get(after_times, layer_pos)

      keys = list(set(before_data.keys()).intersection(after_data.keys()))
      keys.sort()

      for i in keys:
        stacked = np.stack([before_data[i], after_data[i]], axis=2)
        valid_mean = np.any(np.isnan(stacked), axis=2)
        stacked[valid_mean] = np.nan
        newdata_dict[i] = bc.nanmean(stacked, axis=2)

      end_msg = self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='neighborhood seasons', gapflag_offset = n_layers)
      if end_msg is not None:
        break

    return end_msg

  def _fill_gaps_same_time(self, time, layer_pos):
    newdata_dict = self.tmwm_data.get(time, layer_pos)
    return self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='same season')

  def _fill_image(self, time, layer_pos):

    end_msg = self._fill_gaps_same_time(time, layer_pos)

    if end_msg is None:
      end_msg = self._fill_gaps_neib_time(time, layer_pos)

    if end_msg is None:
      end_msg = self._fill_gaps_all_times(time, layer_pos)

    return end_msg

  def _gapfill(self):

    self.time_data_gaps = {}

    self.tmwm_data = _TMWMData(self.time_order, self.time_data, self.time_win_size,
      cpu_max_workers=self.cpu_max_workers, engine=self.engine, gpu_tile_size=self.gpu_tile_size)
    self.tmwm_data.run()

    layer_args = []

    for time in self.time_order:

      nrows, ncols, n_layers = self.time_data[time].shape
      self.time_data_gaps[time] = np.zeros((nrows, ncols, n_layers), dtype='int8')

      for layer_pos in range(0, n_layers):
        layer_args.append((time, layer_pos))

    for end_msg in parallel.ThreadGeneratorLazy(self._fill_image, iter(layer_args), max_workers=self.cpu_max_workers, chunk=self.cpu_max_workers):
      end_msg = True

    gapfilled_data = []
    gapfilled_data_flag = []

    for i in range(0, self.n_years):
      for time in self.time_order:

        time_len = self.time_data[time].shape[2]
        if (time_len <= self.n_years):
          gapfilled_data.append(self.time_data[time][:,:,i])
          gapfilled_data_flag.append(self.time_data_gaps[time][:,:,i])

    gapfilled_data = np.stack(gapfilled_data, axis=2)
    gapfilled_data_flag = np.stack(gapfilled_data_flag, axis=2)

    return gapfilled_data, gapfilled_data_flag

class TLI(ImageGapfill):
  """
  Temporal Linear Interpolation able to gapfill the missing pixels using a linear regression [1]
  over the time using all valid pixels.

  :param fn_files: Raster file paths to be read and gapfilled.
  :param data: 3D array where the last dimension is the time.
  :param verbose: Use ``True`` to print the progress of the gapfilled.

  >>> tli = gapfiller.TLI(fn_files=fn_rasters)
  >>> data_tli = tli.run()
  >>> fn_rasters_tli = tli.save_rasters('./gapfilled_tli')

  [1] `Scikit-learn linear regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_

  """

  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    verbose = True
  ):
    super().__init__(fn_files=fn_files, data=data, verbose=verbose)

  def _temporal_linear_inter(self, data):

    y = data.reshape(-1)
    y_nan = np.isnan(y)
    y_size = data.shape[0]
    y_valid = y[~y_nan]

    n_gaps = np.sum(y_nan.astype('int'))
    data_flag = np.zeros(y.shape)

    if n_gaps > 0 and y_valid.shape[0] > 3:

        X = np.array(range(0, y_size))

        model = LinearRegression()
        model.fit(X[~y_nan].reshape(-1, 1), y_valid)
        y_pred = model.predict(X.reshape(-1, 1))
        rmse = mean_squared_error(y[~y_nan], y_pred[~y_nan], squared=False)

        y[y_nan] = y_pred[y_nan]
        data_flag[y_nan] = rmse

        return np.stack([ y.reshape(data.shape), data_flag.reshape(data.shape) ], axis=1)
    else:
        return np.stack([ data, data_flag ], axis=1)

  def _gapfill(self):
    result = parallel.apply_along_axis(self._temporal_linear_inter, 2, self.data)
    return (result[:,:,:,0], result[:,:,:,1])

class SSA(ImageGapfill):
  """Singular Spectral Analysis
    
  :param fn_files: TODO
  :param data: TODO
  :param window_size: TODO
  :param ngroups: TODO
  :param reconstruct_ngroups: TODO
  :param verbose: TODO
  """

  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    window_size:int = 4,
    ngroups:int = 4,
    reconstruct_ngroups:int = 2,
    season_size:int = 4,
    verbose = True
  ):
    
    self.window_size = window_size
    self.ngroups = ngroups
    self.season_size = season_size
    self.reconstruct_ngroups = reconstruct_ngroups

    super().__init__(fn_files=fn_files, data=data, verbose=verbose)

  def gapfill_ltm(self, ts_data, season_size=12, agg_year=5):
    ts_size = ts_data.shape[1]
    agg_size = (season_size * agg_year)
    i_list = [ (i0, (ts_size if (i0 + agg_size) > ts_size else (i0 + agg_size))) for i0 in range(0,ts_size,agg_size) ]  
    
    arr_ltm = []
    for i0, i1 in i_list:
      ts_year = ts_data[0, i0:i1].reshape(-1, season_size)
      
      if self._perc_gaps(ts_year) == 1.0:
        ts_ltm = np.empty((i1 - i0))
        ts_ltm[:] = np.nan
      else:
        ts_ltm = bc.nanmean(ts_year, axis=0)
        repetions = int((i1 - i0) / season_size)
        ts_ltm = np.tile(ts_ltm, repetions)

      arr_ltm.append(ts_ltm)
    
    ts_ltm = np.concatenate(arr_ltm).reshape(ts_data.shape)
    
    return ts_ltm

  def _all_nan(self, data):
    return np.all(np.isnan(data))

  def _perc_gaps(self, data):
    return np.sum(np.isnan(data).astype('int')) / data.flatten().shape[0]

  def _ssa_reconstruction(self, data):

    ts_data = data.reshape(1,-1).astype('Float32')
    ts_flag = np.zeros(ts_data.shape)

    if self._all_nan(ts_data):
      return np.stack([ ts_data.reshape(data.shape),
                ts_flag.reshape(data.shape) ], axis=1)

    na_mask_initial = np.isnan(ts_data)

    if self._all_nan(ts_data):
      return np.stack([ ts_data.reshape(data.shape),
                ts_flag.reshape(data.shape) ], axis=1)

    ts_gapfilled = ts_data.copy()
    na_mask_no_outiler = np.isnan(ts_gapfilled)
    n_years = int(ts_gapfilled.shape[1] / self.season_size)

    if self._perc_gaps(ts_gapfilled.flatten()) <= 0.80:
      for year in range(5, n_years, 3):
        ts_gapfilled_y = self.gapfill_ltm(ts_data, season_size=self.season_size, agg_year=year)    
        gapfill_mask = np.logical_and(np.isnan(ts_gapfilled),~np.isnan(ts_gapfilled_y))
        ts_gapfilled[gapfill_mask] = ts_gapfilled_y[gapfill_mask]
        gapfill_mask = np.isnan(ts_gapfilled)
    
        if self._perc_gaps(ts_gapfilled) == 0:
          break
    
    if self._perc_gaps(ts_gapfilled.flatten()) < 1.0:
      gapfill_mask = np.isnan(ts_gapfilled)
      ts_mean = bc.nanmean(ts_gapfilled)
      ts_gapfilled[gapfill_mask] = ts_mean

    ssa = SingularSpectrumAnalysis(window_size=self.window_size, groups=self.ngroups)
    ts_components = ssa.fit_transform(ts_gapfilled)
    ts_reconstructed = np.sum(ts_components[0:self.reconstruct_ngroups,:], axis=0)
    ts_flag[na_mask_initial] = 2
    ts_flag[~na_mask_no_outiler] = 1
    
    ssa = SingularSpectrumAnalysis(window_size=self.window_size, groups=self.ngroups)
    ts_components = ssa.fit_transform(ts_gapfilled)
    ts_reconstructed = np.sum(ts_components[0:self.reconstruct_ngroups,:], axis=0)
    
    return np.stack([ts_reconstructed.reshape(data.shape), ts_flag.reshape(data.shape)], axis=1)

  def _gapfill(self):
    result = parallel.apply_along_axis(self._ssa_reconstruction, 2, self.data)
    return (result[:,:,:,0], result[:,:,:,1])

class InPainting(ImageGapfill):
  """
  Approach that uses a inpating technique [1] to gapfill raster data
  using the region neighborhood.
   
  :param fn_files: Raster file paths to be read and gapfilled.
  :param data: 3D array where the last dimension is the time.
  :param space_win: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
  :param data_mask: 2D array indicating a valid areas, equal 1, where in case of gaps should be filled.
  :param mode:  Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA [1]
  :param verbose: Use ``True`` to print the progress of the gapfilled.
  
  >>> # Considerer land_mask as 2D numpy array where 1 indicates land
  >>> inPainting = gapfiller.InPainting(fn_files=fn_rasters, space_win = 10, data_mask=land_mask)
  >>> data_inp = inPainting.run()
  >>> fn_rasters_inp = inPainting.save_rasters('./gapfilled_inp')

  [1] `OpenCV Tutorial - Image Inpainting <https://docs.opencv.org/4.5.2/df/d3d/tutorial_py_inpainting.html>`_
  
  """

  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    verbose:bool = True,
    space_win:int = 10,
    data_mask:np.array = None,
    mode = cv.INPAINT_TELEA
  ):

    self.data_mask = data_mask
    self.space_win = space_win
    self.mode = mode
    super().__init__(fn_files=fn_files, data=data, verbose=verbose)

  def _inpaint(self, data, i=0):

    # necessary for a proper inpaint execution
    data_copy = np.copy(data)
    initial_value = np.nanmedian(data_copy)
    na_data_mask = np.isnan(data_copy)
    data_copy[na_data_mask] = initial_value
    data_gapfilled = cv.inpaint(data_copy.astype('float32'), na_data_mask.astype('uint8'), self.space_win, self.mode)
    
    return (data_gapfilled, i)

  def _gapfill(self):

    data = np.copy(self.data)
    data_flag = np.zeros(data.shape)

    max_workers = multiprocessing.cpu_count()
    n_times = data.shape[2]

    args = []

    for i in range(0, n_times):
      if self._n_gaps(data[:,:,i]) > 0:
        args.append((data[:,:,i], i))
      
    result_set = {}
    for band_data, i in parallel.ThreadGeneratorLazy(self._inpaint, iter(args), max_workers=max_workers, chunk=max_workers*2):
        result_set[f'{i}'] = band_data

    for i in range(0, n_times):
      key = f'{i}'
      
      if (key in result_set):
        band_data_gapfilled = result_set[key]
        
        gapfill_mask = np.logical_and(~np.isnan(band_data_gapfilled), np.isnan(data[:,:,i]))
        gapfill_mask = np.logical_and(gapfill_mask, (self.data_mask == 1))
        
        data[:,:,i][gapfill_mask] = band_data_gapfilled[gapfill_mask]
        data_flag[:,:,i][gapfill_mask] = 1

    return data, data_flag

class SMWM(ImageGapfill):
  """
  Spatial Moving Window Median able to gapfill the missing pixels using the spatial neighborhood
  by a growing window to calculate the median [1]. The smaller windows will be prioritized 
  over larger ones.

  :param fn_files: Raster file paths to be read and gapfilled.
  :param data: 3D array where the last dimension is the time.
  :param verbose: Use ``True`` to print the progress of the gapfilled.
  :param space_win_list: Spatial window possibilities used by the median calculation.

  >>> smwm = gapfiller.SMWM(fn_files=fn_rasters,  space_win_list = [5, 10, 50])
  >>> data_smwm = smwm.run()
  >>> fn_rasters_smwm = smwm.save_rasters('./gapfilled_smwm')

  [1] `Scipy multidimensional median_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html>`_

  """

  def __init__(self,
    fn_files:List = None ,
    data:np.array = None,
    verbose = True,
    space_win_list = range(5, 21, 5)
  ):

    self.space_win_list = space_win_list
    super().__init__(fn_files=fn_files, data=data, verbose=verbose)

  def _spatial_median(self, data, i=0, win_size=5):
    return ndimage.median_filter(data.astype('float32'), size=win_size), i, win_size,

  def _gapfill(self):

    data = np.copy(self.data)
    data_flag = np.zeros(data.shape)

    max_workers = multiprocessing.cpu_count()
    n_times = data.shape[2]

    args = []

    for win_size in self.space_win_list:
      for i in range(0, n_times):
        if self._n_gaps(data[:,:,i]) > 0:
          args.append((data[:,:,i], i, win_size))

    result_set = {}
    for band_data, i, win_size in parallel.ThreadGeneratorLazy(self._spatial_median, iter(args), max_workers=max_workers, chunk=max_workers*2):
        result_set[f'{win_size}-{i}'] = band_data

    for win_size in self.space_win_list:
      for i in range(0, n_times):
        key = f'{win_size}-{i}'
        if (key in result_set):
          band_data_gapfilled = result_set[f'{win_size}-{i}']
          gapfill_mask = np.logical_and(~np.isnan(band_data_gapfilled), np.isnan(data[:,:,i]))
          data[:,:,i][gapfill_mask] = band_data_gapfilled[gapfill_mask]
          data_flag[:,:,i][gapfill_mask] = win_size

    return data, data_flag
