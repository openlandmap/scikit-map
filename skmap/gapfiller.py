'''
Gapfilling approaches using temporal and spatial neighbor pixels
'''
import multiprocessing
import traceback
import warnings
import time
import math
import os
import gc

try:
    from itertools import cycle, islice, chain
    from typing import List, Dict, Union
    from rasterio.windows import Window
    from abc import ABC, abstractmethod
    from pathlib import Path
    from enum import Enum

    from pyts.decomposition import SingularSpectrumAnalysis
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import bottleneck as bc
    import cv2 as cv
    import numpy as np

    from .misc import ttprint
    from .datasets import DATA_ROOT_NAME
    from . import parallel
    from .raster import read_rasters, save_rasters

    class OutlierRemover(Enum):
      """
      Strategy to remove outliers considering the temporal domain.

      """
      Std = 1 #: Standard deviation moving window
      Percentile = 2 #: Percentile calculation

    class ImageGapfill(ABC):
      """
      Abstract class responsible for read/write the raster files in
      all implemented gapfilling methods.

      :param fn_files: Raster file paths to be read and gapfilled. The filename alphabetic order
        is used to infer the temporal order for the read data.
      :param data: 3D array where the last dimension is the time.
      :param outlier_remover: Strategy to remove outliers.
      :param std_win: Temporal window size used to calculate a local median and std.
      :param std_env: Number of std used to define a local envelope around the median.
        Values outside of this envelope are removed.
      :param perc_env: A list containing the lower and upper percentiles used to defined a global
        envelope for the time series. Values outside of this envelope are removed.
      :param n_jobs_io: Number of parallel jobs to read/write raster files.
      :param verbose: Use ``True`` to print the progress of the gapfilled.
      """
      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose:bool = True,
      ):

        if data is None and fn_files is None:
          raise ValueError(f'You should provide at least one of these: data or fn_files')

        self.n_jobs_io = n_jobs_io

        if data is None:
          self.fn_files = fn_files
          self.data, _ = read_rasters(raster_files=fn_files, verbose=verbose, n_jobs=self.n_jobs_io)
        else:
          self.data = data

        self.outlier_remover = outlier_remover
        self.perc_env = perc_env
        self.std_win = std_win
        self.std_env = std_env

        if self.std_win is not None and (self.std_win % 2) == 0:
          raise ValueError(f'The std_win argument must be an odd number')

        self.outlier_fn = None
        if OutlierRemover.Std == outlier_remover:
          self.outlier_fn = self._remove_outliers_std
        elif OutlierRemover.Percentile == outlier_remover:
          self.outlier_fn = self._remove_outliers_perc

        self.verbose = verbose

      def _verbose(self, *args, **kwargs):
        if self.verbose:
          ttprint(*args, **kwargs)

      def _n_gaps(self, data = None):
        if data is None:
          data = self.data

        return np.sum(np.isnan(data).astype('int'))

      def _remove_outliers_perc(self, ts_data):
        ts_data = ts_data.copy()
        lower_pct, higher_pct = np.nanpercentile(ts_data, q = self.perc_env)
        valid_mask = np.logical_and(ts_data >= lower_pct, ts_data <= higher_pct)
        ts_data[~valid_mask] = np.nan
        return ts_data

      def _remove_outliers_std(self, ts_data):

        if np.sum(np.isnan(ts_data).astype('int')) != ts_data.shape[0]:
          ts_data = ts_data.astype('float32')
          ts_size = ts_data.shape[0]

          with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            glob_med = bc.nanmedian(ts_data)

          ts_neib = int((self.std_win - 1) / 2)
          env = self.std_env
          min_len_std = 3

          outliers = []

          for i in range(0, ts_size):
            i0 = 0 if (i - ts_neib) < 0 else (i - ts_neib)
            i1 = ts_size if (i + ts_neib) + 1 > ts_size else (i + ts_neib) + 1

            # Expand the window in the boundaries years
            if i1 == ts_size:
              i0 -= self.std_win - (i1 - i0)

            if i0 == 0:
              i1 += self.std_win - (i1 - i0)

            win_data = ts_data[i0:i1].copy()
            win_data[np.isnan(win_data)] = glob_med
            
            with warnings.catch_warnings():
              warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
              med = bc.nanmedian(win_data)
              std = bc.nanstd(win_data)

            lower = (med - std * env)
            upper = (med + std * env)

            outliers.append(lower > ts_data[i] or upper < ts_data[i])

          outliers = np.array(outliers)
          ts_data[outliers] = np.nan

        return ts_data.astype('float16')

      def _perc_gapfilled(self, gapfilled_data):
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

        if self.outlier_fn:
          self._verbose(f'Removing temporal outliers using {self.outlier_remover}')
          data_noout = parallel.apply_along_axis(self.outlier_fn, 2, self.data)

          self.outlier_mask = np.logical_and(~np.isnan(self.data), np.isnan(data_noout))
          n_outliers = np.sum(self.outlier_mask.astype('int'))
          self._verbose(f'{n_outliers} outliers removed')

          self.data = data_noout

          self._verbose(f'There are {self._n_gaps()} gaps in {self.data.shape}')

        self.gapfilled_data, self.gapfilled_data_flag = self._gapfill()

        gaps_perc = self._perc_gapfilled(self.gapfilled_data)
        self._verbose(f'{gaps_perc*100:.2f}% of the gaps filled in {(time.time() - start):.2f} segs')

        if gaps_perc < 1:
          self._verbose(f'Remained gaps: {self._n_gaps(self.gapfilled_data)}')

        return self.gapfilled_data

      @abstractmethod
      def _gapfill(self):
        pass

      def save_rasters(self,
          out_dir,
          dtype:str = None,
          out_mantain_subdirs:bool = True,
          root_dir_name:str = DATA_ROOT_NAME,
          fn_files:List = None,
          nodata = None,
          spatial_win:Window = None,
          save_flag = True,
        ):
        """
        Save the result in raster files maintaining the same filenames
        of the read rasters.

        :param out_dir: Folder path to save the files.
        :param dtype: Convert the rasters for the specified Numpy ``dtype`` before save. This argument overwrite the values
          retrieved of ``fn_files``
        :param out_mantain_subdirs: Keep the full folder hierarchy of the read raster in the ``out_dir``.
        :param root_dir_name: Keep the relative folder hierarchy of the read raster in the ``out_dir``
          considering of the sub folders of ``root_dir_name``.
        :param fn_files: Raster file paths to retrieve the filenames and folders. Use this parameter in
          situations where the ``data`` parameter is informed in the class constructor. The pixel size,
          crs, extent, image size and nodata for the gapfilled rasters are retrieved from the first valid
          raster of ``fn_files``
        :param nodata: ``Nodata`` value used for the the gapfilled rasters. This argument overwrite the values
          retrieved of ``fn_files``. This argument doesn't affect the flag rasters (gapfill summary), which have
          ``nodata=0``.
        :param spatial_win: Save the gapfilled rasters considering the specified spatial window.
        :param save_flag: Save the flag rasters (gapfill summary).
        """

        if fn_files is not None:
          self.fn_files = fn_files

        if self.fn_files is None:
          raise Exception(f'To use save_rasters you should provide a fn_files list.')

        fn_base_img = None
        for i in range(0, len(self.fn_files)):
          if self.fn_files[i] is not None and self.fn_files[i].is_file():
            fn_base_img = self.fn_files[i]
            break

        n_files = len(self.fn_files)
        n_data = self.gapfilled_data.shape[2]

        if n_files != n_data:
          raise Exception(f'The fn_files incompatible with gapfilled_data shape ({self.gapfilled_data.shape})')

        if not isinstance(out_dir, Path):
          out_dir = Path(out_dir)

        fn_gapfilled_list = []
        fn_flag_list = []

        for i in range(0,n_files):

          src_fn = Path(self.fn_files[i])

          if out_mantain_subdirs:
            cur_out_dir = out_dir.joinpath(str(src_fn.parent).split(root_dir_name)[-1][1:])
            fn_gapfilled = cur_out_dir.joinpath('%s.tif' % src_fn.stem)
            fn_flag = cur_out_dir.joinpath('%s_flag.tif' % src_fn.stem)
          else:
            fn_gapfilled = out_dir.joinpath('%s.tif' % src_fn.stem)
            fn_flag = out_dir.joinpath('%s_flag.tif' % src_fn.stem)

          fn_gapfilled.parent.mkdir(parents=True, exist_ok=True)

          fn_gapfilled_list.append(fn_gapfilled)
          fn_flag_list.append(fn_flag)

        result = save_rasters(fn_base_img, fn_gapfilled_list, self.gapfilled_data,
          dtype = dtype, spatial_win = spatial_win, nodata=nodata)

        if save_flag:
          flag_result = save_rasters(fn_base_img, fn_flag_list, self.gapfilled_data_flag,
          dtype = 'uint8', spatial_win = spatial_win, nodata=0, n_jobs=self.n_jobs_io)
          result = result + flag_result

        self._verbose(f'Number of files saved in {out_dir}: {len(result)}')
        return result

    class _TMWMData():
      def __init__(self,
        time_order: List,
        time_data, time_win_size = 9,
        cpu_max_workers:int = multiprocessing.cpu_count(),
        engine='CPU',
        gpu_tile_size:int = 250
      ):

        self.time_win_size = time_win_size
        self.time_order = time_order
        self.time_data = time_data
        self.cpu_max_workers = cpu_max_workers
        self.gpu_tile_size = gpu_tile_size
        self.engine = engine

        self.gapfilled_data = {}

      def _get_time_window(self, t, n_layers, win_size):

        neib_size = int((win_size - 1) / 2)

        return(
          (t - neib_size) if (t - neib_size) > 0 else 0,
          (t + neib_size) if (t + neib_size) < (n_layers-1) else (n_layers-1),
        )

      def _key_from_layer_pos(self, time, layer_pos, win_size):
        _, _, n_layers = self.time_data[time].shape
        t1, t2 = self._get_time_window(layer_pos, n_layers, win_size)
        return self._key_from_time(time, t1, t2)

      def _key_from_time(self, time, t1, t2):
        return f'{time}_{t1}_{t2}'

      def _calc_nanmedian(self, time, t1, t2):
        with warnings.catch_warnings():
          warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
          result = bc.nanmedian(self.time_data[time][:,:,t1:t2+1].astype('float32'), axis=2)
        
        return self._key_from_time(time, t1, t2), result.astype('float16')

      def _cpu_processing(self, args):
        for key, data in parallel.job(self._calc_nanmedian, iter(args), n_jobs=self.cpu_max_workers, joblib_args={'require': 'sharedmem'}):
          self.gapfilled_data[key] = data

      def _gpu_processing(self, args):

        try:
          import cupy as cp
        except ImportError:
          warnings.warn('GPU engine requires cupy>=9.2.0')

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
              result[win_size] = ( [] if win_size not in result else result[win_size])
              key = self._key_from_layer_pos(t, layer_pos, win_size)
              result[win_size].append(self.gapfilled_data[key])
          except:
            traceback.format_exc()
            continue

        for win_size in list(result.keys()):
          win_stacked = np.stack(result[win_size], axis=2).astype('float32')
          
          with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            result[win_size] = bc.nanmedian(win_stacked, axis=2).astype('float16')

          del win_stacked

        return result

    class TMWM2(ImageGapfill):

      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        season_size = None,
        time_win_size: int=9,
        precomputed_flags:np.array = None, 
        max_same_season_win = None,
        max_neib_season_win = None,
        max_annual_win = None,
        time_win_direction = 'both',
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose = True
      ):

        super().__init__(fn_files=fn_files, data=data,
          outlier_remover=outlier_remover, std_win=std_win,
          std_env=std_env, perc_env=perc_env, n_jobs_io=n_jobs_io,
          verbose=verbose)

        self.time_win_size = time_win_size
        self.season_size = season_size

        self.data = np.ascontiguousarray(self.data.transpose((2,0,1)))
        
        self.time_series_size = self.data.shape[0]
        self.tile_size = (self.data.shape[1], self.data.shape[2])
        self.n_years = int(self.time_series_size / self.season_size)
        self.precomputed_flags = precomputed_flags

        self.time_win_direction = time_win_direction

        self.max_same_season_win = max_same_season_win
        if self.max_same_season_win is None:
          self.max_same_season_win = self.n_years
        
        self.max_neib_season_win = max_neib_season_win
        if self.max_neib_season_win is None:
          self.max_neib_season_win = self.n_years

        self.max_annual_win = max_annual_win
        if self.max_annual_win is None:
          self.max_annual_win = self.n_years

        self.summary_gaps = np.sum(
          np.isnan(self.data).astype('int8'), 
          axis=(1,2)
        )

        if (self.time_win_size % 2) == 0:
          raise ValueError(f'The time_win_size argument must be an odd number')

        if (self.max_same_season_win < self.time_win_size):
          raise ValueError(f'The max_same_season_win argument can not be less than {time_win_size} (time_win_size)')

        if (self.max_neib_season_win < self.time_win_size):
          raise ValueError(f'The max_neib_season_win argument can not be less than {time_win_size} (time_win_size)')

        if (self.max_annual_win < self.time_win_size):
          raise ValueError(f'The max_annual_win argument can not be less than {time_win_size} (time_win_size)')

        self.n_jobs = multiprocessing.cpu_count()
        if self.time_series_size < self.n_jobs:
          self.n_jobs = self.time_series_size

      def _sanitize_win(self, idx):
        return idx[
          np.logical_and(
            idx >= 0,
            idx < self.time_series_size
          )
        ]

      def _season_win(self, i, t_margin):
          
        margin = (self.season_size * t_margin)
        
        i_past = i - margin
        i_futu = i + margin + self.season_size

        if self.time_win_direction == 'past':
          i_futu = i
        elif self.time_win_direction == 'future':
          i_past = i

        return self._sanitize_win(
          np.arange(i_past, i_futu, self.season_size)
        )

      def _annual_win(self, i, t_margin):
        
        win_size = (t_margin * 2) + 1
        start = int(i / self.season_size)
        margin = win_size * self.season_size
        
        i_past = start - margin
        i_futu = start + margin 

        if self.time_win_direction == 'past':
          i_futu = i
        elif self.time_win_direction == 'future':
          i_past = i

        return self._sanitize_win(
          np.arange(i_past, i_futu, 1)
        )

      def _time_windows(self, i):

        win_map = {}
        win_idx = []

        for j in range(0, self.n_years - self.time_win_size):
          t = ((j * 2) + self.time_win_size)
          t_margin = int((t - 1) / 2)
          
          win_size = (t_margin * 2) + 1
          #print(f"{i} {j} {win_size}")

          same_season_win = self._season_win(i, t_margin)
          if len(same_season_win) > 0 and win_size <= self.max_same_season_win:
            #print(f"same_season_win: {i} {j} {win_size} {len(same_season_win)}")
            same_season_idx = t
            win_idx.append(same_season_idx)
            win_map[same_season_idx] = same_season_win

          neib_season_win = np.concatenate([
            self._season_win(i-1, t_margin),
            self._season_win(i+1, t_margin)
          ])
          if len(neib_season_win) > 0 and win_size <= self.max_neib_season_win:
            #print(f"neib_season_win: {i} {j} {win_size} {len(neib_season_win)}")
            neib_season_idx = t + self.n_years
            win_idx.append(neib_season_idx)
            win_map[neib_season_idx] = neib_season_win
          
          annual_win = self._annual_win(i, t_margin)
          if len(annual_win) > 0 and win_size <= self.max_annual_win:
            #print(f"annual_win: {i} {j} {win_size} {len(annual_win)}")
            annual_idx = t + (2 * self.n_years)
            win_idx.append(annual_idx)
            win_map[annual_idx] = annual_win

        win_idx.sort()

        return (win_idx, win_map)

      def _win_choice(self, win_map, idx, i):
        to_gapfill = self.data[i,:,:]
        to_reduce = self.data[win_map[idx]]
        
        availability_mask = np.logical_and(
          np.isnan(to_gapfill), 
          ~np.isnan(bc.nanmin(to_reduce, axis=(0)))
        )

        r = np.empty(self.tile_size, dtype='float32')
        r[:,:] = np.nan
        r[availability_mask] = idx

        return r

      def _flags(self, i, win_idx, win_map):
        #args = [ (win_map, idx, i) for idx in win_idx ]

        win_choice = []
        #for r in parallel.job(self._win_choice, args, n_jobs=len(args), joblib_args={'backend': 'threading'}):
        for idx in win_idx:
          r = self._win_choice(win_map, idx, i)
          win_choice.append(r)

        win_choice = bc.nanmin(np.stack(win_choice, axis=0), axis=0)          
        gc.collect()

        return win_choice

      def _reduce(self, i, to_reduce, mask):
        mask_b = np.broadcast_to(mask, to_reduce.shape)
        to_reduce[~mask_b] = np.nan
        return bc.nanmedian(to_reduce, axis=0), mask

      def _run(self, i):
        
        n_gaps = self.summary_gaps[i]

        try:

          if n_gaps > 0:
              
            win_idx, win_map = self._time_windows(i)

            if self.precomputed_flags is None:
              flags = self._flags(i, win_idx, win_map)
              flags[np.isnan(flags)] = 0
              flags = flags.astype('uint8')
            else:
              flags = self.precomputed_flags[i,:,:]
            
            flags_uniq = np.unique(flags[flags != 0])
            
            args = [ (i, self.data[win_map[idx]], (flags == idx)) for idx in flags_uniq ]

            gapfilled = self.data[i,:,:].copy()
            #for reduced, mask in parallel.job(self._reduce, args, n_jobs=len(args), joblib_args={'backend': 'threading'}):
            for arg in args:
              reduced, mask = self._reduce(*arg)
              gapfilled[mask] = reduced[mask]

            return gapfilled, flags
          
          else:
            return self.data[i,:,:], np.zeros(self.tile_size, dtype='uint8')
            
        except:
          return self.data[i,:,:], np.zeros(self.tile_size, dtype='uint8')

        finally:
          gc.collect()

      def _gapfill(self):
        args = [  (i,)  for i in range(self.time_series_size) ]

        gapfilled_data = []
        gapfilled_data_flag = []
        
        win_idx, win_map = self._time_windows(int(self.time_series_size/2))
        ttprint(f"Deriving {len(win_idx)} gap filling possibilities for each date")

        for gapfilled, flags in parallel.job(self._run, args, n_jobs=self.n_jobs, joblib_args={'backend': 'multiprocessing'}):
          gapfilled_data.append(gapfilled)
          gapfilled_data_flag.append(flags)

        gapfilled_data = np.stack(gapfilled_data, axis=0)
        gapfilled_data_flag = np.stack(gapfilled_data_flag, axis=0)

        return(gapfilled_data, gapfilled_data_flag)


    class TMWM(ImageGapfill):
      """
      Temporal Moving Window Median able to gapfill the missing pixels using the temporal neighborhood
      by a moving window to calculate several median possibilities. The approach prioritizes
      values derived for **1–the same day/month/season**, **2–neighboring days/months/seasons**,
      **and 3–all the year**.

      For example, in the best case scenario a missing pixel on Jan-2005 is filled using a median
      value derived from January of other years, and in the worst case scenario it uses a value derived
      of all the months and available years. If a pixel remains with a missing value, it is because
      there is **no valid data on the entire time series**.

      :param fn_files: Raster file paths to be read and gapfilled.
      :param data: 3D array where the last dimension is the time.
      :param season_size: Season size of a year (for monthly time series it is equal ``12``).
      :param time_win_size: Size of the temporal window used to calculate the median value possibilities.
      :param cpu_max_workers: Number of CPU cores to be used in parallel.
      :param engine: Execute in ``CPU`` [1] or ``GPU`` [2].
      :param gpu_tile_size: Tile size used to split the processing in the GPU.
      :param outlier_remover: Strategy to remove outliers.
      :param std_win: Temporal window size used to calculate a local median and std.
      :param std_env: Number of std used to define a local envelope around the median.
        Values outside of this envelope are removed.
      :param perc_env: A list containing the lower and upper percentiles used to defined a global
        envelope for the time series. Values outside of this envelope are removed.
      :param n_jobs_io: Number of parallel jobs to read/write raster files.
      :param verbose: Use ``True`` to print the progress of the gapfilled.

      Examples
      ========

      >>> from skmap import gapfiller
      >>>
      >>> # For a 4-season time series
      >>> tmwm = gapfiller.TMWM(fn_files=fn_rasters, season_size=4, time_win_size=4)
      >>> data_tmwm = tmwm.run()
      >>>
      >>> fn_rasters_tmwm = tmwm.save_rasters('./gapfilled_tmwm')

      References
      ==========

      [1] `Bootleneck nanmedian <https://kwgoodman.github.io/bottleneck-doc/reference.html#bottleneck.nanmedian>`_

      [2] `CuPY nanmedian <https://docs.cupy.dev/en/stable/reference/generated/cupy.nanmedian.html>`_

      """

      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        season_size = None,
        time_win_size: int=9,
        cpu_max_workers:int = multiprocessing.cpu_count(),
        engine='CPU',
        gpu_tile_size:int = 250,
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose = True
      ):

        super().__init__(fn_files=fn_files, data=data,
          outlier_remover=outlier_remover, std_win=std_win,
          std_env=std_env, perc_env=perc_env, n_jobs_io=n_jobs_io,
          verbose=verbose)

        self.cpu_max_workers = cpu_max_workers
        self.time_win_size = time_win_size
        self.engine = engine
        self.gpu_tile_size = gpu_tile_size
        self.season_size = season_size

        if (self.time_win_size % 2) == 0:
          raise ValueError(f'The time_win_size argument must be an odd number')

      def _do_time_data(self):
        total_times = self.data.shape[2]

        self.time_order = [ str(time) for time in range(0, self.season_size) ]
        self.time_data = {}
        self.n_years = 0

        for time in self.time_order:
          idx = list(range(int(time), total_times, self.season_size))

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
        keys = list(newdata_dict.keys())
        keys.sort()

        for i in newdata_dict.keys():

          gaps_mask = np.isnan(self.time_data[time][:,:,layer_pos])
          n_gaps = np.sum(gaps_mask.astype('int'))

          if n_gaps > 0:
            newdata = newdata_dict[i]
            gapfill_mask = np.logical_and(~np.isnan(newdata),gaps_mask)
            self.time_data[time][:,:,layer_pos][gapfill_mask] = newdata[gapfill_mask]
            self.time_data_gaps[time][:,:,layer_pos][gapfill_mask] = int(i) + gapflag_offset
          else:
              end_msg = f'Layer {layer_pos}: gapfilled all with {i}-year window from {verbose_suffix})'
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
            newdata_dict[i] = np.nanmean(stacked, axis=2)

          end_msg = self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='neighborhood seasons', gapflag_offset = n_layers)
          if end_msg is not None:
            break

        return end_msg

      def _fill_gaps_same_time(self, time, layer_pos):
        newdata_dict = self.tmwm_data.get(time, layer_pos)
        return self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='same season')

      def fill_image(self, time, layer_pos):

        end_msg = self._fill_gaps_same_time(time, layer_pos)

        if end_msg is None:
          end_msg = self._fill_gaps_neib_time(time, layer_pos)

        if end_msg is None:
          end_msg = self._fill_gaps_all_times(time, layer_pos)

        return end_msg

      def _gapfill(self):

        self._do_time_data()
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

        with warnings.catch_warnings():
          warnings.filterwarnings('ignore', 'Mean of empty slice')
          for end_msg in parallel.job(self.fill_image, iter(layer_args), n_jobs=self.cpu_max_workers, joblib_args={'require': 'sharedmem'}):
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
      :param outlier_remover: Strategy to remove outliers.
      :param std_win: Temporal window size used to calculate a local median and std.
      :param std_env: Number of std used to define a local envelope around the median.
        Values outside of this envelope are removed.
      :param perc_env: A list containing the lower and upper percentiles used to defined a global
        envelope for the time series. Values outside of this envelope are removed.
      :param n_jobs_io: Number of parallel jobs to read/write raster files.
      :param verbose: Use ``True`` to print the progress of the gapfilled.

      Examples
      ========

      >>> from skmap import gapfiller
      >>>
      >>> tli = gapfiller.TLI(fn_files=fn_rasters)
      >>> data_tli = tli.run()
      >>>
      >>> fn_rasters_tli = tli.save_rasters('./gapfilled_tli')

      References
      ==========

      [1] `Scikit-learn linear regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_

      """

      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose = True
      ):
        super().__init__(fn_files=fn_files, data=data,
          outlier_remover=outlier_remover, std_win=std_win,
          std_env=std_env, perc_env=perc_env, n_jobs_io=n_jobs_io,
          verbose=verbose)

      def _temporal_linear_inter(self, data):

        y = data.reshape(-1).copy()
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
      """
      Approach that uses a Singular Spectral Analysis (SSA [1]) to gapfill the missing
      values and smooth all the raster data. The missing values are first gapfilled using
      a long-term median strategy derived over values from other days/months/seasons. Later
      the SSA is uses to decompose each time series in multiple components (``ngroups``),
      considering only part of them to reconstruct the output time series.
      (``reconstruct_ngroups``).

      :param fn_files: Raster file paths to be read and gapfilled.
      :param data: 3D array where the last dimension is the time.
      :param season_size: Season size of a year used to calculate
        the long-term median (for monthly time series it is equal ``12``).
      :param max_gap_pct: Max percentage allowed to run the approach. For pixels
        where this condition is satisfied the result is ``np.nan`` for all dates.
      :param ltm_resolution: Number of years used to calculate the long-term median.
      :param window_size: Size of the sliding window (i.e. the size of each word).
        If float, it represents the percentage of the size of each time series and must be between 0 and 1.
        The window size will be computed as ``max(2, ceil(window_size * n_timestamps))`` [1].
      :param ngroups: Number of components used to decompose the time series [1].
      :param reconstruct_ngroups: Number of components used to reconstruct the time series.
      :param outlier_remover: Strategy to remove outliers.
      :param std_win: Temporal window size used to calculate a local median and std.
      :param std_env: Number of std used to define a local envelope around the median.
        Values outside of this envelope are removed.
      :param perc_env: A list containing the lower and upper percentiles used to defined a global
        envelope for the time series. Values outside of this envelope are removed.
      :param n_jobs_io: Number of parallel jobs to read/write raster files.
      :param verbose: Use ``True`` to print the progress of the gapfilled.

      Examples
      ========

      >>> from skmap import gapfiller
      >>>
      >>> # For a 4-season time series
      >>> ssa = gapfiller.SSA(fn_files=fn_rasters, season_size=4)
      >>> data_ssa = ssa.run()
      >>>
      >>> fn_rasters_ssa = ssa.save_rasters('./gapfilled_ssa', dtype='uint8', save_flag=False)

      References
      ==========

      [1] `Pyts SingularSpectrumAnalysis <https://pyts.readthedocs.io/en/stable/generated/pyts.decomposition.SingularSpectrumAnalysis.html>`_
      """

      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        season_size:int = 4,
        max_gap_pct:int = 0.8,
        ltm_resolution:int = 5,
        window_size:int = 4,
        ngroups:int = 4,
        reconstruct_ngroups:int = 2,
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose = True
      ):
        super().__init__(fn_files=fn_files, data=data,
          outlier_remover=outlier_remover, std_win=std_win,
          std_env=std_env, perc_env=perc_env, n_jobs_io=n_jobs_io,
          verbose=verbose)

        self.season_size = season_size
        self.max_gap_pct = max_gap_pct
        self.ltm_resolution = ltm_resolution
        self.window_size = window_size
        self.ngroups = ngroups
        self.reconstruct_ngroups = reconstruct_ngroups

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

        ts_gapfilled = ts_data.copy()
        n_years = int(ts_gapfilled.shape[1] / self.season_size)

        if self._perc_gaps(ts_gapfilled.flatten()) <= self.max_gap_pct:
          for year in range(self.ltm_resolution, n_years, self.ltm_resolution):
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
        ts_flag[na_mask_initial] = 1

        return np.stack([ts_reconstructed.reshape(data.shape), ts_flag.reshape(data.shape)], axis=1)

      def _gapfill(self):
        result = parallel.apply_along_axis(self._ssa_reconstruction, 2, self.data)
        return (result[:,:,:,0], result[:,:,:,1])

    class InPainting(ImageGapfill):
      """
      Approach that uses a inpating technique [1] to gapfill raster data
      using neighborhood values.

      :param fn_files: Raster file paths to be read and gapfilled.
      :param data: 3D array where the last dimension is the time.
      :param space_win: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
      :param data_mask: 2D array indicating a valid areas, equal 1, where in case of gaps should be filled.
      :param mode:  Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA [1]
      :param outlier_remover: Strategy to remove outliers.
      :param std_win: Temporal window size used to calculate a local median and std.
      :param std_env: Number of std used to define a local envelope around the median.
        Values outside of this envelope are removed.
      :param perc_env: A list containing the lower and upper percentiles used to defined a global
        envelope for the time series. Values outside of this envelope are removed.
      :param n_jobs_io: Number of parallel jobs to read/write raster files.
      :param verbose: Use ``True`` to print the progress of the gapfilled.

      Examples
      ========

      >>> from skmap import gapfiller
      >>>
      >>> # Considerer land_mask as 2D numpy array where 1 indicates land
      >>> inPainting = gapfiller.InPainting(fn_files=fn_rasters, space_win = 10, data_mask=land_mask)
      >>> data_inp = inPainting.run()
      >>>
      >>> fn_rasters_inp = inPainting.save_rasters('./gapfilled_inp')

      References
      ==========

      [1] `OpenCV Tutorial - Image Inpainting <https://docs.opencv.org/4.5.2/df/d3d/tutorial_py_inpainting.html>`_

      """

      def __init__(self,
        fn_files:List = None ,
        data:np.array = None,
        space_win = 10,
        data_mask = None,
        mode = cv.INPAINT_TELEA,
        outlier_remover:OutlierRemover = None,
        std_win:int = 3,
        std_env:int = 2,
        perc_env:list = [2,98],
        n_jobs_io = 4,
        verbose = True
      ):
        super().__init__(fn_files=fn_files, data=data,
          outlier_remover=outlier_remover, std_win=std_win,
          std_env=std_env, perc_env=perc_env, n_jobs_io=n_jobs_io,
          verbose=verbose)

        self.data_mask = data_mask
        self.space_win = space_win
        self.mode = mode

      def _inpaint(self, data, i=0):

        # necessary for a proper inpaint execution
        data_copy = np.copy(data)
        
        with warnings.catch_warnings():
          warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
          initial_value = np.nanmedian(data_copy)

        na_data_mask = np.isnan(data_copy)
        data_copy[na_data_mask] = initial_value
        data_gapfilled = cv.inpaint(data_copy.astype('float32'), na_data_mask.astype('uint8'), self.space_win, self.mode)

        return (data_gapfilled, i)

      def _gapfill(self):

        data = np.copy(self.data)
        data_flag = np.zeros(data.shape)

        if self.data_mask is None:
          self.data_mask = np.ones(data.shape[0:2])

        max_workers = multiprocessing.cpu_count()
        n_times = data.shape[2]

        args = []

        for i in range(0, n_times):
          if self._n_gaps(data[:,:,i]) > 0:
            args.append((data[:,:,i], i))

        result_set = {}
        for band_data, i in parallel.job(self._inpaint, args, n_jobs=max_workers):
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

    def time_first_space_later(
      fn_files:List = None,
      data:np.array = None,
      time_strategy:ImageGapfill = TMWM,
      time_args:set = {},
      space_strategy:ImageGapfill = InPainting,
      space_args:set = {},
      space_flag_val = 100
    ):
      """
      Helper function to gapfill all the missing pixels using
      first a temporal strategy (``TMWM``, ``TLI``, ``SSA``) and later
      a spatial strategy (``InPainting``).

      :param fn_files: Raster file paths to be read and gapfilled.
      :param data: 3D array where the last dimension is the time.
      :param time_strategy: One of the implemented temporal gapfilling approaches.
      :param time_args: A ``set`` of parameters for the temporal gapfilling strategy
      :param space_strategy: One of the implemented spatial gapfilling approaches.
      :param space_args: A ``set`` of parameters for the spatial gapfilling strategy.
      :param space_flag_val: The flag value used to indicate which pixels were gapfilled
        by the spatial gapfilling strategy.

      Examples
      ========

      >>> from skmap import gapfiller
      >>>
      >>> # For a 4-season time series
      >>> tfsl = gapfiller.time_first_space_later(
      >>>  fn_files = fn_rasters,
      >>>  time_strategy = gapfiller.TMWM,
      >>>  time_args = { 'season_size': 4 },
      >>>  space_strategy = gapfiller.InPainting,
      >>>  space_args = { 'space_win': 10 }
      >>> )
      >>>
      >>> fn_rasters_tfsl  = tfsl.save_rasters('./gapfilled_tmwm_inpaint', dtype='uint8', fn_files=fn_rasters)

      """
      time_args['fn_files'] = fn_files
      time_args['data'] = data

      time = time_strategy(**time_args)
      time_gapfilled = time.run()
      time_gapfilled_pct = time._perc_gapfilled(time.gapfilled_data)

      if time_gapfilled_pct < 1.0:

        space_args['data'] = time_gapfilled
        space = space_strategy(**space_args)
        space.run()

        space_mask = (space.gapfilled_data_flag == 1)
        time.gapfilled_data_flag[space_mask] = space_flag_val
        space.gapfilled_data_flag = time.gapfilled_data_flag

        return space
      else:
        return time

except ImportError as e:
    from .misc import _warn_deps
    _warn_deps(e, 'gapfiller')
