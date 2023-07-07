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
    from typing import List, Dict, Union
    from rasterio.windows import Window
    from abc import ABC, abstractmethod
    from pathlib import Path
    from enum import Enum

    import bottleneck as bc
    import cv2 as cv
    import numpy as np
    import pyfftw

    from skmap.misc import ttprint
    from skmap.transform import SKMapTransformer
    from skmap import parallel

    class Filler(SKMapTransformer, ABC):
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
        verbose:bool = True
      ):
        self.verbose = verbose

      def _n_gaps(self, data = None):
        nan_data = np.isnan(data)
        gap_mask = np.logical_not(np.all(nan_data, axis=-1))
        return np.sum(nan_data[gap_mask].astype('int'))

      def run(self, data):
        """
        Execute the gapfilling approach.

        """
        n_gaps = None
        if self.verbose:
          n_gaps = self._n_gaps(data)
          self._verbose(f'There are {n_gaps} gaps in {data.shape}')

        start = time.time()
        filled, filled_flag = self._run(data)

        if self.verbose:
          r_gaps = self._n_gaps(filled)
          gaps_perc = (n_gaps - r_gaps) / n_gaps
          self._verbose(f'{gaps_perc*100:.2f}% of the gaps filled in {(time.time() - start):.2f} segs')

          if gaps_perc < 1:
            self._verbose(f'Remained gaps: {r_gaps}')

        return filled, filled_flag

      @abstractmethod
      def _run(self, data):
        pass

    class SeasConv(Filler):    
      """
      :param season_size: number of images per year
      :param att_seas: dB of attenuation for images of opposite seasonality
      :param att_env: dB of attenuation for temporarly far images
      :param n_cpu: number of CPU to be used in parallel
      """
      
      def __init__(self,
        season_size:int,
        att_seas:float = 60,
        att_env:float = 20,
        n_jobs:int = os.cpu_count(),
        verbose = False
      ):

        super().__init__(verbose=verbose)

        self.season_size = season_size
        self.att_seas = att_seas
        self.att_env = att_env
        self.n_jobs = n_jobs
          
      def _compute_conv_mat_row(self, n_imag):            
          
        # Compute a triangular basis function with yaerly periodicity
        conv_mat_row = np.zeros((n_imag))
        
        base_func = np.zeros((self.season_size,))
        period_y = self.season_size/2.0
        slope_y = self.att_seas/10/period_y
        
        for i in np.arange(self.season_size):
          if i <= period_y:
            base_func[i] = -slope_y*i
          else:
            base_func[i] = slope_y*(i-period_y)-self.att_seas/10            
        
        # Compute the envelop to attenuate temporarly far images
        env_func = np.zeros((n_imag,))
        delta_e = n_imag
        slope_e = self.att_env/10/delta_e
        
        for i in np.arange(delta_e):
          env_func[i] = -slope_e*i
        conv_mat_row = 10.0**(np.resize(base_func,n_imag) + env_func)
        
        return conv_mat_row
          
      def _fftw_toeplitz_matmul(self, data, valid_mask, conv_vec):
        plan = 'FFTW_EXHAUSTIVE'
        N_samp = conv_vec.shape[0]
        N_ext = N_samp*2
        N_fft = (np.floor(N_ext/2)+1).astype(int)
        N_imag = data.shape[1]
        a_w = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
        a_w_fft = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
        c_w = pyfftw.empty_aligned(N_ext, dtype='float32')
        c_w_fft = pyfftw.empty_aligned(N_fft, dtype='complex64')
        b_w_fft = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
        b_w = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
        fft_object_a = pyfftw.FFTW(a_w, a_w_fft, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_jobs)
        fft_object_c = pyfftw.FFTW(c_w, c_w_fft, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_jobs)
        fft_object_b = pyfftw.FFTW(b_w_fft, b_w, axes=(0,), flags=(plan,), direction='FFTW_BACKWARD',threads=self.n_jobs)
        c_w = np.zeros(N_ext)
        c_w[0:N_samp] = conv_vec
        c_w[N_samp:] = np.roll(conv_vec[::-1],1)
        fft_object_c(c_w)
        a_w = np.concatenate((data,np.zeros((N_samp,N_imag))))
        fft_object_a(a_w)
        b_w_fft = c_w_fft.reshape(-1,1) * a_w_fft
        fft_object_b(b_w_fft)
        conv = b_w[0:N_samp,:].copy()
        a_w = np.concatenate((valid_mask,np.zeros((N_samp,N_imag))))
        fft_object_a(a_w)
        b_w_fft = c_w_fft.reshape(-1,1) * a_w_fft
        fft_object_b(b_w_fft)
        filled_qa = b_w[0:N_samp,:]
        filled = conv/filled_qa
        filled_qa /= np.sum(c_w)
        return filled, filled_qa
      
      def _run(self, data):
        # Convolution and normalization
        
        try:
          import mkl
          mkl_threads = mkl.get_num_threads()
          mkl.set_num_threads(self.n_jobs)
        except:
          pass
        
        np.seterr(divide='ignore', invalid='ignore')

        orig_shape = data.shape
        data = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2])).T.copy()

        valid_mask = ~np.isnan(data)
        data[~valid_mask] = 0.0
        n_imag = data.shape[0]

        if self.season_size*2 > n_imag:
          warnings.warn("Not enough images available")
        assert (self.att_env/10 + self.att_seas/10) < np.finfo(np.double).precision, "Reduce the total attenuations to avoid numerical issues"

        filled, filled_qa = self._fftw_toeplitz_matmul(
          data, valid_mask.astype(float), 
          self._compute_conv_mat_row(n_imag)
        )
        
        filled[valid_mask] = data[valid_mask]
        filled_qa[valid_mask] = 1.0
        filled_qa = filled_qa * 100
        
        # Return the reconstructed time series and the quality assesment layer
        return np.reshape(filled.T, orig_shape), np.reshape(filled_qa.T, orig_shape) 

    class TMWM2(Filler):

      def __init__(self,
        season_size = None,
        time_win_size: int=9,
        max_same_season_win = None,
        max_neib_season_win = None,
        max_annual_win = None,
        time_win_direction = 'both',
        verbose = True
      ):

        super().__init__(verbose=verbose)

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

    class InPainting(Filler):
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
        space_win = 10,
        data_mask = None,
        mode = cv.INPAINT_TELEA,
        verbose = True
      ):
        super().__init__(verbose=verbose)

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
      time_strategy:Filler = TMWM2,
      time_args:set = {},
      space_strategy:Filler = InPainting,
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
    from skmap.misc import _warn_deps
    _warn_deps(e, 'gapfiller')
