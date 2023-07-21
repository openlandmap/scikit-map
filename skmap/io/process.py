import time
import os
from enum import Enum

try:

  from abc import ABC, abstractmethod
  from skmap import parallel

  from skmap import SKMapRunner, parallel
  from skmap.misc import date_range, nan_percentile
  from skmap.io import RasterData

  from scipy.special import log1p
  from statsmodels.tsa.seasonal import STL
  import statsmodels.api as sm

  import scipy.sparse as sparse
  from scipy.sparse.linalg import splu
  
  import numpy as np
  import bottleneck as bn
  from datetime import datetime
  from pandas import DataFrame
  import pandas as pd

  import cv2 as cv
  import pyfftw

  class Transformer(SKMapRunner, ABC):
    
    def __init__(self,
      name:str,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(verbose=verbose, temporal=temporal)
      self.name = name
      self.name_qa = f'{self.name}{RasterData.TRANSFORM_SEP}qa'

    def _new_info(self, rdata, group, outname, nm):

      info = rdata._info()
      new_info = []

      for index, row in info.iterrows():
        start_dt = row[RasterData.START_DT_COL]
        end_dt = row[RasterData.END_DT_COL]
        raster_file = row[RasterData.PATH_COL]

        name = rdata._set_date(outname, start_dt, end_dt, nm=nm, gr=group)

        new_group = f'{group}.{nm}'

        new_info.append(
          rdata._new_info_row(raster_file, group=new_group, name=name, dates=[start_dt, end_dt])
        )

      return new_info

    def run(self, 
      rdata:RasterData,
      group:str,
      outname:str = 'skmap_{nm}_{gr}_{dt}.tif'
    ):
      """
      Execute the gapfilling approach.
      """

      if outname is None:
        outname = 'skmap_{nm}_{gr}_{dt}.tif'

      array = rdata._array()

      start = time.time()
      result = self._run(array)

      if isinstance(result, tuple) and len(result) >= 2:
        new_array, new_array_qa = result[0], result[1]
      else:
        new_array, new_array_qa = result, None

      new_info = self._new_info(rdata, group, outname, self.name)
      if new_array_qa is not None:
        new_info += self._new_info(rdata, group, outname, self.name_qa)
        new_array = np.concatenate([new_array, new_array_qa], axis=-1)

      return new_array, DataFrame(new_info)

    @abstractmethod
    def _run(self, data):
      pass

  class Derivator(SKMapRunner, ABC):
    
    def __init__(self,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(verbose=verbose, temporal=temporal)

    def run(self, 
      rdata:RasterData,
      group:str,
      outname:str = None
    ):
      """
      Execute the gapfilling approach.
      """

      kwargs = {'rdata': rdata, 'group': group}
      if outname is not None:
        kwargs['outname'] = outname

      start = time.time()
      new_array, new_info = self._run(**kwargs)

      return new_array, new_info

    @abstractmethod
    def _run(self, 
      rdata:RasterData, 
      group:str,
      outname:str
    ):
      pass

  class Filler(Transformer, ABC):
   
    def __init__(self,
      name:str,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(name=name, verbose=verbose, temporal=temporal)

    def _n_gaps(self, data = None):
      nan_data = np.isnan(data)
      gap_mask = np.logical_not(np.all(nan_data, axis=-1))
      return np.sum(nan_data[gap_mask].astype('int'))

    def _run(self, data):
      """
      Execute the gapfilling approach.

      """
      
      n_gaps = None
      if self.verbose:
        n_gaps = self._n_gaps(data)
        self._verbose(f'There are {n_gaps} gaps in {data.shape}')

      start = time.time()
      result = self._gapfill(data)

      if isinstance(result, tuple) and len(result) >= 2:
        filled, filled_qa = result[0], result[1]
      else:
        filled, filled_qa = result, None

      if self.verbose:
        r_gaps = self._n_gaps(filled)
        gaps_perc = (n_gaps - r_gaps) / n_gaps
        self._verbose(f'{gaps_perc*100:.2f}% of the gaps filled in {(time.time() - start):.2f} segs')

        if gaps_perc < 1:
          self._verbose(f'Remained gaps: {r_gaps}')

      return result

    @abstractmethod
    def _gapfill(self, data):
      pass

  class SeasConvFill(Filler):
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
      return_qa:bool = False,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(name='seasconv', verbose=verbose, temporal=True)

      self.season_size = season_size
      self.return_qa = return_qa
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
      in_ts_forward = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
      out_ts_forward = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
      in_conv_forward = pyfftw.empty_aligned(N_ext, dtype='float32')
      out_conv_forward = pyfftw.empty_aligned(N_fft, dtype='complex64')
      out_conv_backward = pyfftw.empty_aligned(N_ext, dtype='float32')
      in_ts_backward = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
      out_ts_backward = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
      plan_conv_forward = pyfftw.FFTW(in_conv_forward, out_conv_forward, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_jobs)
      plan_conv_backward = pyfftw.FFTW(out_conv_forward, out_conv_backward, axes=(0,), flags=(plan,), direction='FFTW_BACKWARD',threads=self.n_jobs)
      plan_ts_forward = pyfftw.FFTW(in_ts_forward, out_ts_forward, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_jobs)
      plan_ts_backward = pyfftw.FFTW(in_ts_backward, out_ts_backward, axes=(0,), flags=(plan,), direction='FFTW_BACKWARD',threads=self.n_jobs)
      in_conv_forward = np.zeros(N_ext)
      in_conv_forward[0:N_samp] = conv_vec
      in_conv_forward[N_samp:] = np.roll(conv_vec[::-1],1)
      plan_conv_forward(in_conv_forward)
      conv_fft = out_conv_forward.copy()
      in_conv_forward = np.zeros(N_ext)
      in_conv_forward[0:N_samp] = 1
      plan_conv_forward(in_conv_forward)
      out_conv_forward = conv_fft * out_conv_forward
      plan_conv_backward(out_conv_forward)
      in_ts_forward = np.concatenate((data,np.zeros((N_samp,N_imag))))
      plan_ts_forward(in_ts_forward)
      in_ts_backward = conv_fft.reshape(-1,1) * out_ts_forward
      plan_ts_backward(in_ts_backward)
      conv = out_ts_backward[0:N_samp,:].copy()
      in_ts_forward = np.concatenate((valid_mask,np.zeros((N_samp,N_imag))))
      plan_ts_forward(in_ts_forward)
      in_ts_backward = conv_fft.reshape(-1,1) * out_ts_forward
      plan_ts_backward(in_ts_backward)
      filled_qa = out_ts_backward[0:N_samp,:]
      filled = conv/filled_qa
      filled_qa /= out_conv_backward.reshape(-1,1)[0:N_samp] # Renormalization of the quality assesmtent vector
      return filled, filled_qa
    
    def _gapfill(self, data):
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
      filled_qa[filled_qa == 0.0] = np.nan
      
      # Return the reconstructed time series and the quality assesment layer
      if self.return_qa:
        return np.reshape(filled.T, orig_shape), np.reshape(filled_qa.T, orig_shape) 
      else:
        return np.reshape(filled.T, orig_shape)

  class InPaintingFill(Filler):
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
      super().__init__(verbose=verbose, temporal=False)

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

  class WhittakerSmooth(Transformer):    
    """
    https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py
    """
    
    def __init__(self,
      lmbd = 1, 
      d = 2,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(name='whittaker', verbose=verbose, temporal=True)

      self.lmbd = lmbd
      self.d = d
      self.n_jobs = n_jobs
    
    def _speyediff(self, N, d, format='csc'):
      """
      (utility function)
      Construct a d-th order sparse difference matrix based on 
      an initial N x N identity matrix
      
      Final matrix (N-d) x N
      """
      
      assert not (d < 0), "d must be non negative"
      shape     = (N-d, N)
      diagonals = np.zeros(2*d + 1)
      diagonals[d] = 1.
      for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
      offsets = np.arange(d+1)
      spmat = sparse.diags(diagonals, offsets, shape, format=format)
      return spmat
    
    def _process_ts(self, data):
      y = data.reshape(-1).copy()
      n_gaps = np.sum((np.isnan(y)).astype('int'))
      
      if n_gaps == 0:
        r = splu(self.coefmat).solve(y)
        return r
      else:
        return y

    def _run(self, data):
      
      m = data.shape[-1]
      E = sparse.eye(m, format='csc')
      D = self._speyediff(m, self.d, format='csc')
      self.coefmat = E + self.lmbd * D.conj().T.dot(D)

      return parallel.apply_along_axis(self._process_ts, 2, data, n_jobs=self.n_jobs)

  class TimeEnum(Enum):
    
    MONTHLY = 1
    MONTHLY_15P = 2
    MONTHLY_LONGTERM = 3

    BIMONTHLY = 4
    BIMONTHLY_15P = 5
    BIMONTHLY_LONGTERM = 6

    QUARTERLY = 7
    YEARLY = 8

  class TimeAggregate(Derivator):
    
    def __init__(self,
      time:list = [ TimeEnum.YEARLY, TimeEnum.MONTHLY_LONGTERM ],
      operations = ['p25', 'p50', 'p75', 'std'],
      rename_operations:dict = {},
      date_overlap:bool = False,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(verbose=verbose, temporal=True)

      self.time = time
      self.operations = operations
      self.rename_operations = rename_operations
      self.date_overlap = date_overlap
      self.n_jobs = n_jobs

      self.percs = []
      self.bn_ops = []

      for op in self.operations:
        if op[0] == 'p':
          self.percs.append(int(op[1:]))
        else:
          method = f'nan{op}'
          if not hasattr(bn, method):
            raise Exception(f'Operation {method} is invalid, since bottleneck.{method} not exists.')
          self.bn_ops.append( (op, getattr(bn, method)) )

    def _op_name(self, op):
      if op in self.rename_operations:
        return self.rename_operations[op]
      else:
        return op

    def _aggregate(self, in_array, tm, dt1, dt2):

      out_array = []
      ops = []

      for op, method in self.bn_ops:
        out_array.append(
          method(in_array, axis=-1)[:, :, np.newaxis]
        )
        ops.append(self._op_name(f'{op}'))

      if len(self.percs) > 0:
        out_array.append(
          nan_percentile(in_array.copy().transpose((2,0,1)), q=self.percs).transpose((1,2,0))
        )
        
        for p in self.percs:
          ops.append(self._op_name(f'p{p}'))

      out_array = np.concatenate(out_array, axis=-1)

      return (out_array, ops, tm, dt1, dt2)

    def _args_yearly(self, rdata, start_dt, end_dt, date_format):
      
      args = []

      for dt1, dt2 in date_range(
        f'{start_dt.year}0101',f'{end_dt.year}1201', 
        'years', 1, return_str=True, ignore_29feb=False, 
        date_format=date_format):

        tm = 'yearly'
        in_array = rdata.filter_date(dt1, dt2, return_array=True, 
          date_format=date_format, date_overlap=self.date_overlap)
        
        if in_array.size > 0:  
          args += [ (in_array, tm, datetime.strptime(dt1, date_format), datetime.strptime(dt2, date_format)) ]

      return args

    def _args_monthly_longterm(self, rdata, start_dt, end_dt, date_format):

      args = []

      for month in range(1,13):
        
        in_array = []
        month = str(month).zfill(2)

        for dt1, dt2 in date_range(
          f'{start_dt.year}{month}01',f'{end_dt.year}{month}01', 
          'months', 1, date_offset=11, return_str=True, 
          ignore_29feb=False, date_format=date_format):
          
          array = rdata.filter_date(dt1, dt2, return_array=True, 
              date_format=date_format, date_overlap=self.date_overlap)
          
          if array.size > 0:
            in_array.append(array)

        tm = f'm{month}'
        if len(in_array) > 0:
          args += [ (np.concatenate(in_array, axis=-1), tm, start_dt, end_dt) ]

      return args

    def _run(self, 
      rdata:RasterData,
      group:str,
      outname:str = 'skmap_aggregate.{gr}.{tm}_{op}_{dt}.tif'
    ):

      date_format = '%Y%m%d'
      start_dt = rdata.info[RasterData.START_DT_COL].min()
      end_dt = rdata.info[RasterData.END_DT_COL].max()

      args = []

      for t in self.time:

        if t == TimeEnum.MONTHLY_LONGTERM:
          args += self._args_monthly_longterm(rdata, start_dt, end_dt, date_format)
        elif t == TimeEnum.YEARLY:
          args += self._args_yearly(rdata, start_dt, end_dt, date_format)
        else:
          raise Exception(f"Aggregation by {t} not implemented")
      
      new_array = []
      new_info = []

      self._verbose(f"Computing {len(args)} "
        + f"time aggregates from {start_dt.year} to {end_dt.year}"
      )

      for out_array, ops, tm, dt1, dt2 in parallel.job(self._aggregate, args, joblib_args={'backend': 'threading'}):
        for op in ops:
          name = rdata._set_date(outname, 
                dt1, dt2, 
                rdata.date_format, rdata.date_style, 
                op=op, tm=tm, gr=group
              )
          
          new_group = f'{group}.{tm}.{op}'

          new_info.append(
            rdata._new_info_row('', name=name, group=new_group, dates=[start_dt, end_dt])
          )

        new_array.append(out_array)
        
      new_array = np.concatenate(new_array, axis=-1)

      return new_array, DataFrame(new_info)

  class TrendAnalysis(Derivator):
    
    def __init__(self,
      season_size:int,
      season_smoother:int = None,
      trend_smoother:int = None,
      log_rescale:tuple = None,
      scale_factor:int = 10000,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(verbose=verbose, temporal=True)
      
      self.season_size = season_size
      self.season_smoother = season_smoother
      self.trend_smoother = trend_smoother
      self.n_jobs = n_jobs

      self.vmin, self.vmax = None, None
      if log_rescale is not None:
        self.vmin, self.vmax = log_rescale

      self.name_misc = [
        ('alpha', 'm', scale_factor), ('alpha', 'sd', scale_factor), 
        ('alpha', 'tv', scale_factor), ('alpha', 'pv', scale_factor), 
        ('beta', 'm', scale_factor), ('beta', 'sd', scale_factor), 
        ('beta', 'tv', scale_factor), ('beta', 'pv', scale_factor), 
        ('r2', 'm', scale_factor)
      ]

      if self.season_smoother is None:
        self.season_smoother = self.season_size + 1
      if self.trend_smoother is None:
        self.trend_smoother = (2 * self.season_size) + 1

    def _trend_regression(self, data):

      has_nan = np.sum(np.isnan(data).astype('int'))
      
      ts_size = data.shape[0]
      out_size = ts_size + 9 # fixed number of ols return values

      if has_nan == 0:
        
        res = STL(data.copy(), period=self.season_size, 
          seasonal=self.season_smoother, trend=self.trend_smoother, robust=True).fit()
        
        y = res.trend

        if self.vmin is not None:
          y[y > self.vmax] = self.vmax
          y[y < self.vmin] = self.vmin
          y = log1p(y / self.vmax)
        
        y_size = y.shape[0]
        X = np.array(range(0, y_size)) / y_size
        
        X = sm.add_constant(X)
        model = sm.OLS(y,X)
        results = model.fit()

        result_stack = np.stack([
          results.params,
          results.bse,
          results.tvalues,
          results.pvalues
        ],axis=1)
        
        return np.concatenate([
          res.trend,
          result_stack[0,:],
          result_stack[1,:],
          np.stack([results.rsquared])
        ])
      
      else: 
        nan_result = np.empty(out_size)
        nan_result[:] = np.nan
        return nan_result

    def _run(self, 
      rdata:RasterData,
      group:str,
      outname:str = 'skmap_{gr}.{nm}_{pr}_{dt}.tif'
    ):

      array = rdata._array()
      info = rdata._info()

      start_dt_min = rdata.info[RasterData.START_DT_COL].min()
      end_dt_max = rdata.info[RasterData.END_DT_COL].max()

      new_array = parallel.apply_along_axis(self._trend_regression, 
        axis=2, arr=array, n_jobs=self.n_jobs)

      new_info = []

      for index, row in info.iterrows():
        start_dt = row[RasterData.START_DT_COL]
        end_dt = row[RasterData.END_DT_COL]

        nm, pr = ('trend', 'm')

        name = rdata._set_date(outname, 
          start_dt, end_dt, nm=nm, pr=pr, gr=group)

        new_group = f'{group}.{nm}.{pr}'

        new_info.append(
          rdata._new_info_row('', group=new_group, name=name, dates=[start_dt, end_dt])
        )

      ts_size = array.shape[2]

      for i, (nm, pr, scale) in zip(range(0, len(self.name_misc)), self.name_misc):
        
        new_array[:,:,ts_size + i] *= scale

        name = rdata._set_date(outname, start_dt_min, 
          end_dt_max, nm=nm, pr=pr, gr=group)

        new_group = f'{group}.{nm}.{pr}'

        new_info.append(
          rdata._new_info_row('', group=new_group, name=name, dates=[start_dt_min, end_dt_max])
        )

      return new_array, DataFrame(new_info)

except ImportError as e:
  from skmap.misc import _warn_deps
  _warn_deps(e, 'skmap.io.derive')