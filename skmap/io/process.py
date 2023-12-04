import time
import os
import warnings
from enum import Enum
from typing import List, Union, TypedDict, Callable
from scipy.linalg import matmul_toeplitz

try:

  from abc import ABC, abstractmethod
  from skmap import parallel

  from skmap import SKMapGroupRunner, SKMapRunner, parallel
  from skmap.misc import date_range, nan_percentile
  from skmap.misc import new_memmap, del_memmap, ref_memmap, load_memmap
  from skmap.io import RasterData

  from scipy.signal import find_peaks
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
  import math
  import gc

  from dateutil.relativedelta import relativedelta

  import pyfftw
  
  #os.environ['NUMEXPR_MAX_THREADS'] = '1'
  #os.environ['NUMEXPR_NUM_THREADS'] = '1'
  import numexpr as ne

  class Transformer(SKMapGroupRunner, ABC):
    
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

    # FIXME: adapt for group_list, ginfo_list
    def run(self, 
      rdata:RasterData,
      group:str,
      outname:str = 'skmap_{nm}_{gr}_{dt}'
    ):

      if outname is None:
        outname = 'skmap_{nm}_{gr}_{dt}'

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

  class Derivator(SKMapGroupRunner, ABC):
    
    def __init__(self,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(verbose=verbose, temporal=temporal)
      
    def run(self, 
      rdata:RasterData,
      group_list:str,
      ginfo_list:str,
      outname:str = None
    ):
      """
      Execute the gapfilling approach.
      """

      kwargs = {
        'rdata': rdata, 
        'group_list': group_list, 
        'ginfo_list': ginfo_list
      }
      if outname is not None:
        kwargs['outname'] = outname

      start = time.time()
      new_array, new_info = self._run(**kwargs)

      return new_array, new_info

    @abstractmethod
    def _run(self, 
      rdata:RasterData, 
      group_list:str,
      ginfo_list:str,
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
      conv_vect_future = [],
      conv_vect_past = [],
      return_qa:bool = False,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):
      super().__init__(name='seasconv', verbose=verbose, temporal=True)
      self.season_size = season_size
      self.return_qa = return_qa
      self.att_seas = att_seas
      self.att_env = att_env
      self.conv_vect_future = conv_vect_future
      self.conv_vect_past = conv_vect_past
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
        
    def _fftw_toeplitz_matmul(self, data, valid_mask):
      tmp_norm_vec = np.ones((data.shape[0], 1));
      tmp_norm_vec[0] = 0.
      norm_vec = matmul_toeplitz((self.conv_vect_past, self.conv_vect_future), tmp_norm_vec, check_finite=False, workers=None)
      filled = matmul_toeplitz((self.conv_vect_past, self.conv_vect_future), data, check_finite=False, workers=None)
      filled_qa = matmul_toeplitz((self.conv_vect_past, self.conv_vect_future), valid_mask, check_finite=False, workers=None)
      conv_vec = np.concatenate((self.conv_vect_past, self.conv_vect_future[-1:0:-1]))
      nz_conv_vec = conv_vec[conv_vec>0]
      min_conv_val = np.min(nz_conv_vec)
      filled = filled/filled_qa
      no_fill_mask = filled_qa < min_conv_val
      filled_qa /= np.max(norm_vec)
      filled[no_fill_mask] = np.nan
      filled_qa[no_fill_mask] = 0
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
        warnings.warn("Less then two years of images available, the time series reconstruction will not take advantage of seasonality")
      half_conv_vect = self._compute_conv_mat_row(n_imag)
      if self.conv_vect_future == []:
        self.conv_vect_future = half_conv_vect
      if self.conv_vect_past == []:
        self.conv_vect_past = half_conv_vect

      filled, filled_qa = self._fftw_toeplitz_matmul(
        data, valid_mask.astype(float))
      filled[valid_mask] = data[valid_mask]
      filled_qa[valid_mask] = 1.0
      filled_qa = filled_qa * 100
      filled_qa[filled_qa == 0.0] = np.nan      
      # Return the reconstructed time series and the quality assesment layer
      if self.return_qa:
        return np.reshape(filled.T, orig_shape), np.reshape(filled_qa.T, orig_shape) 
      else:
        return np.reshape(filled.T, orig_shape)

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
      operations:List = ['p25', 'p50', 'p75', 'std'],
      rename_operations:dict = {},
      post_expression:str = None,
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

      self.post_expression = post_expression

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

    def _aggregate(self, new_idx, ref_array, array_idx, group, tm, dt1, dt2):

      array = load_memmap(**ref_array)

      ops = []
      _idxs = []

      for op, method in self.bn_ops:
        array[:,:,new_idx:new_idx+1] = method(array[:,:,array_idx], axis=-1)[:, :, np.newaxis]
        _idxs.append(new_idx)
        new_idx += 1

        ops.append(self._op_name(f'{op}'))

      if len(self.percs) > 0:
        perc_idx = list(range(new_idx, new_idx + len(self.percs)))
        in_array = array[:,:,array_idx] #array[:,:,array_idx].copy()
        array[:,:,perc_idx] = nan_percentile(in_array.transpose((2,0,1)), q=self.percs).transpose((1,2,0))
        new_idx += len(self.percs)
        _idxs += perc_idx
        
        for p in self.percs:
          ops.append(self._op_name(f'p{p}'))

      if self.post_expression is not None and len(_idxs) > 0:
        for idx in _idxs:
          array[:,:,idx] = ne.evaluate(self.post_expression, local_dict = { 'new_array': array[:,:,idx] })

      return (group, ops, tm, dt1, dt2)

    def _args_monthly(self, rdata, group, start_dt, end_dt, date_format, months = 1, daysp = None):
      
      args = []
      ref_array = ref_memmap(rdata.array)

      for dt1, dt2 in date_range(
        f'{start_dt.year}0101',f'{end_dt.year}1201', 
        'months', months, return_str=True, ignore_29feb=True,
        date_format=date_format):
          
        dt1a, dt2a = dt1, dt2
        if daysp is not None:
          dt1a = datetime.strptime(dt1, date_format)
          dt2a = datetime.strptime(dt2, date_format)
          dt1a = (dt1a - relativedelta(days=daysp)).strftime(date_format)
          dt2a = (dt2a + relativedelta(days=daysp)).strftime(date_format)

        tm = ''
        array_idx = rdata.filter_date(dt1a, dt2a, return_idx=True, 
          date_format=date_format, date_overlap=self.date_overlap)
        
        if len(array_idx) > 0:  
          args += [ (ref_array, array_idx, group, tm, datetime.strptime(dt1, date_format), datetime.strptime(dt2, date_format)) ]

      return args

    def _args_yearly(self, rdata, group, start_dt, end_dt, date_format):
      
      args = []
      ref_array = ref_memmap(rdata.array)

      for dt1, dt2 in date_range(
        f'{start_dt.year}0101',f'{end_dt.year}1201', 
        'years', 1, return_str=True, ignore_29feb=False, 
        date_format=date_format):

        tm = 'yearly'
        array_idx = rdata.filter_date(dt1, dt2, return_idx=True, 
          date_format=date_format, date_overlap=self.date_overlap)
        
        if len(array_idx):  
          args += [ (ref_array, array_idx, group, tm, datetime.strptime(dt1, date_format), datetime.strptime(dt2, date_format)) ]

      return args

    def _args_monthly_longterm(self, rdata, group, start_dt, end_dt, date_format):

      args = []
      ref_array = ref_memmap(rdata.array)

      for month in range(1,13):
        
        array_idx_list = []
        month = str(month).zfill(2)

        for dt1, dt2 in date_range(
          f'{start_dt.year}{month}01',f'{end_dt.year}{month}01', 
          'months', 1, date_offset=11, return_str=True, 
          ignore_29feb=False, date_format=date_format):
          
          array_idx = rdata.filter_date(dt1, dt2, return_idx=True, 
              date_format=date_format, date_overlap=self.date_overlap)
          
          if len(array_idx):
            array_idx_list += array_idx

        tm = f'm{month}'
        if len(array_idx_list) > 0:
          #args += [ (np.concatenate(in_array, axis=-1), tm, start_dt, end_dt) ]
          args += [ (ref_array, array_idx_list, group, tm, start_dt, end_dt) ]

      return args

    def _run(self, 
      rdata:RasterData,
      group_list:list,
      ginfo_list:list,
      outname:str = 'skmap_aggregate.{gr}_{op}_{dt}'
    ):

      args = []

      for group, ginfo in zip(group_list, ginfo_list):

        date_format = '%Y%m%d'
        start_dt = ginfo[RasterData.START_DT_COL].min()
        end_dt = ginfo[RasterData.END_DT_COL].max()

        rdata._active_group = group
      
        for t in self.time:

          if t == TimeEnum.MONTHLY_LONGTERM:
            args += self._args_monthly_longterm(rdata, group, start_dt, end_dt, date_format)
          elif t == TimeEnum.YEARLY:
            args += self._args_yearly(rdata, group, start_dt, end_dt, date_format)
          elif t == TimeEnum.MONTHLY:
            args += self._args_monthly(rdata, group, start_dt, end_dt, date_format, 1)
          elif t == TimeEnum.MONTHLY_15P:
            args += self._args_monthly(rdata, group, start_dt, end_dt, date_format, 1, 15)
          elif t == TimeEnum.BIMONTHLY:
            args += self._args_monthly(rdata, group, start_dt, end_dt, date_format, 2)
          elif t == TimeEnum.BIMONTHLY_15P:
            args += self._args_monthly(rdata, group, start_dt, end_dt, date_format, 2, 15)
          elif t == TimeEnum.QUARTERLY:
            args += self._args_monthly(rdata, group, start_dt, end_dt, date_format, 3)
          else:
            raise Exception(f"Aggregation by {t} not implemented")
        
      n_new_rasters = len(args) * len(self.operations)
      idx_offset = rdata._idx_offset()

      _args = []
      for idx, arg in zip(range(0, n_new_rasters, len(self.operations)), args):
        _arg = list(arg)
        _arg.insert(0, idx_offset + idx)
        _args.append(tuple(_arg))

      args = _args
      new_info = []

      self._verbose(f"Computing {len(args)} "
        + f"time aggregates from {start_dt.year} to {end_dt.year}"
      )

      for group, ops, tm, dt1, dt2 in parallel.job(self._aggregate, args, joblib_args={'backend': 'multiprocessing'}):
        for op in ops:
          
          _group = group
          if tm != '':
            _group = f'{group}.{tm}'

          rdata._active_group = group
          name = rdata._set_date(outname, 
                dt1, dt2, 
                op=op, gr=_group
              )
          
          new_group = f'{_group}.{op}'

          new_info.append(
            rdata._new_info_row(rdata.base_raster, name=name, group=new_group, 
              dates=[dt1, dt2])
          )

      rdata._active_group = None
      
      return None, DataFrame(new_info)

  class PeakAnalysis(Derivator):
    
    def __init__(self,
      season_size:int,
      min_height:float = 0.5,
      min_prominence:float = 0.2,
      min_distance:float = 1.0,
      scale_expr:str = None,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(verbose=verbose, temporal=True)
      
      self.season_size = season_size
      self.min_height = min_height
      self.min_prominence = min_prominence
      self.min_distance = min_distance
      self.scale_expr = scale_expr
      self.n_jobs = n_jobs

      self.name_misc = [
        ('peaks', 'm', 100), ('peaks', 'n', 1), 
      ]

      self.scale_arr = np.array([ scale for _, _, scale in self.name_misc ])

    def _find_peaks(self, data):

      if self.scale_expr is not None:
        data = ne.evaluate(self.scale_expr, { 'data': data })

      has_nan = np.sum(np.isnan(data).astype('int'))
      
      ts_size = data.shape[0]
      idxs = [ (i, i + self.season_size) for i in range(0, ts_size, self.season_size) ]

      n_bands = self.scale_arr.shape
      result = np.empty((len(idxs) * 2))
      
      if has_nan == 0:
        
        peaks, _ = find_peaks(data, height=self.min_height, prominence=self.min_prominence, distance=self.min_distance)
        _peaks = list(peaks)

        o2 = 0

        if len(peaks) > 0:
          for i0, i1 in idxs:
            seas_peaks = list(( i for i in range(i0, i1) if i in _peaks))
            nos = len(seas_peaks)
            
            mean, los = np.nan, 0
            if nos > 0:
              mean = np.mean(data[seas_peaks])
              los = np.sum(data[i0:i1] > mean * 0.5) / self.season_size

            result[o2] = los * self.scale_arr[0]
            result[o2 + 1] = nos * self.scale_arr[1]
            o2 += 2

      return result

    def _unpack(self, i0_0, i0_1, i2, ref_array, idx_offset):
      
      array = load_memmap(**ref_array)
      result = np.apply_along_axis(self._find_peaks, 2, array[i0_0:i0_1, :, i2])
      o2 = list(range(idx_offset, idx_offset + result.shape[2]))
      array[i0_0:i0_1, :, o2] = result
          
      return True

    def _args(self, rdata, ginfo):

      ref_array = ref_memmap(rdata.array)
      max_i0 = rdata.array.shape[0]
      rows_per_job = math.ceil(max_i0 / self.n_jobs)

      idx_offset = rdata._idx_offset()

      args = []
      for i in range(0, max_i0, rows_per_job):
        i0_0, i0_1 = i, (i + rows_per_job)
        if i0_1 > max_i0:
          i0_1 = max_i0

        i2 = ginfo.index
        args.append((i0_0, i0_1, i2, ref_array, idx_offset))

      return args

    def _run(self, 
      rdata:RasterData,
      group_list:list,
      ginfo_list:list,
      outname:str = 'skmap_{gr}.{nm}_{pr}_{dt}'
    ):

      new_info = []

      for group, ginfo in zip(group_list, ginfo_list):

        rdata._active_group = group
        array = rdata._array()

        start_dt_min = ginfo[RasterData.START_DT_COL].min()
        end_dt_max = ginfo[RasterData.END_DT_COL].max()

        ts_size = ginfo.shape[0]
        
        args = self._args(rdata, ginfo)
        
        for r in parallel.job(self._unpack, args, n_jobs=self.n_jobs, joblib_args={'backend': 'multiprocessing'}):
          continue

        for i in range(0, ts_size, self.season_size):
          
          _i = int(i /  self.season_size)
          i0, i1 = (i, i + self.season_size - 1)
          
          start_dt_min = ginfo.iloc[i0][RasterData.START_DT_COL]
          end_dt_max = ginfo.iloc[i1][RasterData.END_DT_COL]
          
          for j, (nm, pr, _) in zip(range(_i, _i + len(self.name_misc)), self.name_misc):
            
            name = rdata._set_date(outname, start_dt_min, 
              end_dt_max, nm=nm, pr=pr, gr=group)

            new_group = f'{group}.{nm}.{pr}'

            new_info.append(
              rdata._new_info_row(rdata.base_raster, group=new_group, name=name, dates=[start_dt_min, end_dt_max])
            )
          
      return None, DataFrame(new_info)

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
        ('r2', 'm', 100)
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
        
        if np.std(data) == 0:
          nan_result = np.empty(out_size)
          nan_result[:] = np.nan
          return nan_result

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
      outname:str = 'skmap_{gr}.{nm}_{pr}_{dt}'
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
          rdata._new_info_row(rdata.base_raster, group=new_group, name=name, dates=[start_dt, end_dt])
        )

      ts_size = array.shape[2]

      for i, (nm, pr, scale) in zip(range(0, len(self.name_misc)), self.name_misc):
        
        new_array[:,:,ts_size + i] *= scale

        name = rdata._set_date(outname, start_dt_min, 
          end_dt_max, nm=nm, pr=pr, gr=group)

        new_group = f'{group}.{nm}.{pr}'

        new_info.append(
          rdata._new_info_row(rdata.base_raster, group=new_group, name=name, dates=[start_dt_min, end_dt_max])
        )

      return new_array, DataFrame(new_info)
    
  class Calc(SKMapRunner):

    def __init__(self,
      expressions:dict,
      mask_group:str = None,
      mask_values:list = [],
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      self.n_jobs = n_jobs

      self.expressions = expressions
      self.mask_group = mask_group
      self.mask_values = mask_values
      self.date_cols = [RasterData.START_DT_COL, RasterData.END_DT_COL]

    def _map(self, ref_array, gmap, new_gmap):
      
      array_dict = {}
      array = load_memmap(**ref_array)
  
      array_mask = None
      if self.mask_group is not None and len(self.mask_values) >= 1:
        idx = gmap[self.mask_group]
        array_mask = np.isin(array[:,:,idx], self.mask_values)
      
      for group in gmap.keys():
        idx = gmap[group]
        array_dict[group] = array[:,:,idx]
        if array_mask is not None and group != self.mask_group:
          array_dict[group][array_mask] = np.nan
          
      for group in self.expressions.keys():
        expression = self.expressions[group]
        if group in gmap:
          idx = gmap[group]
        else:
          idx = new_gmap[group]
        array[:,:,idx] = ne.evaluate(expression, local_dict=array_dict)
      
      fidx = list(gmap.values())[0]

      return(fidx)

    def run(self, 
      rdata:RasterData,
      outname:str = 'skmap_{gr}_{dt}'
    ):

      self.groups = list(rdata.info[RasterData.GROUP_COL].unique())
      n_dates = rdata.info[self.date_cols].value_counts().shape[0]

      self.new_groups = []
      for key in self.expressions.keys():
        if key not in self.groups:
          self.new_groups.append(key)

      args = []

      ref_array = ref_memmap(rdata.array)

      idx_offset = rdata._idx_offset()
      idx_counter = 0
      n_new_groups = len(self.new_groups)
      for _, rows in rdata.info.groupby(self.date_cols):
        gidx = rows.index
        ggroup = list(rdata.info.iloc[gidx]['group'])

        gmap = {}
        new_gmap = {}
        
        for idx, group in zip(gidx, ggroup):
          gmap[group] = idx

        new_group_offset = idx_offset + (idx_counter * n_new_groups)
        for idx, new_group in zip(range(0, n_new_groups), self.new_groups):
          new_gmap[new_group] = (new_group_offset + idx)
        
        args.append((ref_array, gmap, new_gmap))
        idx_counter += 1
      
      new_info = []

      for fidx in parallel.job(self._map, args, n_jobs=self.n_jobs, joblib_args={'backend': 'multiprocessing'}):
        
        row = rdata.info.iloc[fidx]

        start_dt, end_dt = row[RasterData.START_DT_COL], row[RasterData.END_DT_COL]
        group = row[RasterData.GROUP_COL]

        date_format = rdata.date_args[group]['date_format']
        date_style = rdata.date_args[group]['date_style']

        for new_group in self.new_groups:
          name = rdata._set_date(outname, start_dt, end_dt, 
            date_format=date_format, date_style=date_style,  gr=new_group)
          new_info.append(
            rdata._new_info_row(rdata.base_raster, 
              date_format=date_format, date_style=date_style,
              group=new_group, name=name, dates=[start_dt, end_dt]
            )
          )

      return None, DataFrame(new_info)

    def _calc(self, array_dict):

      if self.mask_group is not None and len(self.mask_values) >= 1:
        array_mask = np.isin(array_dict[self.mask_group], self.mask_values)

        for g in array_dict.keys():
          if g != self.mask_group:
            array_dict[g][array_mask] = np.nan

      for group in self.expressions.keys():
        expression = self.expressions[group]
        array_dict[group] = ne.evaluate(expression, local_dict=array_dict)
    
      return array_dict

except ImportError as e:
  from skmap.misc import _warn_deps
  _warn_deps(e, 'skmap.io')