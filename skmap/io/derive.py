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
  
  import numpy as np
  import bottleneck as bn
  from datetime import datetime
  from pandas import DataFrame

  class Derivator(SKMapRunner, ABC):
    
    def __init__(self,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(verbose=verbose, temporal=temporal)

    def run(self, 
      rdata:RasterData,
      outname:str = None
    ):
      """
      Execute the gapfilling approach.
      """

      kwargs = {'rdata': rdata}
      if outname is not None:
        kwargs['outname'] = outname

      start = time.time()
      new_array, new_info = self._run(**kwargs)

      return new_array, new_info

    @abstractmethod
    def _run(self, 
      rdata:RasterData, 
      outname:str
    ):
      pass

  class TimeEnum(Enum):
    
    MONTHLY = 1
    MONTHLY_15P = 1
    MONTHLY_LONGTERM = 3

    BIMONTHLY = 1
    BIMONTHLY_15P = 1
    BIMONTHLY_LONGTERM = 3

    QUARTERLY = 1
    YEARLY = 2

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
      outname:str = 'skmap_derivative.{tm}_{op}_{dt}.tif'
    ):

      date_format = '%Y%m%d'
      start_dt = rdata.info[RasterData.START_DT_COL].min()
      end_dt = rdata.info[RasterData.END_DT_COL].max()

      args = []

      for t in self.time:

        if t == TimeEnum.MONTHLY_LONGTERM:
          args += self._args_monthly_longterm(rdata, start_dt, end_dt, date_format)
        elif TimeEnum.YEARLY in self.time:
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
                op=op, tm=tm
              )
          
          new_info.append(
            rdata._new_info_row('', name=name, dates=[start_dt, end_dt])
          )

        new_array.append(out_array)
        
      new_array = np.concatenate(new_array, axis=-1)

      return new_array, DataFrame(new_info)

  class TrendLinearRegression(Derivator):
    
    def __init__(self,
      season_size:int,
      season_smoother:int = None,
      trend_smoother:int = None,
      trend_log1p:bool = True,
      scale_factor:int = 10000,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(verbose=verbose, temporal=True)
      
      self.season_size = season_size
      self.season_smoother = season_smoother
      self.trend_smoother = trend_smoother
      self.trend_log1p = trend_log1p
      self.n_jobs = n_jobs

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
        if self.trend_log1p:
          y = log1p(res.trend)
        
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
      outname:str = 'skmap_derivative.{nm}_{pr}_{dt}.tif'
    ):

      array = rdata.array
      info = rdata.info

      start_dt_min = rdata.info[RasterData.START_DT_COL].min()
      end_dt_max = rdata.info[RasterData.END_DT_COL].max()

      new_array = parallel.apply_along_axis(self._trend_regression, 
        axis=2, arr=array, n_jobs=self.n_jobs)

      new_info = []

      for index, row in info.iterrows():
        start_dt = row[RasterData.START_DT_COL]
        end_dt = row[RasterData.END_DT_COL]

        name = rdata._set_date(outname, 
          start_dt, end_dt, nm=f'trend', pr='m')

        new_info.append(
          rdata._new_info_row('', name=name, dates=[start_dt, end_dt])
        )

      ts_size = array.shape[2]

      for i, (nm, pr, scale) in zip(range(0, len(self.name_misc)), self.name_misc):
        
        new_array[:,:,ts_size + i] *= scale
        
        name = rdata._set_date(outname, start_dt_min, 
          end_dt_max, nm=nm, pr=pr)

        new_info.append(
          rdata._new_info_row('', name=name, dates=[start_dt_min, end_dt_max])
        )

      return new_array, DataFrame(new_info)

except ImportError as e:
  from skmap.misc import _warn_deps
  _warn_deps(e, 'skmap.io.derive')