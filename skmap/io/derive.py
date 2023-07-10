import time
import os

try:

  from abc import ABC, abstractmethod
  from skmap import parallel

  from skmap import SKMapRunner, parallel
  from skmap.misc import gen_dates, nan_percentile
  from skmap.io import RasterData
  
  import numpy as np
  import bottleneck as bn
  from pandas import DataFrame

  class Derivator(SKMapRunner, ABC):
    
    def __init__(self,
      verbose:bool = True,
      temporal = False
    ):
      super().__init__(verbose=verbose, temporal=temporal)

    def run(self, 
      rdata:RasterData,
      outname:str = 'skmap_derivative_{op}_{dt}.tif'
    ):
      """
      Execute the gapfilling approach.
      """

      start = time.time()
      new_array, new_info = self._run(rdata, outname)

      return new_array, new_info

    @abstractmethod
    def _run(self, 
      rdata:RasterData, 
      outname:str
    ):
      pass

  class TimeAggregate(Derivator):
    
    def __init__(self,
      yealy = True,
      operations = ['p25', 'p50', 'p75', 'std'],
      rename_operations:dict = {},
      date_overlap:bool = False,
      n_jobs:int = os.cpu_count(),
      verbose = False
    ):

      super().__init__(verbose=verbose, temporal=True)

      self.yealy = yealy
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

    def _aggregate(self, in_array, tm):

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

      return out_array, ops, tm

    def _run(self, 
      rdata:RasterData,
      outname:str = 'skmap_derivative.{tm}_{op}_{dt}.tif'
    ):

      date_format = '%Y%m%d'
      start_dt = rdata.info[RasterData.START_DT_COL].min()
      end_dt = rdata.info[RasterData.END_DT_COL].max()

      args = []

      for month in range(1,13):
        
        in_array = []
        month = str(month).zfill(2)

        for dt1, dt2 in gen_dates(
          f'{start_dt.year}{month}01',f'{end_dt.year}{month}01', 
          'months', 1, date_offset=11, return_str=True, 
          ignore_29feb=False, date_format=date_format):
          in_array.append(
            rdata.filter_date(dt1, dt2, return_array=True, 
              date_format=date_format, date_overlap=self.date_overlap)
          )

        tm = f'm{month}'
        args += [ (np.concatenate(in_array, axis=-1), tm) ]

      if self.yealy:
        for dt1, dt2 in gen_dates(
          f'{start_dt.year}0101',f'{end_dt.year}1201', 
          'years', 1, return_str=True, ignore_29feb=False, 
          date_format=date_format):

          tm = 'yearly'
          in_array = rdata.filter_date(dt1, dt2, return_array=True, 
            date_format=date_format, date_overlap=self.date_overlap)
          args += [ (in_array, tm) ]

      new_array = []
      new_info = []

      self._verbose(f"Computing {len(args)} "
        + f"time aggregates from {start_dt.year} to {end_dt.year}"
      )

      for out_array, ops, tm in parallel.job(self._aggregate, args, joblib_args={'backend': 'threading'}):
        for op in ops:
          name = rdata._set_date(outname, 
                start_dt, end_dt, 
                rdata.date_format, rdata.date_style, 
                op=op, tm=tm
              )
          
          new_info.append(
            rdata._new_info_row('', name, [start_dt, end_dt])
          )

        new_array.append(out_array)
        
      new_array = np.concatenate(new_array, axis=-1)

      return new_array, DataFrame(new_info)

except ImportError as e:
  from skmap.misc import _warn_deps
  _warn_deps(e, 'gapfiller')