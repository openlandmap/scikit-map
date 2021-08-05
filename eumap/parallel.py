"""
Parallelization helpers based in thread and process pools
"""
import numpy
import multiprocessing
from typing import Callable, Iterator

import geopandas as gpd
import multiprocessing
from osgeo import osr
import math
import rasterio
from rasterio.windows import Window, from_bounds
import os.path
from pqdm.processes import pqdm as ppqdm
from pqdm.threads import pqdm as tpqdm

from .misc import ttprint
from . import datasets

EUMAP_TILING_SYSTEM_FN = 'eu_tilling system_30km.gpkg'
"""
Filename with the default tiling system for eumap
"""

CPU_COUNT = multiprocessing.cpu_count()
"""
Number of CPU cores available.
"""

def ThreadGeneratorLazy(
  worker:Callable, 
  args:Iterator[tuple], 
  max_workers:int = CPU_COUNT, 
  chunk:int  = CPU_COUNT*2, 
  fixed_args:tuple = ()
):
  """ 
  Execute a function in parallel using a ``ThreadPoolExecutor`` [1].

  :param worker: Function to execute in parallel.
  :param args: Argument iterator where each element is send job of the pool.
  :param max_workers: Number of CPU cores to use in the parallelization.
    By default all cores are used.
  :param chunk: Number of chunks to split the parallelization jobs.
  :param fixed_args: Constant arguments added in ``args`` in each 
    execution of the ``worker`` function.
  :returns: A generator with the return of all workers
  :rtype: Generator

  >>> from eumap.parallel import ThreadGeneratorLazy
  >>> 
  >>> def worker(i, msg):
  >>>   print(f'{i}: {msg}')
  >>>   return f'Worker {i} finished'
  >>> 
  >>> args = iter([ (i,) for i in range(0,5)])
  >>> fixed_args = ("I'm running in parallel", )
  >>> 
  >>> for result in ThreadGeneratorLazy(worker, args, fixed_args=fixed_args):
  >>>   print(result)

  [1] `Python ThreadPoolExecutor class <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_

  """
  import concurrent.futures
  from itertools import islice

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    group = islice(args, chunk)
    futures = {executor.submit(worker, *arg + fixed_args) for arg in group}

    while futures:
      done, futures = concurrent.futures.wait(
        futures, return_when=concurrent.futures.FIRST_COMPLETED
      )

      for future in done:
        yield future.result()

      group = islice(args, chunk)

      for arg in group:
        futures.add(executor.submit(worker,*arg + fixed_args))

def ProcessGeneratorLazy(
  worker:Callable, 
  args:Iterator[tuple], 
  max_workers:int = CPU_COUNT, 
  chunk:int  = CPU_COUNT*2, 
  fixed_args:tuple = ()
):
  """ 
  Execute a function in parallel using a ``ProcessPoolExecutor`` [1].

  :param worker: Function to execute in parallel.
  :param args:     to separate  job of the pool.
  :param max_workers: Number of CPU cores to use in the parallelization.
    By default all cores are used.
  :param chunk: Number of chunks to split the parallelization jobs.
  :param fixed_args: Constant arguments added in ``args`` in each 
    execution of the ``worker`` function.
  :returns: A generator with the return of all workers
  :rtype: Generator

  >>> from eumap.parallel import ProcessGeneratorLazy
  >>> 
  >>> def worker(i, msg):
  >>>   print(f'{i}: {msg}')
  >>>   return f'Worker {i} finished'
  >>> 
  >>> args = iter([ (i,) for i in range(0,5)])
  >>> fixed_args = ("I'm running in parallel", )
  >>> 
  >>> for result in ProcessGeneratorLazy(worker, args, fixed_args=fixed_args):
  >>>   print(result)

  [1] `Python ProcessPoolExecutor class <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_

  """
  import concurrent.futures
  from itertools import islice

  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    group = islice(args, chunk)
    futures = {executor.submit(worker, *arg + fixed_args) for arg in group}

    while futures:
      done, futures = concurrent.futures.wait(
        futures, return_when=concurrent.futures.FIRST_COMPLETED
      )

      for future in done:
        yield future.result()

      group = islice(args, chunk)

      for arg in group:
        futures.add(executor.submit(worker, *arg + fixed_args))

def job(
  worker:Callable, 
  worker_args:Iterator[tuple], 
  n_jobs:int = -1, 
  joblib_args:set = {}
):
  """ 
  Execute a function in parallel using **Joblib**.

  :param worker: Function to execute in parallel.
  :param worker_args: Argument iterator where each element is send
    to separate job.
  :param joblib_args: Number of CPU cores to use in the parallelization.
    By default all cores are used.
  :param joblib_args: Joblib argumets to send to ``Parallel class`` [1].
  :returns: A generator with the return of all workers
  :rtype: Generator

  >>> from eumap import parallel
  >>> 
  >>> def worker(i, msg):
  >>>   print(f'{i}: {msg}')
  >>>   return f'Worker {i} finished'
  >>> 
  >>> msg = ("I'm running in parallel", )
  >>> args = iter([ (i,msg) for i in range(0,5)])
  >>> 
  >>> for result in parallel.job(worker, args, n_jobs=-1, joblib_args={'backend': 'threading'}):
  >>>   print(result)

  [1] `joblib.Parallel class <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib.Parallel>`_

  """
  from joblib import Parallel, delayed

  joblib_args['n_jobs'] = n_jobs

  for worker_result in Parallel(**joblib_args)(delayed(worker)(*args) for args in worker_args):
    yield worker_result

def apply_along_axis(
  worker:Callable, 
  axis, 
  arr:numpy.array, 
  *args:any, 
  **kwargs:any
):
  """ 
  Execute a function through a ``numpy.array`` axis in parallel [1].
  It uses joblib and ``backend=loky``, so avoid to send shared 
  memory objects as arguments.

  :param worker: Function to execute in parallel. It needs to have
    at least one argument (``numpy.array``).
  :param axis: Axis used to execute the worker.
  :param arr: The input array.
  :param args: Additional arguments to the worker.
  :param kwargs: Additional named arguments to the worker.
  :returns: The output array with one dimension less than the input array.
  :rtype: numpy.array

  >>> from eumap import parallel
  >>> 
  >>> def fn(arr, const):
  >>>   return np.sum(arr) + const
  >>> 
  >>> const = 1
  >>> arr = np.ones((100,100,100))
  >>> 
  >>> out = parallel.apply_along_axis(fn, 0, arr, const)
  >>> print(arr.shape, out.shape)
  
  [1] `Best answer from Eric O Lebigot <https://stackoverflow.com/a/45555516>`_

  """
  import numpy as np

  def run(worker, axis, arr, args, kwargs):
    return np.apply_along_axis(worker, axis, arr, *args, **kwargs)

  """
  Like numpy.apply_along_axis(), but takes advantage of multiple
  cores.
  """        
  # Effective axis where apply_along_axis() will be applied by each
  # worker (any non-zero axis number would work, so as to allow the use
  # of `np.array_split()`, which is only done on axis 0):
  effective_axis = 1 if axis == 0 else axis
  if effective_axis != axis:
      arr = arr.swapaxes(axis, effective_axis)

  # Chunks for the mapping (only a few chunks):
  chunks = [(worker, effective_axis, sub_arr, args, kwargs)
            for sub_arr in np.array_split(arr, CPU_COUNT)]
  
  result = []
  for r in job(run, chunks):
    result.append(r)
    
  return np.concatenate(result)


class TilingProcessing():

  def __init__(self, tiling_system_fn = None,
         base_raster_fn = None,
         pixel_precision = 6,
         verbose:bool = True):

    if tiling_system_fn is None:

      if verbose:
        ttprint('Using default eumap tiling system')

      tiling_system_fn = f'eumap_data/{EUMAP_TILING_SYSTEM_FN}'
      if not os.path.isfile(tiling_system_fn):
        datasets.get_data(EUMAP_TILING_SYSTEM_FN)

    self.pixel_precision = pixel_precision
    self.tiles = gpd.read_file(tiling_system_fn)
    self.num_tiles = self.tiles.shape[0]
    self.base_raster = rasterio.open(base_raster_fn)

    tiles_srs = osr.SpatialReference()
    tiles_srs.ImportFromProj4(self.tiles.crs.to_proj4())
    base_raster_srs = osr.SpatialReference()
    base_raster_srs.ImportFromProj4(self.base_raster.crs.to_proj4())

    if base_raster_srs.ImportFromProj4(self.base_raster.crs.to_proj4()) != tiles_srs.ImportFromProj4(self.tiles.crs.to_proj4()):
      raise Exception('Different SpatialReference' +
            f'\n tiling_system_fn ({tiles_srs.ExportToProj4()})'+
            f'\n base_raster_fn ({base_raster_srs.ExportToProj4()})')
    if verbose:
      ttprint(f'{self.num_tiles} tiles available')

  def process_one(self, idx, func, func_args = ()):

    tile = self.tiles.iloc[idx]
    left, bottom, right, top = tile.geometry.bounds

    window = from_bounds(left, bottom, right, top, self.base_raster.transform)

    return func(idx, tile, window, *func_args)

  def process_multiple(self, idx_list, func, func_args = (),
    max_workers:int = multiprocessing.cpu_count(),
    use_threads:bool = True,
    progress_bar:bool = True):
    print(f"func_args: {func_args}")
    args = []
    for idx in idx_list:
      tile = self.tiles.iloc[idx]
      left, bottom, right, top = tile.geometry.bounds

      window = from_bounds(left, bottom, right, top, self.base_raster.transform)

      args.append((idx, tile, window, *func_args))
    if progress_bar:
      pqdm = (tpqdm if use_threads else ppqdm)
      results = pqdm(args,func,n_jobs=max_workers,argument_type='args')

    else:
      WorkerPool = (ThreadGeneratorLazy if use_threads else ProcessGeneratorLazy)

      results = []
      for r in WorkerPool(func, iter(args), max_workers=max_workers, chunk=max_workers*2):
        results.append(r)

    return results

  def process_all(self, func, func_args = (),
    max_workers:int = multiprocessing.cpu_count(),
    use_threads:bool = True):

    idx_list = range(0, self.num_tiles)
    return self.process_multiple(idx_list, func, func_args, max_workers, use_threads)