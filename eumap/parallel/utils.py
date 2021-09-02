"""
Parallelization helpers based in thread/process pools and joblib
"""
import numpy
import multiprocessing
from typing import Callable, Iterator, List

import warnings
from pathlib import Path
import geopandas as gpd
import multiprocessing
from osgeo import osr
import math
import rasterio
from rasterio.windows import Window, from_bounds
import os.path

from ..misc import ttprint
from .. import datasets

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

  Examples
  ========

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

  References
  ==========

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

  Examples
  ========

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

  References
  ==========

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
  Execute a function in parallel using joblib [1].

  :param worker: Function to execute in parallel.
  :param worker_args: Argument iterator where each element is send
    to separate job.
  :param joblib_args: Number of CPU cores to use in the parallelization.
    By default all cores are used.
  :param joblib_args: Joblib argumets to send to ``Parallel class`` [1].
  :returns: A generator with the return of all workers
  :rtype: Generator

  Examples
  ========

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

  References
  ==========

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

  Examples
  ========

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

  References
  ==========

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
  """
  Execute a processing function in parallel considering a tiling system
  and a base raster. It creates a rasterio ``window`` object for each tile
  according to the pixel size of the specified base.

  :param tiling_system_fn: Vector file path with the tiles to read.
  :param base_raster_fn: Raster file path used the retrieve
    the ``affine transformation`` for ``windows``.
  :param verbose: Use ``True`` to print informations about read tiles
    and the base raster.

  """

  def __init__(self,
    tiling_system_fn = 'http://s3.eu-central-1.wasabisys.com/eumap/tiling_system_30km.gpkg',
    base_raster_fn = 'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_eumap_epsg3035_v1.0.tif',
    verbose:bool = False,
    epsg_checking:bool = True
  ):

    from pyproj import CRS

    self.tiles = gpd.read_file(tiling_system_fn)
    self.num_tiles = self.tiles.shape[0]
    self.base_raster = rasterio.open(base_raster_fn)

    tile_epsg = CRS(self.tiles.crs.to_wkt()).to_epsg()
    raster_epsg = CRS(self.base_raster.crs.to_wkt()).to_epsg()

    if epsg_checking and tile_epsg != raster_epsg:
      raise Exception(
        'Different SpatialReference' +
        f'\n tiling_system_fn:\n{self.tiles.crs.to_wkt()}'+
        f'\n base_raster_fn:\n{self.base_raster.crs.to_wkt()}'
      )

    if verbose:
      pixel_size = self.base_raster.transform[0]
      ttprint(f'Pixel size equal {pixel_size} in {Path(base_raster_fn).name}')
      ttprint(f'{self.num_tiles} tiles available in {Path(tiling_system_fn).name}')
      ttprint(f'Using EPSG:{raster_epsg}')

  def _tile_window(self, idx):
    tile = self.tiles.iloc[idx]
    left, bottom, right, top = tile.geometry.bounds

    return tile, from_bounds(left, bottom, right, top, self.base_raster.transform)

  def process_one(self,
    idx:int,
    func:Callable,
    *args:any
  ):
    """
    Process a single tile using the specified function args.

    :param idx: The tile id to process. This idx is generated for all the tiles
      in a sequence starting from ``0``.
    :param func: A function with at least the arguments ``idx, tile, window``.
    :param args: Additional arguments to send to the function.

    Examples
    ========

    >>> from eumap.parallel import TilingProcessing
    >>> from eumap.raster import read_rasters
    >>>
    >>> def run(idx, tile, window, raster_files):
    >>>     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
    >>>     print(f'Tile {idx}: data read {data.shape}')
    >>>
    >>> raster_files = [
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_eumap_epsg3035_v1.0.tif', # winter
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_eumap_epsg3035_v1.0.tif', # spring
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_eumap_epsg3035_v1.0.tif', # summer
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_eumap_epsg3035_v1.0.tif'  # fall
    >>> ]
    >>>
    >>> tiling= TilingProcessing(verbose=True)
    >>> tiling.process_one(0, run, raster_files)

    """
    tile, window = self._tile_window(idx)
    return func(idx, tile, window, *args)

  def process_multiple(self,
    idx_list:List[int],
    func:Callable,
    *args:any,
    max_workers:int = CPU_COUNT,
    use_threads:bool = True,
    progress_bar:bool = False
  ):
    """
    Process in parallel a list of tile using the specified function args.

    :param idx: The tile ids to process. This idx is generated for all the tiles
      in a sequence starting from ``0``.
    :param func: A function with at least the arguments ``idx, tile, window``.
    :param args: Additional arguments to send to the function.
    :param max_workers: Number of CPU cores to use in the parallelization.
      By default all cores are used.
    :param use_threads: If ``True`` the parallel processing uses ``ThreadGeneratorLazy``,
      otherwise it uses ProcessGeneratorLazy.
    :param progress_bar: If ``True`` the parallel processing uses ``pqdm`` [1] presenting
      a progress bar and ignoring the ``use_threads``.

    Examples
    ========

    >>> from eumap.parallel import TilingProcessing
    >>> from eumap.raster import read_rasters
    >>>
    >>> def run(idx, tile, window, raster_files):
    >>>     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
    >>>     print(f'Tile {idx}: data read {data.shape}')
    >>>
    >>> raster_files = [
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_eumap_epsg3035_v1.0.tif', # winter
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_eumap_epsg3035_v1.0.tif', # spring
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_eumap_epsg3035_v1.0.tif', # summer
    >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_eumap_epsg3035_v1.0.tif'  # fall
    >>> ]
    >>>
    >>> tiling= TilingProcessing(verbose=True)
    >>> idx_list = [0,10,100]
    >>> result = tiling.process_multiple(idx_list, run, raster_files)

    References
    ==========

    [1] `Parallel TQDM <https://pqdm.readthedocs.io/en/latest/readme.html>`_

    """

    _args = []

    for idx in idx_list:
      tile, window = self._tile_window(idx)
      _args.append((idx, tile, window, *args))

    if progress_bar:
      try:
        from pqdm.processes import pqdm as ppqdm
        from pqdm.threads import pqdm as tpqdm
        pqdm = (tpqdm if use_threads else ppqdm)
        results = pqdm(iter(_args), func, n_jobs=max_workers, argument_type='args')
      except ImportError as e:
        from ..misc import _warn_deps
        _warn_deps(e, 'progress bar')
        progress_bar = False

    if not progress_bar:
      WorkerPool = (ThreadGeneratorLazy if use_threads else ProcessGeneratorLazy)

      results = []
      for r in WorkerPool(func, iter(_args), max_workers=max_workers, chunk=max_workers*2):
        results.append(r)

    return results

  def process_all(self,
    func:Callable,
    *args:any,
    max_workers:int = CPU_COUNT,
    use_threads:bool = True,
    progress_bar:bool = False
  ):
    """
    Process in parallel all of tile using the specified function args.

    :param func: A function with at least the arguments ``idx, tile, window``.
    :param args: Additional arguments to send to the function.
    :param max_workers: Number of CPU cores to use in the parallelization.
      By default all cores are used.
    :param use_threads: If ``True`` the parallel processing uses ``ThreadGeneratorLazy``,
      otherwise it uses ProcessGeneratorLazy.
    :param progress_bar: If ``True`` the parallel processing uses ``pqdm`` [1] presenting
      a progress bar, ignoring the ``use_threads``.

    Examples
    ========

    >>> from eumap.parallel import TilingProcessing
    >>> from eumap.raster import read_rasters
    >>>
    >>> def run(idx, tile, window, msg):
    >>>     print(f'Tile {idx} => {msg}')
    >>>
    >>> tiling= TilingProcessing(verbose=True)
    >>> msg = "Let's crunch some data."
    >>> result = tiling.process_all(run)

    References
    ==========

    [1] `Parallel TQDM <https://pqdm.readthedocs.io/en/latest/readme.html>`_
    """

    idx_list = range(0, self.num_tiles)
    return self.process_multiple(idx_list, func, *args, max_workers=max_workers, \
      use_threads=use_threads, progress_bar=progress_bar)
