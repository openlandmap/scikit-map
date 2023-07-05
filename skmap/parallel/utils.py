"""
Parallelization helpers based in thread/process pools and joblib
"""
import numpy
import multiprocessing
from typing import Callable, Iterator, List,  Union
from concurrent.futures import as_completed, wait, FIRST_COMPLETED, ProcessPoolExecutor

import warnings
from pathlib import Path
import geopandas as gpd
import numpy as np
import multiprocessing
from osgeo import osr
import math
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from rasterio.windows import Window, from_bounds
import os.path
import psutil
import time
import gc

from ..misc import ttprint

CPU_COUNT = multiprocessing.cpu_count()
"""
Number of CPU cores available.
"""

def _mem_usage():
  mem = psutil.virtual_memory()
  return (mem.used / mem.total)

def _run_task(i, task, mem_usage_limit, mem_check_interval, mem_check, verbose, *args):
  while (_mem_usage() > mem_usage_limit and mem_check):
    if verbose:
      ttprint(f'Memory usage in {_mem_usage():.2f}%, stopping workers for {task.__name__}')
    time.sleep(mem_check_interval)
    gc.collect()
  
  return task(*args)

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

  >>> from skmap.parallel import ThreadGeneratorLazy
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

  >>> from skmap.parallel import ProcessGeneratorLazy
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

  >>> from skmap import parallel
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
  axis:int,
  arr:numpy.array,
  n_jobs:int = CPU_COUNT,
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
  :param n_jobs: Number of parallel jobs to run the worker function
  :param args: Additional arguments to the worker.
  :param kwargs: Additional named arguments to the worker.
  :returns: The output array with one dimension less than the input array.
  :rtype: numpy.array

  Examples
  ========

  >>> from skmap import parallel
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
            for sub_arr in np.array_split(arr, n_jobs)]

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
  :param verbose: Use ``True`` to print information about read tiles
    and the base raster.

  """

  def __init__(self,
    tiling_system_fn = 'http://s3.eu-central-1.wasabisys.com/skmap/tiling_system_30km.gpkg',
    base_raster_fn = 'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif',
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

    >>> from skmap.parallel import TilingProcessing
    >>> from skmap.raster import read_rasters
    >>>
    >>> def run(idx, tile, window, raster_files):
    >>>     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
    >>>     print(f'Tile {idx}: data read {data.shape}')
    >>>
    >>> raster_files = [
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
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

    >>> from skmap.parallel import TilingProcessing
    >>> from skmap.raster import read_rasters
    >>>
    >>> def run(idx, tile, window, raster_files):
    >>>     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
    >>>     print(f'Tile {idx}: data read {data.shape}')
    >>>
    >>> raster_files = [
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
    >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
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

    >>> from skmap.parallel import TilingProcessing
    >>> from skmap.raster import read_rasters
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

  @staticmethod
  def generate_tiles(
    tile_size:int, 
    extent:tuple, 
    crs:str,
    raster_layer_fn:str = None,
  ):
    """
    Generate a custom tiling system based on a regular grid.

    :param tile_size: Single value used to define the width and height of a
      individual tile. It assumes the same unit of ``crs`` (degree for geographic coordinate
      systems and meter for projected coordinate systems). Tiles outside of the image
      are clipped to fit in the informed extent.
    :param extent: Extent definition considering ``minx, miny, maxx, maxy`` according 
      to the ``crs`` argument.
    :param crs: Coordinate reference system for the tile geometries.
      Can be anything accepted by pyproj.CRS.from_user_input(), 
      such as an authority string (EPSG:4326) or a WKT/proj4 string.
    :param raster_layer_fn: If provided, for each tile the ``min``, ``max`` and ``mode`` 
      values are calculated considering the raster pixels inside the tile. It assumes the
      same ``crs`` for the raster layer and tiles.

    :returns: Tiling system with follow columns:
      ``tile_id``, ``minx``, ``miny``, ``maxx``, ``maxy`` and ``geometry``. The additional 
      columns ``raster_min``, ``raster_mode_value``, ``raster_mode_count`` and ``raster_max``
      are returned when a raster layer is provided.
    :rtype: geopandas.GeoDataFrame

    Examples
    ========

    >>> skmap_extent = (900000, 930010, 6540000, 5460010)
    >>> tiling_system = TilingProcessing.generate_tiles(30000, skmap_extent, 'epsg:3035')
    >>> tiling_system.to_file(tiling_system_fn,  driver="GPKG")

    """
    
    minx, miny, maxx, maxy = extent
    
    data = {'tile_id': [], 'minx':[], 'miny':[], 'maxx':[], 'maxy':[], 'geometry':[]}
    tile_id = 0
    
    for x1 in np.arange(minx, maxx, tile_size):
      for y1 in np.arange(miny, maxy, tile_size):
        
        x2 = x1+tile_size
        if x2 > maxx:
          x2 = maxx

        y2 = y1+tile_size
        if y2 > maxy:
          y2 = maxy

        data['tile_id'].append(tile_id)
        data['minx'].append(x1)
        data['miny'].append(y1)
        data['maxx'].append(x2)
        data['maxy'].append(y2)
        data['geometry'].append(Polygon([
            (x1,y1), (x2,y1), 
            (x2,y2), (x1,y2)
        ]))
        
        tile_id += 1

    tiles = gpd.GeoDataFrame(data).set_crs(crs, inplace=True)


    if raster_layer_fn is not None:

      def _raster_values(tile, raster_layer_fn):
        shapes = [ tile['geometry'] ]
        
        try:
          with rasterio.open(raster_layer_fn) as src:
            out_image, out_transform = mask(src, shapes, crop=True, filled=True)
            
            out_image = out_image.astype('float32')
            nodata_val = src.nodatavals[0]
            
            _values, _counts = np.unique(out_image, return_counts=True)
            values, counts = [], []
            for v,c in zip(_values, _counts):
                if (v != nodata_val):
                    values.append(v)
                    counts.append(c)
            
            values = np.array(values)
            counts = np.array(counts)
            m = np.argmax(counts)
            
            tile['raster_min'] = np.min(values)
            tile['raster_mode_value'] = values[m]
            tile['raster_mode_count'] = counts[m]
            tile['raster_max'] = np.max(values)
        except:
          tile['raster_min'] = None
          tile['raster_mode_value'] = None
          tile['raster_mode_count'] = None
          tile['raster_max'] = None
          
        return tile

      args = [ (tiles.loc[i,:], raster_layer_fn) for i in range(0, len(tiles)) ]
      
      result = []
      for t in job(_raster_values, args):
        result.append(t)

      tiles = gpd.GeoDataFrame(result).set_crs(crs, inplace=True)

    return tiles

class TaskSequencer():
  """
  Execute a pipeline of sequential tasks, in a way that the output of 
  one task is used as input for the next task. For each task,
  a pool of workers is created, allowing the execution of all the 
  available workers in parallel, for different portions of the input data

  :param tasks: Task definition list, where each element can be: (1) a ``Callable`` function;
    (2) a tuple containing a ``Callable`` function and the number of workers for the task; or
    (3) a tuple containing a ``Callable`` function, the number of workers and an ``bool``
    indication if the task would respect the ``mem_usage_limit``. The default number of 
    workers is ``1``.
  :param mem_usage_limit: Percentage of memory usage that when reached triggers a momentarily stop 
    of execution for specific tasks. For example, if the ``task_1`` is responsible for reading
    the data and ``task_2`` for processing it, the ``task_1`` definition can receive an 
    ``bool`` indication to respect the ``mem_usage_limit``, allowing the ``task_2`` to process
    the data that has already been read and releasing memory for the next ``task_1`` reads.
  :param wait_timeout: Timeout argument used by ``concurrent.futures.wait``.
  :param verbose: Use ``True`` to print the communication and status of the tasks
  
  Examples
  ========
  
  >>> from skmap.parallel import TaskSequencer
  >>> 
  >>> output = TaskSequencer(
  >>> tasks=[ 
  >>>   task_1, 
  >>>   (task_2, 2)
  >>> ]
  
  Pipeline produced by this example code:

  >>>                ----------      ----------
  >>> input_data ->  | task_1 |  ->  | task_2 |  ->  output_data
  >>>                 ----------      ----------
  >>>                 |              |
  >>>                 |-worker_1     |-worker_1
  >>>                                |-worker_2

  """

  def __init__(self, 
    tasks:Union[List[Callable], List[tuple]],
    mem_usage_limit:float = 0.75,
    wait_timeout:int = 5,
    verbose:bool = False
  ):
  
    self.wait_timeout = wait_timeout
    self.mem_usage_limit = mem_usage_limit
    self.verbose = verbose
    self.mem_check_interval = 10

    self.tasks = []
    self.pipeline = []
    self.mem_checks = []

    for task in tasks:

      pool_size = 1
      mem_check = False

      if type(task) is tuple:
        if len(task) == 2:
          task, pool_size = task
        else:
          task, pool_size, mem_check = task

      self._verbose(f'Starting {pool_size} worker(s) for {task.__name__} (mem_check={mem_check})')

      self.tasks.append(task)
      self.pipeline.append(ProcessPoolExecutor(max_workers = pool_size))
      self.mem_checks.append(mem_check)
        
    self.n_tasks = len(self.tasks)
    self.pipeline_futures = [ set() for i in range(0, self.n_tasks) ]

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)

  def run(self, input_data:List[tuple]):
    """
    Run the task pipeline considering the ``input_data`` argument.

    :param input_data: Input data used to feed the first task.

    :returns: List of returned values produced by the last task and 
      with the same size of the ``input_data`` argument.
    :rtype: List

    Examples
    ========

    >>> from skmap.misc import ttprint
    >>> from skmap.parallel import TaskSequencer
    >>> import time
    >>> 
    >>> def rnd_data(const, size):
    >>>     data = np.random.rand(size, size, size)
    >>>     time.sleep(2)
    >>>     return (const, data)
    >>> 
    >>> def max_value(const, data):
    >>>     ttprint(f'Calculating the max value over {data.shape}')
    >>>     time.sleep(8)
    >>>     result = np.max(data + const)
    >>>     return result
    >>> 
    >>> taskSeq = TaskSequencer(
    >>> tasks=[ 
    >>>      rnd_data, 
    >>>      (max_value, 2)
    >>>  ],
    >>>  verbose=True
    >>> )
    >>> 
    >>> taskSeq.run(input_data=[ (const, 10) for const in range(0,3) ])
    >>> taskSeq.run(input_data=[ (const, 20) for const in range(3,6) ])

    """

    for i_dta in input_data:
      self._verbose(f'Submission to {self.tasks[0].__name__}')
    
      self.pipeline_futures[0].add(
        self.pipeline[0].submit(_run_task, *(
            (0, self.tasks[0], self.mem_usage_limit, 
             self.mem_check_interval, self.mem_checks[0], self.verbose,
             *i_dta))
        )
      )
    
    keep_going = True
    
    while keep_going:
      keep_going_aux = []
      for i in range(0, self.n_tasks - 1):
        
        done, self.pipeline_futures[i] = wait(self.pipeline_futures[i], return_when=FIRST_COMPLETED, timeout=self.wait_timeout)
        
        if len(done) > 0:
          self._verbose(f'{self.tasks[i].__name__} state: done={len(done)} waiting={len(self.pipeline_futures[i])}')

        for f in done:
          nex_i = i+1
          self._verbose(f'Submission to {self.tasks[nex_i].__name__}')
          self.pipeline_futures[nex_i].add(
            self.pipeline[nex_i].submit( _run_task, *(
                 nex_i, self.tasks[nex_i], self.mem_usage_limit, 
                 self.mem_check_interval, self.mem_checks[nex_i], self.verbose,
                 *f.result()))
          )

        keep_going_aux.append(len(self.pipeline_futures[i]) > 0) 
      
      keep_going = any(keep_going_aux)

    self._verbose(f'Waiting {self.tasks[-1].__name__}')
    result = [ future.result() for future in as_completed(self.pipeline_futures[-1]) ]
    
    return result