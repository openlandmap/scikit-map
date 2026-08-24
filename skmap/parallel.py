"""
Parallelization helpers based on Ray
"""

import multiprocessing
from pathlib import Path
from typing import Any, Callable, Iterator, List, Union

import geopandas as gpd
import numpy
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import Polygon

from .misc import ttprint

CPU_COUNT = multiprocessing.cpu_count()
"""
Number of CPU cores available.
"""


def _call_worker(worker, args):
    """Call ``worker(*args)`` (module-level so Ray can serialize it)."""

    return worker(*args)


class SharedArray:
    """A numpy array held in the Ray object store, accessed via an ObjectRef.

    ``shape`` and ``dtype`` are O(1) metadata (no materialization); ``get()``
    materializes the full array in process memory and should be used sparingly.
    """

    def __init__(self, ref, shape, dtype):
        self.ref = ref  # ray.ObjectRef
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self):
        return len(self.shape)

    def get(self):
        """Materialize the full array (use sparingly)."""
        import ray

        return ray.get(self.ref)

    def __repr__(self):
        return f"SharedArray(shape={self.shape}, dtype={self.dtype})"


def put_shared(array):
    """Place ``array`` in the Ray object store and return a ``SharedArray``."""
    import ray

    ray.init(ignore_reinit_error=True)
    arr = np.ascontiguousarray(array)
    return SharedArray(ray.put(arr), arr.shape, arr.dtype)


def get_shared(sa_or_ref):
    """Return the numpy array backing a ``SharedArray`` or raw ``ObjectRef``."""
    import ray

    ref = sa_or_ref.ref if isinstance(sa_or_ref, SharedArray) else sa_or_ref
    return ray.get(ref)


def _remote(fn, *args):
    """Invoke a module-level worker remotely and return its ObjectRef."""
    import ray

    ray.init(ignore_reinit_error=True)
    return ray.remote(fn).remote(*args)


def _stack_bands(band_refs, shape):
    """Stack per-band arrays into a single ``(N, H*W)`` array (read assembly)."""
    import ray

    bands = [ray.get(r) for r in band_refs]
    return np.stack(bands, axis=0).reshape(shape)


def _assemble(refs, new_shape, out_idx_list, n_in):
    """Build a new array: copy the first ``n_in`` input bands, then write the
    returned output slices at their absolute band indices.

    ``refs`` is ``[ref_in, *slice_refs]`` (a single list so Ray passes the
    ObjectRefs by reference rather than dereferencing top-level args).
    """
    import ray

    arr_in = ray.get(refs[0])
    out = np.empty(new_shape, dtype=arr_in.dtype)
    out[:n_in] = arr_in
    for idx, sref in zip(out_idx_list, refs[1:]):
        out[idx] = ray.get(sref)
    return out


def _select_bands(refs, keep_idx, shape):
    """Return a new array with only the bands in ``keep_idx`` (drop)."""
    import ray

    return ray.get(refs[0])[keep_idx, :]


def _concat(refs, shapes):
    """Concatenate several arrays along the band axis (group-run concat)."""
    import ray

    return np.concatenate([ray.get(r) for r in refs], axis=0)


def job(
    worker: Callable,
    worker_args: Iterator[tuple],
    n_jobs: int = -1,
    **kwargs,
):
    """
    Execute a function in parallel using Ray [1].

    :param worker: Function to execute in parallel.
    :param worker_args: Argument iterator where each element is send
      to separate job.
    :param n_jobs: Number of parallel jobs to run the worker function.
      By default all cores are used.
    :returns: A generator with the return of all workers, in submission order.
    :rtype: Generator

    Examples
    ========

    >>> from skmap import parallel
    >>>
    >>> def worker(i, msg):
    ...   print(f'{i}: {msg}')
    ...   return f'Worker {i} finished'
    >>>
    >>> msg = ("I'm running in parallel", )
    >>> args = iter([ (i,msg) for i in range(0,5)])
    >>>
    >>> for result in parallel.job(worker, args): # doctest: +SKIP
    ...   print(result)

    References
    ==========

    [1] `Ray Core <https://docs.ray.io/en/latest/ray-core/walkthrough.html>`_

    """
    import warnings

    import ray

    if "joblib_args" in kwargs:
        warnings.warn(
            "joblib_args is deprecated and ignored; Ray is the only backend",
            DeprecationWarning,
            stacklevel=2,
        )

    ray.init(ignore_reinit_error=True)

    # Map n_jobs to a per-task CPU allocation so that at most n_jobs tasks
    # run concurrently (n_jobs=-1 uses all available CPUs).
    total_cpus = int(ray.cluster_resources().get("CPU", CPU_COUNT))
    if n_jobs <= 0 or n_jobs >= total_cpus:
        num_cpus = 1
    else:
        num_cpus = max(1, total_cpus // n_jobs)

    remote_call = ray.remote(num_cpus=num_cpus)(_call_worker)
    refs = [remote_call.remote(worker, args) for args in worker_args]

    # ray.get preserves submission order
    for result in ray.get(refs):
        yield result


def apply_along_axis(
    worker: Callable,
    axis: int,
    arr: numpy.array,
    n_jobs: int = CPU_COUNT,
    *args: any,
    **kwargs: any,
):
    """
    Execute a function through a ``numpy.array`` axis in parallel [1].
    It uses Ray, so avoid to send shared memory objects as arguments.

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

    >>> import multiprocessing
    >>> import numpy as np
    >>> from skmap import parallel
    >>>
    >>> def fn(arr, const):
    ...   return np.sum(arr) + const
    >>>
    >>> const = 1
    >>> arr = np.ones((100,100,100))
    >>>
    >>> out = parallel.apply_along_axis(fn, 0, arr, 4, const) # doctest: +SKIP
    >>> print(arr.shape, out.shape) # doctest: +SKIP

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
    chunks = [
        (worker, effective_axis, sub_arr, args, kwargs)
        for sub_arr in np.array_split(arr, n_jobs)
    ]

    result = []
    for r in job(run, chunks):
        result.append(r)

    result = np.concatenate(result)
    if effective_axis != axis:
        # Undo the swap so the reduced axis returns to its original position
        # (numpy.apply_along_axis semantics: axis 0 -> (out, ...)).
        result = result.swapaxes(axis, effective_axis)
    return result


class TilingProcessing:
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

    def __init__(
        self,
        tiling_system_fn="http://s3.eu-central-1.wasabisys.com/skmap/tiling_system_30km.gpkg",
        base_raster_fn="http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif",
        verbose: bool = False,
        epsg_checking: bool = True,
    ) -> None:
        from pyproj import CRS

        self.tiles = gpd.read_file(tiling_system_fn)
        self.num_tiles = self.tiles.shape[0]
        with rasterio.open(base_raster_fn) as base_raster:
            self.base_transform = base_raster.transform
            base_crs = base_raster.crs

        tile_epsg = CRS(self.tiles.crs.to_wkt()).to_epsg()
        raster_epsg = CRS(base_crs.to_wkt()).to_epsg()

        if epsg_checking and tile_epsg != raster_epsg:
            raise Exception(
                "Different SpatialReference"
                + f"\n tiling_system_fn:\n{self.tiles.crs.to_wkt()}"
                + f"\n base_raster_fn:\n{base_crs.to_wkt()}"
            )

        if verbose:
            pixel_size = self.base_transform[0]
            ttprint(f"Pixel size equal {pixel_size} in {Path(base_raster_fn).name}")
            ttprint(
                f"{self.num_tiles} tiles available in {Path(tiling_system_fn).name}"
            )
            ttprint(f"Using EPSG:{raster_epsg}")

    def _tile_window(self, idx):
        tile = self.tiles.iloc[idx]
        left, bottom, right, top = tile.geometry.bounds

        return tile, from_bounds(left, bottom, right, top, self.base_transform)

    def process_one(self, idx: int, func: Callable, *args: any):
        """
        Process a single tile using the specified function args.

        :param idx: The tile id to process. This idx is generated for all the tiles
          in a sequence starting from ``0``.
        :param func: A function with at least the arguments ``idx, tile, window``.
        :param args: Additional arguments to send to the function.

        Examples
        ========

        >>> from skmap.parallel import TilingProcessing
        >>> from skmap.io.base import read_rasters
        >>>
        >>> def run(idx, tile, window, raster_files):
        ...     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
        ...     print(f'Tile {idx}: data read {data.shape}')
        >>>
        >>> raster_files = [
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
        ... ]
        >>>
        >>> tiling= TilingProcessing(verbose=True) # doctest: +SKIP
        >>> tiling.process_one(0, run, raster_files) # doctest: +SKIP

        """
        tile, window = self._tile_window(idx)
        return func(idx, tile, window, *args)

    def process_multiple(
        self,
        idx_list: List[int],
        func: Callable,
        *args: any,
        max_workers: int = CPU_COUNT,
        use_threads: bool = True,
        progress_bar: bool = False,
    ):
        """
        Process in parallel a list of tile using the specified function args.

        :param idx: The tile ids to process. This idx is generated for all the tiles
          in a sequence starting from ``0``.
        :param func: A function with at least the arguments ``idx, tile, window``.
        :param args: Additional arguments to send to the function.
        :param max_workers: Number of CPU cores to use in the parallelization.
          By default all cores are used.
        :param use_threads: Deprecated and ignored (Ray always uses processes).
        :param progress_bar: Deprecated and ignored.

        Examples
        ========

        >>> from skmap.parallel import TilingProcessing
        >>> from skmap.io.base import read_rasters
        >>>
        >>> def run(idx, tile, window, raster_files):
        ...     data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
        ...     print(f'Tile {idx}: data read {data.shape}')
        >>>
        >>> raster_files = [
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
        ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
        ... ]
        >>>
        >>> tiling= TilingProcessing(verbose=True) # doctest: +SKIP
        >>> idx_list = [0,10,100]
        >>> result = tiling.process_multiple(idx_list, run, raster_files) # doctest: +SKIP

        """

        _args = []

        for idx in idx_list:
            tile, window = self._tile_window(idx)
            _args.append((idx, tile, window, *args))

        return list(job(func, iter(_args), n_jobs=max_workers))

    def process_all(
        self,
        func: Callable,
        *args: any,
        max_workers: int = CPU_COUNT,
        use_threads: bool = True,
        progress_bar: bool = False,
    ):
        """
        Process in parallel all of tile using the specified function args.

        :param func: A function with at least the arguments ``idx, tile, window``.
        :param args: Additional arguments to send to the function.
        :param max_workers: Number of CPU cores to use in the parallelization.
          By default all cores are used.
        :param use_threads: Deprecated and ignored (Ray always uses processes).
        :param progress_bar: Deprecated and ignored.

        Examples
        ========

        >>> from skmap.parallel import TilingProcessing
        >>> from skmap.io.base import read_rasters
        >>>
        >>> def run(idx, tile, window, msg):
        ...     print(f'Tile {idx} => {msg}')
        >>>
        >>> tiling= TilingProcessing(verbose=True) # doctest: +SKIP
        >>> msg = "Let's crunch some data."
        >>> result = tiling.process_all(run) # doctest: +SKIP

        """

        idx_list = range(0, self.num_tiles)
        return self.process_multiple(
            idx_list,
            func,
            *args,
            max_workers=max_workers,
            use_threads=use_threads,
            progress_bar=progress_bar,
        )

    @staticmethod
    def generate_tiles(
        tile_size: int,
        extent: tuple,
        crs: str,
        raster_layer_fn: str = None,
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

        >>> from skmap.parallel import TilingProcessing
        >>> skmap_extent = (900000, 930010, 6540000, 5460010)
        >>> tiling_system = TilingProcessing.generate_tiles(30000, skmap_extent, 'epsg:3035')
        >>> tiling_system.to_file(tiling_system_fn,  driver="GPKG") # doctest: +SKIP

        """

        minx, miny, maxx, maxy = extent

        data = {
            "tile_id": [],
            "minx": [],
            "miny": [],
            "maxx": [],
            "maxy": [],
            "geometry": [],
        }
        tile_id = 0

        for x1 in np.arange(minx, maxx, tile_size):
            for y1 in np.arange(miny, maxy, tile_size):
                x2 = x1 + tile_size
                if x2 > maxx:
                    x2 = maxx

                y2 = y1 + tile_size
                if y2 > maxy:
                    y2 = maxy

                data["tile_id"].append(tile_id)
                data["minx"].append(x1)
                data["miny"].append(y1)
                data["maxx"].append(x2)
                data["maxy"].append(y2)
                data["geometry"].append(
                    Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                )

                tile_id += 1

        tiles = gpd.GeoDataFrame(data).set_crs(crs, inplace=True)

        if raster_layer_fn is not None:

            def _raster_values(tile, raster_layer_fn):
                shapes = [tile["geometry"]]

                try:
                    with rasterio.open(raster_layer_fn) as src:
                        out_image, out_transform = mask(
                            src, shapes, crop=True, filled=True
                        )

                        out_image = out_image.astype("float32")
                        nodata_val = src.nodatavals[0]

                        _values, _counts = np.unique(out_image, return_counts=True)
                        values, counts = [], []
                        for v, c in zip(_values, _counts):
                            if v != nodata_val:
                                values.append(v)
                                counts.append(c)

                        values = np.array(values)
                        counts = np.array(counts)
                        m = np.argmax(counts)

                        tile["raster_min"] = np.min(values)
                        tile["raster_mode_value"] = values[m]
                        tile["raster_mode_count"] = counts[m]
                        tile["raster_max"] = np.max(values)
                except:
                    tile["raster_min"] = None
                    tile["raster_mode_value"] = None
                    tile["raster_mode_count"] = None
                    tile["raster_max"] = None

                return tile

            args = [(tiles.loc[i, :], raster_layer_fn) for i in range(0, len(tiles))]

            result = []
            for t in job(_raster_values, args):
                result.append(t)

            tiles = gpd.GeoDataFrame(result).set_crs(crs, inplace=True)

        return tiles


class TaskSequencer:
    """
    Execute a pipeline of sequential tasks, in a way that the output of
    one task is used as input for the next task. Each task is run in
    parallel over the input data using ``skmap.parallel.job`` (Ray).

    :param tasks: Task definition list, where each element can be: (1) a ``Callable`` function;
      (2) a tuple containing a ``Callable`` function and the number of workers for the task; or
      (3) a tuple containing a ``Callable`` function, the number of workers and an ``bool``
      indication if the task would respect the ``mem_usage_limit``. The default number of
      workers is ``1``.
    :param mem_usage_limit: Deprecated and ignored (no backpressure in the Ray model).
    :param wait_timeout: Deprecated and ignored.
    :param verbose: Use ``True`` to print the communication and status of the tasks

    Examples
    ========

    .. code-block:: python

       from skmap.parallel import TaskSequencer

       output = TaskSequencer(
           tasks=[
             task_1,
             (task_2, 2)
           ]
       )

    Pipeline produced by this example code::

                       ----------      ----------
        input_data ->  | task_1 |  ->  | task_2 |  ->  output_data
                        ----------      ----------
                        |              |
                        |-worker_1     |-worker_1
                                       |-worker_2

    """

    def __init__(
        self,
        tasks: Union[List[Callable], List[tuple]],
        mem_usage_limit: float = 0.75,
        wait_timeout: int = 5,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        self.tasks = []
        self.pool_sizes = []

        for task in tasks:
            pool_size = 1

            if type(task) is tuple:
                if len(task) == 2:
                    task, pool_size = task
                else:
                    task, pool_size, _ = task

            self._verbose(f"Starting {pool_size} worker(s) for {task.__name__}")

            self.tasks.append(task)
            self.pool_sizes.append(pool_size)

        self.n_tasks = len(self.tasks)

    def _verbose(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            ttprint(*args, **kwargs)

    def run(self, input_data: List[tuple]):
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
        ...     data = np.random.rand(size, size, size)
        ...     time.sleep(2)
        ...     return (const, data)
        >>>
        >>> def max_value(const, data):
        ...     ttprint(f'Calculating the max value over {data.shape}')
        ...     time.sleep(8)
        ...     result = np.max(data + const)
        ...     return result
        >>>
        >>> taskSeq = TaskSequencer(
        ...     tasks=[
        ...         rnd_data,
        ...         (max_value, 2)
        ...     ],
        ...     verbose=True
        ... )
        [...] Starting 1 worker(s) for rnd_data
        [...] Starting 2 worker(s) for max_value
        >>>
        >>> taskSeq.run(input_data=[ (const, 10) for const in range(0,3) ]) # doctest: +SKIP
        >>> taskSeq.run(input_data=[ (const, 20) for const in range(3,6) ]) # doctest: +SKIP


        """

        data = list(input_data)
        for task, pool_size in zip(self.tasks, self.pool_sizes):
            self._verbose(f"Running {task.__name__} with {pool_size} worker(s)")
            data = list(job(task, iter(data), n_jobs=pool_size))

        return data
