'''
Parallel block-wise processing and result aggregation for large raster datasets
'''

try:
    import pygeos as pg
    import rasterio as rio
    import rasterio.features as rfeatures
    import numpy as np
    import sys
    from shapely import speedups
    if speedups.available:
        speedups.enable()
    import shapely.geometry as g
    from multiprocessing.pool import ThreadPool
    import multiprocessing as mp
    import threading
    from datetime import datetime

    from typing import Union, Tuple, Iterable, Callable, Iterator

    def _read_block(
        src: Union[rio.DatasetReader, Iterable[rio.DatasetReader]],
        window: rio.windows.Window,
        geometry: Union[dict, None]=None,
        band: int=1,
    ):
        if isinstance(src, rio.DatasetReader):
            src = [src]
        mask = src[0].read_masks(band, window=window).astype(bool)
        for s in src[1:]:
            mask = mask & s.read_masks(band, window=window).astype(bool)
        if not mask.any():
            return None
        if geometry is not None:
            gmask = rfeatures.rasterize(
                [geometry],
                out_shape=(window.height, window.width),
                transform=src[0].window_transform(window),
                fill=0,
            ).astype(bool)
            if not gmask.any():
                return None
            mask = mask & gmask
        if not mask.any():
            return None
        data = np.stack([
            s.read(band, window=window)
            for s in src
        ])
        return data[:, mask], mask, window

    def _id(x):
        return x

    class _RasterBlockFunction:

        def __init__(self,
            func: Callable,
            # agg_func: Callable=_id, # should not be done here
            return_data_only: bool=False,
        ):
            self.func = func
            # self.agg_func = agg_func # should not be done here
            self.return_data_only = return_data_only

        def __call__(self, args):
            data, mask, window = args
            result = self.func(*data)
            # result = self.agg_func(result) # should not be done here
            if self.return_data_only:
                return result
            return result, mask, window

    class RasterBlockReader:
        """
        Thread-parallel reader for large rasters.

        If ``reference_file`` is not ``None``, builds an R-tree index [1] of the block geometries read from the ``reference_file`` on initialization. All rasters read with the initialized reader are assumed to have identical geotransforms and block structures to the reference.

        :param reference_file: Path (URL) of the reference raster.

        For full usage examples please refer to the block processing tutorial notebook [2].

        Examples
        ========

        >>> from skmap.parallel.blocks import RasterBlockReader
        >>> from skmap.misc import ttprint
        >>>
        >>> fp = 'https://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_landcover.hcl_lucas.corine.rf_p_30m_0..0cm_2019_skmap_epsg3035_v0.1.tif'
        >>>
        >>> ttprint('initializing reader')
        >>> reader = RasterBlockReader(fp)
        >>> ttprint('reader initialized')

        References
        ==========

        [1] `pygeos STRTree <https://pygeos.readthedocs.io/en/latest/strtree.html>`_

        [2] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

        """

        def __init__(self,
            reference_file: str=None,
        ):
            self.reference = None
            if reference_file is not None:
                self._build_rtree(reference_file)

        def _open(self,
            src_path,
        ):
            src = rio.open(src_path)
            if all((
                src.crs == self.reference.crs,
                src.transform == self.reference.transform,
                src.width == self.reference.width,
                src.height == self.reference.height,
            )):
                return src

            return rio.vrt.WarpedVRT(
                src,
                resampling=rio.enums.Resampling.nearest,
                crs=self.reference.crs,
                transform=self.reference.transform,
                width=self.reference.width,
                height=self.reference.height,
            )

        def _build_rtree(self, reference_file):
            self.reference = rio.open(reference_file)
            self.block_windows = np.array([
                tup[1]
                for tup in self.reference.block_windows()
            ])
            boxes = self._blocks2boxes(self.block_windows)
            self.rtree = pg.strtree.STRtree(boxes)
            __ = self.rtree.query(boxes[0])

        def _get_block_indices(self,
            geometry: dict,
        ):
            _geom = pg.from_shapely(g.shape(geometry))
            return self.rtree.query(_geom)

        def _blocks2boxes(self, block_windows):
            coords = np.array([
                self.reference.window_bounds(bw)
                for bw in block_windows
            ])
            return pg.box(*coords.T)

        def read_overlay(self,
            src_path: Union[str, Iterable[str]],
            geometry: dict,
            band: int=1,
            geometry_mask: bool=True,
            max_workers: int=mp.cpu_count(),
            optimize_threadcount: bool=True,
        ) -> Iterator[Tuple[np.ndarray, np.ndarray, rio.windows.Window]]:
            """
            Thread-parallel reading of large rasters within a bounding geometry.

            Only blocks that intersect with ``geometry`` are read. Returns a generator yielding ``(data, mask, window)`` tuples for each block, where ``data`` are the stacked pixel values of all rasters at ``mask==True``,
            ``mask`` is the reduced (via bitwise ``and``) block data mask for all rasters, and ``window`` is the ``rasterio.windows.Window`` [1] for the block within the transform of the ``reference_file``.
            All rasters read with the initialized reader are assumed to have identical geotransforms and block structures to the ``reference_file`` used for initialization.
            If the reader was initialized with ``reference_file==None``, the first file in ``src_path`` is used as the reference and the block R-tree is built before yielding data from the first block.

            :param src_path:             Path(s) (or URLs) of the raster file(s) to read.
            :param geometry:             The bounding geometry within which to read raster blocks, given as a dictionary (with the GeoJSON geometry schema).
            :param band:                 Index of band to read from all rasters.
            :param geometry_mask:        Indicates wheather or not to use the geometry as a data mask. If ``False``, the block data will be returned in its entirety, regardless if some of it falls outside of the ``geometry``.
            :param max_workers:          Maximum number of worker threads to use, defaults to ``multiprocessing.cpu_count()``.
            :param optimize_threadcount: Wheather or not to optimize number of workers. If ``True``, the number of worker threads will be iteratively increased until the average read time per block stops decreasing or ``max_workers`` is reached. If ``False``, ``max_workers`` will be used as the number of threads.

            :returns: Generator yielding ``(data, mask, window)`` tuples for each block.
            :rtype: Iterator[Tuple(np.ndarray, np.ndarray, rasterio.windows.Window)]

            For full usage examples please refer to the block processing tutorial notebook [2].

            Examples
            ========

            >>> geom = {
            >>>     'type': 'Polygon',
            >>>     'coordinates': [[
            >>>         [4765389, 2441103],
            >>>         [4764441, 2439352],
            >>>         [4767369, 2438696],
            >>>         [4761659, 2441949],
            >>>         [4765389, 2441103],
            >>>     ]],
            >>> }
            >>> block_data_gen = reader.read_overlay(fp)
            >>> data, mask, window = next(block_data_gen)

            References
            ==========

            [1] `Rasterio Window <https://rasterio.readthedocs.io/en/latest/api/rasterio.windows.html>`_

            [2] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

            """

            if isinstance(src_path, str):
                src_path = [src_path]

            if self.reference is None:
                self._build_rtree(src_path[0])

            block_idx = self._get_block_indices(geometry)

            sources = {}

            def _read_worker(window: rio.windows.Window):
                import threading
                tname = threading.current_thread().name
                if tname not in sources:
                    sources[tname] = [
                        self._open(sp)
                        for sp in src_path
                    ]
                return _read_block(
                    sources[tname],
                    window,
                    (geometry if geometry_mask else None),
                )

            n_workers = max_workers

            try:
                if optimize_threadcount:
                    n_workers = 2
                    t_block_best = np.inf

                    first_results = []

                    while block_idx.size > 0 and n_workers <= max_workers:
                        batch_idx, block_idx = block_idx[:n_workers], block_idx[n_workers:]
                        with ThreadPool(n_workers) as pool:
                            t_start = datetime.now()
                            first_results += pool.map(
                                _read_worker,
                                self.block_windows[batch_idx],
                            )
                            dt = datetime.now() - t_start
                        dt = (dt.seconds + dt.microseconds * 1e-6) / batch_idx.size
                        if dt >= t_block_best:
                            n_workers -= 1
                            break
                        t_block_best = dt
                        n_workers += 1

                    print(f'reader using {n_workers} threads')

                    for block_data in first_results:
                        if block_data is not None:
                            yield block_data

                with ThreadPool(n_workers) as pool:
                    for block_data in pool.imap_unordered(
                        _read_worker,
                        self.block_windows[block_idx],
                    ):
                        if block_data is not None:
                            yield block_data
            except GeneratorExit:
                pass

            with ThreadPool(n_workers) as pool:
                __ = pool.map(
                    lambda key: [s.close() for s in sources[key]],
                    sources,
                )
                sources = {}

    class RasterBlockAggregator:
        """
        Class for aggregating results of block wise raster processing into a single result.

        :param reader: RasterBlockReader instance to use for reading rasters.

        For full usage examples please refer to the block processing tutorial notebook [1].

        Examples
        ========

        >>> from skmap.parallel.blocks import RasterBlockReader, RasterBlockAggregator
        >>>
        >>> fp = 'https://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_landcover.hcl_lucas.corine.rf_p_30m_0..0cm_2019_skmap_epsg3035_v0.1.tif'
        >>>
        >>> reader = RasterBlockReader(fp)
        >>> aggregator = RasterBlockAggregator(reader)

        References
        ==========

        [1] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

        """

        def __init__(self,
            reader: RasterBlockReader=None,
        ):
            self.reader = reader

        def aggregate(self,
            src_path: Union[str, Iterable[str]],
            geometry: dict,
            block_func: Callable,
            agg_func: Callable=np.mean,
            **kwargs,
        ):
            """
            Aggregates results of block wise raster processing into a single result.

            :param src_path:             Path(s) (or URLs) of the raster file(s) to read. If aggregator is initialized with ``reader=None``, the first file in ``src_path`` will be used to initialize a new reader.
            :param geometry:             The bounding geometry within which to read raster blocks, given as a dictionary (with the GeoJSON geometry schema).
            :param block_func:           Callable to perform on the data for each block.
            :param agg_func:             Callable to produce an aggregation of block-wise results.
            :param **kwargs:             Additional keyword arguments passed to ``RasterBlockReader.read_overlay()``.

            :returns: The result of ``agg_func`` called with block-wise ``block_func`` results as the argument.

            For full usage examples please refer to the block processing tutorial notebook [1].

            Examples
            ========

            >>> geom = {
            >>>     'type': 'Polygon',
            >>>     'coordinates': [[
            >>>         [4765389, 2441103],
            >>>         [4764441, 2439352],
            >>>         [4767369, 2438696],
            >>>         [4761659, 2441949],
            >>>         [4765389, 2441103],
            >>>     ]],
            >>> }
            >>>
            >>> def urban_fabric_area(lc):
            >>>     return (lc==1) * 9e-4 # spatial resolution is 30x30 m
            >>>
            >>> result = agg.aggregate(
            >>>     fp, geom,
            >>>     block_func=urban_fabric_area,
            >>>     agg_func=np.sum,
            >>> )

            References
            ==========

            [1] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

            """

            if isinstance(src_path, str):
                src_path = [src_path]

            if self.reader is None:
                self.reader = RasterBlockReader(src_path[0])

            (*block_results,) = map(
                _RasterBlockFunction(
                    block_func,
                    return_data_only=True,
                    # agg_func=agg_func, # should not be done here
                ),
                self.reader.read_overlay(
                    src_path,
                    geometry,
                    **kwargs,
                ),
            )

            result = agg_func(block_results)
            return result

    class RasterBlockWriter:
        """
        Class for writing results of block wise raster processing results into a new raster file.

        :param reader: RasterBlockReader instance to use for reading rasters.

        For full usage examples please refer to the block processing tutorial notebook [1].

        Examples
        ========

        >>> from skmap.parallel.blocks import RasterBlockReader, RasterBlockWriter
        >>>
        >>> fp = 'https://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_landcover.hcl_lucas.corine.rf_p_30m_0..0cm_2019_skmap_epsg3035_v0.1.tif'
        >>>
        >>> reader = RasterBlockReader(fp)
        >>> writer = RasterBlockWriter(reader)

        References
        ==========

        [1] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

        """

        def __init__(self,
            reader: RasterBlockReader=None,
        ):
            self.reader = reader

        def write(self,
            src_path: Union[str, Iterable[str]],
            dst_path: str,
            geometry: dict,
            block_func: Callable=_id,
            geometry_mask: bool=True,
            reader_kwargs: dict={},
            **kwargs,
        ):
            """
            Writes block wise calculation results to new raster file.

            Performs ``block_func`` on all blocks of file(s) listed in ``src_path`` that intersect with ``geometry`` and writes the results to a new raster.

            :param src_path:             Path(s) (or URLs) of the raster file(s) to read. If aggregator is initialized with ``reader=None``, the first file in ``src_path`` will be used to initialize a new reader.
            :param dst_path:             Path to write the result raster to.
            :param geometry:             The bounding geometry within which to read raster blocks, given as a dictionary (with the GeoJSON geometry schema).
            :param block_func:           Callable to perform on the data for each block. Result must retain the shape of input data. Defaults to the identity function.
            :param geometry_mask:        Indicates wheather or not to use the geometry as a data mask. If ``False``, calculation will be performed on all of the block data, regardless if some of it falls outside of the ``geometry``.
            :param reader_kwargs:        Additional keyword arguments passed to ``RasterBlockReader.read_overlay()``.
            :param **kwargs:             Additional raster profile keyword arguments passed to the ``rasterio`` dataset writer [1].

            For full usage examples please refer to the block processing tutorial notebook [2].

            Examples
            ========

            >>> geom = {
            >>>     'type': 'Polygon',
            >>>     'coordinates': [[
            >>>         [4765389, 2441103],
            >>>         [4764441, 2439352],
            >>>         [4767369, 2438696],
            >>>         [4761659, 2441949],
            >>>         [4765389, 2441103],
            >>>     ]],
            >>> }
            >>>
            >>> def is_urban_fabric(lc):
            >>>     return lc == 1
            >>>
            >>> writer.write(fp, 'urban_fabric.tif', geom, is_urban_fabric, dtype='uint8', nodata=0)

            References
            ==========

            [1] `Writing datasets with Rasterio <https://rasterio.readthedocs.io/en/latest/quickstart.html#saving-raster-data>`_

            [2] `Raster block processing tutorial <../notebooks/06_raster_block_processing.html>`_

            """

            if isinstance(src_path, str):
                src_path = [src_path]

            if self.reader is None:
                self.reader = RasterBlockReader(src_path[0])

            profile = self.reader.reference.profile

            block_indices = self.reader._get_block_indices(geometry)
            out_window_geometry = self.reader._blocks2boxes(self.reader.block_windows[block_indices])
            out_window_geometry = pg.set_operations.coverage_union_all(out_window_geometry)
            out_bounds = pg.measurement.bounds(out_window_geometry)

            out_window = rio.windows.from_bounds(
                *out_bounds,
                transform=profile['transform'],
            )
            out_transform = self.reader.reference.window_transform(out_window)
            profile.update(
                transform=out_transform,
                width=out_window.width,
                height=out_window.height,
            )
            profile.update(**kwargs)

            with rio.open(
                dst_path, 'w',
                **profile,
            ) as dst:
                for data, mask, window in map(
                    _RasterBlockFunction(block_func, return_data_only=False),
                    self.reader.read_overlay(
                        src_path,
                        geometry,
                        geometry_mask=geometry_mask,
                        **reader_kwargs,
                    )
                ):
                    window_bounds = self.reader.reference.window_bounds(window)
                    window = rio.windows.from_bounds(*window_bounds, transform=dst.transform)
                    out = np.full_like(mask, profile['nodata'], dtype=profile['dtype'])
                    out[mask] = data.astype(profile['dtype'])
                    dst.write(out, 1, window=window)

except ImportError as e:
    from ..misc import _warn_deps
    _warn_deps(e, 'blocks')
