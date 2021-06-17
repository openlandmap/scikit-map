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

from typing import Union, Tuple, Iterable, Callable

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

class RasterBlockFunction:

    def __init__(self,
        func: Callable,
        agg_func: Callable=_id,
        return_data_only: bool=False,
    ):
        self.func = func
        self.agg_func = agg_func
        self.return_data_only = return_data_only

    def __call__(self, args):
        data, mask, window = args
        result = self.func(*data)
        result = self.agg_func(result)
        if self.return_data_only:
            return result
        return result, mask, window

class RasterReader:

    def __init__(self,
        reference_file: str=None,
    ):
        self.reference = None
        if reference_file is not None:
            self.reference = rio.open(reference_file)
            self.block_windows = np.array([
                tup[1]
                for tup in self.reference.block_windows()
            ])
            boxes = self.blocks2boxes(self.block_windows)
            self.rtree = pg.strtree.STRtree(boxes)
            __ = self.rtree.query(boxes[0])

    def get_block_indices(self,
        geometry: dict,
    ):
        _geom = pg.from_shapely(g.shape(geometry))
        return self.rtree.query(_geom)

    def blocks2boxes(self, block_windows):
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
    ):
        block_idx = self.get_block_indices(geometry)

        if isinstance(src_path, str):
            src_path = [src_path]

        if self.reference is None:
            self.reference = rio.open(reference_file)

        sources = {}

        def _read_worker(window: rio.windows.Window):
            import threading
            tname = threading.current_thread().name
            if tname not in sources:
                sources[tname] = [
                    rio.open(sp)
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

class RasterAggregator:

    def __init__(self,
        reader: RasterReader=None,
    ):
        self.reader = reader

    def aggregate(self,
        src_path: Union[str, Iterable[str]],
        geometry: dict,
        block_func: Callable,
        agg_func: Callable=np.mean,
        **kwargs,
    ):
        if self.reader is None:
            self.reader = RasterReader(src_path)

        (*block_results,) = map(
            RasterBlockFunction(
                block_func,
                return_data_only=True,
                agg_func=agg_func,
            ),
            self.reader.read_overlay(
                src_path,
                geometry,
                **kwargs,
            ),
        )

        result = agg_func(block_results)
        return result

class RasterWriter:

    def __init__(self,
        reader: RasterReader=None,
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
        if self.reader is None:
            self.reader = RasterReader(src_path)

        profile = self.reader.reference.profile

        block_indices = self.reader.get_block_indices(geometry)
        out_window_geometry = self.reader.blocks2boxes(self.reader.block_windows[block_indices])
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
                RasterBlockFunction(block_func, return_data_only=False),
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
