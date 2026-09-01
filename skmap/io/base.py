"""
Raster data input and output
"""
from contextlib import ExitStack, contextmanager
from rasterio.io import DatasetWriter

import copy
import math
import os
import tempfile
import time
import warnings
from base64 import b64decode, encodebytes
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from skmap.compute import ComputeBackend
from uuid import uuid4

import numpy
import numpy as np
import pandas as pd
import rasterio
import requests
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from osgeo import gdal
from pandas import DataFrame, Series, to_datetime
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

import skmap_bindings as sb
from skmap import SKMapBase, SKMapGroupRunner, SKMapRunner, parallel
from skmap.misc import (
    _eval,
    date_range,
    make_tempdir,
    ttprint,
)

_INT_DTYPE = (
    "uint8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "int",
    "uint",
)


def _nodata_replacement(dtype: str):
    if dtype in _INT_DTYPE:
        return np.iinfo(dtype).max
    else:
        return np.nan


def _fit_in_dtype(data: NDArray, dtype: str, nodata: int) -> NDArray:
    if dtype in _INT_DTYPE:
        data = np.rint(data)

        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max

        data = np.where((data < min_val), min_val, data)
        data = np.where((data > max_val), max_val, data)

        if nodata == min_val:
            data = np.where((data == nodata), (min_val + 1), data)
        elif nodata == max_val:
            data = np.where((data == nodata), (max_val - 1), data)

    return data


def _read_raster(
    raster_idx,
    raster_files,
    band,
    window,
    dtype,
    data_mask,
    expected_shape,
    try_without_window,
    scale,
    gdal_opts,
    overview,
    verbose,
):

    for key in gdal_opts.keys():
        gdal.SetConfigOption(key, gdal_opts[key])

    raster_file = raster_files[raster_idx]
    ds, band_data = None, None
    nodata = None

    try:
        ds = rasterio.open(raster_file)

        if verbose:
            ttprint("Start reading")
        array_mm = None
        if overview is not None and overview > 1:
            h = window.height if window is not None else ds.height
            w = window.width if window is not None else ds.width
            array_mm = ds.read(
                band,
                out_dtype=dtype,
                out_shape=(1, math.ceil(h / overview), math.ceil(w / overview)),
                window=window,
            )
        else:
            array_mm = ds.read(band, out_dtype=dtype, window=window)
        if verbose:
            ttprint("End reading")

        # Normalise to 2-D (H, W): rasterio returns (1, H, W) for a band list.
        if array_mm.ndim == 3:
            array_mm = array_mm[0]

        # if band_data.size == 0 and try_without_window:
        #  band_data = ds.read(band, out=array_mm[:,:,raster_idx])

        # band_data = band_data.astype(dtype)
        nodata = ds.nodatavals[0]
        data_exists = True
        # print(f"Data was read: {raster_file}")
    except Exception as ex:
        if verbose:
            ttprint(f"Exception: {ex}")
        # traceback.i(print)_exc()

        if window is not None:
            if verbose:
                ttprint(f"ERROR: Failed to read {raster_file} window {window}")
            array_mm = np.empty((int(window.height), int(window.width)))
            array_mm = _nodata_replacement(dtype)

        if expected_shape is not None:
            if verbose:
                ttprint(f"Full nan image for {raster_file}")
            array_mm = np.empty(expected_shape)
            array_mm = _nodata_replacement(dtype)

    if data_exists:
        if data_mask is not None:
            if len(data_mask.shape) == 3:
                data_mask = data_mask[:, :, 0]

            if data_mask.shape == array_mm.shape:
                array_mm[np.logical_not(data_mask)] = np.nan
            else:
                ttprint(
                    f"WARNING: incompatible data_mask shape {data_mask.shape} != {array_mm.shape}"
                )

        if nodata is not None:
            if verbose:
                ttprint("Start _nodata_replacement")
            array_mm[array_mm == nodata] = _nodata_replacement(dtype)
            if verbose:
                ttprint("End _nodata_replacement")

    if scale != 1.0:
        if verbose:
            ttprint("Start scaling")
        array_mm = array_mm * scale
        if verbose:
            ttprint("End scaling")

    return array_mm, raster_idx, data_exists


def _read_auth_raster(raster_files, url_pos, bands, username, password, dtype, nodata):
    url = raster_files[url_pos]

    data = None
    ds_params = None

    try:
        data = requests.get(url, auth=(username, password), stream=True)

        with rasterio.io.MemoryFile(data.content) as memfile:
            if verbose:
                ttprint(f"Reading {url} to {memfile.name}")

            with memfile.open() as ds:
                if bands is None:
                    bands = range(1, ds.count + 1)

                if nodata is None:
                    nodata = ds.nodatavals[0]

                data = ds.read(bands)
                if isinstance(data, np.ndarray):
                    data = data.astype(dtype)
                    data[data == nodata] = _nodata_replacement(dtype)

                nbands, x_size, y_size = data.shape

                ds_params = {
                    "driver": ds.driver,
                    "width": x_size,
                    "height": y_size,
                    "count": nbands,
                    "dtype": ds.dtypes[0],
                    "crs": ds.crs,
                    "transform": ds.transform,
                }

    except:
        ttprint(f"Invalid raster file {url}")
        # traceback.print_exc()
        pass

    return url_pos, data, ds_params

@contextmanager
def _new_raster(base_raster, raster_file, data, window=None, dtype=None, nodata=None, overview=None):
    if not isinstance(raster_file, Path):
        raster_file = Path(raster_file)

    raster_file.parent.mkdir(parents=True, exist_ok=True)

    if len(data.shape) < 3:
        data = np.stack([data], axis=2)

    x_size, y_size, nbands = data.shape

    with rasterio.open(base_raster, "r") as base_raster:
        if dtype is None:
            dtype = base_raster.dtypes[0]

        if nodata is None:
            nodata = base_raster.nodata

        transform = base_raster.transform

        if window is not None:
            transform = rasterio.windows.transform(window, transform)

        if overview is not None and overview > 1:
            transform = transform * rasterio.Affine.scale(overview, overview)

        with rasterio.open(
            raster_file,
            "w",
            driver="GTiff",
            height=x_size,
            width=y_size,
            count=nbands,
            dtype=dtype,
            crs=base_raster.crs,
            compress="LZW",
            transform=transform,
            nodata=nodata,
        ) as dataset:
            yield dataset


def _save_raster(
    fn_base_raster: str,
    raster_file: str,
    ref_array,
    i: int,
    spatial_win: Window | None = None,
    dtype: str | None = None,
    nodata=None,
    fit_in_dtype=False,
    overview=None,
    on_each_outfile: Callable | None = None,
):
    # if len(data.shape) < 3:
    #  data = np.stack([data], axis=2)

    # _, _, nbands = data.shape

    array = parallel.get_shared(ref_array)

    with rasterio.open(fn_base_raster) as src:
        h, w = src.height, src.width
    if spatial_win is not None:
        h, w = spatial_win.height, spatial_win.width
    if overview is not None and overview > 1:
        h = math.ceil(h / overview)
        w = math.ceil(w / overview)
    band = np.array(array[i, :].reshape(h, w))  # writable copy (object store is read-only)

    with _new_raster(
        fn_base_raster, raster_file, band, spatial_win, dtype, nodata, overview
    ) as new_raster: # type: DatasetWriter
        band_dtype = new_raster.dtypes[0]

        if fit_in_dtype:
            band = _fit_in_dtype(band, band_dtype, new_raster.nodata)

        band[np.isnan(band)] = new_raster.nodata
        new_raster.write(band.astype(band_dtype), indexes=1)

    if on_each_outfile is not None:
        on_each_outfile(raster_file)

    return raster_file


def save_rasters_cpp(
    base_raster: Union[List, str],
    out_data: np.array,
    out_files: Union[List, str],
    out_dir: str = ".",
    out_idx: List = None,
    out_s3: Union[List, str] = None,
    window: Window = None,
    nodata: int = None,
    dtype: type = np.int16,
    n_jobs: int = 8,
    gdal_opts: dict = {},
    gdal_co: str = {
        "COMPRESS": "deflate",
        "ZLEVEL": "9",
        "TILED": "TRUE",
        "BLOCKXSIZE": "1024",
        "BLOCKYSIZE": "1024",
    },
    verbose=False,
):
    """Write a stack of 2D raster arrays to GeoTIFF files in parallel via the C++ bindings."""

    if isinstance(out_files, str):
        out_files = [out_files]
    if len(out_files) < n_jobs:
        n_jobs = len(out_files)

    n_layers = len(out_files)

    if window is None:
        ds = rasterio.open(base_raster)
        window = rasterio.windows.Window(0, 0, ds.width, ds.height)
    if out_idx is None:
        out_idx = list(range(0, n_layers))
    if out_s3 is not None:
        out_dir = str(make_tempdir())
    if nodata is None:
        ds = rasterio.open(base_raster)
        nodata = int(ds.nodatavals[0])

    creation_options = [f"{k}={v}" for k, v in gdal_co.items()]

    if isinstance(base_raster, str):
        base_raster = [base_raster for i in out_files]

    write_fn = sb.writeInt16Data
    if dtype == np.uint8:
        write_fn = sb.writeByteData
    elif dtype == np.uint16:
        write_fn = sb.writeUInt16Data
    elif dtype == np.float32:
        write_fn = sb.writeData

    if verbose:
        ttprint(f"Saving {n_layers} layers using window={window} to ")

    write_fn(
        out_data,
        n_jobs,
        gdal_opts,
        base_raster,
        out_dir,
        out_files,
        out_idx,
        window.col_off,
        window.row_off,
        window.width,
        window.height,
        nodata,
        creation_options,
    )

    if verbose:
        ttprint("End")

    if out_s3 is not None:
        # S3 upload is now handled by the caller via the MinIO client
        # (skmap.misc.s3_upload_file); the C++ bindings no longer shell out.
        return out_s3
    else:
        return [str(Path(out_dir).joinpath(f"{o}.tif")) for o in out_files]


def _read_shape(raster_files, band, window, bounds, overview):
    """Return ``(height, width, window)`` for a read, converting ``bounds`` to a window."""
    ds = rasterio.open(raster_files[-1])
    if bounds is not None and len(bounds) == 4:
        bounds = shape(
            rasterio.warp.transform_geom(
                src_crs="EPSG:4326",
                dst_crs=ds.crs,
                geom=box(*bounds),
            )
        ).bounds
        window = from_bounds(*bounds, ds.transform).round_lengths()

    b = band[0] if isinstance(band, (list, tuple)) else band
    if overview is not None and overview > 1:
        overviews = ds.overviews(b)
        if overview not in overviews:
            raise ValueError(
                f"Overview {overview} is invalid for {raster_files[-1]}.\n"
                f"Use one of overviews: {overviews}"
            )
        h = window.height if window is not None else ds.height
        w = window.width if window is not None else ds.width
        return (math.ceil(h / overview), math.ceil(w / overview), window)
    if window is not None:
        return window.height, window.width, window
    return ds.height, ds.width, window


def _resolve_read_params(
    raster_files, band, extent, extent_epsg, dtype, n_layers, overview,
    ram_fraction, verbose,
):
    """Resolve the read window (from ``extent``) and overview factor (RAM-fit).

    Returns ``(window, overview, out_height, out_width)``.  ``overview`` is
    ``None`` for a full-resolution read, or the COG overview factor to use.
    """
    import psutil

    ds = rasterio.open(raster_files[0])
    b = band[0] if isinstance(band, (list, tuple)) else band

    window = None
    if extent is not None and len(extent) == 4:
        src_crs = extent_epsg if extent_epsg is not None else ds.crs
        bnds = rasterio.warp.transform_bounds(src_crs, ds.crs, *extent)
        window = from_bounds(*bnds, ds.transform).round_lengths()
        # Clip to the raster grid and cast to ints (the C++ readData binding
        # takes unsigned integer offsets).
        window = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        window = rasterio.windows.Window(
            int(window.col_off), int(window.row_off),
            int(window.width), int(window.height),
        )

    w = window.width if window is not None else ds.width
    h = window.height if window is not None else ds.height
    overviews = ds.overviews(b)
    dtype_bytes = np.dtype(dtype).itemsize
    full_bytes = w * h * n_layers * dtype_bytes
    available = psutil.virtual_memory().available * ram_fraction

    if overview is not None:
        if overview not in overviews:
            raise ValueError(
                f"Overview {overview} is invalid for {raster_files[0]}. "
                f"Use one of: {overviews}"
            )
    elif full_bytes > available:
        for ov in sorted(overviews):
            ov_bytes = math.ceil(w / ov) * math.ceil(h / ov) * n_layers * dtype_bytes
            if ov_bytes <= available:
                if verbose:
                    ttprint(
                        f"Full read ~{full_bytes / 1e9:.2f} GB exceeds available "
                        f"RAM (~{available / 1e9:.2f} GB); using overview x{ov} "
                        f"(~{ov_bytes / 1e9:.2f} GB)"
                    )
                overview = ov
                break
        else:
            if overviews:
                overview = max(overviews)
                if verbose:
                    ttprint(
                        f"Full read ~{full_bytes / 1e9:.2f} GB exceeds available "
                        f"RAM; even the coarsest overview x{overview} does not fit"
                    )
            else:
                raise MemoryError(
                    f"Reading {n_layers} rasters at {w}x{h} {dtype} needs "
                    f"~{full_bytes / 1e9:.2f} GB but only ~{available / 1e9:.2f} GB "
                    f"RAM is available and the rasters have no COG overviews. "
                    f"Provide a smaller extent or rasters with overviews."
                )

    if overview is not None and overview > 1:
        out_h = math.ceil(h / overview)
        out_w = math.ceil(w / overview)
    else:
        out_h, out_w = h, w
    return window, overview, out_h, out_w


def _cpp_read_ok(
    dtype, bounds, data_mask, scale, expected_shape, try_without_window, overview, max_rasters
):
    """Return True if the C++ readData path can satisfy the request."""
    return (
        dtype == "float32"
        and bounds is None
        and data_mask is None
        and scale == 1.0
        and expected_shape is None
        and not try_without_window
        and max_rasters is None
    )


def _read_rasters_cpp(raster_files, band, window, n_jobs, dtype, gdal_opts, verbose, overview=None):
    """Read a stack of rasters into a ``(N, H*W)`` array via the C++ bindings."""
    n_layers = len(raster_files)
    if window is None:
        ds = rasterio.open(raster_files[0])
        window = rasterio.windows.Window(0, 0, ds.width, ds.height)
    if overview is not None and overview > 1:
        buf_w = math.ceil(window.width / overview)
        buf_h = math.ceil(window.height / overview)
    else:
        buf_w, buf_h = window.width, window.height
    out_data = np.empty((n_layers, buf_w * buf_h), dtype=dtype)
    out_idx = list(range(0, n_layers))

    if verbose:
        ttprint(
            f"Reading {n_layers} layers using window={window} overview={overview} "
            f"and array={out_data.shape}"
        )

    sb.readData(
        out_data,
        n_jobs,
        raster_files,
        out_idx,
        int(window.col_off),
        int(window.row_off),
        int(window.width),
        int(window.height),
        band,
        gdal_opts,
        None,
        np.nan,
        overview if overview is not None else 0,
    )

    if verbose:
        ttprint("End")

    return out_data


def read_rasters(
    raster_files: Union[List, str] = [],
    band: Union[List, int] = 1,
    window: Window | None = None,
    bounds: [] = None,
    dtype: str = "float32",
    n_jobs: int = 8,
    data_mask: numpy.array = None,
    scale: float = 1.0,
    expected_shape=None,
    try_without_window: bool = False,
    gdal_opts: dict = {},
    overview=None,
    max_rasters=None,
    backend: Union[str, "ComputeBackend"] = None,
    verbose=False,
) -> NDArray:
    """Read raster files into a single ``(N, H*W)`` array (files-first, pixels flattened).

    The ``nodata`` value is replaced by ``np.nan`` for float dtypes and by the
    lowest value in range for integer dtypes.

    :param raster_files: Raster paths (a single path is also accepted).
    :param band: Band index (or list of indices) to read.
    :param window: Spatial window to read. ``None`` reads the full extent.
    :param bounds: Bounding box (EPSG:4326) converted to a window.
    :param dtype: Output dtype. The C++ backend supports ``float32`` only.
    :param n_jobs: Number of parallel workers (python) / threads (cpp).
    :param data_mask: Mask array; pixels where it is 0 become ``np.nan``.
    :param scale: Multiply the read data by this factor.
    :param expected_shape: Shape used to build an empty array when a raster is missing.
    :param try_without_window: Retry without the window if the windowed read fails.
    :param gdal_opts: GDAL configuration options.
    :param overview: Overview level to read (COG files).
    :param max_rasters: Deprecated no-op (kept for backward compatibility).
    :param backend: ``"python"`` or ``"cpp"``. ``None`` auto-selects ``"cpp"``
      when the request is float32 with no python-only features, else ``"python"``.
      An explicit ``"cpp"`` with unsupported features falls back to ``"python"``
      with a warning.  An explicit ``"cpp"`` also keeps the result in-process
      (no Ray); the auto-selected cpp path keeps the Ray object-store model.
    :param verbose: Print reading progress.

    :returns: A :class:`skmap.parallel.SharedArray` of shape ``(N, H*W)``
      (``N`` files, ``H*W`` flattened pixels).  It is held in the Ray object
      store (call ``.get()`` to materialize it) except for explicit
      ``backend='cpp'`` reads, which return a local array directly.
    """
    if data_mask is not None and dtype not in ("float16", "float32"):
        raise Exception("The data_mask requires dtype as float")

    if isinstance(raster_files, str):
        raster_files = [raster_files]
    if isinstance(band, int):
        band = [band]
    if isinstance(raster_files[0], Path):
        raster_files = [str(r) for r in raster_files]
    if len(raster_files) < n_jobs:
        n_jobs = len(raster_files)

    cpp_explicit = False
    if backend is None:
        backend = (
            "cpp"
            if _cpp_read_ok(
                dtype, bounds, data_mask, scale, expected_shape,
                try_without_window, overview, max_rasters,
            )
            else "python"
        )
    else:
        backend = str(backend).lower()
        cpp_explicit = backend == "cpp"
        if backend == "cpp" and not _cpp_read_ok(
            dtype, bounds, data_mask, scale, expected_shape,
            try_without_window, overview, max_rasters,
        ):
            warnings.warn(
                "backend='cpp' requested but the request uses python-only "
                "features (non-float32 dtype, bounds, data_mask, scale, "
                "overview, expected_shape, try_without_window or max_rasters); "
                "falling back to the python backend",
                stacklevel=2,
            )
            backend = "python"

    if backend == "cpp":
        out = _read_rasters_cpp(
            raster_files, band, window, n_jobs, dtype, gdal_opts, verbose, overview
        )
        # An explicit backend='cpp' keeps the result in-process (no Ray); the
        # auto-selected cpp path keeps the default Ray object-store model.
        return parallel.put_shared(out, local=cpp_explicit)

    if verbose:
        ttprint(f"Reading {len(raster_files)} raster file(s) using {n_jobs} workers")

    height, width, window = _read_shape(raster_files, band, window, bounds, overview)

    args = [
        (
            raster_idx,
            raster_files,
            band,
            window,
            dtype,
            data_mask,
            expected_shape,
            try_without_window,
            scale,
            gdal_opts,
            overview,
            verbose,
        )
        for raster_idx in range(0, len(raster_files))
    ]

    # Workers return band arrays (Ray return values = ObjectRefs); a single
    # _stack_bands worker assembles them in the object store so the main
    # process never materializes the full array.
    band_refs = []
    for array, raster_idx, data_exists in parallel.job(
        _read_raster,
        args,
        n_jobs=n_jobs,
    ):
        if not data_exists:
            raster_file = raster_files[raster_idx]
            raise Exception(f"The raster {raster_file} not exists")
        band_refs.append((raster_idx, parallel.put_shared(array.reshape(-1)).ref))

    band_refs.sort(key=lambda x: x[0])
    refs = [r for _, r in band_refs]
    n = len(raster_files)
    out_ref = parallel._remote(parallel._stack_bands, refs, (n, height * width))
    return parallel.SharedArray(out_ref, (n, height * width), np.dtype(dtype))




def read_auth_rasters(
    raster_files: List,
    username: str,
    password: str,
    bands=None,
    dtype: str = "float16",
    n_jobs: int = 4,
    return_base_raster: bool = False,
    nodata=None,
    verbose: bool = False,
):
    """
    Read raster files trough a authenticate HTTP service, aggregating them into
    a single array. For raster files without authentication it's better
    to use read_rasters.

    The ``nodata`` value is replaced by ``np.nan`` in case of ``dtype=float*``,
    and for ``dtype=*int*`` it's replaced by the the lowest possible value
    inside the range (for ``int16`` this value is ``-32768``).

    :param raster_files: A list with the raster urls.
    :param username: Username to provide to the basic access authentication.
    :param password: Password to provide to the basic access authentication.
    :param bands: Which bands needs to be read. By default is ``None`` reading all
      the bands.
    :param dtype: Convert the read data to specific ``dtype``. By default it reads in
      ``float16`` to save memory, however pay attention in the precision limitations for
      this ``dtype`` [1].
    :param n_jobs: Number of parallel jobs used to read the raster files.
    :param return_base_raster: Return an empty raster with the same properties
      of the read rasters ``(height, width, n_bands, crs, dtype, transform)``.
    :param nodata: Use this value if the nodata property is not defined in the
      read rasters.
    :param verbose: Use ``True`` to print the reading progress.

    :returns: A 4D array, where the first dimension refers to the bands and the last
      dimension to read files. If ``return_base_raster=True`` the second value
      will be a base raster path.
    :rtype: Numpy.array or Tuple[Numpy.array, Path]

    Examples
    ========

    >>> from skmap.io.base import read_auth_rasters
    >>>
    >>> # Do the registration in
    >>> # https://glad.umd.edu/ard/user-registration
    >>> username = '<YOUR_USERNAME>'
    >>> password = '<YOUR_PASSWORD>'
    >>> raster_files = [
    ...     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/850.tif',
    ...     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/851.tif',
    ...     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/852.tif',
    ...     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/853.tif'
    ... ]
    >>>
    >>> data, base_raster = read_auth_rasters(
    ...     raster_files,
    ...     username,
    ...     password,
    ...     return_base_raster=True,
    ...     verbose=True
    ... ) # doctest: +SKIP
    >>> print(f'Data: shape={data.shape}, dtype={data.dtype} and base_raster={base_raster}') # doctest: +SKIP

    References
    ==========

    [1] `Float16 Precision <https://github.com/numpy/numpy/issues/8063>`_
    """

    if verbose:
        ttprint(
            f"Reading {len(raster_files)} remote raster files using {n_jobs} workers"
        )

    args = [
        (raster_files, url_pos, bands, username, password, dtype, nodata)
        for url_pos in range(0, len(raster_files))
    ]

    raster_data = {}
    fn_base_raster = None

    for url_pos, data, ds_params in parallel.job(
        _read_auth_raster, args, n_jobs=n_jobs
    ):
        if data is not None:
            raster_data[url_pos] = data

            if return_base_raster and fn_base_raster is None:
                with tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ) as base_raster:
                    with rasterio.open(
                        base_raster.name,
                        "w",
                        driver=ds_params["driver"],
                        width=ds_params["width"],
                        height=ds_params["height"],
                        count=ds_params["count"],
                        crs=ds_params["crs"],
                        dtype=ds_params["dtype"],
                        transform=ds_params["transform"],
                    ) as ds:
                        fn_base_raster = ds.name

    raster_data_arr = []
    for i in range(0, len(raster_files)):
        if i in raster_data:
            raster_data_arr.append(raster_data[i])

    raster_data = np.stack(raster_data_arr, axis=-1)
    del raster_data_arr

    if return_base_raster:
        if verbose:
            ttprint(f"The base raster is {fn_base_raster}")
        return raster_data, fn_base_raster
    else:
        return raster_data


def save_rasters(
    base_raster: str,
    raster_files: List,
    array,
    window: Window = None,
    bounds: [] = None,
    overview: int = None,
    dtype: str = None,
    nodata=None,
    array_idx: List = [],
    fit_in_dtype: bool = False,
    n_jobs: int = 8,
    on_each_outfile: Callable = None,
    verbose: bool = False,
):
    """
    Save a ``(N, H*W)`` array in multiple raster files using as reference one base raster.
    The first dimension (bands) is used to split the array in different rasters. GeoTIFF is
    the only output format supported. It always replaces the ``np.nan`` value
    by the specified ``nodata``.

    :param base_raster: The base raster path used to retrieve the
      parameters ``(height, width, n_bands, crs, dtype, transform)`` for the
      new rasters.
    :param raster_files: A list containing the paths for the new raster. It creates
      the folder hierarchy if not exists.
    :param array: ``(N, H*W)`` data array (files-first, pixels flattened).
    :param window: Save the data considering a spatial window, even if the ``base_rasters``
      refers to a bigger area. For example, it's possible to have a base raster covering the whole
      Europe and save the data using a window that cover just part of Wageningen. By default is
      ``None`` saving the raster data in position ``0, 0`` of the raster grid.
    :param dtype: Convert the data to a specific ``dtype`` before save it. By default is ``None``
      using the same ``dtype`` from the base raster.
    :param nodata: Use the specified value as ``nodata`` for the new rasters. By default is ``None``
      using the same ``nodata`` from the base raster.
    :param fit_in_dtype: If ``True`` the values outside of ``dtype`` range are truncated to the minimum
      and maximum representation. It's also change the minimum and maximum data values, if they exist,
      to avoid overlap with ``nodata`` (see the ``_fit_in_dtype`` function). For example, if
      ``dtype='uint8'`` and ``nodata=0``, all data values equal to ``0`` are re-scaled to ``1`` in the
      new rasters.
    :param n_jobs: Number of parallel jobs used to save the raster files.
    :param verbose: Use ``True`` to print the saving progress.

    :returns: A list containing the path for new rasters.
    :rtype: List[Path]

    Examples
    ========

    >>> import rasterio
    >>> from skmap.io.base import read_rasters, save_rasters
    >>>
    >>> # skmap COG layers - NDVI seasons for 2019
    >>> raster_files = [
    ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
    ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
    ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
    ...     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
    ... ]
    >>>
    >>> # Transform for the EPSG:3035
    >>> eu_transform = rasterio.open(raster_files[0]).transform # doctest: +SKIP
    >>> # Bounding box window over Wageningen, NL
    >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform) #doctest: +SKIP
    >>>
    >>> data, _ = read_rasters(raster_files=raster_files, window=window, verbose=True) #doctest: +SKIP
    >>>
    >>> # Save in the current execution folder
    >>> raster_files = [
    ...     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_wageningen_epsg3035_v1.0.tif',
    ...     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_wageningen_epsg3035_v1.0.tif',
    ...     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_wageningen_epsg3035_v1.0.tif',
    ...     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_wageningen_epsg3035_v1.0.tif'
    ... ] # doctest: +SKIP
    >>>
    >>> save_rasters(raster_files[0], raster_files, data, window=window, verbose=True) #doctest: +SKIP

    """

    if type(raster_files) == str:
        raster_files = [raster_files]

    # if len(data.shape) < 3:
    #  data = np.stack([data], axis=2)

    if len(array_idx) == 0:
        array_idx = list(range(0, array.shape[0]))

    if len(array_idx) != len(raster_files):
        raise Exception(
            f"The array shape {array.shape} is incompatible with the raster_files size {len(raster_files)}."
        )

    ds = rasterio.open(base_raster)
    if bounds is not None and len(bounds) == 4:
        bounds = shape(
            rasterio.warp.transform_geom(
                src_crs="EPSG:4326",
                dst_crs=ds.crs,
                geom=box(*bounds),
            )
        ).bounds
        window = from_bounds(*bounds, ds.transform).round_lengths()
        if verbose:
            ttprint(f"Transform {bounds} into {window}")

    if verbose:
        ttprint(f"Saving {len(raster_files)} raster files using {n_jobs} workers")

    ref_array = parallel.put_shared(array).ref

    args = [
        (base_raster, raster_file, ref_array, i, window, dtype, nodata, fit_in_dtype, overview)
        for raster_file, i in zip(raster_files, array_idx)
    ]

    # batch_size = math.floor(len(args) / n_jobs)
    # if batch_size <= 0:
    #  batch_size = 'auto'

    out_files = []
    for out_raster in parallel.job(
        _save_raster,
        args,
        n_jobs=n_jobs,
    ):
        if on_each_outfile is not None:
            on_each_outfile(out_raster)
        out_files.append(out_raster)
        continue

    return out_files


class RasterData(SKMapBase):
    """High-level accessor over a catalog of raster layers with spatial/temporal filtering and I/O."""

    PLACEHOLDER_DT = "{dt}"
    INTERVAL_DT_SEP = "_"

    GROUP_COL = "group"
    NAME_COL = "name"
    PATH_COL = "input_path"
    BAND_COL = "input_band"
    TEMPORAL_COL = "temporal"
    DT_COL = "date"
    START_DT_COL = "start_date"
    END_DT_COL = "end_date"

    TRANSFORM_SEP = "."

    def __init__(
        self,
        raster_files: Union[List, str, dict],
        raster_mask: str = None,
        raster_mask_val=np.nan,
        max_rasters: int = None,
        verbose=False,
        backend: Union[str, "ComputeBackend"] = "numpy",
    ) -> None:
        from skmap.compute import get_backend

        self.backend = get_backend(backend)

        if isinstance(raster_files, str):
            raster_files = {"default": [raster_files]}
        elif isinstance(raster_files, list):
            raster_files = {"default": raster_files}

        self.raster_files = raster_files

        self.verbose = verbose

        self.raster_mask = raster_mask
        self.raster_mask_val = raster_mask_val

        rows = []
        for group in raster_files.keys():
            if isinstance(raster_files[group], str):
                rows.append([group, raster_files[group], 1, None, None])
            else:
                for r in raster_files[group]:
                    if isinstance(r, tuple):
                        if len(r) == 2:
                            rows.append([group, r[0], r[1], None, None])
                        elif len(r) == 4:
                            rows.append([group, r[0], r[1], r[2], r[3]])
                        else:
                            raise Exception(
                                f"Wrong tuple size {len(r)}. Please provide 2 or 4 size tuple."
                            )
                    else:
                        rows.append([group, r, 1, None, None])

        self.info = DataFrame(
            rows,
            columns=[
                RasterData.GROUP_COL,
                RasterData.PATH_COL,
                RasterData.BAND_COL,
                RasterData.START_DT_COL,
                RasterData.END_DT_COL,
            ],
        )

        self.info[RasterData.TEMPORAL_COL] = self.info.apply(
            lambda r: RasterData.PLACEHOLDER_DT in str(r[RasterData.PATH_COL]), axis=1
        )
        self.info[RasterData.NAME_COL] = self.info.apply(
            lambda r: Path(str(r[RasterData.PATH_COL]).split("?")[0]).stem
            if not r[RasterData.TEMPORAL_COL]
            else None,
            axis=1,
        )

        self.date_args = {}
        self._active_group = None

        # Lazily populated by ``read()``; ``None`` means the layers are not
        # loaded in memory (paths only) so overlay can sample from files.
        self.array = None
        self.base_raster = None
        self.window = None
        self.overview = None
        self.extent = None
        self.extent_epsg = None

        has_date = ~self.info[RasterData.START_DT_COL].isnull().any()

        if has_date:
            self.info[RasterData.TEMPORAL_COL] = True
            for g in self.info[RasterData.GROUP_COL].unique():
                self.date_args[g] = {
                    "date_style": "interval",
                    "date_format": "%Y%m%d",
                    "ignore_29feb": True,
                }

        self.info.reset_index(drop=True, inplace=True)
        self.max_rasters = max_rasters

    def _new_info_row(
        self,
        raster_file: str,
        name: str,
        group: str = None,
        dates: list = [],
        date_format=None,
        date_style=None,
        ignore_29feb=True,
    ):
        row = {}

        if group is None or "default" in group:
            group = "default"

        row[RasterData.PATH_COL] = raster_file
        row[RasterData.NAME_COL] = name
        row[RasterData.GROUP_COL] = group
        row[RasterData.BAND_COL] = 1

        if self._active_group is not None:
            if date_style is None:
                date_style = self.date_args[self._active_group]["date_style"]
            if date_format is None:
                date_format = self.date_args[self._active_group]["date_format"]

            self.date_args[group] = self.date_args[self._active_group]
        else:
            self.date_args[group] = {
                "date_style": date_style,
                "date_format": date_format,
                "ignore_29feb": ignore_29feb,
            }

        if len(dates) > 0 and date_style is not None:
            row[RasterData.TEMPORAL_COL] = True

            dt1, dt2 = (dates[0], dates[1])

            if isinstance(dt1, str):
                dt1 = datetime.strptime(dt1, date_format)
            if isinstance(dt2, str):
                dt2 = datetime.strptime(dt2, date_format)
            row[RasterData.START_DT_COL] = dt1
            row[RasterData.END_DT_COL] = dt2

        else:
            row[RasterData.PATH_COL] = raster_file
            row[RasterData.NAME_COL] = name

        return row

    @classmethod
    def from_info(
        cls,
        info,
        backend: Union[str, "ComputeBackend"] = "numpy",
        verbose: bool = False,
        date_format: str = "%Y%m%d",
        ignore_29feb: bool = True,
    ):
        """Build a RasterData from a pre-built ``info`` DataFrame (classmethod).

        Used by the layer sources (:mod:`skmap.io.sources`) to construct a lazy
        RasterData whose ``info`` already carries dates, groups, names and any
        extra per-layer variable columns.  Unlike ``__init__``, the incoming
        ``temporal`` column is preserved as-is (static layers stay ``False``).
        """
        from skmap.compute import get_backend

        obj = cls.__new__(cls)
        obj.backend = get_backend(backend)
        obj.verbose = verbose
        obj.raster_mask = None
        obj.raster_mask_val = np.nan
        obj.max_rasters = None
        obj.raster_files = {}
        obj.info = info.reset_index(drop=True)
        obj.date_args = {}
        obj._active_group = None
        obj.array = None
        obj.base_raster = None
        obj.window = None
        obj.overview = None
        obj.extent = None
        obj.extent_epsg = None

        if RasterData.TEMPORAL_COL not in obj.info.columns:
            obj.info[RasterData.TEMPORAL_COL] = obj.info.apply(
                lambda r: RasterData.PLACEHOLDER_DT in str(r[RasterData.PATH_COL]),
                axis=1,
            )

        # date_args for the groups that actually carry dates (temporal groups)
        dated = ~obj.info[RasterData.START_DT_COL].isnull()
        for g in obj.info.loc[dated, RasterData.GROUP_COL].unique():
            obj.date_args[g] = {
                "date_style": "interval",
                "date_format": date_format,
                "ignore_29feb": ignore_29feb,
            }

        return obj

    @classmethod
    def from_yaml(
        cls,
        path: str,
        base_path: str = None,
        date_format: str = "%Y%m%d",
        ignore_29feb: bool = True,
        backend: Union[str, "ComputeBackend"] = "numpy",
        verbose: bool = False,
    ):
        """Build a lazy RasterData from a YAML layer catalogue (classmethod).

        See :mod:`skmap.io.sources` for the YAML schema and the expansion
        rules.  The result holds paths + dates only (no ``.read()``); its
        ``info`` carries one column per ``{variable}`` referenced in the path
        templates (e.g. ``band``, ``year``, ``start_month``, ``end_month``,
        ``perc``) so runners can group by multiple columns.
        """
        from skmap.io.sources import YamlSource

        return YamlSource(
            path,
            base_path=base_path,
            date_format=date_format,
            ignore_29feb=ignore_29feb,
        ).to_rasterdata(backend=backend, verbose=verbose)

    @classmethod
    def from_stac(
        cls,
        url: str,
        collections: Union[str, List[str]],
        datetime: str = None,
        bbox: List[float] = None,
        bands: List[str] = None,
        max_items: int = None,
        limit: int = 500,
        date_format: str = "%Y%m%d",
        ignore_29feb: bool = True,
        backend: Union[str, "ComputeBackend"] = "numpy",
        verbose: bool = False,
    ):
        """Build a lazy RasterData from a STAC catalogue (classmethod).

        Queries the per-collection ``/items`` endpoint and yields one ``info``
        row per data asset (``roles`` contains ``"data"``).  The result holds
        remote COG hrefs + dates only (no ``.read()``); its ``info`` carries
        ``collection``, ``asset``, ``year``, ``gsd`` and ``epsg`` columns.
        See :class:`skmap.io.sources.StacSource` for the query rules.
        """
        from skmap.io.sources import StacSource

        return StacSource(
            url=url,
            collections=collections,
            datetime=datetime,
            bbox=bbox,
            bands=bands,
            max_items=max_items,
            limit=limit,
            date_format=date_format,
            ignore_29feb=ignore_29feb,
        ).to_rasterdata(backend=backend, verbose=verbose)

    def _set_date(
        self,
        text,
        dt1,
        dt2,
        date_format=None,
        date_style=None,
        ignore_29feb=None,
        **kwargs,
    ):
        if "gr" in kwargs and "default" in kwargs.get("gr"):
            gr = ""

        if date_format is None:
            date_format = self.date_args[self._active_group]["date_format"]

        if date_style is None:
            date_style = self.date_args[self._active_group]["date_style"]

        if ignore_29feb and "%j" in date_format:
            dt1 = dt1 + relativedelta(leapdays=-1)
            dt2 = dt2 + relativedelta(leapdays=-1)

        if date_style == "start_date":
            dt = f"{dt1.strftime(date_format)}"
        elif date_style == "end_date":
            dt = f"{dt2.strftime(date_format)}"
        else:
            dt = f"{dt1.strftime(date_format)}"
            dt += f"{RasterData.INTERVAL_DT_SEP}"
            dt += f"{dt2.strftime(date_format)}"

        return _eval(str(text), {**kwargs, **locals()})

    def timespan(
        self,
        start_date,
        end_date,
        date_unit,
        date_step,
        date_style: str = "interval",
        date_format: str = "%Y%m%d",
        ignore_29feb=True,
        group: [list, str] = [],
    ):
        """Return the ``(start_date, end_date)`` span covered by the loaded layers."""

        if isinstance(group, str):
            group = [group]

        to_drop = []
        to_add = []

        for _group, ginfo in self.info.groupby(RasterData.GROUP_COL):
            if len(group) > 0 and _group not in group:
                continue

            self.date_args[_group] = {
                "date_style": date_style,
                "date_format": date_format,
                "ignore_29feb": ignore_29feb,
            }

            dates = date_range(
                start_date,
                end_date,
                date_unit=date_unit,
                date_step=date_step,
                date_format=date_format,
                ignore_29feb=ignore_29feb,
            )

            def fun(r):
                if r[RasterData.TEMPORAL_COL]:
                    names, start, end = [], [], []
                    for dt1, dt2 in dates:
                        names.append(
                            self._set_date(
                                r[RasterData.PATH_COL],
                                dt1,
                                dt2,
                                date_format=date_format,
                                date_style=date_style,
                                ignore_29feb=ignore_29feb,
                            )
                        )
                        start.append(dt1)
                        end.append(dt2)
                    return Series([names, start, end])
                else:
                    return Series([[r[RasterData.PATH_COL]], [None], [None]])

            temporal_cols = [
                RasterData.PATH_COL,
                RasterData.START_DT_COL,
                RasterData.END_DT_COL,
            ]

            ginfo[temporal_cols] = ginfo.apply(fun, axis=1)
            ginfo = ginfo.explode(temporal_cols)
            ginfo[RasterData.NAME_COL] = ginfo.apply(
                lambda r: Path(r[RasterData.PATH_COL]).stem, axis=1
            )

            to_drop.append(ginfo.index)
            to_add.append(ginfo)

        for idx in to_drop:
            self.info = self.info.drop(index=idx)

        self.info = pd.concat([self.info] + to_add).reset_index(drop=True)

        return self

    def _has_base_raster(self) -> bool:
        """Return True if at least one referenced raster is reachable."""
        for filepath in list(self.info[RasterData.PATH_COL]):
            if "http" in filepath:
                res = requests.head(filepath)
                if res.status_code == 200:
                    return True
            else:
                if Path(filepath).exists():
                    return True
        return False

    def read(
        self,
        extent: list = None,
        extent_epsg=None,
        dtype: str = "float32",
        expected_shape=None,
        overview: int = None,
        ram_fraction: float = 0.7,
        n_jobs: int = 4,
        scale: float = 1,
        gdal_opts: dict = {},
    ):
        """Read the selected layers into a SharedArray and return ``self``.

        :param extent: ``(minx, miny, maxx, maxy)`` bounding box to read, in
          ``extent_epsg`` (defaults to the rasters' own CRS). ``None`` reads
          the full extent.
        :param extent_epsg: EPSG code / CRS of ``extent``. Defaults to the
          rasters' own CRS.
        :param overview: COG overview factor to read (e.g. ``2``). ``None``
          auto-selects: if the full read would not fit in RAM (estimated from
          width, height, dtype and ``ram_fraction`` of available memory), the
          finest overview that fits is used and logged.
        :param ram_fraction: fraction of available RAM the read is allowed to
          use (default 0.7).

        Side effects: sets ``self.window``, ``self.overview``, ``self.extent``,
        ``self.extent_epsg``, ``self.base_raster`` and ``self.array``.
        """

        self.extent = extent
        self.extent_epsg = extent_epsg

        self.base_raster = self._base_raster()
        raster_files = []

        # FIXME: add supporting for band_list
        for band, rows in self.info.groupby(RasterData.BAND_COL):
            # keep http(s) URLs as strings: Path() would mangle "https://" -> "https:/"
            raster_files += [
                r if "http" in str(r) else Path(r) for r in rows[RasterData.PATH_COL]
            ]

        # Resolve the read window (from extent) and the overview factor
        # (explicit or RAM-fit) once, so mask and data reads agree.
        window, overview, out_h, out_w = _resolve_read_params(
            raster_files, band, extent, extent_epsg, dtype,
            len(raster_files), overview, ram_fraction, self.verbose,
        )
        self.window = window
        self.overview = overview

        data_mask = None
        if self.raster_mask is not None:
            self._verbose(
                f"Masking {self.raster_mask_val} values considering {Path(self.raster_mask).name}"
            )
            data_mask = read_rasters(
                [self.raster_mask],
                window=window,
                overview=overview,
                gdal_opts=gdal_opts,
            )
            data_mask = data_mask.get().reshape(out_h, out_w)
            if self.raster_mask_val is np.nan:
                data_mask = np.logical_not(np.isnan(data_mask))
            else:
                data_mask = data_mask != self.raster_mask_val

        self._verbose(
            f"RasterData with {len(raster_files)} rasters"
            + f" and {len(self.info[RasterData.GROUP_COL].unique())} group(s)"
        )

        self.array = read_rasters(
            raster_files,
            band=band,
            window=window,
            data_mask=data_mask,
            dtype=dtype,
            expected_shape=expected_shape,
            n_jobs=n_jobs,
            overview=overview,
            scale=scale,
            gdal_opts=gdal_opts,
            verbose=self.verbose,
            max_rasters=self.max_rasters,
            # explicit backend='cpp' reads via the C++ bindings only (no Ray)
            backend="cpp" if self.backend.name == "cpp" else None,
        )

        self._spatial_shape = (out_h, out_w)

        # The array is rebuilt positionally (rows 0..N-1); reset info.index to match.
        self.info = self.info.reset_index(drop=True)

        self._verbose(f"Read array shape: {self.array.shape}")

        return self

    def run(
        self,
        process: SKMapRunner,
        group: [list, str] = [],
        outname: str = None,
        drop_input: bool = False,
        backend: Union[str, "ComputeBackend"] = None,
    ):
        """Execute a function over the loaded raster data, yielding per-tile results.

        ``backend`` optionally overrides the compute backend for this run only
        (it does not mutate ``self.backend``).
        """

        # Propagate the compute backend so the runner uses it instead of
        # hardcoded libraries.  A per-run override wins over the object default.
        if backend is not None:
            from skmap.compute import get_backend

            process.backend = get_backend(backend)
        else:
            process.backend = self.backend

        # Reset the fallback log so the summary below reflects only this run.
        process.backend.reset_fallbacks()

        if isinstance(process, SKMapGroupRunner):
            if drop_input:
                input_groups = self.info[
                    self.info[RasterData.TEMPORAL_COL] == process.temporal
                ][RasterData.GROUP_COL].unique().tolist()
            self._group_run(process, group, outname)
            if drop_input:
                self.drop(input_groups)
        else:
            process_name = process.__class__.__name__

            start = time.time()
            self._verbose(f"Running {process_name}" + f" on {self.array.shape}")

            # Snapshot the input groups so drop_input can remove the consumed
            # covariates even when no explicit `group` is passed (e.g. Prediction).
            input_groups = self.info[RasterData.GROUP_COL].unique().tolist()

            kwargs = {"rdata": self}
            if outname is not None:
                kwargs["outname"] = outname

            _, new_info = process.run(**kwargs)

            if new_info.shape[0] > 0:
                idx_offset = self._idx_offset()
                new_info.index = list(range(idx_offset, idx_offset + new_info.shape[0]))
                self.info = pd.concat([self.info, new_info])

            self._verbose(
                "Execution"
                + f" time for {process_name}: {(time.time() - start):.2f} segs"
            )

            if drop_input:
                self.drop(group if group else input_groups)

        self._report_fallbacks(process)

        return self

    def _report_fallbacks(self, process):
        """Print one summary line if the backend fell back to numpy/scipy.

        Note: fallbacks recorded inside Ray workers (e.g. TimeAggregate's
        per-tile ``_aggregate``) are not propagated back to the main process,
        so this summary reflects main-process operations only.
        """
        fb = getattr(process.backend, "fallbacks", None)
        if not fb:
            return
        ops = sorted({op for op, _ in fb})
        self._verbose(
            f"{process.backend.name} backend fell back to numpy/scipy for: {ops}"
        )

    def _group_run(
        self,
        process: SKMapGroupRunner,
        group: [list, str] = [],
        outname: str = None,
    ):
        if isinstance(group, str):
            group = [group]

        to_add_info = []

        group_list = []
        ginfo_list = []

        for _group, ginfo in self.info.groupby(RasterData.GROUP_COL):
            if ginfo[RasterData.TEMPORAL_COL].iloc[0] != process.temporal:
                self._verbose(
                    f"Skipping {process.__class__.__name__} for {_group} group"
                )
                continue

            if len(group) > 0 and _group not in group:
                continue

            expr_group = f'{RasterData.GROUP_COL} == "{_group}"'
            ginfo = self.info.query(expr_group)

            group_list.append(_group)
            ginfo_list.append(ginfo)

        process_name = process.__class__.__name__

        start = time.time()
        self._verbose(f"Running {process_name}" + f" {len(group_list)} groups")

        new_array, new_info = process.run(self, group_list, ginfo_list, outname)

        if new_array is not None:
            new_ref = parallel.put_shared(
                new_array, local=self.backend.name == "cpp"
            ).ref
            out_ref = parallel._remote(
                parallel._concat,
                [self.array.ref, new_ref],
                [self.array.shape, new_array.shape],
            )
            self.array = parallel.SharedArray(
                out_ref,
                (self.array.shape[0] + new_array.shape[0], self.array.shape[1]),
                self.array.dtype,
            )
            # Free the local copy now that the data lives in the object store;
            # otherwise a large result is held twice until this frame returns.
            del new_array

        to_add_info.append(new_info)

        self._verbose(
            "Execution" + f" time for {process_name}: {(time.time() - start):.2f} segs"
        )

        self._active_group = None

        if len(to_add_info) > 0:
            new_info = pd.concat(to_add_info)

            idx_offset = self._idx_offset()
            new_info.index = list(range(idx_offset, idx_offset + new_info.shape[0]))
            self.info = pd.concat([self.info, new_info])

        return self

    def drop(self, group):
        """Return a copy with the named layers removed."""

        if isinstance(group, str):
            group = [group]

        self._verbose(f"Dropping data and info for groups: {group}")
        idx = self.info[self.info[RasterData.GROUP_COL].isin(group)].index
        keep_idx = [i for i in range(self.array.shape[0]) if i not in set(idx)]
        out_ref = parallel._remote(
            parallel._select_bands,
            [self.array.ref],
            keep_idx,
            (len(keep_idx), self.array.shape[1]),
        )
        self.array = parallel.SharedArray(
            out_ref, (len(keep_idx), self.array.shape[1]), self.array.dtype
        )
        self.info = self.info.drop(idx).reset_index(drop=True)

        return self

    def rename(self, groups: dict):
        """Return a copy with layers renamed according to a mapping."""

        self.info[RasterData.GROUP_COL] = self.info[RasterData.GROUP_COL].replace(
            groups
        )
        self.info[RasterData.NAME_COL] = self.info[RasterData.NAME_COL].replace(groups)
        for old_group in groups.keys():
            new_group = groups[old_group]
            self.date_args[new_group] = self.date_args[old_group]
            del self.date_args[old_group]
        return self

    def filter_date(
        self,
        start_date,
        end_date=None,
        date_format="%Y-%m-%d",
        date_overlap=False,
        include_non_temporal=False,
        by_start_date=False,
        return_array=False,
        return_copy=True,
        return_idx=False,
    ):
        """Return a copy keeping only layers whose date falls within ``[start, end]``.

        :param include_non_temporal: when ``True``, also keep layers without a
            date (static layers with ``None`` in the start-date column), so
            they are preserved alongside the date-filtered temporal layers.
        :param by_start_date: when ``True``, a layer is kept when its
            ``start_date`` falls within ``[start_date, end_date]`` (the
            ``end_date`` column is ignored).  This assigns cross-year
            composites (e.g. a Dec-to-Mar winter composite) to the year they
            start in.
        """

        start_dt_col, end_dt_col = (RasterData.START_DT_COL, RasterData.END_DT_COL)
        info_main = self.info

        if RasterData.DT_COL in info_main.columns:
            start_dt_col, end_dt_col = (RasterData.DT_COL, None)

        if by_start_date:
            if end_date is None:
                raise ValueError("by_start_date requires end_date")
            dt_mask = (
                info_main[start_dt_col] >= to_datetime(start_date, format=date_format)
            ) & (info_main[start_dt_col] <= to_datetime(end_date, format=date_format))
        elif date_overlap:
            dt_mask = np.logical_or(
                info_main[start_dt_col] >= to_datetime(start_date, format=date_format),
                info_main[end_dt_col] >= to_datetime(start_date, format=date_format),
            )
        else:
            dt_mask = info_main[start_dt_col] >= to_datetime(
                start_date, format=date_format
            )

        if not by_start_date and end_date is not None and end_dt_col is not None:
            if date_overlap:
                dt_mask_end = np.logical_or(
                    info_main[end_dt_col] <= to_datetime(end_date, format=date_format),
                    info_main[start_dt_col]
                    <= to_datetime(end_date, format=date_format),
                )
            else:
                dt_mask_end = info_main[end_dt_col] <= to_datetime(
                    end_date, format=date_format
                )

            dt_mask = np.logical_and(dt_mask, dt_mask_end)

        if include_non_temporal:
            dt_mask = np.logical_or(dt_mask, info_main[start_dt_col].isnull())

        return self._filter(
            info_main[dt_mask],
            return_array=return_array,
            return_copy=return_copy,
            return_idx=return_idx,
        )

    def filter_contains(
        self, text, return_array=False, return_copy=True, return_idx=False
    ):
        """Return a copy keeping only layers whose name contains the given substring(s)."""

        return self.filter(
            f'{self.NAME_COL}.str.contains("{text}")',
            return_array=return_array,
            return_copy=return_copy,
            return_idx=return_idx,
        )

    def filter(self, expr, return_array=False, return_copy=True, return_idx=False):
        """Return a copy keeping only layers matching a boolean/query expression."""

        return self._filter(
            self.info.query(expr),
            return_array=return_array,
            return_copy=return_copy,
            return_idx=return_idx,
        )

    def _filter(
        self,
        info,
        return_info=False,
        return_array=False,
        return_copy=True,
        return_idx=False,
    ):
        # Active filters
        if self._active_group is not None:
            info = info.query(f'{RasterData.GROUP_COL} == "{self._active_group}"')

        if return_idx:
            return list(info.index)
        elif return_array:
            return None if self.array is None else self.array.get()[info.index, :]
        elif return_info:
            return info
        elif return_copy:
            rdata = copy.copy(self)
            if self.array is not None:
                rdata.array = parallel.put_shared(
                    self.array.get()[info.index, :], local=self.backend.name == "cpp"
                )
            rdata.info = info.reset_index(drop=True)
            # Deep-copy the mutable dicts so the filtered copy cannot corrupt
            # the original (regression fix: they were shared by reference).
            rdata.date_args = copy.deepcopy(self.date_args)
            rdata.raster_files = copy.deepcopy(self.raster_files)
            return rdata
        else:
            if self.array is not None:
                self.array = parallel.put_shared(
                    self.array.get()[info.index, :], local=self.backend.name == "cpp"
                )
            return self

    def _array(self):
        return self._filter(self.info, return_array=True)

    def _info(self):
        return self._filter(self.info, return_info=True)

    def get_groups(self) -> List[str]:
        """Return the group names, excluding the shared ``common`` group.

        A RasterData organised by year (plus a ``common`` group) can drive
        per-group prediction and overlay. A dataset with only ``common``
        layers returns ``["common"]``.
        """
        groups = sorted(
            set(self.info[RasterData.GROUP_COL].unique()) - {"common", "otf"}
        )
        if not groups and "common" in set(self.info[RasterData.GROUP_COL].unique()):
            groups = ["common"]
        return groups

    def _band_index(self, name: str, group: str = None) -> int:
        """Return the array band index (row position) of a named layer.

        The array is always ``(N, H*W)`` with band ``k`` corresponding to the
        ``k``-th row of ``self.info``, so positional lookup (not the possibly
        stale ``info.index`` values) is used.
        """
        info = self.info.reset_index(drop=True)
        mask = info[RasterData.NAME_COL] == name
        if group is not None:
            mask &= info[RasterData.GROUP_COL] == group
        hits = np.flatnonzero(mask.values)
        if len(hits) == 0:
            raise KeyError(
                f"Layer {name!r} not found in RasterData groups {self.get_groups()}"
            )
        return int(hits[0])

    def _get_covs_idx(self, covs_lst: List[str]) -> np.ndarray:
        """Map covariate names to band indices per group (``common`` falls back).

        Returns an int matrix of shape ``(len(covs_lst), n_groups)`` where each
        column is a group and each row a covariate; a covariate missing from a
        group falls back to the ``common`` group.
        """
        groups = self.get_groups()
        info = self.info.reset_index(drop=True)
        covs_idx = np.zeros((len(covs_lst), len(groups)), np.int32)
        for j, g in enumerate(groups):
            ginfo = info[info[RasterData.GROUP_COL] == g]
            common = info[info[RasterData.GROUP_COL] == "common"]
            for i, c in enumerate(covs_lst):
                hits = ginfo[ginfo[RasterData.NAME_COL] == c]
                if len(hits):
                    covs_idx[i, j] = int(hits.index[0])
                else:
                    chits = common[common[RasterData.NAME_COL] == c]
                    if len(chits):
                        covs_idx[i, j] = int(chits.index[0])
                    else:
                        raise KeyError(
                            f"Covariate {c!r} not found in group {g!r} or common"
                        )
        return covs_idx

    def get_years(self) -> List[int]:
        """Sorted unique years of the temporal layers (from ``start_date``).

        Returns an empty list when there are no temporal layers or no
        ``start_date`` column.  Static-only catalogues have no years.
        """
        info = self.info
        if (
            RasterData.TEMPORAL_COL not in info.columns
            or "start_date" not in info.columns
        ):
            return []
        temporal = info[info[RasterData.TEMPORAL_COL]]
        if temporal.empty or temporal["start_date"].isna().all():
            return []
        return sorted(temporal["start_date"].dt.year.dropna().astype(int).unique())

    def _get_covs_idx_by_year(
        self, covs_lst: List[str], years: List = None
    ) -> tuple:
        """Map covariate names to band indices per *year*.

        Returns ``(covs_idx, years)`` where ``covs_idx`` has shape
        ``(len(covs_lst), n_years)``.  Temporal covariates are matched by
        ``name`` **and** ``start_date`` year (resolving year-agnostic names
        such as ``ndvi_winter`` that repeat once per year); static covariates
        (the ``common`` group) use the same band index for every year.

        A static-only catalogue (no temporal layers) yields ``n_years == 1``
        with a ``None`` year placeholder, so a single prediction is produced
        from the static covariates.

        :raises ValueError: if temporal layers exist but lack ``start_date``
          (year-based prediction requires dates).
        """
        info = self.info.reset_index(drop=True)
        temporal = info[info[RasterData.TEMPORAL_COL]]

        if (
            not temporal.empty
            and (
                "start_date" not in info.columns
                or temporal["start_date"].isna().all()
            )
        ):
            raise ValueError(
                "Prediction requires a 'start_date' on temporal layers "
                "to build per-year feature matrices; got temporal layers "
                "without dates."
            )

        if years is None:
            years = self.get_years()
        if not years:
            # static-only catalogue: one prediction unit, all covs from common.
            years = [None]

        common = info[info[RasterData.GROUP_COL] == "common"]
        covs_idx = np.zeros((len(covs_lst), len(years)), np.int32)
        for j, y in enumerate(years):
            if y is None:
                year_info = info.iloc[0:0]
            else:
                year_info = temporal[temporal["start_date"].dt.year == y]
            for i, c in enumerate(covs_lst):
                hits = year_info[year_info[RasterData.NAME_COL] == c]
                if len(hits):
                    covs_idx[i, j] = int(hits.index[0])
                else:
                    chits = common[common[RasterData.NAME_COL] == c]
                    if len(chits):
                        covs_idx[i, j] = int(chits.index[0])
                    else:
                        raise KeyError(
                            f"Covariate {c!r} not found for year {y!r} "
                            "or in the common group"
                        )
        return covs_idx, years

    @property
    def valid_pixels(self) -> np.ndarray:
        """Boolean mask over ``H*W`` of pixels with no NaN in any band."""
        return ~np.isnan(self.array.get()).any(axis=0)

    def select_valid(self) -> np.ndarray:
        """Return the ``(n_valid, n_bands)`` array of non-NaN pixel rows."""
        return self.array.get()[:, self.valid_pixels].T

    def expand_valid(self, values, nodata=np.nan) -> np.ndarray:
        """Expand a ``(n_valid,)`` / ``(n_valid, k)`` array back to ``H*W``."""
        valid = self.valid_pixels
        values = np.asarray(values)
        if values.ndim == 1:
            out = np.full(valid.shape[0], nodata, dtype=values.dtype)
            out[valid] = values
            return out
        out = np.full((valid.shape[0], values.shape[1]), nodata, dtype=values.dtype)
        out[valid] = values
        return out

    def _base_raster(self):
        for _, row in self.info.iterrows():
            path = row[RasterData.PATH_COL]
            if "http" in str(path):
                res = requests.head(path)
                if res.status_code == 200:
                    return path
            elif os.path.isfile(path):
                return path

        raise Exception("No base raster is available.")

    def to_dir(
        self,
        out_dir: Union[Path, str],
        group_expr: str = None,
        dtype: str = None,
        nodata=None,
        fit_in_dtype: bool = False,
        n_jobs: int = 4,
        return_outfiles=False,
        on_each_outfile: Callable = None,
    ):
        """Write the selected layers to a local directory as GeoTIFFs."""

        if isinstance(out_dir, str):
            out_dir = Path(out_dir)

        info = self.info
        if group_expr is not None:
            info = self.info.query(group_expr)

        if info.size == 0:
            ttprint("No rasters to save. Double check group_expr arg.")
            return self

        base_raster = self._base_raster()
        outfiles = [
            out_dir.joinpath(f"{name}.tif") for name in list(info[RasterData.NAME_COL])
        ]

        self._verbose(f"Saving rasters in {out_dir}")

        save_rasters(
            base_raster,
            outfiles,
            self.array,
            array_idx=info.index,
            window=self.window,
            overview=self.overview,
            dtype=dtype,
            nodata=nodata,
            fit_in_dtype=fit_in_dtype,
            n_jobs=n_jobs,
            on_each_outfile=on_each_outfile,
            verbose=self.verbose,
        )

        if return_outfiles:
            return outfiles
        else:
            return self

    def to_s3(
        self,
        host: Union[str, list],
        access_key: str,
        secret_key: str,
        path: str,
        secure: bool = True,
        tmp_dir: str = None,
        group_expr: str = None,
        dtype: str = None,
        nodata=None,
        fit_in_dtype: bool = False,
        n_jobs: int = None,
        verbose_cp=False,
    ):
        """Write the selected layers to an S3-compatible bucket as GeoTIFFs."""

        from minio import Minio

        bucket = path.split("/")[0]
        prefix = "/".join(path.split("/")[1:])

        if tmp_dir is None:
            tmp_dir = Path(tempfile.TemporaryDirectory().name)
            tmp_dir = tmp_dir.joinpath(prefix)

        def _to_s3(outfile) -> None:
            _host = host
            if isinstance(host, list):
                ih = int.from_bytes(str(outfile.stem).encode(), "little") % len(host)
                _host = host[ih]

            client = Minio(_host, access_key, secret_key, secure=secure)
            name = f"{outfile.name}"

            if verbose_cp:
                ttprint(f"Copying {outfile} to http://{host}/{bucket}/{prefix}/{name}")

            client.fput_object(bucket, f"{prefix}/{name}", outfile)
            os.remove(outfile)

        outfiles = self.to_dir(
            tmp_dir,
            group_expr=group_expr,
            dtype=dtype,
            nodata=nodata,
            fit_in_dtype=fit_in_dtype,
            n_jobs=n_jobs,
            return_outfiles=True,
            on_each_outfile=_to_s3,
        )

        name = outfiles[len(outfiles) - 1].name
        last_url = f"http://{host}/{bucket}/{prefix}/{name}"

        self._verbose(f"{len(outfiles)} rasters copied to s3")
        self._verbose(f"Last raster in s3: {last_url}")

        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cleanup()

    def __del__(self) -> None:
        # Avoid printing during GC; just drop the ObjectRef if present.
        try:
            self._cleanup()
        except Exception:
            pass

    def _cleanup(self):
        """Drop the ObjectRef so Ray can GC the object-store array."""
        if hasattr(self, "array"):
            try:
                del self.array
            except Exception:
                pass

    def _get_titles(self, img_title, bands):
        f_arr = self.filter(f"group=={bands}")

        if img_title == "date":
            titles = list(
                f_arr.info["start_date"].astype(str)
                + " - "
                + f_arr.info["end_date"].astype(str)
            )
        elif img_title == "index":
            titles = [str(i) for i in range(f_arr.info.shape[0])]
        elif img_title == "name":
            titles = f_arr.info["name"].to_list()
            # titles = []
            # n = 20
            # for name in list(f_arr.info['name']):
            #   titles.append('\n'.join(name[i:i+n] for i in range(0, len(name), n)))
        elif isinstance(img_title, str) and "{" in img_title:
            # template with {field} placeholders, e.g. "Land Cover {year} {name}"
            titles = []
            for _, row in f_arr.info.iterrows():
                fields = row.to_dict()
                sd, ed = fields.get("start_date"), fields.get("end_date")
                if sd is not None and ed is not None:
                    fields["date"] = f"{sd} - {ed}"
                y = fields.get("year")
                if y is not None and not pd.isna(y):
                    fields["year"] = int(y)
                titles.append(img_title.format(**fields))
        else:
            titles = [img_title] * f_arr.info.shape[0]
        return titles

    def point_query(
        self,
        x: list,
        y: list,
        cols: int = 3,
        titles: list = None,
        label_xaxis: str = "index",
        return_data: bool = False,
    ):
        """
        Makes point queries on dataset and provide plots and data

        :param x: longitude value(s) of the given point(s)
        :param y: latitude value(s) of the given point(s)
        :param cols: column count of the desired layout. Default is 3.
        :param titles: list of the titles that will be placed on top of the each graph
        :param label_xaxis: labels of the x axes. it could be `index`, `name`,`date` or None.
        :param return_data: If the user wants to access the data sampled from rasters, this
          needs to be set to True. Default is False

        Examples
        ========

        >>> import geopandas as gpd
        >>> from skmap.data import toy
        >>> rasterdata = toy.ndvi_rdata(gappy=False) #doctest: +SKIP
        >>> points = gpd.read_file('./skmap/data/toy/samples/samples.gpkg')
        >>> rasterdata.point_query(x=points.geometry.x.to_list(), y=points.geometry.y.to_list() , label_xaxis='index', cols=3, titles=points.label) #doctest: +SKIP
        """
        from matplotlib import pyplot

        df = pd.DataFrame()
        df["x"], df["y"], df["title"] = x, y, titles
        bbox = rasterio.open(self._base_raster()).bounds
        # filtering points based on the bounds of the base raster
        df = df[
            (bbox.left <= df["x"])
            & (df["x"] <= bbox.right)
            & (bbox.bottom <= df["y"])
            & (df["y"] <= bbox.top)
        ]

        with rasterio.open(self._base_raster()) as src:
            row_id, col_id = rasterio.transform.rowcol(src.transform, df.x, df.y)
            pix = row_id * src.width + col_id
        df["data"] = np.array(self.array.get()[:, pix].T).tolist()
        # if data is required no need to create figures
        if return_data:
            return df.data.to_numpy()

        labels_x = self._get_titles(label_xaxis)
        fig, axs = pyplot.subplots(
            ncols=cols,
            nrows=math.ceil(len(x) / cols),
            figsize=(6 * cols, 2 * math.ceil(len(x) / cols)),
            sharex=True,
            sharey=True,
        )
        mgc = df.shape[0]  # maximum graph count
        for i, ax in enumerate(axs.flatten()):
            if i < mgc:
                ax.plot(labels_x, df.data[i], "-o", markersize=4, color="blue", lw=1)
                ax.set_title(titles[i], fontsize=10)
                ax.tick_params(axis="x", rotation=90)
            else:
                ax.axis("off")
        pyplot.tight_layout()
        pyplot.close()
        return fig

    def _vminmax(self, vmm, arr):
        """
        To check and calculate the boundaries of the data. If the bounds are supplied
        it will return it, If not function will return the 1 and 99% of the data as bounds.
        :param vmm: supplied min/max bounds of data
        :param arr: the data will be used to generate a image
        """
        if vmm[0]:
            return vmm
        return np.nanquantile(arr.flatten(), [0.02, 0.98])

    def _op_io(self, figure):
        """
        converts figure to image and ascii representation of it to use with
        HTML embeded animation.
        :param figure: matplotlib figure object
        """
        buffer = BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        img64 = encodebytes(buffer.getvalue()).decode("ascii")
        return img64

    def _mutate_and_save(self, img, arr, titletext, textfontsize):
        """Mutate a base shot and render it to PNG in a single worker call.

        Keeps the mutated figure inside the worker process so it is never
        pickled back to the caller (unpickled matplotlib figures have
        read-only spines and fail on ``savefig``).
        """

        return self._op_io(self._mutate_baseshot(img, arr, titletext, textfontsize))

    def _percent_clip(self, arr):
        """
        To calculate and scale the band upper and lower limits to generate a composite
        image from 3 bands. returns the scaled data
        :param arr: the data usually single band data in np.array format.
        """
        return (arr - np.nanpercentile(arr, 1)) / (
            np.nanpercentile(arr, 99) - np.nanpercentile(arr, 1)
        )

    def _mutate_baseshot(self, img, arr, titletext, textfontsize):
        """
        takes imshow generated mock image copies it and replaces the nested image data.
        :param img: the mock image, generated with pyplot.imshow
        :param arr: the scaled data for the frame
        :param title_params: it is a dict. It will be used for the title generation on the frame.
        """
        c_img = deepcopy(img)
        c_img.set_data(arr)
        if titletext:
            c_img._axes.set_title(
                label=titletext, fontdict=dict(fontsize=textfontsize), pad=1
            )
        return c_img.get_figure()

    def _gen_baseshot(
        self,
        arr,
        scaling: int = 1,
        img_style: dict = None,
        cbar_props: dict = None,
        composite=False,
    ):
        from matplotlib import pyplot
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # base figure with predefined style

        # no axis labels
        tick_params = dict(left=False, labelleft=False, labelbottom=False, bottom=False)

        # scaling the figsize based on the passed array shape
        # the base figsize is 3.15 inc = 8cm almost half short side of a A4 page
        fig, ax = pyplot.subplots()
        rc, cc = arr.shape[:2]
        fig.set_size_inches(scaling * 3.15, scaling * 3.15 * rc / cc)

        # generation of basedata based on the array shape
        basedata = np.zeros(rc * cc).reshape(rc, cc)
        if composite:
            basedata = np.zeros(rc * cc * 3).reshape(rc, cc, 3)

        # crafting the base image
        ax.tick_params(**tick_params)
        ax.margins(x=0)

        if img_style:
            baseimg = ax.imshow(
                basedata, **img_style
            )  # img_style = dict(vmin=, vmax=, cmap=)
        else:
            baseimg = ax.imshow(basedata)
        # if there will be a colorbar there will be a colorbar
        if cbar_props:  # cbar_props is dict(label='text')
            div = make_axes_locatable(ax)
            pyplot.colorbar(
                baseimg,
                orientation="horizontal",
                label=cbar_props["label"],
                cax=div.append_axes("bottom", size="3%", pad=0.05),
            )
        pyplot.close()
        return baseimg

    def _band_manage(self, groups):
        """
        to structure the band(s) data based on the provided band information.
        either single or multiple band data.
        :params groups: list of band names.
        """

        h, w = self._spatial_shape

        if len(groups) == 1:  # single band raster
            arr = self.filter(f"group=={groups}").array.get()
            arr = arr.reshape(arr.shape[0], h, w).transpose(1, 2, 0)
        elif len(groups) == 3:  # composite
            arr = []
            band1 = self.filter(f"group=={groups}[0]", return_array=True)
            band2 = self.filter(f"group=={groups}[1]", return_array=True)
            band3 = self.filter(f"group=={groups}[2]", return_array=True)
            band1 = band1.reshape(band1.shape[0], h, w).transpose(1, 2, 0)
            band2 = band2.reshape(band2.shape[0], h, w).transpose(1, 2, 0)
            band3 = band3.reshape(band3.shape[0], h, w).transpose(1, 2, 0)

            alpha = np.ones(band3.shape)
            mask = np.any(np.isnan(np.stack([band1, band2, band3], axis=-1)), axis=-1)
            alpha[mask] = 0

            for i in range(band1.shape[2]):
                arr.append(
                    np.stack(
                        [
                            np.clip(self._percent_clip(band1[:, :, i]), 0, 1),
                            np.clip(self._percent_clip(band2[:, :, i]), 0, 1),
                            np.clip(self._percent_clip(band3[:, :, i]), 0, 1),
                            alpha[:, :, i],
                        ],
                        axis=2,
                    )
                )
        else:
            raise Exception("""The band count should either be one or three.
                      Current plotting capabilites are limited to single
                      or composite image generation.""")
        return arr

    def plot(
        self,
        groups: list = None,
        cmap: str = "Spectral_r",
        cbar_title: str = None,
        img_title_text: str or list = "index",
        img_title_fontsize: int = 10,
        vminmax: tuple = (None, None),
        to_img: str = None,
        dpi: int = 100,
        layout_col: int = 4,
    ):
        """
        Generates a grid plot to view and save with a colorscale with a desired layout.
        :param cmap                 : This sets the colorscale with given matplotlib.colormap. Default is Spectral_r
        :param cbar_title           : This sets the colorbar title if the cbar exists in the plot. Default is None.
        :param img_title_text       : This sets the image titles that will be display on top of the each image. Default is `index`.
        :param img_ltitle_fontsize  : This sets the fontsize of the image label which will be on top of the image. Default is 10.
        :param v_minmax             : This sets the loower and upper limits of the data that will be plot and the colorbar. Default is None and will be calculated on he fly.
        :param groups                : This used for to generate composite plot. Pass one or tree group names (groups) which will be used to generate. Default is None.
        :param to_img               : This sets the directory adn the format of the file where the generated image will be saved. Default is None.
        :param dpi                  : dot per inch value to save the figure. If the `to_img` param provided
        :param layout_col           : This controls the column count that will be used in the grid plot. Default is 3.
        """
        from matplotlib import pyplot

        if not groups:
            groups = [self.info.group.to_list()[0]]

        arr = self._band_manage(groups=groups)

        if isinstance(img_title_text, str):
            img_title_text = self._get_titles(img_title_text, groups)

        if len(groups) == 3:
            img_cnt = len(arr)
            composite = True
            baseimg = self._gen_baseshot(arr[0][:, :, 0])
        elif len(groups) == 1:
            img_cnt = arr.shape[2]
            composite = False
            vminmax = self._vminmax(vminmax, arr)
            baseimg = self._gen_baseshot(arr[:, :, 0])

        if img_cnt < layout_col:
            layout_col = img_cnt

        layout_row = math.ceil(img_cnt / layout_col)

        set_h = baseimg.get_size()[0] / baseimg.get_figure()._dpi
        set_w = baseimg.get_size()[1] / baseimg.get_figure()._dpi
        if set_w > set_h:
            set_w = set_w * 3.15 / set_h
            set_h = 3.15
        else:
            set_h = set_h * 3.15 / set_w
            set_w = 3.15
        grd_fig, grd_axs = pyplot.subplots(
            nrows=layout_row,
            ncols=layout_col,
            gridspec_kw=dict(wspace=0.1, hspace=0.1),
            figsize=(
                set_w * layout_col + (layout_col - 1) * 0.1,
                set_h * layout_row + (layout_row - 1) * 0.1,  # + 1
            ),
        )

        def _preprocess(arr_, ind, composite):
            if composite:
                return np.flipud(arr_[ind])
            else:
                return np.flipud(arr_[:, :, ind])

        matrix_params = dict(vmin=vminmax[0], vmax=vminmax[1], cmap=cmap)

        def gen_pane(
            ind, arr, ax, composite, matrix_params, img_title_text, img_title_fontsize
        ) -> None:
            try:
                ax.pcolorfast(
                    _preprocess(arr, ind, composite=composite), **matrix_params
                )
                ax.set_title(img_title_text[ind], fontsize=img_title_fontsize, pad=1)
                ax.tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )
            except IndexError:
                ax.axis("off")

        try:
            for i, ax in enumerate(grd_axs.flatten()):
                gen_pane(
                    i,
                    arr,
                    ax,
                    composite,
                    matrix_params,
                    img_title_text,
                    img_title_fontsize,
                )

        except AttributeError:
            gen_pane(
                0,
                arr,
                grd_axs,
                composite,
                matrix_params,
                img_title_text,
                img_title_fontsize,
            )
        pyplot.close()

        if not composite:
            grd_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            w, h = grd_fig.get_size_inches()
            cbar_ax = grd_fig.add_axes([0.1, 1 + (3.15 / h) * 0.1, 0.8, 0.16 / h])
            cbar_ax = grd_fig.colorbar(
                pyplot.imshow(
                    arr[:, :, 0], vmin=vminmax[0], vmax=vminmax[1], cmap=cmap
                ),
                orientation="horizontal",
                cax=cbar_ax,
                ticklocation="top",
            ).set_label(label=cbar_title)
            pyplot.tight_layout()
            pyplot.close()

        if to_img:
            grd_fig.savefig(
                to_img, format=f"{to_img.split('.')[-1]}", dpi=dpi, bbox_inches="tight"
            )
        return grd_fig

    def _idx_offset(self):
        return self.info.index.max() + 1

    def animate(
        self,
        cmap: str = "Spectral_r",
        groups: list = None,
        scaling: float = 2,
        cbar_title: str = None,
        img_title_text: str or list = "index",
        img_title_fontsize: int = 10,
        vminmax: tuple = (None, None),
        interval: int = 250,
        to_gif: str = None,
        n_jobs: int = 4,
    ):
        """
        Generates an animation with the given band(s) and saves it.

        :param cmap: colormap name that will derived from ``matplotlib.colormaps()``
        :param groups: used to select the band(s) or to generate a composite
            image that will be used as animation frame. Default is ``None``,
            which selects the first band on ``RasterData``.
        :param scaling: scaling can be used to increase/decrease the frame size.
            Default is 2.
        :param cbar_title: title for the colorbar.
        :param img_title_text: title text for each frame, or ``"index"``.
        :param img_title_fontsize: font size for the image title.
        :param vminmax: ``(vmin, vmax)`` tuple for clipping.
        :param interval: frame interval in milliseconds.
        :param to_gif: path to save a GIF if not ``None``.
        :param n_jobs: number of parallel jobs.
        """

        from IPython.display import HTML
        from matplotlib._animation_data import DISPLAY_TEMPLATE, JS_INCLUDE, STYLE_INCLUDE
        from PIL import Image

        if not groups:
            groups = [self.info.group.to_list()[0]]
        arr = self._band_manage(groups=groups)

        if isinstance(img_title_text, str):
            img_title_text = self._get_titles(img_title_text, groups)

        if len(groups) == 3:
            img_cnt = len(arr)
            baseimg = self._gen_baseshot(
                arr=arr[0][:, :, 0], scaling=scaling, composite=True
            )
            args = [
                (baseimg, arr[i], img_title_text[i], img_title_fontsize)
                for i in range(img_cnt)
            ]
        elif len(groups) == 1:
            img_cnt = arr.shape[2]
            vminmax = self._vminmax(vminmax, arr)
            baseimg = self._gen_baseshot(
                arr=arr[:, :, 0],
                scaling=scaling,
                img_style=dict(vmin=vminmax[0], vmax=vminmax[1], cmap=cmap),
                cbar_props=dict(label=cbar_title),
                composite=False,
            )
            args = [
                (baseimg, arr[:, :, i], img_title_text[i], img_title_fontsize)
                for i in range(img_cnt)
            ]

        int_img = [
            j
            for j in parallel.job(
                self._mutate_and_save, args, n_jobs=n_jobs
            )
        ]

        if to_gif is not None:
            with ExitStack() as stack:
                imgs = (
                    stack.enter_context(Image.open(BytesIO(b64decode(img))))
                    for img in int_img
                )
                img = next(imgs)
                img.save(
                    to_gif,
                    format="GIF",
                    append_images=imgs,
                    save_all=True,
                    duration=interval,
                    loop=0,
                )

        template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
        embedded_frames = "\n" + "".join(
            template.format(i, "png", imgdata.replace("\n", "\\\n"))
            for i, imgdata in enumerate(int_img)
        )
        mode_dict = dict(once_checked="", loop_checked="checked", reflect_checked="")

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "temp.html")
            with open(path, "w") as of:
                of.write(JS_INCLUDE + STYLE_INCLUDE)
                of.write(
                    DISPLAY_TEMPLATE.format(
                        id=uuid4().hex,
                        Nframes=img_cnt,
                        fill_frames=embedded_frames,
                        interval=interval,
                        **mode_dict,
                    )
                )
            html_rep = path.read_text()
        return HTML(html_rep)
