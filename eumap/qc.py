'''
Dataset quality control utilities
'''


from typing import Iterable, Union
import warnings
from pathlib import Path
import rasterio as rio
from shapely import geometry as g
import geopandas as gp
import requests
import numpy as np
from operator import add
from functools import reduce

from .parallel import blocks
from .datasets.catalogue import _Resource

_LANDMASK_REF = 'https://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_land.mask_pflugmacher2019.landcover.12_f_30m_s0..0m_2014..2016_skmap_epsg3035_v0.1.tif'

def _test_field_nonempty(val):
    if isinstance(val, str):
        val = val.strip()
    return val not in ['', []]

class Test:
    """
    Class for performing QC against GH datasets.

    :param bounds:      Iterable of bounding coordinates to check quality within, if ``None`` the bounds of the raster will be used
    :param verbose:     Verbosity of the checks
    :param crs:         CRS of the bounding coordinates (defaults to the raster coordinates)

    Examples
    ========

    >>> from skmap.qc import Test
    >>>
    >>> bounds = (4751935.0, 2420238.0, 4772117.0, 2444223.0)
    >>> test = Test(bounds, verbose=True)
    >>> dataset_url = 'https://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_landcover.hcl_lucas.corine.rf_p_30m_0..0cm_2019_skmap_epsg3035_v0.1.tif'
    >>>
    >>> available = test.availability(dataset_url)
    >>> coverage_fraction = test.raster_land_coverage(dataset_url, include_ice=True, include_wetlands=True)

    """

    def __init__(self,
        bounds: Iterable,
        crs: bool=None,
        verbose: bool=False,
    ):
        self.bounds = bounds
        self.verbose = verbose
        self.crs = crs

    def accessibility(self,
        dataset_url: str,
    ) -> bool:
        """
        Check if a remote resource is available.

        :param dataset_url:     URL of remote resource

        :returns:               True if dataset is accessible, otherwise False
        """

        resp = requests.get(dataset_url, stream=True)
        result = resp.status_code < 400
        del resp

        if self.verbose:
            if result:
                acc = 'accessible'
            else:
                acc = 'inaccessible'
            print(
                f'Dataset {acc}:',
                dataset_url,
                sep='\n',
            )

        return result

    def raster_land_coverage(self,
        dataset_path: Union[str, Path],
        include_ice: bool=False,
        include_wetlands: bool=False,
    ) -> float:
        """
        Check completeness of a remote raster resource against land mask derived from [1].

        :param dataset_path:        Path or URL to raster dataset
        :param include_ice:        Include snow and ice in landmask
        :param include_wetlands:    Include wetlands in landmask

        :returns: Fraction of pixels which are not nodata accross the landmask

        References
        ==========

        [1] `Pan-European land cover map 2015 (Pflugmacher et al., 2019) <https://doi.pangaea.de/10.1594/PANGAEA.896282>`_

        """

        with rio.open(dataset_path) as src:
            nodata = src.nodata

        def _get_counts(landmask, data):
            idx_landmask = landmask < 10
            if include_ice:
                idx_landmask = idx_landmask | (landmask == 12)
            if include_wetlands:
                idx_landmask = idx_landmask | (landmask == 11)

            idx_data = (data != nodata) & idx_landmask

            return np.array([
                idx_data.sum(),
                idx_landmask.sum(),
            ])


        reader = blocks.RasterBlockReader(_LANDMASK_REF)
        agg = blocks.RasterBlockAggregator(reader)

        _crs = self.crs
        if _crs is None:
            _crs = reader.reference.crs

        _gdf = gp.GeoDataFrame(
            geometry=[g.box(*self.bounds)],
            crs=_crs,
        )

        if _gdf.crs != reader.reference.crs:
            _gdf = _gdf.to_crs(reader.reference.crs)

        _geom = g.mapping(_gdf.geometry[0])

        counts = agg.aggregate(
            [_LANDMASK_REF, dataset_path], _geom,
            block_func=_get_counts,
            agg_func=lambda results: reduce(add, results),
        )

        result = counts[0] / counts[1]

        if self.verbose:
            print(
                f'Completeness {result*100}% for dataset:',
                dataset_path,
                sep='\n',
            )

        return result
    
    def metadata_consistency(self,
        resource:_Resource,
    ) -> dict:
        META_KEYS = (
            'title',
            'abstract',
            'theme',
            'authors',
        )
        result = []
        
        for meta_key in META_KEYS:
            try:
                meta_val = resource.meta[meta_key]
                assert _test_field_nonempty(meta_val)
                result.append(True)
            except (KeyError, AssertionError):
                result.append(False)
                if self.verbose:
                    print('Missing metadata:', meta_key)
        
        if self.verbose:
            print('All metadata present:', all(result))
            
        return dict(zip(META_KEYS, result))