from typing import List, Union
from dateutil.relativedelta import relativedelta

import requests
import numpy as np
import bottleneck as bc
import joblib
from pathlib import Path
import gc
import os

import warnings
from skmap import parallel
from skmap.raster import read_auth_rasters, save_rasters
from skmap.misc import _warn_deps, _eval, nan_percentile, ttprint, find_files, GoogleSheet

class GLADLandsat():
  """
    
    Automation to download and process Landsat ARD images provided GLAD [1,2]. 

    :param username: Username to access the files [3].
    :param password: Password to access the files [3].
    :param parallel_download: Number of files to download in parallel.
    :param filter_additional_qa: Use ``True`` to remove pixels flagged as additional 
      cloud (``11 to 17`` - table 3 in [2]).
    :param verbose: Use ``True`` to print the progress of all steps.

    References
    ==========

    [1] `User Manual - GLAD Landsat ARD <https://glad.geog.umd.edu/ard/user-manual>`_

    [2] `Remote Sensing paper (Potapov, et al, 2020) <https://doi.org/10.3390/rs12030426>`_

    [3] `User Registration - GLAD Landsat ARD <https://glad.geog.umd.edu/ard/user-registration>`_

  """
  def __init__(self,
    username:str,
    password:str,
    parallel_download:int = 4,
    filter_additional_qa:bool = True,
    verbose:bool = True,
  ):
  
    self.username = username
    self.password = password
    self.base_url = 'https://glad.umd.edu/dataset/landsat_v1.1'

    self.parallel_download = parallel_download
    self.verbose = verbose
    self.filter_additional_qa = filter_additional_qa

    self.bqa_idx = 7

    self.max_spectral_val = 40000

    super().__init__()

  def _glad_id(self, year_interval_id):
    year, interval_id = year_interval_id.split('-')
    return (int(year) - 1980) * 23 + int(interval_id)

  def _glad_urls(self, tile, start, end):
  
    i0 = self._glad_id(start)
    i1 = self._glad_id(end)

    urls = []
    lat = tile.split('_')[1]

    for i in range(i0, (i1+1)):
      urls.append(f'{self.base_url}/{lat}/{tile}/{i}.tif')

    return urls

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)

  def read(self, 
    tile:str, 
    start:str, 
    end:str, 
    clear_sky:bool = True, 
    min_clear_sky:float = 0.2
  ):
    """
    
    Download and read multiple dates for a specific tile. The images are
    read on-the-fly without save the download files in disk.

    :param tile: GLAD tile id to be processed according to [1].
    :param start: Start date format (``{YEAR}-{INTERVAL_ID}`` - ``2000-1``). 
      The ``INTERVAL_ID`` ranges from 1 to 23 according to [2].
    :param end: End date format (``{YEAR}-{INTERVAL_ID}`` - ``2000-1``).
      The ``INTERVAL_ID`` ranges from 1 to 23 according to [2].
    :param clear_sky: Use ``True`` to keep only the clear sky pixels, which are
      ``1, 2, 5 and 6`` according to quality flag band (Table 3 in [3]). For
      ``filter_additional_qa=False`` the pixels ``11 to 17`` are also considered
      clear sky.
    :param min_clear_sky: Minimum percentage of clear sky pixels in the tile to
      keep a specific data, otherwise remove it from the result.

    :returns: The read data, the accessed URLs and the Path for a empty base raster.
    :rtype: Tuple[Numpy.array, List[str], Path]
    
    Examples
    ========

    >>> from skmap.datasets.eo import GLADLandsat
    >>> 
    >>> # Do the registration in
    >>> # https://glad.umd.edu/ard/user-registration
    >>> username = '<YOUR_USERNAME>'
    >>> password = '<YOUR_PASSWORD>'
    >>> glad_landsat = GLADLandsat(username, password, verbose=True)
    >>> data, urls, base_raster = glad_landsat.read('092W_47N', '2020-6', '2020-10')
    >>> print(f'Data shape: {data.shape}')

    References
    ==========

    [1] `File glad_ard_tiles.shp (GLAD Landsat ARD Tools V1.1) <https://glad.geog.umd.edu/ard/software-download>`_
    
    [2] `File 16d_intervals.xlsx (GLAD Landsat ARD Tools V1.1) <https://glad.geog.umd.edu/ard/software-download>`_

    [3] `Remote Sensing paper (Potapov, et al, 2020) <https://doi.org/10.3390/rs12030426>`_

    """

    url_list = self._glad_urls(tile, start, end)

    data, base_raster = read_auth_rasters(
      raster_files = url_list,
      username = self.username,
      password = self.password,
      verbose = self.verbose,
      return_base_raster = True,
      nodata = 0, 
      n_jobs = self.parallel_download
    )

    if (isinstance(data, np.ndarray)):
      self._verbose(f'Read data {data.shape}.')

      if clear_sky:
        self._verbose(f'Removing cloud and cloud shadow pixels.')
        clear_sky_mask, clear_sky_pct = self._clear_sky_mask(data)
        data[~clear_sky_mask] = np.nan
        clear_sky_idx = np.where(clear_sky_pct >= min_clear_sky)[0]
        data = data[:,:,:,clear_sky_idx]

        if len(clear_sky_idx) == 0:
            url_list = []

    return data, url_list, base_raster

  def _clear_sky_mask(self, data):

    qa_data = data[self.bqa_idx,:,:,:]

    if(np.nanmax(qa_data) > 255):
      qa_data /= 100

    clear_sky1 = np.logical_and(qa_data >= 1, qa_data <= 2)
    clear_sky2 = np.logical_and(qa_data >= 5, qa_data <= 6)
    clear_sky_mask = np.logical_or(clear_sky1, clear_sky2)

    if not self.filter_additional_qa:
      clear_sky3 = np.logical_and(qa_data >= 11, qa_data <= 17) 
      clear_sky_mask = np.logical_or(clear_sky_mask, clear_sky3)
    
    clear_sky_mask = np.broadcast_to(clear_sky_mask, shape=data.shape)

    nrow, ncols = data.shape[1], data.shape[2]
    clear_sky_pct = []

    for i in range(0, data.shape[3]):
      clear_sky_count = bc.nansum(clear_sky_mask[0,:,:,i].astype(np.int))
      clear_sky_pct.append(clear_sky_count / (nrow * ncols))

    return clear_sky_mask, np.array(clear_sky_pct)

  def _save_percentile_agg(self, result, base_raster, tile, start, p, output_dir, unit8):
    
    if not isinstance(output_dir, Path):
      output_dir = Path(output_dir)

    output_files = []

    fn_raster_list = []
    data = []

    for i in range(0, result.shape[0]):
      
      band_data = result[i,:,:,:]
      dtype = 'uint16'

      if unit8:
        scale, conv = self.max_spectral_val, 255
        
        valid_mask = ~np.isnan(band_data)
        band_data[valid_mask] = ((band_data[valid_mask] / scale) * conv)
        band_data[valid_mask] = np.floor(band_data[valid_mask])

        dtype = 'uint8'
      
      for paux in p:
        p_dir = output_dir.joinpath(f'{start}').joinpath(f'B{i+1}_P{int(paux)}')
        fn_raster_list.append(p_dir.joinpath(f'{tile}.tif'))
      
      data.append(band_data)
    
    data = np.concatenate(data, axis=2)
    
    output_files += save_rasters(fn_base_raster=base_raster, data=data, fn_raster_list=fn_raster_list, 
      dtype=dtype, nodata=0, fit_in_dtype=True, n_jobs=self.parallel_download, verbose=self.verbose)

    return output_files

  def _calc_save_count(self, data, max_val, base_raster, tile, start, output_dir):
    band_data = data[0,:,:,:]
    band_data = band_data.copy()
    band_valid_mask = np.logical_and(band_data >= 1.0, band_data <= max_val)
    band_data[~band_valid_mask] = np.nan

    band_count = bc.nansum(np.logical_not(np.isnan(band_data)).astype('uint8'), axis=2)
    band_count = np.stack([band_count], axis=2)

    if not isinstance(output_dir, Path):
      output_dir = Path(output_dir)

    fn_raster = output_dir.joinpath(f'{start}') \
                      .joinpath(f'B1_COUNT') \
                      .joinpath(f'{tile}.tif')

    dtype = 'uint8'
    save_rasters(fn_base_raster=base_raster, data=band_count, fn_raster_list=[ fn_raster ], 
      dtype=dtype, nodata=0, n_jobs=1, verbose=self.verbose)

    return fn_raster

  def percentile_agg(self, 
    tile:str, 
    start:str, 
    end:str, 
    p:List, 
    clear_sky:bool = True, 
    min_clear_sky:bool = 0.2, 
    n_jobs = 7, 
    output_dir:Path = None, 
    unit8:bool = True
  ):
    """
    
    Download, read and aggregate multiple dates in different percentiles.

    :param tile: GLAD tile id to be processed according to [1].
    :param start: Start date format (``{YEAR}-{INTERVAL_ID}`` - ``2000-1``). 
      The ``INTERVAL_ID`` ranges from 1 to 23 according to [2].
    :param end: End date format (``{YEAR}-{INTERVAL_ID}`` - ``2000-1``).
      The ``INTERVAL_ID`` ranges from 1 to 23 according to [2].
    :param p: A list with the percentiles values between 0 and 100.
    :param clear_sky: Use ``True`` to keep only the clear sky pixels, which are
      ``1, 2, 5 and 6`` according to quality flag band (Table 3 in [3]). For
      ``filter_additional_qa=False`` the pixels ``11 to 17`` are also considered
      clear sky.
    :param min_clear_sky: Minimum percentage of clear sky pixels in the tile to
      keep a specific data, otherwise remove it from the result.
    :param n_jobs: Number of jobs to process the spectral bands in parallel. More
      then ``7`` is meaningless and don't improve the performance.
    :param output_dir: If provided save the result to this folder. By default is 
      ``None`` and no files are saved to disk.
    :param unit8: Use ``True`` to convert the read data to ``unit8``.

    :returns: The read data, the Path for a empty base raster and the saved 
      files (only if ``output_dir=True``).
    :rtype: Tuple[Numpy.array, List[str], Path]

    Examples
    ========
    
    >>> from skmap.datasets.eo import GLADLandsat
    >>> 
    >>> # Do the registration here
    >>> # https://glad.umd.edu/ard/user-registration
    >>> username = '<YOUR_USERNAME>'
    >>> password = '<YOUR_PASSWORD>'
    >>> glad_landsat = GLADLandsat(username, password, verbose=True)
    >>> data, base_raster, _ = glad_landsat.percentile_agg('092W_47N', '2020-6', '2020-10', 
    >>>                         p=[25,50,75], output_dir='./glad_landsat_ard_percentiles')
    >>> print(f'Shape of data: {data.shape}')

    References
    ==========
    
    [1] `File glad_ard_tiles.shp (GLAD Landsat ARD Tools V1.1) <https://glad.geog.umd.edu/ard/software-download>`_
    
    [2] `File 16d_intervals.xlsx (GLAD Landsat ARD Tools V1.1) <https://glad.geog.umd.edu/ard/software-download>`_

    [3] `Remote Sensing paper (Potapov, et al, 2020) <https://doi.org/10.3390/rs12030426>`_

    """

    def _run(band_data, p, i, max_val):
      if (i == 6):
        max_val =  65000

      band_data = band_data.copy()
      band_valid_mask = np.logical_and(band_data >= 1.0, band_data <= max_val)
      band_data[~band_valid_mask] = np.nan

      perc_data = nan_percentile(band_data.transpose(2,0,1), p)
      perc_data = np.stack(perc_data, axis=0)

      return (perc_data, i)

    data, url_list, base_raster = self.read(tile, start, end, clear_sky, min_clear_sky)

    if len(url_list) > 0:
      self._verbose(f'Aggregating by {len(p)} percentiles.')
      
      args = []
      
      for i in range(0, self.bqa_idx):
        args.append( (data[i,:,:,:], p, i, self.max_spectral_val) )

      result = {}
      
      for perc_data, i in parallel.job(_run, args, n_jobs=n_jobs):
        result[i] = perc_data

      result = [result[i] for i in range(0,len(args))]
      if len(result) > 0:
        result = np.stack(result, axis=3).transpose(3,1,2,0)

        output_files = []
        if output_dir is not None:
          self._verbose(f'Saving the result in {output_dir}.')
          output_files = self._save_percentile_agg(result, base_raster, tile, start, p, output_dir, unit8)

        self._verbose(f'Counting clear_sky pixels')
        output_file_count = self._calc_save_count(data, self.max_spectral_val, base_raster, tile, start, output_dir)
        output_files.append(output_file_count)

        del data
        gc.collect()

        self._verbose(f'Removing {base_raster}')
        os.remove(base_raster)

        return result, base_raster, output_files
    
    else:
      raise RuntimeError('All the images were read, but no clear_sky data was found.')

try:
  
  import pystac
  import rasterio
  import mimetypes
  import pandas as pd
  import matplotlib.pyplot as plt
  from minio import Minio
  from PIL import Image
  from itertools import chain
  from datetime import datetime
  from pyproj import Transformer
  from matplotlib.colors import ListedColormap
  from shapely.geometry import Polygon, mapping, shape
  from pystac.extensions.item_assets import ItemAssetsExtension

  class STACGenerator():
    """
    
    Generator able to access a remote Google Spreadsheet [1] containing several
    raster layer metadata (e.g. name, description, cloud-optimized GeoTIFF URL)
    and produce multiple SpatioTemporal Asset Catalogs (STAC) instances in a 
    local folder and / or remote S3 bucket [2,3].

    The COG files need to be publicly accessible to HTTP and compatible the Geo-harmonizer 
    file naming convention [4]. The thumbnails are produced for every COG 
    according to color scheme defined by columns ``thumb_cmap``, ``thumb_vmin``, ``thumb_vmax``. 

    :param gsheet: Object representation of a Google Spreadsheet containing the metadata.
    :param url_date_format: Date format expected in the COG URL (``strftime``).
    :param cog_level: COG overview level used to generate the thumbnail.
    :param thumb_overwrite: Overwrite the thumbnail files if exists.
    :param asset_id_delim: Field delimiter used to split the COG filename [4].
    :param asset_id_fields: Fields retrieved from COG filename used to compose the asset id.
    :param catalogs: Used to pass a dictionary (``catalog_id`` as key and 
      ``pystac.catalog.Catalog`` as value) for update operation in pre-existing catalogs.
    :param verbose: Use ``True`` to print the progress of all steps.


    References
    ==========

    [1] `ODSE Raster layer metadata example <https://docs.google.com/spreadsheets/d/10tAhEpZ7TYPD0UWhrI0LHcuIzGZNt5AgSjx2Bu-FciU>`_  

    [2] `ODSE STAC Catalog <https://s3.eu-central-1.wasabisys.com/stac/odse/catalog.json>`_

    [3] `ODSE STAC Browser <http://stac.opendatascience.eu>`_

    [4] `Geo-harmonizer file naming convention <https://gitlab.com/geoharmonizer_inea/spatial-layers>`_

  """
    def __init__(self,
      gsheet:GoogleSheet,
      url_date_format = '%Y.%m.%d',
      cog_level = 7,
      thumb_overwrite = False,
      asset_id_delim = '_',
      asset_id_fields = [1,3,5],
      catalogs = None,
      verbose = False
    ):

      self.cog_level = cog_level
      self.thumb_overwrite = thumb_overwrite
      self.gsheet = gsheet
      self.url_date_format = url_date_format
      self.verbose = verbose
      self.asset_id_delim = asset_id_delim
      self.asset_id_fields = asset_id_fields

      coll_columns = gsheet.collections.columns
      self.additional_url_cols = list(coll_columns[pd.Series(coll_columns).str.startswith('url')])

      self.gdal_env = {
        "GDAL_CACHEMAX": 1024_000_000,  # 1 Gb in bytes
        "GDAL_DISABLE_READDIR_ON_OPEN": False,
        "VSI_CACHE": True,
        "VSI_CACHE_SIZE": 1024_000_000,
        "GDAL_HTTP_MAX_RETRY": 3,
        "GDAL_HTTP_RETRY_DELAY": 60
      }

      self.fields = {
        'collection': ['id', 'title', 'description', 'license','keywords'],
        'provider': ['name', 'description', 'roles', 'url'],
        'catalog': ['id', 'title', 'description'],
        'common_metadata': ['constellation', 'platform', 'instruments', 'gsd'],
        'internal': ['start_date', 'end_date', 'date_step', 'date_unit', 'date_style','catalog', 'providers', 'main_url']
      }

      self.fields['internal'] += self.additional_url_cols

      self.fields_lookup = dict.fromkeys(
        set(chain(*[self.fields[key] for key in self.fields.keys()]))
        , True
      )

      self.providers = self._providers()
      if catalogs is None:
        self.catalogs = self._catalogs()
      else:
        self.catalogs = catalogs
      self._populate()

    def _verbose(self, *args, **kwargs):
      if self.verbose:
        ttprint(*args, **kwargs)

    def _providers(self):
      providers = {}
      for i,p in self.gsheet.providers.iterrows():
        providers[p['name']] = pystac.Provider(**self._kargs(p, 'provider'))
      return providers;

    def _catalogs(self):
      catalogs = {}
      for i,p in self.gsheet.catalogs.iterrows():
        catalogs[p['id']] = pystac.Catalog(**self._kargs(p, 'catalog'))
      return catalogs;

    def _fetch_collection(self, key, i, row, bbox_footprint_results):

      items = []
      for start_date, end_date in self._gen_dates(**row):
        main_url = self._parse_url(row['main_url'], start_date, end_date, row['date_unit'], row['date_style'])
        additional_urls = []
        for ac_url in self.additional_url_cols:
          if row[ac_url]:
            additional_urls.append(self._parse_url(row[ac_url], start_date, end_date, row['date_unit'], row['date_style']))
        
        bbox, footprint = bbox_footprint_results[main_url]

        items.append(self._new_item(row, start_date, end_date, main_url, bbox, footprint, additional_urls))

      return (key, row, items)

    def _populate(self):

      self.new_collections = {}
      groups = self.gsheet.collections.groupby('catalog')
      
      args = []
      for key in groups.groups.keys():
        for i, row in groups.get_group(key).iterrows():
          for start_date, end_date in self._gen_dates(**row):
            main_url = self._parse_url(row['main_url'], start_date, end_date, row['date_unit'], row['date_style'])
            args.append((main_url,))

      bbox_footprint_results = {}
      for url, bbox, footprint in parallel.job(self._bbox_and_footprint, args, n_jobs=-1, joblib_args={'backend': 'multiprocessing'}):
        bbox_footprint_results[url] = (bbox, footprint)

      collection_args = []
      for key in groups.groups.keys():
        for i, row in groups.get_group(key).iterrows():
          collection_args.append((key, i, row, bbox_footprint_results))
          
      for key, row, items in parallel.job(self._fetch_collection, collection_args, n_jobs=-1, joblib_args={'backend': 'multiprocessing'}):
        collection = self._new_collection(row, items)
        if collection is None:
          self._verbose(f"Faile to create the collection {row['id']}")
        else:

          item = self.catalogs[key].get_child(collection.id)
          if item is not None:
            self.catalogs[key].remove_child(collection.id)

          self.catalogs[key].add_child(collection)
          if key not in self.new_collections:
            self.new_collections[key] = []
          self.new_collections[key].append(collection)

          self._verbose(f"Creating collection {collection.id} with {len(items)}")

    def _get_val(self, dicty, key):
      if key in dicty:
        return dicty[key]
      else:
        return None

    def _is_data(self, asset):
      for r in asset.roles:
        if r == 'data':
          return True
      return False

    def _generate_thumbs(self, output_dir='./stac', thumb_base_url=None):

      for key, colls in self.new_collections.items():
        
        args = []
        catalog = self.catalogs[key]

        for coll in colls: 
          for item in coll.get_all_items():
            
            assets = [ a for a in item.assets.items() ]
            collection = item.get_collection()
            cmap = self._get_val(collection.extra_fields, 'thumb_cmap')
            vmin = self._get_val(collection.extra_fields, 'thumb_vmin')
            vmax = self._get_val(collection.extra_fields, 'thumb_vmax')
            
            if ',' in cmap:
              colors = cmap.split(',')
              cmap = ListedColormap(colors)


            for key, asset in assets:
              if self._is_data(asset):
                thumb_fn = Path(output_dir) \
                  .joinpath(catalog.id) \
                  .joinpath(item.collection_id) \
                  .joinpath(item.id) \
                  .joinpath(Path(asset.href).stem + '.png')
                
                thumb_fn.parent.mkdir(parents=True, exist_ok=True)
                if 'main' in asset.extra_fields:
                  args.append((asset.href, str(thumb_fn), item.id, True, cmap, vmin, vmax))
                else:
                  args.append((asset.href, str(thumb_fn), item.id, False))

        self._verbose(f"Generating {len(args)} thumbnails for catalog {catalog.id}")

        for thumb_fn, item_id, is_thumb_url in parallel.job(self._thumbnail, args, n_jobs=-1):
          item = catalog.get_item(item_id, True)
          
          if thumb_fn is not None:
            thumd_id = self._asset_id(thumb_fn, self.asset_id_delim, self.asset_id_fields) + '_preview'
            roles = []
            if is_thumb_url:
              thumd_id = 'thumbnail'
              roles = ['thumbnail']

            if thumb_base_url is not None:
              thumb_fn = str(thumb_fn).replace(str(output_dir), thumb_base_url)

            item.add_asset(thumd_id, pystac.Asset(href=thumb_fn, media_type=pystac.MediaType.PNG, roles=roles))
              
    def _thumbnail(self, url, thumb_fn, item_id, is_thumb_url=False, cmap = 'binary', vmin = None, vmax = None):
      
      if not self.thumb_overwrite and Path(thumb_fn).exists():
        #self._verbose(f'Skipping {thumb_fn}')
        return (thumb_fn, item_id, is_thumb_url)

      r = requests.head(url)
      if not (r.status_code == 200):
        return(None, item_id, is_thumb_url)

      with rasterio.open(url) as src:
        try:
          oviews = src.overviews(1)
          if len(oviews) == 0:
            return(None, item_id, is_thumb_url)

          cog_level = self.cog_level
          if cog_level >= len(oviews):
            cog_level = -1
          
          oview = oviews[cog_level]

          result = src.read(1, out_shape=(1, src.height // oview, src.width // oview)).astype('float32')
          result[result==src.nodata] = np.nan

          if vmin is None:
            perc = np.nanpercentile(result,[8,92])
            vmin, vmax = perc[0], perc[1]

          fig, ax = plt.subplots(figsize=(1, 1))

          ax.axis('off')
          plt.imsave(thumb_fn, result, cmap=cmap, vmin=vmin, vmax=vmax)

          basewidth = 675
          img = Image.open(thumb_fn)
          wpercent = (basewidth/float(img.size[0]))
          hsize = int((float(img.size[1])*float(wpercent)))
          img = img.resize((basewidth,hsize), Image.ANTIALIAS)
          img.save(thumb_fn)

          return (thumb_fn, item_id, is_thumb_url)
        except:
          return(None, item_id, is_thumb_url)

    def _new_collection(self, row, items):  
      
      if len(items) > 0:
        unioned_footprint = shape(items[0].geometry)
        collection_bbox = list(unioned_footprint.bounds)
        
        start_date = items[0].properties['start_datetime']
        end_date = items[-1].properties['end_datetime']

        collection_interval = sorted([
          datetime.strptime(start_date,"%Y-%m-%d"), 
          datetime.strptime(end_date,"%Y-%m-%d")
        ])

        collection = pystac.Collection(
          extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[collection_bbox]), 
            temporal=pystac.TemporalExtent(intervals=[collection_interval])
          ),
          providers=[ self.providers[p] for p in row['providers'] ],
          stac_extensions=['https://stac-extensions.github.io/item-assets/v1.0.0/schema.json'],
          **self._kargs(row, 'collection', True)
        )

        #itemasset_ext = ItemAssetsExtension.ext(collection)
        #print(itemasset_ext.item_assets)

        collection.add_items(items)
      
        return collection
      else:
        return None

    def _asset_id(self, url, delim = None, id_fields = None):
      
      if delim is None:
        return Path(url).name

      fields = Path(url).name.split(delim)
      result = []
      for f in id_fields:
        if f < len(fields):
          result.append(fields[f])

      return delim.join(result)

    def _new_item(self, row, start_date, end_date, main_url, bbox, footprint, additional_urls = []):
      
      start_date_str = start_date.strftime("%Y-%m-%d")
      end_date_str = end_date.strftime("%Y-%m-%d")

      start_date_url_str = start_date.strftime(self.url_date_format)
      end_date_url_str = end_date.strftime(self.url_date_format)

      item_id = f'{row["id"]}_{start_date_url_str}..{end_date_url_str}'

      item = pystac.Item(id=item_id,
                      geometry=footprint,
                      bbox=bbox,
                      datetime=start_date,
                      properties={'start_datetime': start_date_str, 'end_datetime': end_date_str},
                      stac_extensions=["https://stac-extensions.github.io/eo/v1.0.0/schema.json"])

      item.common_metadata.gsd = row['gsd']
      item.common_metadata.instruments = row['instruments']
      
      if 'platform' in row and row['platform']:
        item.common_metadata.platform = row['platform']
      if 'constellation' in row:
        item.common_metadata.constellation = row['constellation']

      #eo_ext = EOExtension.ext(item)
      #eo_ext.apply(bands=[
      #Band.create(name='EVI', description='Enhanced vegetation index', common_name='evi')
      #])

      item.add_asset(self._asset_id(main_url, self.asset_id_delim, self.asset_id_fields), \
        pystac.Asset(href=main_url, media_type=pystac.MediaType.GEOTIFF, roles=['data'], extra_fields={'main': True}))
      for aurl in additional_urls:
        item.add_asset(self._asset_id(aurl, self.asset_id_delim, self.asset_id_fields), \
          pystac.Asset(href=aurl, media_type=pystac.MediaType.GEOTIFF, roles=['data']))

      return item

    def _gen_dates(self, start_date, end_date, date_unit, date_step, ignore_29feb, **kwargs):

      result = []
      
      if date_unit == 'static': 
        result.append((
          start_date, 
          end_date
        ))
      elif date_unit == 'custom_multiannual': 
      ## Irregular/custom date iteration
        
        for year_range in date_step.split(','):
          year_range_arr = year_range.split('..')
          start_year = year_range_arr[0]
          end_year = year_range_arr[1]

          result.append((
            datetime.strptime(f'{start_year}.01.01', self.url_date_format),
            datetime.strptime(f'{end_year}.12.31', self.url_date_format)
          ))

      elif date_unit == 'custom_predefined': 
      ## Irregular/custom date iteration
        
        for dt_range in date_step.split(','):
          dt_range_arr = dt_range.split('..')
          start_year = dt_range_arr[0]
          end_year = dt_range_arr[1]
          result.append((
            datetime.strptime(f'{start_year}', self.url_date_format),
            datetime.strptime(f'{end_year}', self.url_date_format)
          ))

      else:

        dt1 = start_date
        while(dt1 <= end_date):
        ## Regular date iteration
          
          if date_unit == 'custom_intraannual': 
          ## Regular yearly iteration and irregular/custom intraannual date iteration

            for date_range in date_step.split(','):
              date_step_arr = date_range.split('..')
              start_dt = date_step_arr[0]
              end_dt = date_step_arr[1]

              year = int(dt1.strftime('%Y'))
              y, y_m1, y_p1 = str(year), str((year - 1)), str((year + 1))

              start_dt = start_dt.replace('{year}', y) \
                                 .replace('{year_minus_1}', y_m1) \
                                 .replace('{year_plus_1}', y_p1)

              end_dt = end_dt.replace('{year}', y) \
                             .replace('{year_minus_1}', y_m1) \
                             .replace('{year_plus_1}', y_p1)

              result.append((
                datetime.strptime(start_dt, self.url_date_format),
                datetime.strptime(end_dt, self.url_date_format)
              ))

            dt1 = dt1 + relativedelta(years=+1)

          else:
            ## Regular date iteration (yearly, monthly, daily, etc)
            delta_args = {}
            delta_args[date_unit] = int(date_step) # TODO: Threat the value "month"
            
            dt1n = dt1 + relativedelta(**delta_args)
            dt2 = dt1n + relativedelta(days=-1)
          
            if (ignore_29feb.lower() == 'true' and dt2.strftime("%m") == '02' and dt2.strftime("%d") == '29'):
              dt2 = dt2 + relativedelta(days=-1)
                
            result.append((dt1, dt2))       
            dt1 = dt1n
      
      return result

    def _kargs(self, row, key, add_extra_fields=False):
      _args = {}
      for f in self.fields[key]:
        if f in row:
          _args[f] = row[f]

      if add_extra_fields:
        _args['extra_fields'] = {}
        for ef in row.keys():
          if ef not in self.fields_lookup and row[ef]:
            _args['extra_fields'][ef] = row[ef]

      return _args

    def _parse_url(self, url, dt1, dt2, date_unit = 'months', date_style = 'interval'):
      
      date_format = self.url_date_format
      if (date_unit == 'years' or date_unit == 'custom_multiannual'):
        date_format = '%Y'

      if (date_style == 'start_date'):
        dt = f'{dt1.strftime(date_format)}'
      elif (date_style == 'end_date'):
        dt = f'{dt2.strftime(date_format)}'
      else:
        dt = f'{dt1.strftime(date_format)}..{dt2.strftime(date_format)}'

      item_id = str(Path(url).name) \
                  .replace('{dt}','') \
                  .replace('__', '_')

      return _eval(url, locals())

    def _bbox_and_footprint(self, raster_fn):

      self._verbose(f'Accesing COG bounds ({raster_fn})')

      r = requests.head(raster_fn)
      if not (r.status_code == 200):
        self._verbose(f'The file {raster_fn} not exists')
        return(None, None, None)

      with rasterio.Env(**self.gdal_env) as rio_env:
        with rasterio.open(raster_fn) as ds:

          transformer = Transformer.from_crs(ds.crs, "epsg:4326", always_xy=True)

          bounds = ds.bounds
          left_wgs84, bottom_wgs84 = transformer.transform(bounds.bottom, bounds.left)
          right_wgs84, top_wgs84 = transformer.transform(bounds.top, bounds.right)

          bbox = [left_wgs84, bottom_wgs84, right_wgs84, top_wgs84]
          footprint = Polygon([
            [left_wgs84, bottom_wgs84],
            [left_wgs84, top_wgs84],
            [right_wgs84, top_wgs84],
            [right_wgs84, bottom_wgs84]
          ])

          return (raster_fn, bbox, mapping(footprint))

    def save_all(self,
      output_dir:str = 'stac',
      catalog_type = pystac.CatalogType.SELF_CONTAINED,
      thumb_base_url = None
    ):
      """
    
      Save the STAC instance to local folder.

      :param output_dir: Destination folder.
      :param catalog_type: Normalization strategy defined 
        by ``pystac.CatalogType``.
      :param thumb_base_url: Base urls for the thumbnail 
        files. Useful in cases where the COG files are 
        hosted in a different location (S3 bucket) of 
        STAC files.
      
      Examples
      ========

      >>> from skmap.misc import GoogleSheet
      >>> from skmap.datasets.eo import STACGenerator
      >>> 
      >>> # Generate your key follow the instructions in https://docs.gspread.org/en/latest/oauth2.html
      >>> key_file = '<GDRIVE_KEY>'
      >>> # Public accessible Google Spreadsheet (Anyone on the internet with this link can view)
      >>> url = 'https://docs.google.com/spreadsheets/d/10tAhEpZ7TYPD0UWhrI0LHcuIzGZNt5AgSjx2Bu-FciU'
      >>> 
      >>> gsheet = GoogleSheet(key_file, url, verbose=True)
      >>> stac_generator = STACGenerator(gsheet, asset_id_fields=[1,2,3,5], catalogs=catalogs, verbose=True)
      >>> stac_generator.save_all(output_dir='stac_odse', thumb_base_url=f'https://s3.eu-central-1.wasabisys.com/stac')

      """

      output_dir = Path(output_dir)
      self._generate_thumbs(output_dir, thumb_base_url)

      for key, catalog in self.catalogs.items():
        catalog.normalize_and_save(
          root_href=str(output_dir.joinpath(catalog.id)), 
          catalog_type=catalog_type
        )

    def _s3_fput_object(self, fpath, output_dir, client, s3_bucket_name, s3_prefix):
      if(fpath.is_file()):
        object_name = fpath.relative_to(output_dir.name)
        if s3_prefix:
          object_name = Path(s3_prefix).joinpath(object_name)
        content_type,_ = mimetypes.guess_type(fpath)
        client.fput_object(s3_bucket_name, str(object_name), str(fpath), content_type=content_type)
        return True
      else:
        return False

    def save_and_publish_all(self,
      s3_host:str,
      s3_access_key:str,
      s3_access_secret:str,
      s3_bucket_name:str,
      s3_prefix:str = '',
      output_dir:str = 'stac',
      catalog_type = pystac.CatalogType.SELF_CONTAINED
    ):
      """
    
      Save the STAC instance to local folder and upload all the files
      to a s3 bucket.

      :param s3_host: Hostname of a S3 service.
      :param s3_access_key: Access key (aka user ID) of S3 service.
      :param s3_access_secret: Secret key (aka user ID) of S3 service.
      :param s3_bucket_name: Name of the bucket.
      :param s3_prefix: Object name prefix (URL part) in the bucket.
      :param output_dir: Destination folder.
      :param catalog_type: Normalization strategy defined 
        by ``pystac.CatalogType``.

      Examples
      ========

      >>> s3_host = "<S3_HOST>"
      >>> s3_access_key = "<s3_access_key>"
      >>> s3_access_secret = "<s3_access_secret>"
      >>> s3_bucket_name = 'stac'
      >>>
      >>> # Generate your key follow the instructions in https://docs.gspread.org/en/latest/oauth2.html
      >>> key_file = '<GDRIVE_KEY>'
      >>> # Public accessible Google Spreadsheet (Anyone on the internet with this link can view)
      >>> url = 'https://docs.google.com/spreadsheets/d/10tAhEpZ7TYPD0UWhrI0LHcuIzGZNt5AgSjx2Bu-FciU'
      >>> 
      >>> gsheet = GoogleSheet(key_file, url)
      >>> stac_generator = STACGenerator(gsheet, verbose=True)
      >>> stac_generator.save_and_publish_all(s3_host, s3_access_key, s3_access_secret, s3_bucket_name)

      """

      thumb_base_url = f'https://{s3_host}/{s3_bucket_name}'
      self.save_all(output_dir, catalog_type=catalog_type, thumb_base_url=thumb_base_url)

      output_dir = Path(output_dir)
      files = find_files([output_dir, '*.*'])
      
      client = Minio(s3_host, s3_access_key, s3_access_secret, secure=True)
      
      args = [ (fpath, output_dir, client, s3_bucket_name, s3_prefix) for fpath in files ]
      self._verbose(f"Copying {len(files)} files to {s3_host}/{s3_bucket_name}/{s3_prefix}")

      for r in parallel.job(self._s3_fput_object, args, n_jobs=10, joblib_args={'backend': 'threading'}):
        continue
      self._verbose(f"End")

except Exception as e:
  _warn_deps(e, 'eo.STACGenerator')

try:

  from pystac import Catalog
  from whoosh.qparser import QueryParser
  from whoosh.index import create_in
  from whoosh import fields

  class STACIndex():
    """
    
    Enable full-text search for static SpatioTemporal Asset Catalogs (STAC) using whoosh.

    :param catalog: STAC catalog instance ``pystac.catalog.Catalog``.
    :param index_dir: Output folder to store the whoosh index files.
    :param verbose: Use ``True`` to print the progress of all steps.


  """
    def __init__(self,
      catalog:Catalog,
      index_dir = 'stac_index',
      verbose = False
    ):

      self.catalog = catalog
      self.index_dir = index_dir
      self.verbose = verbose

      self.ix = self._index()

    def _index(self):
      Path(self.index_dir).mkdir(parents=True, exist_ok=True)
      ix = create_in(self.index_dir, fields.Schema(title=fields.TEXT(stored=True), 
                                                   id=fields.ID(stored=True), description=fields.TEXT))
      
      self._verbose(f'Retriving all collections from {self.catalog.title}')
      collections = list(self.catalog.get_collections())
      
      self._verbose(f'Creating index for {len(collections)} collections')
      writer = ix.writer()
      for collection in collections:
        writer.add_document(
            title=collection.title, 
            id=collection.id, 
            description=collection.title + ' ' + collection.description
        )

      writer.commit()

      return ix

    def search(self, 
      query, 
      field='title'
    ):
      """
      
      Full-text Search over the title and / or description of the collections.

      :param query: Search text.
      :param field: Can be ``title`` or ``description``.

      """
    
      result = []
      with self.ix.searcher() as searcher:
        parser = QueryParser(field, self.ix.schema)
        myquery = parser.parse(query)
        
        i = 0
        for r in searcher.search(myquery):
            result.append({
                'pos': i,
                'id': r['id'],
                'title': r['title']
            })
            i += 1
      
      return result

    def _verbose(self, *args, **kwargs):
      if self.verbose:
        ttprint(*args, **kwargs)

except Exception as e:
  _warn_deps(e, 'eo.STACGenerator')