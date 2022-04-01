"""
Earth observation data acess
"""
from typing import List, Union

import requests
import numpy as np
import bottleneck as bc
import joblib
from pathlib import Path
import gc
import os

from .. import parallel
from ..raster import read_auth_rasters, save_rasters
from ..misc import nan_percentile, ttprint

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

    >>> from eumap.datasets.eo import GLADLandsat
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
    
    >>> from eumap.datasets.eo import GLADLandsat
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

        del data
        gc.collect()

        self._verbose(f'Removing {base_raster}')
        os.remove(base_raster)

        return result, base_raster, output_files
    
    else:
      raise RuntimeError('All the images were read, but no clear_sky data was found.')