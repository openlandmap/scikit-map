import requests
import numpy as np
import bottleneck as bc
import joblib
from pathlib import Path
import gc

from .. import parallel
from ..raster import read_rasters_remote, save_rasters
from ..misc import nan_percentile, ttprint

class LandsatARD():
  
  def __init__(self,
    username:str,
    password:str,
    parallel_download:int = 4,
    verbose:bool = True,
  ):
  
    self.username = username
    self.password = password
    self.base_url = 'https://glad.umd.edu/dataset/landsat_v1.1'

    self.parallel_download = parallel_download
    self.verbose = verbose

    self.bqa_idx = 7

    self.max_spectral_val = 40000
    self.max_thermal_val = 100

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

  def read(self, tile, start, end, clear_sky = True, min_clear_sky = 0.2):
    data, url_list, ds_list = read_rasters_remote(
      url_list = self._glad_urls(tile, start, end),
      username = self.username,
      password = self.password,
      verbose = self.verbose,
      return_ds = True,
      nodata = 0, 
      n_jobs = self.parallel_download
    )

    self._verbose(f'Read data {data.shape}.')

    if clear_sky:
      self._verbose(f'Removing cloud and cloud shadow pixels.')
      clear_sky_mask, clear_sky_pct = self._clear_sky_mask(data)
      data[~clear_sky_mask] = np.nan
      clear_sky_idx = np.where(clear_sky_pct >= min_clear_sky)[0]
      data = data[:,:,:,clear_sky_idx]

      if len(clear_sky_idx) == 0:
          url_list, ds_list = [], []

    return data, url_list, ds_list

  def _clear_sky_mask(self, data):

    qa_data = data[self.bqa_idx,:,:,:]

    clear_sky1 = np.logical_and(qa_data >= 1, qa_data <= 2)
    clear_sky2 = np.logical_and(qa_data >= 5, qa_data <= 6)
    clear_sky3 = np.logical_and(qa_data >= 11, qa_data <= 17)

    clear_sky_mask = np.logical_or(
              np.logical_or(clear_sky1, clear_sky2), 
              clear_sky3
    )
    clear_sky_mask = np.broadcast_to(clear_sky_mask, shape=data.shape)

    nrow, ncols = data.shape[1], data.shape[2]
    clear_sky_pct = []

    for i in range(0, data.shape[3]):
      clear_sky_count = bc.nansum(clear_sky_mask[0,:,:,i].astype(np.int))
      clear_sky_pct.append(clear_sky_count / (nrow * ncols))

    return clear_sky_mask, np.array(clear_sky_pct)

  def _save_percentile_agg(self, result, ds, tile, start, p, output_dir, unit8):
    
    if not isinstance(output_dir, Path):
      output_dir = Path(output_dir)

    for i in range(0, result.shape[0]):
      
      band_data = result[i,:,:,:]
      dtype = 'uint16'

      if unit8:
        scale, conv = self.max_spectral_val, 255
        if (i == 6):
          scale, conv = self.max_thermal_val, 1
        
        valid_mask = ~np.isnan(band_data)
        band_data[valid_mask] = ((band_data[valid_mask] / scale) * conv)
        band_data[valid_mask] = np.floor(band_data[valid_mask])

        dtype = 'uint8'
      
      fn_raster_list = []
      for paux in p:
        p_dir = output_dir.joinpath(f'{start}').joinpath(f'B{i+1}_P{int(paux)}')
        fn_raster_list.append(p_dir.joinpath(f'{tile}.tif'))
      
      save_rasters(fn_base_raster=ds, data=band_data, fn_raster_list=fn_raster_list, 
        data_type=dtype, n_jobs=len(p), verbose=self.verbose)

  def percentile_agg(self, tile, start, end, p, clear_sky = True, min_clear_sky = 0.2, 
    n_jobs = 6, output_dir = None, unit8 = True):

    def _run(band_data, p, i):
      
      max_val = self.max_spectral_val
      if (i == 6):
        max_val =  65000

      band_valid_mask = np.logical_and(band_data >= 1.0, band_data <= max_val)
      band_data[~band_valid_mask] = np.nan

      perc_data = nan_percentile(band_data.transpose(2,0,1), p)
      perc_data = np.stack(perc_data, axis=0)

      return (perc_data, i)

    data, url_list, ds_list = self.read(tile, start, end, clear_sky, min_clear_sky)

    self._verbose(f'Aggregating by {len(p)} percentiles.')

    if len(ds_list) > 0:
      args = []
      
      for i in range(0, self.bqa_idx):
        args.append( (data[i,:,:,:], p, i) )

      result = {}
      
      for perc_data, i in parallel.ThreadGeneratorLazy(_run, iter(args), max_workers=n_jobs, chunk=n_jobs*2):
        result[i] = perc_data

      result = [result[i] for i in range(0,len(args))]
      if len(result) > 0:
        result = np.stack(result, axis=3).transpose(3,1,2,0)

        if output_dir is not None:
          self._verbose(f'Saving the result in {output_dir}.')
          self._save_percentile_agg(result, ds_list[0], tile, start, p, output_dir, unit8)

        del data
        gc.collect()

        return result, ds_list[0]
    
    else:
      raise RuntimeError('All the images were read, but no clear_sky data was found.')