import numpy as np

from typing import List, Union
from pyeumap.misc import ttprint, find_files
from pyeumap import parallel

import rasterio

def read_rasters(	
									raster_dirs:List = [], 
									raster_files:List = [], 
									raster_ext = 'tif', 
									spatial_win = None,
									dtype = 'float16', 
									n_jobs = 4, 
									verbose = False,
									try_without_window = False
								):
	if len(raster_dirs) == 0 and len(raster_files) == 0:
		raise Exception('The raster_dirs and raster_files params can be empty at same time.')

	if len(raster_files) == 0:
		raster_files = find_files(raster_dirs, f'*.{raster_ext}')

	if verbose:
		ttprint(f'Reading {len(raster_files)} raster files using {n_jobs} workers')

	def _read_raster(raster_pos):
		raster_file = raster_files[raster_pos]
		band_data = None

		with rasterio.open(raster_file) as raster_ds:
			try:
				band_data = raster_ds.read(1, window=spatial_win)
				if band_data.size == 0 and try_without_window:
					band_data = raster_ds.read(1)
			except:
				if spatial_win is not None:
					ttprint(f'ERROR: Failed to read {raster_file} window {spatial_win}.')
					band_data = np.empty((int(spatial_win.width), int(spatial_win.height)))
					band_data[:] = np.nan
		return raster_pos, band_data, raster_ds.nodatavals[0]

	raster_data = {}
	args = [ (raster_pos,) for raster_pos in range(0,len(raster_files)) ]

	for raster_pos, band_data, nodata in parallel.ThreadGeneratorLazy(_read_raster, iter(args), max_workers=n_jobs, chunk=n_jobs*2):
		raster_file = raster_files[raster_pos]

		if (isinstance(band_data, np.ndarray)):
			
			band_data = band_data.astype(dtype)
			band_data[band_data == nodata] = np.nan
			
			#if (verbose and np.isnan(np.min(band_data))):
			#	ttprint(f'Layer {raster_file} has NA values (nodata={nodata})')
		
		else:
			raise Exception(f'The raster {raster_file} was not found.')
		raster_data[raster_pos] = band_data
	
	raster_data = [raster_data[i] for i in range(0,len(raster_files))]
	raster_data = np.ascontiguousarray(np.stack(raster_data, axis=2))
	return raster_data, raster_files

def create_raster(
										fn_base_raster, 
										fn_raster, 
										data, 
										spatial_win = None, 
										data_type = None, 
										raster_format = 'GTiff', 
										nodata = 0
									):
	
	if len(data.shape) < 3:
		data = np.stack([data], axis=2)

	x_size, y_size, nbands = data.shape
	
	with rasterio.open(fn_base_raster, 'r') as base_raster:

		if data_type is None:
			data_type = base_raster.dtypes[0]
					
		transform = base_raster.transform
		
		if spatial_win is not None:
			transform = rasterio.windows.transform(spatial_win, transform)
			
		return rasterio.open(fn_raster, 'w', 
						driver=raster_format, 
						width=x_size, 
						height=y_size, 
						count=nbands,
						dtype=data_type, 
						crs=base_raster.crs,
						compress='LZW',
						transform=transform)

def write_new_raster(
											fn_base_raster, 
											fn_new_raster, 
											data, 
											spatial_win = None, 
											data_type = None, 
											raster_format = 'GTiff', 
											nodata = 0
										):

	if len(data.shape) < 3:
		data = np.stack([data], axis=2)

	_, _, nbands = data.shape

	with create_raster(fn_base_raster, fn_new_raster, data, spatial_win, data_type, raster_format) as new_raster:
		new_raster.nodata = nodata
		for band in range(0, nbands):
			new_raster.write(data[:,:,band].astype(new_raster.dtypes[band]), indexes=(band+1))
