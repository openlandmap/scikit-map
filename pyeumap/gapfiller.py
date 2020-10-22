from .misc import ttprint
from .datasets import DATA_ROOT_NAME
from scipy.ndimage import median_filter, spline_filter

from typing import List, Dict

import numpy as np
import gdal
import osr
import os

_OUT_DIR = os.path.join(os.getcwd(), 'gapfilled')

class TimeGapFiller():

	def __init__(self,
		fn_times_layers: Dict,
		time_order: List,
		out_dir: str=_OUT_DIR,
		root_dir_name: str=DATA_ROOT_NAME,
	):

		self.fn_times_layers = fn_times_layers
		self.time_order = time_order
		self.max_n_layers_per_time = 0
		self.out_dir = out_dir
		self.root_dir_name = root_dir_name

		self.ltm_data = {}
		self.time_data = {}

		for time in time_order:
			ttprint(f'Reading {len(self.fn_times_layers[time])} layers on {time}')
			self.time_data[time] = self.read_layers(self.fn_times_layers[time])
			time_shape = self.time_data[time].shape
			if time_shape[2] > self.max_n_layers_per_time:
				self.max_n_layers_per_time = time_shape[2]

			ttprint(f'Data shape: {time_shape}')

	def read_layers(self, fn_layers):
		result = []

		for fn_layer in fn_layers:
			ds = gdal.Open(str(fn_layer))
			band_data = ds.GetRasterBand(1).ReadAsArray()
			if (isinstance(band_data, np.ndarray)):
				nodata = ds.GetRasterBand(1).GetNoDataValue()
				band_data = band_data.astype('Float32')
				band_data[band_data == nodata] = np.nan
			else:
				band_data = np.empty((ds.RasterXSize, ds.RasterXSize))
				band_data[:] = np.nan
			result.append(band_data)

		result = np.stack(result, axis=2)

		return result

	def _data_to_new_img(self, fn_base_img, fn_new_img, data, data_type = None, img_format = 'GTiff', nodata = 0):

	  driver = gdal.GetDriverByName(img_format)
	  base_ds = gdal.Open( str(fn_base_img) )

	  x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
	  x_size, y_size, nbands = data.shape

	  out_srs = osr.SpatialReference()
	  out_srs.ImportFromWkt(base_ds.GetProjectionRef())

	  if data_type is None:
	    data_type = base_ds.GetRasterBand(1).DataType

	  new_ds = driver.Create(fn_new_img, x_size, y_size, nbands, data_type)
	  new_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
	  new_ds.SetProjection(out_srs.ExportToWkt())

	  for band in range(0, nbands):
	    new_band = new_ds.GetRasterBand((band+1))
	    new_band.WriteArray(data[:,:,band],0,0)
	    new_band.SetNoDataValue(nodata)

	  new_ds.FlushCache()

	def run(self):

		self.time_data_gaps = {}

		self.timeseries_data = []
		self.timeseries_gaps = []
		self.timeseries_fn = []

		for time in self.time_order:
			self.time_data[time]
			ttprint(f'Filling the gaps with long-term median for {time}')
			ltm_data = np.nanmedian(self.time_data[time], axis=2)

			_, _, n_layers = self.time_data[time].shape
			self.time_data_gaps[time] = []

			for t in range(0, n_layers):
				nan_mask = np.isnan(self.time_data[time][:,:,t])

				self.time_data[time][:,:,t][nan_mask] = ltm_data[nan_mask]
				self.time_data_gaps[time].append(nan_mask)

		for t in range(0, self.max_n_layers_per_time):
			for time in self.time_order:
				if t < self.time_data[time].shape[2]:
					self.timeseries_data.append(self.time_data[time][:,:,t])
					self.timeseries_gaps.append(self.time_data_gaps[time][t])
					src_fn = str(self.fn_times_layers[time][t])
					out_fn = os.path.join(
						self.out_dir,
						src_fn.split(self.root_dir_name)[-1][1:],
					)
					self.timeseries_fn.append(out_fn)

		self.timeseries_data = np.stack(self.timeseries_data, axis=2)
		self.timeseries_gaps = np.stack(self.timeseries_gaps, axis=2)

		ltm_data = np.nanmedian(self.timeseries_data, axis=2)

		_, _, n_layers = self.timeseries_data.shape

		ttprint(f'Filling the gaps with long-term median for all time series')
		for t in range(0, n_layers):
			nan_mask = np.isnan(self.timeseries_data[:,:,t])
			self.timeseries_data[:,:,t][nan_mask] = ltm_data[nan_mask]

		ttprint(f'Saving the results')
		fn_base_img = self.fn_times_layers[time][0]
		for t in range(0, n_layers):
			out_fn = self.timeseries_fn[t]
			out_img_dir = os.path.dirname(out_fn)
			if not os.path.isdir(out_img_dir):
				os.makedirs(out_img_dir)
			self._data_to_new_img(fn_base_img, out_fn, self.timeseries_data[:,:,t:t+1])
