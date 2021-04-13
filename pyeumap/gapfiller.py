from .misc import ttprint
from .datasets import DATA_ROOT_NAME
import multiprocessing
from itertools import cycle, islice
import math
import traceback
import rasterio
import threading
from rasterio.windows import Window

from . import parallel
from sklearn.linear_model import LinearRegression

from typing import List, Dict

from itertools import chain
import numpy as np
from osgeo import gdal
from osgeo import osr
import os

import bottleneck as bc
from .misc import ttprint
from pyeumap.raster import read_rasters, write_new_raster

_OUT_DIR = os.path.join(os.getcwd(), 'gapfilled')

class GapFilledData():

	def __init__(self, time_order: List, time_data, time_win_size = 5, 
		cpu_max_workers:int = multiprocessing.cpu_count(), 
		engine='CPU',
		gpu_tile_size:int = 250):

		self.time_win_size = time_win_size 
		self.time_order = time_order
		self.time_data = time_data
		self.cpu_max_workers = cpu_max_workers
		self.gpu_tile_size = gpu_tile_size
		self.engine = engine

		self.gapfilled_data = {}

	def _get_time_window(self, t, max_win_size, win_size):
		return(
			(t - win_size) if (t - win_size) > 0 else 0,
			(t + win_size) if (t + win_size) < (max_win_size-1) else (max_win_size-1),
		)

	def _key_from_layer_pos(self, time, layer_pos, win_size):
		_, _, n_layers = self.time_data[time].shape
		t1, t2 = self._get_time_window(layer_pos, n_layers, win_size)
		return self._key_from_time(time, t1, t2)

	def _key_from_time(self, time, t1, t2):
		return f'{time}_{t1}_{t2}'

	def _calc_nanmedian(self, time, t1, t2):
		#return self._key_from_time(time, t1, t2), np.nanmedian(self.time_data[time][:,:,t1:t2], axis=2)
		return self._key_from_time(time, t1, t2), bc.nanmedian(self.time_data[time][:,:,t1:t2], axis=2)
		#med = nanPercentile(np.transpose(self.time_data[time][:,:,t1:t2], axes=[2, 0, 1]), [50])
		#return self._key_from_time(time, t1, t2), med[0]

	def _cpu_processing(self, args):
		for key, data in parallel.ThreadGeneratorLazy(self._calc_nanmedian, iter(args), max_workers=self.cpu_max_workers, chunk=self.cpu_max_workers):
			self.gapfilled_data[key] = data

	def _gpu_processing(self, args):

		import cupy as cp

		x_size, y_size, _ = self.time_data[self.time_order[0]].shape

		for x1 in chain(range(0, x_size, self.gpu_tile_size), [self.gpu_tile_size]):
			for y1 in chain(range(0, x_size, self.gpu_tile_size), [self.gpu_tile_size]):
				
				gpu_data = {}

				x2 = ( (x_size-1) if (x1 + self.gpu_tile_size) >= x_size else (x1 + self.gpu_tile_size))
				y2 = ( (y_size-1) if (y1 + self.gpu_tile_size) >= y_size else (y1 + self.gpu_tile_size))

				for time in self.time_order:
					gpu_data[time] = cp.asarray(self.time_data[time][x1:x2,y1:y2,:])
					
				for time, t1, t2 in iter(args):
					key = self._key_from_time(time, t1, t2)
					if key not in self.gapfilled_data:
						self.gapfilled_data[key] = np.zeros((x_size, y_size), dtype='float32')

					gpu_median = cp.nanmedian(gpu_data[time][:,:,t1:t2], axis=2)
					self.gapfilled_data[key][x1:x2,y1:y2] = cp.asnumpy(gpu_median)
					gpu_median = None

				gpu_data = None

	def run(self):
		self.gapfilled_data = {}
		self.available_windows = {}

		args = set()

		for time in self.time_order:
			
			_, _, n_layers = self.time_data[time].shape
			self.available_windows[time] = []

			for layer_pos in range(0, n_layers):

				self.available_windows[time].append([])

				for w in range(self.time_win_size, n_layers, self.time_win_size):
					t1, t2 = self._get_time_window(layer_pos, n_layers, w)
					self.available_windows[time][layer_pos].append(w)
					args.add((time, t1, t2))

		ttprint(f'Calculating {len(args)} gap filling possibilities')

		if self.engine == 'CPU':
			ttprint(f'Using cpu engine')
			self._cpu_processing(args)
		elif  self.engine == 'GPU':
			ttprint(f'Using GPU engine')
			self._gpu_processing(args)
		
		ttprint(f'Possibilities calculated')

	def get(self, time, layer_pos):
		
		time_list = (time if type(time) == list else [time])
		result = {}

		for t in time_list:
			try:
				for win_size in (self.available_windows[t][layer_pos]):
					result[win_size] = ( [] if t not in result else result[win_size])
					key = self._key_from_layer_pos(t, layer_pos, win_size)
					result[win_size].append(self.gapfilled_data[key])
			except:
				traceback.format_exc()
				continue
		
		for win_size in list(result.keys()):
			result[win_size] = np.nanmean(np.stack(result[win_size], axis=2), axis=2)

		return result

class TimeGapFiller():

	def __init__(self,
		fn_times_layers: Dict,
		time_order: List,
		time_win_size: int=8,
		spatial_win:Window = None,
		out_dir: str=_OUT_DIR,
		out_mantain_subdirs: bool=True,
		root_dir_name: str=DATA_ROOT_NAME,
		cpu_max_workers:int = multiprocessing.cpu_count(),
		engine='CPU',
		verbose = True,
		gpu_tile_size:int = 250, 
	):

		self.fn_times_layers = fn_times_layers
		self.time_order = time_order
		self.max_n_layers_per_time = 0
		self.out_dir = out_dir
		self.root_dir_name = root_dir_name
		self.cpu_max_workers = cpu_max_workers
		self.time_win_size = time_win_size
		self.spatial_win = spatial_win
		self.engine = engine
		self.out_mantain_subdirs = out_mantain_subdirs
		self.gpu_tile_size = gpu_tile_size

		self.time_data = {}
		
		self.verbose = verbose
		
	def read_layers(self):
		for time in self.time_order:
			self.time_data[time], _ = read_rasters(raster_files = self.fn_times_layers[time], verbose=self.verbose)
			if self.verbose:
				ttprint(f'{time} data shape: {self.time_data[time].shape}')
			time_shape = self.time_data[time].shape
			if time_shape[2] > self.max_n_layers_per_time:
				self.max_n_layers_per_time = time_shape[2]

		if self.verbose:
			ttprint('Reading process finished')

	def _get_neib_times(self, time):
		
		total_times = len(self.time_order)
		i = self.time_order.index(time)

		time_order_rev = self.time_order.copy()
		time_order_rev.reverse()
		
		neib_times = []

		for j in range(1, math.ceil(total_times / 2) + 1):

			# Setup slicer in the right positions
			after = islice(cycle(self.time_order), i, None)
			before = islice(cycle(time_order_rev), total_times-i, None)
			next(after)

			before_tim = [next(before) for t in range(0,j)]
			after_tim = [next(after) for t in range(0,j)]

			neib_times.append((before_tim, after_tim))

		return neib_times

	def _fill_gaps(self, time, layer_pos, newdata_dict, verbose_suffix='', gapflag_offset = 0):
		
		end_msg = None

		for i in newdata_dict.keys():

			gaps_mask = np.isnan(self.time_data[time][:,:,layer_pos])
			newdata = newdata_dict[i][gaps_mask]
			
			gaps_pct = np.count_nonzero(~np.isnan(newdata))
			newdata_pct = np.count_nonzero(gaps_mask.flatten())

			if newdata_pct != 0:
				gapfilled_pct =  gaps_pct / newdata_pct
				
				self.time_data[time][:,:,layer_pos][gaps_mask] = newdata
				self.time_data_gaps[time][:,:,layer_pos][gaps_mask] = int(i) + gapflag_offset

				if gapfilled_pct == 1:
					end_msg = f'Layer {layer_pos}: reached {100*gapfilled_pct:.2f}% with {i}-year window from {verbose_suffix})'
					break

		return end_msg

	def _fill_gaps_all_times(self, time, layer_pos):

		_, _, n_layers = self.time_data[time].shape

		end_msg = None
		all_data = self.gapfilled_data.get(self.time_order, layer_pos)
		end_msg = self._fill_gaps(time, layer_pos, all_data, verbose_suffix='all seasons', gapflag_offset = n_layers*2)
		
		return end_msg

	def _fill_gaps_neib_time(self, time, layer_pos):

		newdata_dict = {}
		end_msg = None
		_, _, n_layers = self.time_data[time].shape

		for before_times, after_times in self._get_neib_times(time):

			before_data = self.gapfilled_data.get(before_times, layer_pos)
			after_data = self.gapfilled_data.get(after_times, layer_pos)

			keys = list(set(before_data.keys()).intersection(after_data.keys()))
			keys.sort()

			for i in keys:
				stacked = np.stack([before_data[i], after_data[i]], axis=2)
				valid_mean = np.any(np.isnan(stacked), axis=2)
				stacked[valid_mean] = np.nan
				newdata_dict[i] = np.nanmean(stacked, axis=2)

			end_msg = self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='neighborhood seasons', gapflag_offset = n_layers)
			if end_msg is not None:
				break

		return end_msg

	def _fill_gaps_same_time(self, time, layer_pos):
		newdata_dict = self.gapfilled_data.get(time, layer_pos)
		return self._fill_gaps(time, layer_pos, newdata_dict, verbose_suffix='same season')

	def fill_layer(self, time, layer_pos):
		
		end_msg = self._fill_gaps_same_time(time, layer_pos)
		
		if end_msg is None:
			end_msg = self._fill_gaps_neib_time(time, layer_pos)

		if end_msg is None:
			end_msg = self._fill_gaps_all_times(time, layer_pos)
		
		return end_msg
	
	def _write_data(self, time):
		fn_base_img = self.fn_times_layers[time][0]
		_, _, n_layers = self.time_data[time].shape

		for t in range(0, n_layers):

			src_fn = self.fn_times_layers[time][t]

			if self.out_mantain_subdirs:
				out_dir = os.path.join(self.out_dir, str(src_fn.parent).split(self.root_dir_name)[-1][1:])
				out_fn_data = os.path.join(out_dir, ('%s.tif' % src_fn.stem))
				out_fn_flag = os.path.join(out_dir, ('%s_flag.tif' % src_fn.stem))
			else:
				out_fn_data = os.path.join(self.out_dir, ('%s.tif' % src_fn.stem))
				out_fn_flag = os.path.join(self.out_dir, ('%s_flag.tif' % src_fn.stem))

			out_img_dir = os.path.dirname(out_fn_data)

			if not os.path.isdir(out_img_dir):
				try:
					os.makedirs(out_img_dir)
				except:
					continue

			write_new_raster(fn_base_img, out_fn_data, self.time_data[time][:,:,t:t+1], data_type='uint8')
			write_new_raster(fn_base_img, out_fn_flag, self.time_data_gaps[time][:,:,t:t+1], data_type='uint8')
		
		return True
		
	def save_to_img(self):
		if self.verbose:
			ttprint(f'Saving the results')
		
		args = [ (time,) for time in self.time_order]
		for end_msg in parallel.ThreadGeneratorLazy(self._write_data, iter(args), max_workers=len(self.time_order), chunk=len(self.time_order)):
			end_msg = True

		if self.verbose:
			ttprint(f'Saving proces finished')

	def run(self):

		self.time_data_gaps = {}

		self.gapfilled_data = GapFilledData(self.time_order, self.time_data, self.time_win_size, 
			cpu_max_workers=self.cpu_max_workers, engine=self.engine, gpu_tile_size=self.gpu_tile_size)
		self.gapfilled_data.run()

		layer_args = []

		for time in self.time_order:
			
			nrows, ncols, n_layers = self.time_data[time].shape
			self.time_data_gaps[time] = np.zeros((nrows, ncols, n_layers), dtype='int8')

			for layer_pos in range(0, n_layers):
				layer_args.append((time, layer_pos))
		
		if self.verbose:
			ttprint(f'Filling the gaps')
		for end_msg in parallel.ThreadGeneratorLazy(self.fill_layer, iter(layer_args), max_workers=self.cpu_max_workers, chunk=self.cpu_max_workers):
			end_msg = True
