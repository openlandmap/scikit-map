from pyeumap.misc import ttprint

from typing import List, Union
import joblib

import multiprocessing
import geopandas as gpd
import numpy as np
import gdal
import osr
import math
import rasterio
import re

from . import parallel
from pathlib import Path
from geopandas import GeoDataFrame

import gc
import concurrent.futures
from concurrent.futures import as_completed

import uuid
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict

_automl_enabled = False
try:
	from autosklearn.classification import AutoSklearnClassifier
	_automl_enabled = True
except ImportError:
	pass

class LandMapper():

	def __init__(self, points:GeoDataFrame, 
				feat_col_prfxs:List[str], 
				target_col:str, 
				estimator:Union[BaseEstimator, None] = None,
				imputer:BaseEstimator = SimpleImputer(missing_values=np.nan, strategy='mean'),
				eval_strategy = 'train_val_split', 
				val_samples_pct = 0.2, 
				min_samples_per_class = 0.05,
				scoring = None, 
				weight_col = None, 
				cv = 5, 
				param_grid = {}, 
				group_col = None, 
				fill_nodata=True, 
				refit=True,
				pred_method='predict',
				verbose = True):

		if not isinstance(points, gpd.GeoDataFrame):
			points = gpd.read_file(points)
				
		self.verbose = verbose
		self.pts = points
		self.target_col = target_col
		self.imputer = imputer
		self.weight_col = weight_col
		self.group_col = group_col
		self.pred_method = pred_method
		self.min_samples_per_class = min_samples_per_class
		self.refit = refit

		if eval_strategy is None:
			self.refit = True    
        
		if estimator is None:
			if _automl_enabled:
				self.estimator = AutoSklearnClassifier(**autosklearn_kwargs)
			else:
				self.estimator = RandomForestClassifier(n_estimators=100)
		else:
			self.estimator = estimator

		self.feature_cols = []
		for feat_prfx in feat_col_prfxs:
			self.feature_cols += list(self.pts.columns[self.pts.columns.str.startswith(feat_prfx)])
		
		classes_pct = (self.pts[self.target_col].value_counts() / self.pts[target_col].count())
		self.rows_to_remove = self.pts[self.target_col].isin(classes_pct[classes_pct >= min_samples_per_class].axes[0])
		nrows, _ = self.pts[~self.rows_to_remove].shape
		if nrows > 0:
			self.pts = self.pts[self.rows_to_remove]
			if self.verbose:
				ttprint(f'Removing {nrows} sampes due min_samples_per_class condition (< {min_samples_per_class})')
				
		self.features_weight = None
		self.weight_idx = None
		if self.weight_col is not None:
			if self.verbose:
				ttprint(f'Using {self.weight_col} as weight')
			
			self.feature_cols.append(self.weight_col)
			self.weight_idx = self.pts[self.feature_cols].columns.get_loc(self.weight_col)
			
		self.groups = None
		if self.group_col is not None:
			self.groups = self.pts[self.group_col]

		self.scoring = scoring
		self.cv = cv
		self.param_grid = param_grid

		self.eval_strategy = eval_strategy
		self.val_samples_pct = val_samples_pct

		self.features = np.ascontiguousarray(self.pts[self.feature_cols].to_numpy(), dtype=np.float32)
		self.target = np.ascontiguousarray(self.pts[self.target_col].to_numpy(), dtype=np.float32)
		self.features_raw = self.features

		if self.weight_idx is not None:
			self.features_weight = self.features[:, self.weight_idx]
			self.feature_cols.remove(self.weight_col)
			self.features = np.delete(self.features, self.weight_idx, 1)
		
		if fill_nodata:
			self.features = self._fill_nodata(self.features, fit_and_tranform = True)
	
	@staticmethod
	def load_instance(fn_joblib):
		return joblib.load(fn_joblib)

	def save_instance(self, fn_joblib, no_train_data = False, compress='lz4'):
		if no_train_data:
			prop_to_del = [ 'pts', 'features', 'target', 'features_raw', \
											'features_weight', 'weight_idx', \
											'groups', 'scoring', 'rows_to_remove']
			for prop in prop_to_del:
				if self.verbose:
					ttprint(f'Removing {prop} attribute')
				delattr(self, prop)
		
		return joblib.dump(self, fn_joblib, compress=compress)

	def _fill_nodata(self, data, fit_and_tranform = False):
		nodata_idx = self._nodata_idx(data)
		num_nodata_idx = np.sum(nodata_idx)
		pct_nodata_idx = num_nodata_idx / data.size * 100

		result = data

		if (num_nodata_idx > 0):
			ttprint(f'Filling the missing values ({pct_nodata_idx:.2f}% / {num_nodata_idx} values)...')
			
			if fit_and_tranform:
				result = self.imputer.fit_transform(data)
			else:
				result = self.imputer.transform(data)
			
		return result

	def _nodata_idx(self, data):
		if np.isnan(self.imputer.missing_values):
			return np.isnan(data)
		else:
			return (data == self.imputer.missing_values)
	
	def _njobs_cv(self):
		njobs_cv = 1
		if self.estimator.n_jobs != -1:
			njobs_cv = math.ceil(multiprocessing.cpu_count() / self.estimator.n_jobs)
		
		if self.verbose:
			ttprint(f'Using {njobs_cv} jobs for cross validation')
			
		return njobs_cv
	
	def _best_params(self):
		if self.verbose:
			ttprint('Hyperparameter optimization result:')
			means = self.grid_search.cv_results_['mean_test_score']*-1
			stds = self.grid_search.cv_results_['std_test_score']
			for mean, std, params in zip(means, stds, self.grid_search.cv_results_['params']):
				ttprint(f" {mean:.5f} (+/-{2*std:.05f}) from {params}")
			ttprint(f'Best: {self.grid_search.best_score_:.5f} using {self.grid_search.best_params_}')
			
		return self.grid_search.best_params_
	
	def _train_val_split(self):
		
		if self.weight_idx != None:
			_, new_weight_idx = self.features.shape
			self.features = np.insert(self.features)

		train_feat, val_feat, train_targ, val_targ = train_test_split(self.features, self.target, test_size=self.val_samples_pct)
		train_weight = None
		
		if self.weight_idx != None:
			
			#train_weight_idx = train_feat[self.feature_cols].columns.get_loc(self.weight_col)
			train_weight = train_feat[:, new_weight_idx]
			train_feat = np.delete(train_feat, new_weight_idx, 1)
			
			#val_weight_idx = val_feat[self.feature_cols].columns.get_loc(self.weight_col)
			val_weight = val_feat[:, new_weight_idx]
			val_feat = np.delete(val_feat, new_weight_idx, 1)
			
			self.features = np.delete(self.features, new_weight_idx, 1)
			
		if _automl_enabled and isinstance(self.estimator, AutoSklearnClassifier):
			self.estimator.fit(train_feat, train_targ)
		else:
			self.estimator.fit(train_feat, train_targ, sample_weight=train_weight)

		self.eval_targ = val_targ
		self.eval_pred = self.estimator.predict(val_feat)

	def _cv_kfold(self):

		self.eval_targ = self.target
		self.eval_pred = cross_val_predict(self.estimator, self.features, self.target, method=self.pred_method, n_jobs=self._njobs_cv(), \
			cv=self.cv, groups=self.groups, verbose=self.verbose, fit_params= { 'sample_weight': self.features_weight })

	def _grid_search_cv(self):
		if self.verbose:
			ttprint(f'Finding the best hyperparameters {self.param_grid.keys()}')
	
		self.grid_search = GridSearchCV(self.estimator, self.param_grid, 
											cv=self.cv,
											scoring=self.scoring,
											verbose=self.verbose, 
											refit = False,
											n_jobs= self._njobs_cv()
											)
		self.grid_search.fit(self.features, self.target, groups=self.groups, sample_weight=self.features_weight)
		self.estimator.set_params(**self._best_params())
		
		self._cv_kfold()

	def _class_optimal_th(self, curv_precision, curv_recall, curv_th):
		# Removing elements where the precision or recall are zero
		nonzero_mask = np.logical_and((curv_precision != 0.0), (curv_recall != 0.0))
		optimal_idx = np.argmax(1 - np.abs(curv_precision[nonzero_mask] - curv_recall[nonzero_mask]))
		return curv_recall[optimal_idx], curv_precision[optimal_idx], curv_th[optimal_idx]

	def _classification_report_prob(self):
		classes, cnt = np.unique(self.eval_targ, return_counts=True)

		me = {
			'log_loss': {},
			'pr_auc': {},
			'support': {},
			'opti_th': {},
			'opti_recall': {},
			'opti_precision': {},
			'curv_recall': {},
			'curv_precision': {},
			'curv_th': {},
		}

		for c in classes:
				c_mask = (self.eval_targ == c)
				me['log_loss'][c] = metrics.log_loss(self.eval_targ[c_mask], self.eval_pred[c_mask], labels=classes)

		for c_idx, c in enumerate(classes):
			me['support'][c] = cnt[c_idx]

			c_targ = (self.eval_targ == c).astype(int)
			c_pred = self.eval_pred[:, c_idx]

			curv_precision, curv_recall, curv_th = metrics.precision_recall_curve(c_targ,c_pred)
			me['curv_precision'][c], me['curv_recall'][c], me['curv_th'][c] = curv_precision, curv_recall, curv_th

			me['pr_auc'][c] = metrics.auc(me['curv_recall'][c], me['curv_precision'][c])
			me['opti_precision'][c], me['opti_recall'][c], me['opti_th'][c] = self._class_optimal_th(curv_precision, curv_recall, curv_th)

		report = '     log_loss   pr_auc   optimal_prob  optimal_precision  optimal_recall  support\n'
		report += '\n'
		for c in classes:
			report += f"{int(c)}  "
			report += f"{me['log_loss'][c]:.4f}     "
			report += f"{me['pr_auc'][c]:.4f}   "
			report += f"{me['opti_th'][c]:.4f}        "
			report += f"{me['opti_precision'][c]:.4f}              "
			report += f"{me['opti_recall'][c]:.4f}         "
			report += f"{me['support'][c]}\n"

		report += '\n'

		report += f"Total                                   "
		report += f"                                  {np.sum(cnt)}\n"

		self.prob_metrics = me

		return report

	def _estimator_metrics(self):
		if self.pred_method == 'predict':
			self.cm = metrics.confusion_matrix(self.eval_targ, self.eval_pred)
			self.overall_acc = metrics.accuracy_score(self.eval_targ, self.eval_pred)
			self.classification_report = metrics.classification_report(self.eval_targ, self.eval_pred)
		elif self.pred_method == 'predict_proba':
			self.log_loss = metrics.log_loss(self.eval_targ, self.eval_pred)
			self.classification_report = self._classification_report_prob()
	
	def train(self):
		method_name = '_%s' % (self.eval_strategy)
		if self.eval_strategy is None or hasattr(self, method_name):
			
			if self.eval_strategy is not None:
				if self.verbose:
					ttprint('Training and evaluating the model')
			
				train_method = getattr(self, method_name)
				train_method()
			
				self._estimator_metrics()
			
			if self.refit:
				if self.verbose:
					ttprint('Training the final model using all data')

				if _automl_enabled and isinstance(self.estimator, AutoSklearnClassifier):
					self.estimator.fit(self.features, self.target)
				else:
					self.estimator.fit(self.features, self.target, sample_weight=self.features_weight)

		else:
			ttprint(f'{self.eval_strategy} is a invalid validation strategy')

	def _feature_idx(self, fn_layer):
		return self.feature_cols.index(fn_layer.stem)

	def _new_image(self, fn_base_img, fn_new_img, data, spatial_win = None, data_type = None, img_format = 'GTiff', nodata = 0):
		
		x_size, y_size, nbands = data.shape
		
		with rasterio.open(fn_base_img, 'r') as base_img:

			if data_type is None:
				data_type = base_img.dtypes[0]
						
			transform = base_img.transform
			
			if spatial_win is not None:
				transform = rasterio.windows.transform(spatial_win, transform)
				
			return rasterio.open(fn_new_img, 'w', 
							driver=img_format, 
							width=x_size, 
							height=y_size, 
							count=nbands,
							dtype=data_type, 
							crs=base_img.crs,
							compress='LZW',
							transform=transform)

	def _data_to_new_img(self, fn_base_img, fn_new_img, data, spatial_win = None, data_type = None, img_format = 'GTiff', nodata = 0):

		_, _, nbands = data.shape

		with self._new_image(fn_base_img, fn_new_img, data, spatial_win, data_type, img_format) as dst:
			dst.nodata = 0
			for band in range(0, nbands):
				
				dst.write(data[:,:,band].astype(dst.dtypes[band]), indexes=(band+1))
	
	def _find_layers(self, dirs_layers):
		fn_layers = []
		
		for dirs_layer in dirs_layers:
			for fn_layer in list(Path(dirs_layer).glob('**/*.tif')):
				if fn_layer.stem in self.feature_cols:
					fn_layers.append(fn_layer)
				elif self.verbose:
					ttprint(f'Ignoring {fn_layer}')

		fn_layers.sort(key=self._feature_idx)

		return fn_layers

	def _get_data(self, fn_layer, spatial_win):
		if self.verbose:
			ttprint(f'Reading {fn_layer}')
		
		band_data = None
		with rasterio.open(fn_layer) as ds:

			if 'tile' in str(fn_layer):
				band_data = ds.read(1)
			else:
				band_data = ds.read(1, window=spatial_win)
		
		return fn_layer, band_data, ds.nodatavals[0]

	def _read_layers(self, fn_layers, spatial_win, inmem_calc_func, dict_layers_newnames):
		result = []
		
		max_workers = math.ceil(multiprocessing.cpu_count()/2)
		
		args = [ (fn_layer,spatial_win) for fn_layer in fn_layers]
		
		fn_layers = []
		for fn_layer, band_data, nodata in parallel.ThreadGeneratorLazy(self._get_data, iter(args), max_workers=max_workers, chunk=max_workers*2):
			if (isinstance(band_data, np.ndarray)):
				band_data = band_data.astype('Float16')
				band_data[band_data == nodata] = np.nan
				if (np.isnan(np.min(band_data))):
					ttprint(f'Layer {fn_layer} has NA values (nodata={nodata})')
			else:
				ttprint(f'Layer {fn_layer} not found')
			
			fn_layers.append(fn_layer)
			result.append(band_data)
		
		input_data = np.stack(result, axis=2)
		
		if inmem_calc_func is not None:
			fn_layers, input_data = inmem_calc_func(fn_layers, input_data, spatial_win)
		
		return self._reorder_data(fn_layers, dict_layers_newnames, input_data)

	def _reorder_data(self, fn_layers, dict_layers_newnames, input_data ):
		feature_cols_set = set(self.feature_cols)
		layernames = []
		
		for fn_layer in fn_layers:
			layername = fn_layer.stem
			for newname in dict_layers_newnames.keys():
				layername = re.sub(dict_layers_newnames[newname], newname, layername)
			if layername not in feature_cols_set:
				raise Exception(f"Layer {layername} does not exist as feature_cols.\nUse dict_layers_newnames param to match their names")
			layernames.append(layername)
		
		sorted_input_data = []
		for feature_col in self.feature_cols:
			try:
				idx = layernames.index(feature_col)
				sorted_input_data.append(input_data[:,:,idx])
			except:
				raise Exception(f"The feature {feature_col} was not provided")
		
		return np.stack(sorted_input_data, axis=2)
	
	def predict_points(self, input_points):
		
		input_feat = np.ascontiguousarray(input_points[self.feature_cols].to_numpy(), dtype=np.float32)

		n_points, n_features = input_feat.shape

		if self.verbose:
			ttprint(f'Predicing {n_points} points')
		
		return self.estimator.predict(input_feat)
	
	def predict_multi(self, fn_layers_list:List = None, fn_result_list:List = None, spatial_win = None, \
		data_type = 'float32', fill_nodata=False, estimate_uncertainty=False, inmem_calc_func = None, dict_layers_newnames_list:list = []):
		
		data_pool = {}
		result_pool = {}
		
		reading_futures = []
		processing_futures = []
		writing_futures = []
		
		def reading_fn(i):
			ttprint(f'reading {i}')
			fn_layers = fn_layers_list[i]
			dict_layers_newnames = dict_layers_newnames_list[i]
			
			input_data = self._read_layers(fn_layers, spatial_win, inmem_calc_func, dict_layers_newnames)
			
			input_data_key = str(uuid.uuid4())
			data_pool[input_data_key] = input_data
			processing_futures.append(processing_pool.submit(processing_fn, i, input_data_key))
			
		def processing_fn(i, input_data_key):
			ttprint(f'procesing {i}')
			
			input_data = data_pool[input_data_key]
			x_size, y_size, n_features = input_data.shape
			input_data = input_data.reshape(-1, n_features)
			
			if self.verbose:
				ttprint(f'Predicing {x_size * y_size} pixels')

			result = self.estimator.predict_proba(input_data)
			_, n_classes = result.shape

			del data_pool[input_data_key]
			gc.collect()
			
			nan_mask = np.any(np.isnan(input_data), axis=1)
			input_data[nan_mask,:] = 0
			result[nan_mask] = np.nan
			result = result.reshape(x_size, y_size, n_classes)
			result = (result * 100).astype('int8')
			
			result_pool[input_data_key] = result
			writing_futures.append(writting_pool.submit(wrinting_fn, i, input_data_key))
		
		def wrinting_fn(i, input_data_key):
			
			result = result_pool[input_data_key]
			fn_result = fn_result_list[i]
			fn_layers = fn_layers_list[i]
			
			if self.verbose:
				ttprint(f'Saving the result in {fn_result}')
			
			self._data_to_new_img(fn_layers[0], fn_result, result, spatial_win = spatial_win)
			
			del result_pool[input_data_key]
			gc.collect()
			return Path(fn_result)
		
		reading_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
		processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
		writting_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
		
		for i in range(0, len(fn_layers_list)):
			reading_futures.append(reading_pool.submit(reading_fn, (i)))
		
		for future in as_completed(reading_futures):
			print(future.result())

		for future in as_completed(processing_futures):
			print(future.result())
		
		output_fn_files = []
		
		for future in as_completed(writing_futures):
			output_fn_files.append(future.result())

		reading_pool.shutdown(wait=False)
		processing_pool.shutdown(wait=False)
		writting_pool.shutdown(wait=False)
		
		return output_fn_files
		
	def predict(self, dirs_layers:List = None, fn_layers:List = None, fn_result:str = None, spatial_win = None, \
		data_type = 'float32', fill_nodata=False, estimate_uncertainty=False, inmem_calc_func = None, dict_layers_newnames={}):

		if dirs_layers is None and fn_layers is None:
			ttprint(f'Please, inform dirs_layers or fn_layers')
			return

		if fn_layers is None:
			fn_layers = self._find_layers(dirs_layers)
		
		input_data = self._read_layers(fn_layers, spatial_win, inmem_calc_func, dict_layers_newnames)
		
		x_size, y_size, n_features = input_data.shape
		input_data = input_data.reshape(-1, n_features)

		nan_mask = None
		if fill_nodata:
			input_data = self.fill_nodata(input_data)
		else:
			nan_mask = np.any(np.isnan(input_data), axis=1)
			input_data[nan_mask,:] = 0

		if self.verbose:
			ttprint(f'Predicing {x_size * y_size} pixels')

		result = self.estimator.predict_proba(input_data)
		_, n_classes = result.shape
		
		result[nan_mask] = np.nan
		result = result.reshape(x_size, y_size, n_classes)
		result = (result * 100).astype('int8')
		
		if self.verbose:
			ttprint(f'Saving the result in {fn_result}')
		self._data_to_new_img(fn_layers[0], fn_result, result, spatial_win = spatial_win)

		if estimate_uncertainty:
			#result = self.estimator.predict_proba(input_data)
			#result = np.maximum(class_proba, 1e-15)
			#n_classes = self.pts[self.target_col].unique().size

			result = np.maximum(result, 1e-15)
			
			relative_entropy = -1 * result * np.log2(result)
			relative_entropy = 100 * relative_entropy.sum(axis=-1) / np.log2(n_classes)
			if not fill_nodata:
				relative_entropy[nan_mask] = 255
			relative_entropy = relative_entropy.round().astype(np.uint8)

			out_ext = Path(fn_result).suffix
			fn_uncertainty = fn_result.replace(out_ext, '_uncertainty'+out_ext)
			self._data_to_new_img(
				fn_layers[0], fn_uncertainty,
				relative_entropy.reshape(x_size, y_size, 1),
				spatial_win = spatial_win,
				data_type='byte', nodata=255
			)