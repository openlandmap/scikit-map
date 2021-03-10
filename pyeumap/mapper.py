from pyeumap.misc import ttprint, data_to_new_img

from typing import List, Union
import joblib

import multiprocessing
import geopandas as gpd
import numpy as np
import gdal
import osr
import os
import math
import rasterio
import re
import time

from pyeumap import parallel
from pathlib import Path

from geopandas import GeoDataFrame
from pandas import DataFrame

import gc
import concurrent.futures
from concurrent.futures import as_completed

import uuid
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold,BaseCrossValidator
from sklearn.model_selection import cross_val_predict

_automl_enabled = False
try:
	from autosklearn.classification import AutoSklearnClassifier
	_automl_enabled = True
except ImportError:
	pass

DEFAULT = {
	'META_ESTIMATOR': LogisticRegression(),
	'ESTIMATOR': RandomForestClassifier(),
	'CV': KFold(5)
}

#imputer:BaseEstimator = SimpleImputer(missing_values=np.nan, strategy='mean')

class LandMapper():

	def __init__(self, 
				points:Union[DataFrame, Path], 
				feat_col_prfxs:List[str], 
				target_col:str,
				weight_col:Union[str, None] = None, 
				nodata_imputer:Union[BaseEstimator, None] = None,
				estimator:Union[BaseEstimator, None] = DEFAULT['ESTIMATOR'],
				estimator_list:Union[List[BaseEstimator], None] = None,
				meta_estimator:BaseEstimator = DEFAULT['META_ESTIMATOR'],
				hyperpar_selection:Union[BaseEstimator, None] = None,
				hyperpar_selection_list:Union[BaseEstimator, None] = None,
				hyperpar_selection_meta:Union[List[BaseEstimator], None] = None,
				feature_selection:Union[BaseEstimator, None] = None,
				feature_selections_list:Union[BaseEstimator, None] = None,
				cv:BaseCrossValidator = DEFAULT['CV'],
				cv_njobs:int = 1,
				cv_group_col:str = None,
				min_samples_per_class:float = 0.05,
				pred_method:str = 'predict',
				verbose:bool = True):

		self.verbose = verbose
		self.pts = self._pts(points)
		self.target_col = target_col
		
		self.feature_cols = self._feature_cols(feat_col_prfxs)
		self.samples_weight = self._get_column_if_exists(weight_col, 'weight_col')

		self.nodata_imputer = nodata_imputer
		self.estimator_list = self._set_list(estimator, estimator_list, 'estimator')
		self.hyperpar_selection_list = self._set_list(hyperpar_selection, hyperpar_selection_list)
		self.feature_selections_list = self._set_list(feature_selection, feature_selections_list)
		self.meta_estimator, self.meta_features = self._meta_estimator(meta_estimator)
		self.hyperpar_selection_meta = hyperpar_selection_meta

		self.cv = cv
		self.cv_njobs = cv_njobs
		self.cv_groups = self._get_column_if_exists(cv_group_col, 'cv_group_col')

		self.pred_method = self._pred_method(pred_method)
      
		#if _automl_enabled:
		#	self.estimator_list = [ AutoSklearnClassifier(**autosklearn_kwargs) ]

		self._min_samples_restriction(min_samples_per_class)
		self.features = np.ascontiguousarray(self.pts[self.feature_cols].to_numpy(), dtype=np.float32)
		self.target = np.ascontiguousarray(self.pts[self.target_col].to_numpy(), dtype=np.float32)

		if self.nodata_imputer is not None:
			self.features = self._impute_nodata(self.features, fit_and_tranform = True)
	
	def _pts(self, points):
		if isinstance(points, Path):
			suffix = points.suffix
			if suffix == '.csv':
				return gpd.read_csv(points)
			elif suffix == '.gz':
				return gpd.read_csv(points, compression='gzip')
			else:
				return gpd.read_file(points)
		elif isinstance(points, DataFrame):
			return points
		else:
			return points

	def _feature_cols(self, feat_col_prfxs):
		feature_cols = []
		for feat_prfx in feat_col_prfxs:
			feature_cols += list(self.pts.columns[self.pts.columns.str.startswith(feat_prfx)])
		return feature_cols

	def _get_column_if_exists(self, column_name, param_name):
		if column_name is not None:
			if column_name in self.pts.columns:
				return self.pts[self.column_name]
			else:	
				self._verbose(f'Ignoring {param_name}, because {column_name} column not exists.')
		else:
			return None
			 # features_weight

	def _set_list(self, obj, obj_list, var_name = None):
		empty_obj = (obj is None)
		empty_list = (obj_list is None or len(obj_list) == 0)
		if not empty_list:
			return obj_list
		elif not empty_obj and empty_list:
			return [obj]
		elif var_name is not None:
			raise Exception(f'You should provide at least one of these: {var_name} or {var_name}_list.')
		else:
			return []

	def _meta_estimator(self, meta_estimator):
		if len(self.estimator_list) > 1:
			return meta_estimator, []
		else:
			return None, None

	def _pred_method(self, pred_method):
		if self.meta_estimator is not None:
			return 'predict_proba'
		else:
			return pred_method

	def _min_samples_restriction(self, min_samples_per_class):
		classes_pct = (self.pts[self.target_col].value_counts() / self.pts[self.target_col].count())
		rows_to_remove = self.pts[self.target_col].isin(classes_pct[classes_pct >= min_samples_per_class].axes[0])
		nrows, _ = self.pts[~rows_to_remove].shape
		if nrows > 0:
			self.pts = self.pts[rows_to_remove]
			self._verbose(f'Removing {nrows} samples due min_samples_per_class condition (< {min_samples_per_class})')

	def _impute_nodata(self, data, fit_and_tranform = False):
		nodata_idx = self._nodata_idx(data)
		num_nodata_idx = np.sum(nodata_idx)
		pct_nodata_idx = num_nodata_idx / data.size * 100

		if (num_nodata_idx > 0):
			self._verbose(f'Filling the missing values ({pct_nodata_idx:.2f}% / {num_nodata_idx} values)...')
			
			if fit_and_tranform:
				data = self.nodata_imputer.fit_transform(data)
			else:
				data = self.nodata_imputer.transform(data)
			
		return data

	def _nodata_idx(self, data):
		if np.isnan(self.nodata_imputer.missing_values):
			return np.isnan(data)
		else:
			return (data == self.nodata_imputer.missing_values)

	def _verbose(self, *args, **kwargs):
		if self.verbose:
			ttprint(*args, **kwargs)

	def _best_params(self, hyperpar_selection):
			
		means = hyperpar_selection.cv_results_['mean_test_score']*-1
		stds = hyperpar_selection.cv_results_['std_test_score']
		
		for mean, std, params in zip(means, stds, hyperpar_selection.cv_results_['params']):
			self._verbose(f" {mean:.5f} (+/-{2*std:.05f}) from {params}")
		self._verbose(f'Best: {hyperpar_selection.best_score_:.5f} using {hyperpar_selection.best_params_}')
			
		return hyperpar_selection.best_params_

	def _class_optimal_th(self, curv_precision, curv_recall, curv_th):
		# Removing elements where the precision or recall are zero
		nonzero_mask = np.logical_and((curv_precision != 0.0), (curv_recall != 0.0))
		optimal_idx = np.argmax(1 - np.abs(curv_precision[nonzero_mask] - curv_recall[nonzero_mask]))
		return curv_recall[optimal_idx], curv_precision[optimal_idx], curv_th[optimal_idx]

	def _classification_report_prob(self):
		classes, cnt = np.unique(self.target, return_counts=True)

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
				c_mask = (self.target == c)
				me['log_loss'][c] = metrics.log_loss(self.target[c_mask], self.eval_pred[c_mask], labels=classes)

		for c_idx, c in enumerate(classes):
			me['support'][c] = cnt[c_idx]

			c_targ = (self.target == c).astype(int)
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
	
	def _calc_eval_metrics(self):
		
		self.eval_metrics = {}

		if self.pred_method == 'predict':
			self.eval_metrics['confusion_matrix'] = metrics.confusion_matrix(self.target, self.eval_pred)
			self.eval_metrics['overall_acc'] = metrics.accuracy_score(self.target, self.eval_pred)
			self.eval_report = metrics.classification_report(self.target, self.eval_pred)
		elif self.pred_method == 'predict_proba':
			self.eval_metrics['log_loss'] = metrics.log_loss(self.target, self.eval_pred)
			self.eval_report = self._classification_report_prob()

	def _fit_params(self, estimator):
		if isinstance(estimator, Pipeline):
			return {'estimator__sample_weight': self.samples_weight}
		else:
			return {'sample_weight': self.samples_weight}

	def _is_keras_classifier(self, estimator):
		return isinstance(estimator, Pipeline) and isinstance(estimator['estimator'], KerasClassifier)

	def _binarizer_target_if_needed(self, estimator):
		if 	self.pred_method == 'predict_proba' and self._is_keras_classifier(estimator):
			le = preprocessing.LabelBinarizer()
			target = le.fit_transform(self.target)
			return target
		else:
			return self.target

	def _do_hyperpar_selection(self, hyperpar_selection, estimator, features):
		
		estimator_name = type(estimator).__name__
		self._verbose(f'Optimizing hyperparameters for {estimator_name}')

		cv_njobs = self.cv_njobs
		if isinstance(self.cv, int):
			cv_njobs = self.cv
		elif isinstance(self.cv, BaseCrossValidator):
			cv_njobs = self.cv.n_splits

		hyperpar_selection.set_params(
			cv = self.cv,
			refit = False,
			n_jobs = self.cv_njobs
		)
		
		hyperpar_selection.fit(features, self.target, groups=self.cv_groups, **self._fit_params(estimator))
		estimator.set_params(**self._best_params(hyperpar_selection))

	def _do_cv_prediction(self, estimator, features):

		target = self.target
		cv_njobs = self.cv_njobs

		target = self._binarizer_target_if_needed(estimator)

		if isinstance(self.cv, int):
			cv_njobs = self.cv
		elif isinstance(self.cv, BaseCrossValidator):
			cv_njobs = self.cv.n_splits

		return cross_val_predict(estimator, features, target, method=self.pred_method, n_jobs=self.cv_njobs, \
													cv=self.cv, groups=self.cv_groups, verbose=self.verbose, fit_params = self._fit_params(estimator))

	def _do_cv_evaluation(self):
		self._verbose(f'Calculating evaluation metrics')

		if self.meta_estimator is not None:
			self.eval_pred = self._do_cv_prediction(self.meta_estimator, self.meta_features)
		else:
			self.eval_pred = self._do_cv_prediction(self.estimator_list[0], self.features)

		self._calc_eval_metrics()

	def _calc_meta_features(self):
		self._verbose(f'Calculating meta-features')

		for estimator in self.estimator_list:
			self.meta_features.append(self._do_cv_prediction(estimator, self.features))

		self.meta_features = np.concatenate(self.meta_features, axis=1)
		self._verbose(f' Meta-features shape: {self.meta_features.shape}')

	def train(self):

		# Hyperparameter optization for all estimators
		for hyperpar_selection, estimator in zip(self.hyperpar_selection_list, self.estimator_list):
			self._do_hyperpar_selection(hyperpar_selection, estimator, self.features)

		# Meta-features calculation to feed the meta-estimator
		if self.meta_estimator is not None:
			self._calc_meta_features()

		# Hyperparameter optization for the meta-estimator
		if self.hyperpar_selection_meta is not None:
			self._do_hyperpar_selection(self.hyperpar_selection_meta, self.meta_estimator, self.meta_features)

		# CV calculation using the final estimator
		self._do_cv_evaluation()

		# Training the final estimators
		for estimator in self.estimator_list:
			estimator_name = type(estimator).__name__
			self._verbose(f'Training {estimator_name} using all samples')

			target = self._binarizer_target_if_needed(estimator)
			estimator.fit(self.features, target, **self._fit_params(estimator))

		# Training the final meta-estimator
		self._verbose(f'Training meta-estimator using all samples')
		self.meta_estimator.fit(self.meta_features, self.target, **self._fit_params(self.meta_estimator))

	def _feature_idx(self, fn_layer):
		return self.feature_cols.index(fn_layer.stem)
	
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

			try:
				if 'tile' in str(fn_layer):
					band_data = ds.read(1)
				else:
					band_data = ds.read(1, window=spatial_win)
			except:
					band_data = None
					ttprint(f'ERROR: Failed to read {fn_layer} window {spatial_win}.')
					band_data = np.empty((int(spatial_win.width), int(spatial_win.height)))
					band_data[:] = np.nan
			
			if (band_data.shape[0] != band_data.shape[1]):
				pad_dif = band_data.shape[0] - band_data.shape[1]
				band_data_pad = np.pad(band_data, pad_width=(pad_dif,pad_dif), mode='edge')
				band_data_pad = band_data_pad[0:spatial_win.width, 0:spatial_win.height]
				ttprint(f'WARNING: Inconsistent block size {fn_layer}, padding the boundaries.')
				band_data = band_data_pad
			
		return fn_layer, band_data, ds.nodatavals[0]

	def _read_layers(self, fn_layers, spatial_win, inmem_calc_func, dict_layers_newnames, allow_aditional_layers=False):
		result = []
		
		max_workers = 5
		
		args = [ (fn_layer,spatial_win) for fn_layer in fn_layers]
		
		fn_layers = []
		for fn_layer, band_data, nodata in parallel.ThreadGeneratorLazy(self._get_data, iter(args), max_workers=max_workers, chunk=max_workers*2):
			if (isinstance(band_data, np.ndarray)):
				band_data = band_data.astype('Float16')
				band_data[band_data == nodata] = np.nan
				
				if (np.isnan(np.min(band_data))):
					# Slope layer hack
					ttprint(f'Layer {fn_layer} has NA values (nodata={nodata})')
					if fn_layer.name == 'dtm_slope.percent_gedi.eml_m_30m_0..0cm_2000..2018_eumap_epsg3035_v0.2.tif':
						ttprint(f'Layer {fn_layer} filling nodata=100')
						band_data[np.isnan(band_data)] = 100
			else:
				ttprint(f'Layer {fn_layer} not found')
			
			fn_layers.append(fn_layer)
			result.append(band_data)
		
		input_data = np.ascontiguousarray(np.stack(result, axis=2))
		
		feature_cols_set = set(self.feature_cols)
		layernames = []
		for fn_layer in fn_layers:
			layername = fn_layer.stem
			for newname in dict_layers_newnames.keys():
				layername = re.sub(dict_layers_newnames[newname], newname, layername)
			if not allow_aditional_layers and layername not in feature_cols_set:
				raise Exception(f"Layer {layername} does not exist as feature_cols.\nUse dict_layers_newnames param to match their names")
			layernames.append(layername)

		if inmem_calc_func is not None:
			layernames, input_data = inmem_calc_func(layernames, input_data, spatial_win)
		
		return self._reorder_data(layernames, dict_layers_newnames, input_data)

	def _reorder_data(self, layernames, dict_layers_newnames, input_data ):
		
		sorted_input_data = []
		for feature_col in self.feature_cols:
			try:
				idx = layernames.index(feature_col)
				sorted_input_data.append(input_data[:,:,idx])
			except:
				raise Exception(f"The feature {feature_col} was not provided")
		
		return np.stack(sorted_input_data, axis=2)
	
	def _predict(self, input_data):
		
		estimators_pred = []

		for estimator in self.estimator_list:
			
			start = time.time()
			estimator_name = type(estimator).__name__
			self._verbose(f'Executing {estimator_name}')
			
			if self._is_keras_classifier(estimator):
				
				n_elements, _ = input_data.shape
				pred_batch_size = int(n_elements/2)
				
				self._verbose(f'batch_size={pred_batch_size}')
				estimator['estimator'].set_params(batch_size=pred_batch_size)

			estimator_pred_method = getattr(estimator, self.pred_method)
			estimators_pred.append(estimator_pred_method(input_data))
			self._verbose(f'{estimator_name} prediction time: {time.time() - start}')

		if self.meta_estimator is None:
			
			estimator_pred = estimators_pred[0]
			relative_entropy_pred = None

			if self.pred_method == 'predict_proba':
				_, n_classes = estimator_pred.shape
				classes_proba = np.maximum(estimator_pred, 1e-15)
			
				relative_entropy_pred = -1 * classes_proba * np.log2(classes_proba)
				relative_entropy_pred = relative_entropy_pred.sum(axis=1) / np.log2(n_classes)

			return estimator_pred, relative_entropy_pred

		else:
			
			start = time.time()
			meta_estimator_name = type(self.meta_estimator).__name__
			self._verbose(f'Executing {meta_estimator_name}')

			input_meta_features = np.concatenate(estimators_pred, axis=1)
			std_meta_features = np.std(np.stack(estimators_pred, axis=2), axis=2)

			meta_estimator_pred_method = getattr(self.meta_estimator, self.pred_method)
			meta_estimator_pred = meta_estimator_pred_method(input_meta_features)
			self._verbose(f'{meta_estimator_name} prediction time: {time.time() - start}')

			return meta_estimator_pred, std_meta_features 

	def predict_points(self, input_points):
		
		input_data = np.ascontiguousarray(input_points[self.feature_cols].to_numpy(), dtype=np.float32)

		n_points, _ = input_data.shape
		self._verbose(f'Predicting {n_points} points')
		
		return self._predict(input_data)
	
	def predict_multi(self, fn_layers_list:List = None, fn_result_list:List = None, spatial_win = None, \
		data_type = 'float32', fill_nodata=False, estimate_uncertainty=False, inmem_calc_func = None, dict_layers_newnames_list:list = [],
		allow_aditional_layers=False, hard_classes=True):
		
		data_pool = {}
		result_pool = {}
		
		reading_futures = []
		processing_futures = []
		writing_futures = []
		
		def reading_fn(i):
			ttprint(f'reading {i}')
			fn_layers = fn_layers_list[i]
			dict_layers_newnames = dict_layers_newnames_list[i]
			
			start = time.time()
			input_data = self._read_layers(fn_layers, spatial_win, inmem_calc_func, dict_layers_newnames, allow_aditional_layers)
			ttprint(f'## Benchmark ## Reading time: {time.time() - start}')

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

			# Invalid spectral_indices values
			#input_data[~np.isfinite(input_data)] = 0
			#~np.isfinite(input_data)
			nan_mask = np.isnan(input_data)#np.logical_or(np.isnan(input_data), ~np.isfinite(input_data))
			input_data[nan_mask] = 0

			start = time.time()
			result, uncert = self._predict(input_data)
			ttprint(f'## Benchmark ## Processing time: {time.time() - start}')

			_, n_classes = result.shape

			del data_pool[input_data_key]
			gc.collect()
			
			nan_mask = np.any(nan_mask, axis=1)
			input_data[nan_mask,:] = 0
			
			result[nan_mask] = np.nan
			result = result.reshape(x_size, y_size, n_classes)
			result = (result * 100).round().astype('int8')

			uncert[nan_mask] = np.nan
			uncert = uncert.reshape(x_size, y_size, n_classes)
			uncert = (uncert * 100).round().astype('int8')
			
			result_pool[input_data_key] = (result, uncert)
			writing_futures.append(writting_pool.submit(wrinting_fn, i, input_data_key))
		
		def wrinting_fn(i, input_data_key):
			
			result, uncert = result_pool[input_data_key]
			fn_result = fn_result_list[i]
			fn_layers = fn_layers_list[i]
			
			out_files = []

			if self.verbose:
				ttprint(f'Saving the result in {fn_result}')
			start = time.time()
			if hard_classes:
				namask = np.all((result == 0), axis=2)
				argmax = np.argmax(result, axis=2)
				
				result_hcl_uncert = np.take_along_axis(uncert, np.stack([argmax], axis=2), axis=2)
				result_hcl_prob = np.take_along_axis(result, np.stack([argmax], axis=2), axis=2)
				
				result_hcl = argmax + 1
				result_hcl[namask] = 0
				result_hcl_uncert[namask] = 0
				result_hcl_prob[namask] = 0

				fn_hardclasses = fn_result.replace('.tif', '_hcl.tif')
				fn_hardclasses_uncer = fn_result.replace('.tif', '_hcl_uncertainty.tif')
				fn_hardclasses_prob = fn_result.replace('.tif', '_hcl_prob.tif')

				data_to_new_img(fn_layers[0], fn_hardclasses, np.stack([result_hcl], axis=2), spatial_win = spatial_win)
				data_to_new_img(fn_layers[0], fn_hardclasses_uncer, result_hcl_uncert, spatial_win = spatial_win)
				data_to_new_img(fn_layers[0], fn_hardclasses_prob, result_hcl_prob, spatial_win = spatial_win)

				out_files.append(Path(fn_hardclasses))
				out_files.append(Path(fn_hardclasses_uncer))

			fn_uncertainty = fn_result.replace('.tif', '_uncertainty.tif')

			for b in range(0, result.shape[2]):

				fn_result_b = fn_result.replace('.tif', f'_b{b}.tif')
				fn_hardclasses_uncer_b = fn_result.replace('.tif', f'_b{b}_uncertainty.tif')

				data_to_new_img(fn_layers[0], fn_result_b, result[:,:,b:b+1], spatial_win = spatial_win)
				data_to_new_img(fn_layers[0], fn_hardclasses_uncer_b, uncert[:,:,b:b+1], spatial_win = spatial_win)
			
				out_files.append(Path(fn_result_b))
				out_files.append(Path(fn_hardclasses_uncer_b))

			ttprint(f'## Benchmark ## Saving time: {time.time() - start}')
			del result_pool[input_data_key]
			gc.collect()
			
			return out_files
		
		reading_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
		processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
		writting_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
		
		for i in range(0, len(fn_layers_list)):
			reading_futures.append(reading_pool.submit(reading_fn, (i)))
		
		for future in as_completed(reading_futures):
			print(future.result())

		for future in as_completed(processing_futures):
			print(future.result())
		
		output_fn_files = []
		
		for future in as_completed(writing_futures):
			output_fn_files += future.result()

		reading_pool.shutdown(wait=False)
		processing_pool.shutdown(wait=False)
		writting_pool.shutdown(wait=False)

		return output_fn_files
		
	def predict(self, dirs_layers:List = None, fn_layers:List = None, fn_result:str = None, spatial_win = None, \
		data_type = 'float32', fill_nodata=False, estimate_uncertainty=False, inmem_calc_func = None, dict_layers_newnames={}):

		if dirs_layers is None and fn_layers is None:
			self._verbose(f'Please, inform dirs_layers or fn_layers')
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

		pred_result, pred_uncer = self._predict(input_data)

		_, n_classes = pred_result.shape

		pred_result[nan_mask] = np.nan
		pred_result = pred_result.reshape(x_size, y_size, n_classes)
		pred_result = (pred_result * 100).astype('int8')
		data_to_new_img(fn_layers[0], fn_result, pred_result, spatial_win = spatial_win)

		output_fn_files = [fn_result]

		if pred_uncer is not None:
			pred_uncer[nan_mask] = np.nan
			pred_uncer = pred_uncer.reshape(x_size, y_size, n_classes)
			pred_uncer = (pred_uncer * 100).astype('int8')
			
			out_ext = Path(fn_result).suffix
			fn_uncertainty = fn_result.replace(out_ext, '_uncertainty' + out_ext)
			
			data_to_new_img(fn_layers[0], fn_uncertainty, pred_uncer, spatial_win = spatial_win)
			output_fn_files.append(fn_uncertainty)

		return output_fn_files

	@staticmethod
	def load_instance(fn_joblib):
		if not isinstance(fn_joblib, Path):
			fn_joblib = Path(fn_joblib)

		landmapper = joblib.load(fn_joblib)
		for estimator in landmapper.estimator_list:
			if landmapper._is_keras_classifier(estimator):
				fn_keras = fn_joblib.parent.joinpath(f'{fn_joblib.stem}_kerasclassifier.h5')
				estimator['estimator'].model = load_model(fn_keras)
		
		return landmapper

	def save_instance(self, fn_joblib, no_train_data = False, compress='lz4'):
		if not isinstance(fn_joblib, Path):
			fn_joblib = Path(fn_joblib)

		if no_train_data:
			prop_to_del = [ 'pts', 'features', 'target', 'samples_weight', 'cv_groups']
			for prop in prop_to_del:
				if self.verbose:
					ttprint(f'Removing {prop} attribute')
				delattr(self, prop)
		
		for estimator in self.estimator_list:
			if self._is_keras_classifier(estimator):
				basedir = fn_joblib.parent
				fn_keras = fn_joblib.parent.joinpath(f'{fn_joblib.stem}_kerasclassifier.h5')
				estimator['estimator'].model.save(fn_keras)
				estimator['estimator'].model = None

		return joblib.dump(self, fn_joblib, compress=compress)