from pyeumap.misc import ttprint

from typing import List, Union

import multiprocessing
import geopandas as gpd
import numpy as np
import gdal
import osr
import math

from pathlib import Path
from geopandas import GeoDataFrame

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

_automl_enabled = False
try:
	from autosklearn.classification import AutoSklearnClassifier
	_automl_enabled = True
except ImportError:
	pass

class LandMapper():

	def __init__(self, points:GeoDataFrame, feat_col_prfxs:List[str], target_col:str,
		estimator:Union[BaseEstimator, None] = None,
		imputer:BaseEstimator = SimpleImputer(missing_values=np.nan, strategy='mean'),
		eval_strategy = 'train_val_split', val_samples_pct = 0.2, min_samples_per_class = 0.05,
		weight_col = None, cv = 5, param_grid = {}, verbose = True,
		**autosklearn_kwargs):

		if not isinstance(points, gpd.GeoDataFrame):
			points = gpd.read_file(points)

		self.verbose = verbose
		self.pts = points
		self.target_col = target_col
		self.imputer = imputer
		self.weight_col = weight_col
		self.min_samples_per_class = min_samples_per_class

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

		if self.weight_col is not None:
			self.feature_cols.append(self.weight_col)
			self.weight_idx = self.pts[self.feature_cols].columns.get_loc(self.weight_col)
			print(self.weight_idx)

		self.features = self.pts[self.feature_cols].to_numpy().astype('float16')
		self.target = self.pts[self.target_col].to_numpy().astype('float16')

		self.cv = cv
		self.param_grid = param_grid

		self.eval_strategy = eval_strategy
		self.val_samples_pct = val_samples_pct

		self.features_raw = self.features
		self.features = self.fill_nodata(self.features, fit_and_tranform = True)


	def fill_nodata(self, data, fit_and_tranform = False):
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

	def _grid_search_cv(self):

		if self.verbose:
			ttprint('Training and evaluating the model')

		search_cpu_count = 1
		estimator_cpu_count = self.estimator.n_jobs
		if estimator_cpu_count != -1:
			search_cpu_count = math.ceil(multiprocessing.cpu_count() / estimator_cpu_count)

		weight = None
		if self.weight_col != None:
			weight = self.features[:,self.weight_idx]
			ttprint(f'Using {self.weight_col} as weight')
			self.features = np.delete(self.features, self.weight_idx, 1)

		self.grid_search = GridSearchCV(self.estimator, self.param_grid, cv=self.cv,
															scoring="accuracy",
															return_train_score=True,
															verbose=self.verbose, refit = True,
															n_jobs=search_cpu_count)

		self.grid_search.fit(self.features, self.target, sample_weight=weight)

	def _train_val_split(self):
		train_feat, val_feat, train_targ, val_targ = train_test_split(self.features, self.target, test_size=self.val_samples_pct)

		train_feat_weight = None
		features_weight = None
		if self.weight_col != None:
			ttprint(f'Using {self.weight_col} as weight')

			train_feat_weight = train_feat[:,self.weight_idx]
			train_feat = np.delete(train_feat, self.weight_idx, 1)
			val_feat = np.delete(val_feat, self.weight_idx, 1)

			features_weight = self.features[:,self.weight_idx]
			self.features = np.delete(self.features, self.weight_idx, 1)

		if self.verbose:
			ttprint('Training and evaluating the model')

		if isinstance(self.estimator, AutoSklearnClassifier):
			self.estimator.fit(train_feat, train_targ)
		else:
			self.estimator.fit(train_feat, train_targ, sample_weight=train_feat_weight)

		pred_targ = self.estimator.predict(val_feat)
		self.cm = confusion_matrix(val_targ, pred_targ)
		self.overall_acc = accuracy_score(val_targ, pred_targ)
		self.classification_report = classification_report(val_targ, pred_targ)

		if not isinstance(self.estimator, AutoSklearnClassifier):
			if self.verbose:
				ttprint('Training the final model using all data')
			self.estimator.fit(self.features, self.target, sample_weight=features_weight)

	def train(self):
		method_name = '_%s' % (self.eval_strategy)
		if hasattr(self, method_name):
			train_method = getattr(self, method_name)
			train_method()
		else:
			ttprint(f'{self.eval_strategy} is a invalid validation strategy')

	def _feature_idx(self, fn_layer):
		return self.feature_cols.index(fn_layer.stem)

	def _data_to_new_img(self, fn_base_img, fn_new_img, data, data_type = None, img_format = 'GTiff', nodata = 0, options = ["TILED=YES", "COMPRESS=LZW"]):
		driver = gdal.GetDriverByName(img_format)
		base_ds = gdal.Open( str(fn_base_img) )

		x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
		nbands, y_size, x_size = data.shape

		out_srs = osr.SpatialReference()
		out_srs.ImportFromWkt(base_ds.GetProjectionRef())

		if data_type is None:
			data_type = base_ds.GetRasterBand(1).DataType

		new_ds = driver.Create(fn_new_img, x_size, y_size, nbands, data_type, options = options)
		new_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
		new_ds.SetProjection(out_srs.ExportToWkt())

		for band in range(0, nbands):
			new_band = new_ds.GetRasterBand((band+1))
			new_band.WriteArray(data[band,:,:],0,0)
			new_band.SetNoDataValue(nodata)

		new_ds.FlushCache()

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

	def read_data(self, fn_layers):
		result = []

		for fn_layer in fn_layers:
			if self.verbose:
				ttprint(f'Reading {fn_layer}')

			ds = gdal.Open(str(fn_layer))
			nodata = ds.GetRasterBand(1).GetNoDataValue()
			band_data = ds.GetRasterBand(1).ReadAsArray().astype('Float16')
			band_data[band_data == nodata] = self.imputer.missing_values
			result.append(band_data)

		result = np.stack(result, axis=2)

		return result

	def predict(self, dirs_layers:List, fn_result:str, data_type = gdal.GDT_Float32, fill_nodata=False, estimate_uncertainty=False):

		fn_layers = self._find_layers(dirs_layers)
		input_data = self.read_data(fn_layers)

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

		result = self.estimator.predict(input_data)
		result[nan_mask] = np.nan
		result = result.reshape(1, x_size, y_size)

		if self.verbose:
			ttprint(f'Saving the result in {fn_result}')
		self._data_to_new_img(fn_layers[0], fn_result, result, data_type = data_type)

		if estimate_uncertainty:
			class_proba = self.estimator.predict_proba(input_data)
			class_proba = np.maximum(class_proba, 1e-15)
			n_classes = self.pts[self.target_col].unique().size

			relative_entropy = -1 * class_proba * np.log2(class_proba)
			relative_entropy = 100 * relative_entropy.sum(axis=-1) / np.log2(n_classes)
			if not fill_nodata:
				relative_entropy[nan_mask] = 255
			relative_entropy = relative_entropy.round().astype(np.uint8)

			out_ext = Path(fn_result).suffix
			fn_uncertainty = fn_result.replace(out_ext, '_uncertainty'+out_ext)
			self._data_to_new_img(
				fn_layers[0], fn_uncertainty,
				relative_entropy.reshape(1, x_size, y_size),
				data_type=gdal.GDT_Byte, nodata=255
			)
