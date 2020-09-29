from .misc import ttprint

from typing import List

import geopandas as gpd
import numpy as np
import gdal
import osr

from pathlib import Path
from geopandas import GeoDataFrame

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class LandMapper():

	def __init__(self, points:GeoDataFrame, feat_col_prfxs:List[str], target_col:str, 
		estimator:BaseEstimator = RandomForestClassifier(n_estimators=100), 
		imputer:BaseEstimator = SimpleImputer(missing_values=np.nan, strategy='mean'),
		eval_strategy = 'train_val_split', val_samples_pct = 0.2, min_samples_per_class = 0.05):

		if not isinstance(points, gpd.GeoDataFrame):
			points = gpd.read_file(points)

		self.pts = points
		self.target_col = target_col
		self.estimator = estimator
		self.imputer = imputer
		self.min_samples_per_class = min_samples_per_class

		self.feature_cols = []
		for feat_prfx in feat_col_prfxs:
			self.feature_cols += list(self.pts.columns[self.pts.columns.str.startswith(feat_prfx)])


		classes_pct = (self.pts[self.target_col].value_counts() / self.pts[target_col].count())
		self.rows_to_remove = self.pts[self.target_col].isin(classes_pct[classes_pct > min_samples_per_class].axes[0])
		nrows, _ = self.pts[~self.rows_to_remove].shape
		if nrows > 0:
			self.pts = self.pts[self.rows_to_remove]
			ttprint(f'Removing {nrows} sampes due min_samples_per_class = {min_samples_per_class}')

		self.features = self.pts[self.feature_cols].to_numpy().astype('float16')
		self.target = self.pts[self.target_col].to_numpy().astype('float16')

		self.eval_strategies = enumerate(['train_val_split'])
		if eval_strategy not in self.eval_strategies:
			ttprint(f'{eval_strategy} is a invalid validation strategy')
 
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

	def _train_val_split(self):
		train_feat, val_feat, train_targ, val_targ = train_test_split(self.features, self.target, test_size=self.val_samples_pct)
		ttprint('Training and evaluating the model')
		self.estimator.fit(train_feat, train_targ)

		pred_targ = self.estimator.predict(val_feat)
		self.cm = confusion_matrix(val_targ, pred_targ)
		self.overall_acc = accuracy_score(val_targ, pred_targ)

		ttprint('Training the final model using all data')
		self.estimator.fit(self.features, self.target)

	def train(self):
		method_name = '_%s' % (self.eval_strategy)
		if hasattr(self, method_name):
			train_method = getattr(self, method_name)
			train_method()
		else:
			ttprint(f'{self.eval_strategy} is a invalid validation strategy')

	def _feature_idx(self, fn_layer):
		return self.feature_cols.index(fn_layer.stem)

	def _data_to_new_img(self, fn_base_img, fn_new_img, data, data_type = None, img_format = 'GTiff', nodata = 0):
    
	  driver = gdal.GetDriverByName(img_format)
	  base_ds = gdal.Open( str(fn_base_img) )

	  x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
	  nbands, y_size, x_size = data.shape
	  
	  out_srs = osr.SpatialReference()
	  out_srs.ImportFromWkt(base_ds.GetProjectionRef())

	  if data_type is None:
	    data_type = base_ds.GetRasterBand(1).DataType

	  new_ds = driver.Create(fn_new_img, x_size, y_size, nbands, data_type)
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
				else:
					print(f'Ignoring {fn_layer}')

		fn_layers.sort(key=self._feature_idx)

		return fn_layers

	def read_data(self, fn_layers):
		result = []
		
		for fn_layer in fn_layers:
			ttprint(f'Reading {fn_layer}')
			ds = gdal.Open(str(fn_layer))
			nodata = ds.GetRasterBand(1).GetNoDataValue()
			band_data = ds.GetRasterBand(1).ReadAsArray().astype('Float16')
			band_data[band_data == nodata] = self.imputer.missing_values
			result.append(band_data)
		
		result = np.stack(result, axis=2)

		return result

	def predict(self, dirs_layers:List, fn_result:str, data_type = gdal.GDT_Float32):
		
		fn_layers = self._find_layers(dirs_layers)
		input_data = self.read_data(fn_layers)
		
		x_size, y_size, n_features = input_data.shape
		input_data = input_data.reshape(-1, n_features)

		input_data = self.fill_nodata(input_data)

		ttprint(f'Predicing {x_size * y_size} pixels')
		result = self.estimator.predict(input_data)
		result = result.reshape(1, x_size, y_size)
		
		ttprint(f'Saving the result in {fn_result}')
		self._data_to_new_img(fn_layers[0], fn_result, result, data_type = data_type)