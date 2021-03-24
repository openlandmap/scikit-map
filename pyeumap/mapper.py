from abc import ABC, abstractmethod

from pyeumap.misc import ttprint, find_files
from pyeumap.raster import read_rasters, write_new_raster

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

from enum import Enum
from pyeumap import parallel
from pathlib import Path

from geopandas import GeoDataFrame
from pandas import DataFrame

import gc
from concurrent.futures import as_completed, ThreadPoolExecutor

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

class PredictionStrategyType(Enum):
    Lazy = 1
    Eager = 2

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

    self.nodata_imputer = nodata_imputer
    self.estimator_list = self._set_list(estimator, estimator_list, 'estimator')
    self.hyperpar_selection_list = self._set_list(hyperpar_selection, hyperpar_selection_list)
    self.feature_selections_list = self._set_list(feature_selection, feature_selections_list)
    self.meta_estimator, self.meta_features = self._meta_estimator(meta_estimator)
    self.hyperpar_selection_meta = hyperpar_selection_meta

    self.cv = cv
    self.cv_njobs = cv_njobs

    self.pred_method = self._pred_method(pred_method)
      
    #if _automl_enabled:
    # self.estimator_list = [ AutoSklearnClassifier(**autosklearn_kwargs) ]

    self._min_samples_restriction(min_samples_per_class)
    self.features = np.ascontiguousarray(self.pts[self.feature_cols].to_numpy(), dtype=np.float32)
    self.target = target = np.ascontiguousarray(self.pts[self.target_col].to_numpy(), dtype=np.float32)

    self._target_transformation()

    self.samples_weight = self._get_column_if_exists(weight_col, 'weight_col')
    self.cv_groups = self._get_column_if_exists(cv_group_col, 'cv_group_col')

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
        return self.pts[column_name]
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

  def _target_transformation(self):
  
    self._verbose(f"Transforming {self.target_col}:")

    self.target_le = preprocessing.LabelEncoder()
    self.target = self.target_le.fit_transform(self.target)
    
    self.target_classes = {
      'original': self.target_le.classes_,
      'transformed': self.target_le.transform(self.target_le.classes_)
    }

    self._verbose(f" -Original classes: {self.target_classes['original']}")
    self._verbose(f" -Transformed classes: {self.target_classes['transformed']}")

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
    if  self.pred_method == 'predict_proba' and self._is_keras_classifier(estimator):
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

  def _feature_idx(self, fn_layer):
    return self.feature_cols.index(fn_layer.stem)

  def _reorder_layers(self, layernames, dict_layers_newnames, input_data):
    
    sorted_input_data = []
    for feature_col in self.feature_cols:
      try:
        idx = layernames.index(feature_col)
        sorted_input_data.append(input_data[:,:,idx])
      except:
        raise Exception(f"The feature {feature_col} was not provided")
    
    return np.stack(sorted_input_data, axis=2)

  def _read_layers(self, dirs_layers:List = [], fn_layers:List = [], spatial_win = None, 
    allow_aditional_layers = False, inmem_calc_func = None, dict_layers_newnames={}):
    
    n_jobs = 5
    
    raster_data, raster_files = read_rasters(raster_dirs=dirs_layers, raster_files=fn_layers, \
                                      spatial_win=spatial_win, n_jobs=n_jobs, verbose=self.verbose)

    feature_cols_set = set(self.feature_cols)
    layernames = []

    for raster_file in raster_files:
      
      layername = raster_file.stem
      
      for newname in dict_layers_newnames.keys():
        layername = re.sub(dict_layers_newnames[newname], newname, layername)
      
      if not allow_aditional_layers and layername not in feature_cols_set:
        raise Exception(f"Layer {layername} does not exist as feature_cols.\nUse dict_layers_newnames param to match their names")
      
      layernames.append(layername)

    if inmem_calc_func is not None:
      layernames, raster_data = inmem_calc_func(layernames, raster_data, spatial_win)
    
    return self._reorder_layers(layernames, dict_layers_newnames, raster_data), raster_files

  def _write_layer(self, fn_base_layer, fn_output, input_data_shape, output_data, 
    nan_mask, separate_probs = True, spatial_win = None, scale=1, 
    dtype = 'uint8', new_suffix = None):

    if len(output_data.shape) < 2:
      n_classes = 1
    else:
      _, n_classes = output_data.shape

    if not isinstance(output_data, np.floating):
      output_data = output_data.astype('float32')

    output_data[nan_mask] = np.nan
    output_data = output_data.reshape(input_data_shape[0], input_data_shape[1], n_classes)
    output_data = (output_data * scale).astype(dtype)
    
    if new_suffix is not None:
      out_ext = Path(fn_output).suffix
      fn_output = fn_output.replace(out_ext, new_suffix + out_ext)

    fn_output_list = []

    if separate_probs:
      
      for i in range(0,n_classes):
        out_ext = Path(fn_output).suffix
        fn_output_c = fn_output.replace(out_ext, f'_b{i+1}' + out_ext)

        write_new_raster(fn_base_layer, fn_output_c, output_data[:,:,i:i+1], spatial_win = spatial_win)
        fn_output_list += [ fn_output_c ]

    else:
      write_new_raster(fn_base_layer, fn_output, output_data, spatial_win = spatial_win)
      fn_output_list += [ fn_output ]
    
    return fn_output_list

  def _write_layers(self, fn_base_layer, fn_output, input_data_shape, pred_result, 
    pred_uncer, nan_mask, spatial_win, separate_probs, hard_class):

    fn_out_files = []

    if self.pred_method != 'predict_proba':
      separate_probs = False

    fn_pred_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_result, nan_mask, 
      separate_probs = separate_probs, spatial_win = spatial_win, scale=100)
    fn_out_files += fn_pred_files

    if self.pred_method == 'predict_proba':

      if pred_uncer is not None:
        
        fn_uncer_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_uncer, nan_mask, 
          separate_probs = separate_probs, spatial_win = spatial_win, scale=100, new_suffix = '_uncertainty')
        fn_out_files += fn_uncer_files

      if hard_class:

        pred_argmax = np.argmax(pred_result, axis=1)
        pred_argmax_prob = np.take_along_axis(pred_result, np.stack([pred_argmax], axis=1), axis=1)
        
        if pred_uncer is not None and pred_uncer.ndim > 2:
          pred_argmax_uncer = np.take_along_axis(pred_uncer, np.stack([pred_argmax], axis=1), axis=1)

        pred_argmax += 1

        fn_hcl_file = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax, nan_mask, 
          separate_probs = False, spatial_win = spatial_win, new_suffix = '_hcl')
        fn_out_files += fn_hcl_file

        fn_hcl_prob_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax_prob, nan_mask, 
          separate_probs = False, spatial_win = spatial_win, scale=100, new_suffix = '_hcl_prob')
        fn_out_files += fn_hcl_prob_files

        if pred_uncer is not None and pred_uncer.ndim > 2:
          fn_hcl_uncer_file = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax_uncer, nan_mask, 
            separate_probs = False, spatial_win = spatial_win, scale=100, new_suffix = '_hcl_uncertainty')
          fn_out_files += fn_hcl_uncer_file

    return fn_out_files

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
    if self.meta_estimator is not None:
      self._verbose(f'Training meta-estimator using all samples')
      self.meta_estimator.fit(self.meta_features, self.target, **self._fit_params(self.meta_estimator))

  def _relative_entropy(self, pred_result):
    _, n_classes = pred_result.shape

    classes_proba = np.maximum(pred_result, 1e-15)
      
    relative_entropy_pred = -1 * classes_proba * np.log2(classes_proba)
    relative_entropy_pred = relative_entropy_pred.sum(axis=1) / np.log2(n_classes)

    return relative_entropy_pred

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
      self._verbose(f'{estimator_name} prediction time: {(time.time() - start):.2f} segs')

    if self.meta_estimator is None:
      
      estimator_pred = estimators_pred[0]
      relative_entropy_pred = None

      #Produce almost the same data of the invese of the
      #highest probability (hcl_prob.tif)
      #if self.pred_method == 'predict_proba':
      #  relative_entropy_pred = self._relative_entropy(estimator_pred)
      #  relative_entropy_pred = relative_entropy_pred.astype('float32')

      return estimator_pred.astype('float32'), relative_entropy_pred

    else:
      
      start = time.time()
      meta_estimator_name = type(self.meta_estimator).__name__
      self._verbose(f'Executing {meta_estimator_name}')

      input_meta_features = np.concatenate(estimators_pred, axis=1)
      std_meta_features = np.std(np.stack(estimators_pred, axis=2), axis=2)

      meta_estimator_pred_method = getattr(self.meta_estimator, self.pred_method)
      meta_estimator_pred = meta_estimator_pred_method(input_meta_features)
      self._verbose(f'{meta_estimator_name} prediction time: {(time.time() - start):.2f} segs')

      return meta_estimator_pred.astype('float32'), std_meta_features.astype('float32')

  def predict_points(self, input_points):
    
    input_data = np.ascontiguousarray(input_points[self.feature_cols].to_numpy(), dtype=np.float32)

    n_points, _ = input_data.shape
    self._verbose(f'Predicting {n_points} points')
    
    return self._predict(input_data)
  
  def predict(self, dirs_layers:List = [], fn_layers:List = [], fn_output:str = None, 
    spatial_win = None, data_type = 'float32', fill_nodata = False, separate_probs = True, 
    hard_class = True, inmem_calc_func = None, dict_layers_newnames = {}):

    n_jobs = 4

    input_data, fn_layers = self._read_layers(dirs_layers, fn_layers, spatial_win, \
      inmem_calc_func=inmem_calc_func, dict_layers_newnames=dict_layers_newnames)

    x_size, y_size, n_features = input_data.shape
    
    input_data_shape = input_data.shape
    input_data = input_data.reshape(-1, input_data_shape[2])

    nan_mask = None
    if fill_nodata:
      input_data = self.fill_nodata(input_data)
    else:
      nan_mask = np.any(np.isnan(input_data), axis=1)
      input_data[nan_mask,:] = 0

    fn_base_layer = fn_layers[0]

    pred_result, pred_uncer = self._predict(input_data)

    fn_out_files = self._write_layers(fn_base_layer, fn_output, input_data_shape, pred_result, \
      pred_uncer, nan_mask, spatial_win, separate_probs, hard_class)
    
    fn_out_files.sort()

    return fn_out_files

  def predict_multi(self, dirs_layers_list:List = [], fn_layers_list:List = [], fn_output_list:List = [], spatial_win = None,
    data_type = 'float32', fill_nodata = False, separate_probs = True, hard_class = True, 
    inmem_calc_func = None, dict_layers_newnames_list:list = [], allow_aditional_layers=False,
    prediction_strategy_type = PredictionStrategyType.Lazy):

    PredictionStrategyClass = LazyLoadPrediction
    if PredictionStrategyType.Eager == prediction_strategy_type:
      PredictionStrategyClass = EagerLoadPrediction

    prediction_strategy = PredictionStrategyClass(self, dirs_layers_list, fn_layers_list, fn_output_list,
        spatial_win, data_type, fill_nodata, separate_probs, hard_class, inmem_calc_func,
        dict_layers_newnames_list, allow_aditional_layers)

    return prediction_strategy.run()

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

class PredictionStrategy(ABC):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               data_type = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_aditional_layers=False):
    
    self.landmapper = landmapper

    self.fn_layers_list = self._fn_layers_list(dirs_layers_list, fn_layers_list)
    self.fn_output_list = fn_output_list
    
    self.spatial_win = spatial_win
    self.data_type = data_type
    self.fill_nodata = fill_nodata
    self.separate_probs = separate_probs
    self.hard_class = hard_class
    self.inmem_calc_func = inmem_calc_func
    self.dict_layers_newnames_list = dict_layers_newnames_list
    self.allow_aditional_layers = allow_aditional_layers

    super().__init__()

  def _fn_layers_list(self, dirs_layers_list, fn_layers_list):
    if len(fn_layers_list) == 0:
      for dirs_layers in dirs_layers_list:
        fn_layers_list.append(find_files(dirs_layers, '*.tif'))

    return fn_layers_list

  @abstractmethod
  def run(self):
    pass

class EagerLoadPrediction(PredictionStrategy):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               data_type = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_aditional_layers=False):
    
    super().__init__(landmapper, dirs_layers_list, fn_layers_list, fn_output_list,
      spatial_win, data_type, fill_nodata, separate_probs, hard_class, inmem_calc_func,
      dict_layers_newnames_list, allow_aditional_layers)

    self.reading_futures = []
    self.writing_futures = []

    self.reading_pool = ThreadPoolExecutor(max_workers = 5)
    self.writing_pool = ThreadPoolExecutor(max_workers = 5)

  def reading_fn(self, i):
    
    fn_layers = self.fn_layers_list[i]
    
    dict_layers_newnames = {}
    if len(self.dict_layers_newnames_list) > 0:
      self.dict_layers_newnames_list[i]
    
    start = time.time()
    input_data, _ = self.landmapper._read_layers(fn_layers=fn_layers, 
      spatial_win=self.spatial_win, inmem_calc_func = self.inmem_calc_func, 
      dict_layers_newnames = dict_layers_newnames)
    
    self.landmapper._verbose(f'{i+1}) Reading time: {(time.time() - start):.2f} segs')

    input_shape = input_data.shape
    input_data = input_data.reshape(-1, input_shape[-1])

    return (i, input_shape, input_data)

  def writing_fn(self, i, pred_result, pred_uncer, nan_mask, input_shape):
    
    fn_layers = self.fn_layers_list[i]
    fn_output = self.fn_output_list[i]
    
    fn_base_layer = fn_layers[0]

    start = time.time()
    fn_out_files = self.landmapper._write_layers(fn_base_layer, fn_output, input_shape, pred_result, \
      pred_uncer, nan_mask, self.spatial_win, self.separate_probs, self.hard_class)
    self.landmapper._verbose(f'{i+1}) Saving time: {(time.time() - start):.2f} segs')

    return fn_out_files

  def run(self):

    output_fn_files = []

    for i in range(0, len(self.fn_layers_list)):
      self.reading_futures.append(
        self.reading_pool.submit(self.reading_fn, (i))
      )
    
    positions, input_shape, input_data = zip(*[ future.result() for future in as_completed(self.reading_futures) ])
    input_data = np.concatenate(input_data, axis=0)

    self.landmapper._verbose(f'{i+1}) Predicting {input_data.shape[0]} pixels')

    nan_mask = np.isnan(input_data)
    input_data[nan_mask] = 0

    start = time.time()
    pred_result, pred_uncer = self.landmapper._predict(input_data)
    self.landmapper._verbose(f'{i+1}) Predicting time: {(time.time() - start):.2f} segs')

    del input_data
    gc.collect()
    
    nan_mask = np.any(nan_mask, axis=1)    

    n_elements, _ = pred_result.shape
    year_data_size = n_elements / len(positions)

    for i in positions:

      i0 = int(i * year_data_size)
      i1 = int((i+1) * year_data_size)
      
      pred_result_year = pred_result[i0:i1,:]
      pred_uncer_year = pred_uncer[i0:i1,:]
      nan_mask_year = nan_mask[i0:i1]
      input_shape_year = input_shape[i]

      self.writing_futures.append(
        self.writing_pool.submit(self.writing_fn, i, pred_result_year, pred_uncer_year, nan_mask_year, input_shape_year)
      )

    output_fn_files = []
    for future in as_completed(self.writing_futures):
      output_fn_files += future.result()

    self.reading_pool.shutdown(wait=False)
    self.writing_pool.shutdown(wait=False)

    output_fn_files.sort()

    return output_fn_files

class LazyLoadPrediction(PredictionStrategy):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               data_type = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_aditional_layers=False,
               reading_pool_size = 1, 
               processing_pool_size = 1, 
               writing_pool_size = 1):
    
    super().__init__(landmapper, dirs_layers_list, fn_layers_list, fn_output_list,
      spatial_win, data_type, fill_nodata, separate_probs, hard_class, inmem_calc_func,
      dict_layers_newnames_list, allow_aditional_layers)

    self.data_pool = {}
    self.result_pool = {}
    
    self.reading_futures = []
    self.processing_futures = []
    self.writing_futures = []

    self.reading_pool = ThreadPoolExecutor(max_workers = reading_pool_size)
    self.processing_pool = ThreadPoolExecutor(max_workers = processing_pool_size)
    self.writing_pool = ThreadPoolExecutor(max_workers = writing_pool_size)

  def reading_fn(self, i):
    
    fn_layers = self.fn_layers_list[i]
    
    dict_layers_newnames = {}
    if len(self.dict_layers_newnames_list) > 0:
      self.dict_layers_newnames_list[i]
    
    start = time.time()
    input_data, _ = self.landmapper._read_layers(fn_layers=fn_layers, 
      spatial_win=self.spatial_win, inmem_calc_func = self.inmem_calc_func, 
      dict_layers_newnames = dict_layers_newnames)
    
    self.landmapper._verbose(f'{i+1}) Reading time: {(time.time() - start):.2f} segs')

    input_data_key = str(uuid.uuid4())
    self.data_pool[input_data_key] = input_data

    self.processing_futures.append(
      self.processing_pool.submit(self.processing_fn, i, input_data_key)
    )
    
  def processing_fn(self, i, input_data_key):
    
    input_data = self.data_pool[input_data_key]
    x_size, y_size, n_features = input_data.shape
    input_data = input_data.reshape(-1, n_features)
    
    self.landmapper._verbose(f'{i+1}) Predicting {x_size * y_size} pixels')

    nan_mask = np.isnan(input_data)
    input_data[nan_mask] = 0

    start = time.time()
    pred_result, pred_uncer = self.landmapper._predict(input_data)
    self.landmapper._verbose(f'{i+1}) Predicting time: {(time.time() - start):.2f} segs')

    del self.data_pool[input_data_key]
    gc.collect()
    
    nan_mask = np.any(nan_mask, axis=1)
    
    input_data_shape = (x_size, y_size, n_features)
    
    self.result_pool[input_data_key] = (pred_result, pred_uncer, nan_mask, input_data_shape)
    self.writing_futures.append(
      self.writing_pool.submit(self.wrinting_fn, i, input_data_key)
    )
  
  def wrinting_fn(self, i, input_data_key):
    
    pred_result, pred_uncer, nan_mask, input_data_shape = self.result_pool[input_data_key]
    fn_layers = self.fn_layers_list[i]
    fn_output = self.fn_output_list[i]
    
    fn_base_layer = fn_layers[0]

    start = time.time()
    fn_out_files = self.landmapper._write_layers(fn_base_layer, fn_output, input_data_shape, pred_result, \
      pred_uncer, nan_mask, self.spatial_win, self.separate_probs, self.hard_class)
    self.landmapper._verbose(f'{i+1}) Saving time: {(time.time() - start):.2f} segs')

    del self.result_pool[input_data_key]
    gc.collect()
    
    return fn_out_files
  
  def run(self):

    output_fn_files = []

    for i in range(0, len(self.fn_layers_list)):
      self.reading_futures.append(
        self.reading_pool.submit(self.reading_fn, (i))
      )
    
    reading_results = [ future for future in as_completed(self.reading_futures) ]
    processing_results = [ future for future in as_completed(self.processing_futures) ]
    output_fn_files = sum([ future.result() for future in as_completed(self.writing_futures) ], [])

    self.reading_pool.shutdown(wait=False)
    self.processing_pool.shutdown(wait=False)
    self.writing_pool.shutdown(wait=False)

    output_fn_files.sort()

    return output_fn_files
