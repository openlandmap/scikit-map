'''
Overlay and spatial prediction fully compatible with ``scikit-learn``.
'''
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum

from concurrent.futures import as_completed, ThreadPoolExecutor
import concurrent.futures
import traceback
import joblib
import math
import time
import uuid
import gc
import re
import os

from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold, BaseCrossValidator
from sklearn.utils.validation import has_fit_parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import metrics

from rasterio.windows import Window
from pandas import DataFrame
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from . import parallel
from .misc import ttprint, find_files
from .raster import read_rasters, write_new_raster

import warnings
# warnings.filterwarnings('ignore') # should not ignore all warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

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

def build_ann(
  input_shape,
  output_shape,
  n_layers = 3,
  n_neurons = 32,
  activation = 'relu',
  dropout_rate = 0.0,
  learning_rate = 0.0001,
  output_activation = 'softmax',
  loss = 'categorical_crossentropy'
):
  """
  Helper function to create a pretty standard Artificial Neural
  Network-ANN using ``tensorflow``. It's based in a ``Sequential``
  model, which connects multiple hidden layers
  (``Dense=>Dropout=>BatchNormalization``) and uses a ``Nadam``
  optimizer. Developed to be used together with ``KerasClassifier``.

  :param input_shape: The input data shape.
  :param output_shape: The output data shape.
  :param n_layers: Number of hidden layers.
  :param n_neurons: Number of neurons for the hidden layers.
  :param activation: Activation function for the input and hidden layers.
  :param dropout_rate: Dropout rate for the ``BatchNormalization``.
  :param learning_rate: Learning rate for the optimized.
  :param output_activation: Activation function for the output layer.
  :param loss: Loss function used for the Optimizer.

  :returns: The ANN model
  :rtype: Sequential

  Examples
  ========

  >>> from skmap.mapper import build_ann
  >>> from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
  >>>
  >>> ann = KerasClassifier(build_ann, input_shape=(-1, 180), output_shape=33,
  >>>                       epochs=3, batch_size=64, shuffle=True, verbose=1)

  """

  try:
    import tensorflow as tf
    tf.autograph.set_verbosity(0)

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Nadam

  except ImportError as e:
    warnings.warn('build_ann requires tensorflow>=2.5.0')

  model = Sequential()
  model.add(Dense(input_shape, activation=activation))

  for i in range(0, n_layers):
    model.add(Dense(n_neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

  model.add(Dense(output_shape, activation=output_activation))
  model.compile(loss=loss,
      optimizer=Nadam(learning_rate=learning_rate),
  )

  return model

class PredictionStrategyType(Enum):
  """
  Strategy to read multiple raster files during the prediction

  """
  Lazy = 1 #: Load one year while predict other.
  Eager = 2 #: First load all years, then predict all.

class LandMapper():

  """
    Spatial prediction implementation based in supervised machine
    learning models and point samples.

    It's fully compatible with ``scikit-learn`` [1] supporting:

    1. Classification models,
    2. Seamless training using point samples,
    3. Data imputation,
    4. Hyper-parameter optimization,
    5. Feature selection,
    6. Ensemble machine learning (EML) and prediction uncertainty,
    7. AutoML through ``auto-sklearn`` [2],
    8. Accuracy assessment through cross-validation,
    9. Seamless raster prediction (read and write).

    :param points: Point samples used to train the ML model. It supports ``pandas.DataFrame`` and
     a path for plain CSV ``(*.csv)`` or compressed csv file ``(*.gz)``, which are read
     through ``pandas.read_csv`` [3]. All the other extensions are read by ``geopandas`` as GIS vector files [4].
    :param target_col: Column name used to retrieve the target values for the training.
    :param feat_cols: List of column names used to retrieve the feature/covariates for the training.
    :param feat_col_prfxs: List of column prefixes used to derive the ``feat_cols`` list, avoiding to provide
      dozens/hundreds of column names.
    :param weight_col: Column name used to retrieve the ``sample_weight`` for the training.
    :param nodata_imputer: Transformer used to input missing values filling all ``np.nan`` in the point
      samples. All ``sklearn.impute`` classes are supported [1].
    :param estimator: The ML model used by the class. The default model is a ``RandomForestClassifier``,
      however all the ``sklearn`` model are supported [1]. For ``estimator=None`` it tries to use ``auto-sklearn``
      to find the best model and hyper-parameters [2].
    :param estimator_list: A list of models used by the EML implementation. The models output are used to
      feed the ``meta_estimator`` model and to derive the prediction uncertainty. This argument has
      prevalence over ``estimator``.
    :param meta_estimator: Model used to derive the prediction output in the EML implementation. The default model
      here is a ``LogisticRegression``, however all the ``sklearn`` model are supported [1].
    :param hyperpar_selection: Hyper-parameter optimizer used by ``estimator`` model.
    :param hyperpar_selection_list: A list of hyper-parameter optimizers used by ``estimator_list`` models, provided
      in the same order. This argument has prevalence over ``hyperpar_selection``.
    :param hyperpar_selection_meta: Hyper-parameter optimizer used by ``meta_estimator`` model.
    :param feature_selection: Feature selection algorithm used by ``estimator`` model.
    :param feature_selections_list: A list of feature selection algorithm used by ``estimator_list`` models, provided
      in the same order. This argument has prevalence over ``feature_selection``.
    :param cv: Cross validation strategy used by all models. The default strategy is a ``5-Fold cv``,
      however all the ``sklearn`` model are supported [1].
    :param cv_njobs: Number of CPU cores to be used in parallel during the cross validation.
    :param cv_group_col: Column name used to split the train/test set during the cross validation. Use this argument
      to perform a ``spatial CV`` by block/tiles.
    :param min_samples_per_class: Minimum percentage of samples according to ``target_col`` to keep the class in the
      training.
    :param pred_method: Use ``predict_prob`` to predict probabilities and uncertainty, otherwise it predicts only
      the dominant class.
    :param apply_corr_factor: Apply a correction factor (``rmse / averaged_sd``) in the prediction uncertainty output.
    :param verbose:bool: Use ``True`` to print the progress of all steps.
    :param \*\*autosklearn_kwargs: Named arguments supported by ``auto-sklearn`` [2].

    For **usage examples** access the ``skmap`` tutorials [5,6].

    References
    ==========

    [1] `Sklearn API Reference <https://scikit-learn.org/stable/modules/classes.html>`_

    [2] `Auto-sklearn API <https://automl.github.io/auto-sklearn/master/api.html>`_

    [3] `Pandas read_csv function <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

    [4] `Geopandas read_file function <https://geopandas.org/docs/reference/api/geopandas.read_file.html>`_

    [5] `Land Cover Mapping <../notebooks/03_landcover_mapping.html>`_

    [6] `Land Cover Mapping (Advanced) <../notebooks/04_landcover_mapping_advanced.html>`_

  """

  def __init__(
    self,
    points:Union[DataFrame, Path],
    target_col:str,
    feat_cols:Union[List, None] = [],
    feat_col_prfxs:Union[List, None] = [],
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
    min_samples_per_class:float = 0,
    pred_method:str = 'predict',
    verbose:bool = True,
    apply_corr_factor:bool = False,
    **autosklearn_kwargs
  ):

    self.verbose = verbose
    self.pts = self._pts(points)
    self.target_col = target_col

    self.feat_col_prfxs = feat_col_prfxs
    self.feature_cols = self._feature_cols(feat_cols, feat_col_prfxs)

    self.nodata_imputer = nodata_imputer
    self.is_automl_estimator = (_automl_enabled and estimator is None)

    if self.is_automl_estimator:
      self.estimator_list = [ AutoSklearnClassifier(**autosklearn_kwargs) ]
    else:
      self.estimator_list = self._set_list(estimator, estimator_list, 'estimator')

    self.hyperpar_selection_list = self._set_list(hyperpar_selection, hyperpar_selection_list)
    self.feature_selections_list = self._set_list(feature_selection, feature_selections_list)
    self.meta_estimator, self.meta_features = self._meta_estimator(meta_estimator)
    self.hyperpar_selection_meta = hyperpar_selection_meta

    self.cv = cv
    self.cv_njobs = cv_njobs

    self.pred_method = self._pred_method(pred_method)

    self._min_samples_restriction(min_samples_per_class)
    self.features = np.ascontiguousarray(self.pts[self.feature_cols].to_numpy(), dtype=np.float32)
    self.target = target = np.ascontiguousarray(self.pts[self.target_col].to_numpy(), dtype=np.float32)

    self._target_transformation()

    self.samples_weight = self._get_column_if_exists(weight_col, 'weight_col')
    self.cv_groups = self._get_column_if_exists(cv_group_col, 'cv_group_col')

    self.corr_factor = 1
    self.apply_corr_factor = apply_corr_factor

    if self.nodata_imputer is not None:
      self.features = self._impute_nodata(self.features, fit_and_tranform = True)

  def _pts(self, points):
    if isinstance(points, Path):
      suffix = points.suffix
      if suffix == '.csv':
        return pd.read_csv(points)
      elif suffix == '.gz':
        return pd.read_csv(points, compression='gzip')
      else:
        return gpd.read_file(points)
    elif isinstance(points, DataFrame):
      return points
    else:
      return points

  def _feature_cols(self, feat_cols, feat_col_prfxs):
    feature_cols = []
    if len(feat_cols) > 0:
      feature_cols = list(self.pts.columns[self.pts.columns.isin(feat_cols)])
    elif len(feat_col_prfxs) > 0:
      for feat_prfx in feat_col_prfxs:
        feature_cols += list(self.pts.columns[self.pts.columns.str.startswith(feat_prfx)])
    else:
      raise Exception(f'You should provide at least one of these: feat_cols or feat_col_prfxs.')

    if len(feature_cols) == 0:
      raise Exception(f'None feature was found. Check the provided feat_cols or feat_col_prfxs.')

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
    if self.meta_estimator is not None and is_classifier(self.meta_estimator):
      return 'predict_proba'
    else:
      return pred_method

  def _min_samples_restriction(self, min_samples_per_class):
    classes_pct = (self.pts[self.target_col].value_counts() / self.pts[self.target_col].count())
    rows_to_keep = self.pts[self.target_col].isin(classes_pct[classes_pct >= min_samples_per_class].axes[0])
    nrows, _ = self.pts[~rows_to_keep].shape
    if nrows > 0:
      removed_cls = self.pts[~rows_to_keep][self.target_col].unique()
      self.pts = self.pts[rows_to_keep]
      self._verbose(f'Removing {nrows} samples ({self.target_col} in {removed_cls}) '+
        f'due min_samples_per_class condition (< {min_samples_per_class})')

  def _target_transformation(self):

    if self._is_classifier():
      self._verbose(f"Transforming {self.target_col}:")

      self.target_le = preprocessing.LabelEncoder()
      # Change to starting from 1
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

    means = hyperpar_selection.cv_results_['mean_test_score']
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

  def _is_classifier(self):
    if self.meta_estimator is not None:
      return is_classifier(self.meta_estimator)
    else:
      return is_classifier(self.estimator_list[0])

  def _calc_eval_metrics(self):

    self.eval_metrics = {}

    if self.pred_method == 'predict':
      if self._is_classifier():
        self.eval_metrics['confusion_matrix'] = metrics.confusion_matrix(self.target, self.eval_pred)
        self.eval_metrics['overall_acc'] = metrics.accuracy_score(self.target, self.eval_pred)
        self.eval_report = metrics.classification_report(self.target, self.eval_pred)
      else:
        self.eval_metrics['r2'] = metrics.r2_score(self.target, self.eval_pred)  
        self.eval_metrics['rmse'] = metrics.mean_squared_error(self.target, self.eval_pred, squared=False)  
    elif self.pred_method == 'predict_proba':
      self.eval_metrics['log_loss'] = metrics.log_loss(self.target, self.eval_pred)
      self.eval_report = self._classification_report_prob()

  def _fit_params(self, estimator):
    if isinstance(estimator, Pipeline):
      return {'estimator__sample_weight': self.samples_weight}
    if self.is_automl_estimator:
      ttprint('LandMapper is using AutoSklearnClassifier, which not supports fit_params (ex: sample_weight)')
      return {}
    elif has_fit_parameter(estimator, "sample_weight"):
      return {'sample_weight': self.samples_weight}
    else:
      return {}

  def _is_catboost_model(self,estimator):
    try:
      from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as e:
      return False

    if isinstance(estimator,Pipeline):
      return isinstance(estimator,Pipeline) and (
        isinstance(estimator['estimator'], CatBoostClassifier)
        or
        isinstance(estimator['estimator'], CatBoostRegressor)
      )
    else:
      return isinstance(estimator,CatBoostClassifier) or isinstance(estimator,CatBoostRegressor)

  def _is_keras_model(self, estimator):
    try:
      from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
      return isinstance(estimator, Pipeline) and ( 
        isinstance(estimator['estimator'], KerasClassifier) 
        or 
        isinstance(estimator['estimator'], KerasRegressor) 
      )
    except ImportError as e:
      return False

  def _binarizer_target_if_needed(self, estimator):
    if  self.pred_method == 'predict_proba' and self._is_keras_model(estimator):
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
      if self.apply_corr_factor:
        self.corr_factor = self._correction_factor()
        self._verbose(f'Correction factor equal to {self.corr_factor:.6f}')
    else:
      self.eval_pred = self._do_cv_prediction(self.estimator_list[0], self.features)

    self._calc_eval_metrics()

  def _calc_meta_features(self):
    self._verbose(f'Calculating meta-features')

    for estimator in self.estimator_list:
      self.meta_features.append(self._do_cv_prediction(estimator, self.features))

    if self._is_classifier():
      self.meta_features = np.concatenate(self.meta_features, axis=1)
    else:
      self.meta_features = np.stack(self.meta_features, axis=1)

    print(self.meta_features.shape)
    self._verbose(f' Meta-features shape: {self.meta_features.shape}')

  def _correction_factor(self, per_class = False):
    
    n_estimators = len(self.estimator_list)
    n_meta_features = self.meta_features.shape[1]
    n_classes = int(n_meta_features / n_estimators)
    meta_features = self.meta_features.reshape(-1, n_classes, n_estimators)
    
    avg_std_axis, multioutput = None, 'uniform_average' 
    if per_class:
        avg_std_axis, multioutput = (0, 'raw_values')
    
    prep = preprocessing.LabelBinarizer()
    target_bin = prep.fit_transform(self.target)
    
    avg_std = np.mean(np.std(meta_features, axis=-1), axis=avg_std_axis)
    rmse = metrics.mean_squared_error(target_bin, self.eval_pred, multioutput=multioutput, squared=False)
    
    # See https://stats.stackexchange.com/questions/242787/how-to-interpret-root-mean-squared-error-rmse-vs-standard-deviation/375674
    return (rmse / avg_std)

  def _feature_idx(self, fn_layer):
    return self.feature_cols.index(fn_layer.stem)

  def _reorder_layers(self, layernames, dict_layers_newnames, input_data, raster_files):

    sorted_input_data = []
    sorted_raster_files = []
    for feature_col in self.feature_cols:
      try:
        idx = layernames.index(feature_col)
        sorted_input_data.append(input_data[:,:,idx])
        if idx < len(raster_files):
          sorted_raster_files.append(raster_files[idx])
      except:
        raise Exception(f"The feature {feature_col} was not provided")

    return np.stack(sorted_input_data, axis=2), sorted_raster_files

  def _read_layers(self, dirs_layers:List = [], fn_layers:List = [], spatial_win = None,
    dtype = 'Float32', allow_additional_layers = False, inmem_calc_func = None, dict_layers_newnames={},
    n_jobs_io = 5, verbose_renaming=True):

    raster_data, raster_files = read_rasters(raster_dirs=dirs_layers, raster_files=fn_layers, \
                                      spatial_win=spatial_win, dtype=dtype, n_jobs=n_jobs_io, \
                                      verbose=self.verbose,)

    feature_cols_set = set(self.feature_cols)
    layernames = []

    for raster_file in raster_files:

      layername = raster_file.stem

      for newname in dict_layers_newnames.keys():
        layername_aux = layername
        layername = re.sub(dict_layers_newnames[newname], newname, layername)
        if layername_aux != layername and verbose_renaming:
          self._verbose(f'Renaming {layername_aux} to {layername}')

      if not allow_additional_layers and layername not in feature_cols_set:
        raise Exception(f"Layer {layername} does not exist as feature_cols.\nUse dict_layers_newnames param to match their names")

      layernames.append(layername)

    if inmem_calc_func is not None:
      layernames, raster_data = inmem_calc_func(layernames, raster_data, spatial_win)

    return self._reorder_layers(layernames, dict_layers_newnames, raster_data, raster_files)

  def _write_layer(self, fn_base_layer, fn_output, input_data_shape, output_data,
    nan_mask, separate_probs = True, spatial_win = None, scale=1,
    dtype = 'float32', new_suffix = None):

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
      fn_output = Path(str(fn_output).replace(out_ext, new_suffix + out_ext))

    fn_output_list = []

    if separate_probs:

      for i in range(0,n_classes):
        out_ext = Path(fn_output).suffix
        fn_output_c = Path(str(fn_output).replace(out_ext, f'_b{i+1}' + out_ext))

        write_new_raster(fn_base_layer, fn_output_c, output_data[:,:,i:i+1], 
          spatial_win = spatial_win, dtype=dtype, nodata=255)
        fn_output_list += [ fn_output_c ]

    else:
      write_new_raster(fn_base_layer, fn_output, output_data, 
        spatial_win = spatial_win, dtype=dtype, nodata=255)
      fn_output_list += [ fn_output ]

    return fn_output_list

  def _write_layers(self, fn_base_layer, fn_output, input_data_shape, pred_result,
    pred_uncer, nan_mask, spatial_win, separate_probs, hard_class, dtype = 'float32'):

    fn_out_files = []

    if self.pred_method != 'predict_proba':
      separate_probs = False
      scale = 1
    else:
      scale = 100

    fn_pred_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_result, nan_mask,
      separate_probs = separate_probs, spatial_win = spatial_win, dtype = dtype, scale = scale)
    fn_out_files += fn_pred_files

    if self.pred_method == 'predict_proba':

      if pred_uncer is not None:

        fn_uncer_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_uncer, nan_mask,
          separate_probs = separate_probs, spatial_win = spatial_win, dtype = 'uint8', scale=100, new_suffix = '_uncertainty')
        fn_out_files += fn_uncer_files

      if hard_class:

        pred_argmax = np.argmax(pred_result, axis=1)
        pred_argmax_prob = np.take_along_axis(pred_result, np.stack([pred_argmax], axis=1), axis=1)

        if pred_uncer is not None and pred_uncer.ndim > 2:
          pred_argmax_uncer = np.take_along_axis(pred_uncer, np.stack([pred_argmax], axis=1), axis=1)

        pred_argmax += 1

        fn_hcl_file = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax, nan_mask,
          separate_probs = False, spatial_win = spatial_win, dtype = dtype, new_suffix = '_hcl')
        fn_out_files += fn_hcl_file

        fn_hcl_prob_files = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax_prob, nan_mask,
          separate_probs = False, spatial_win = spatial_win, dtype = 'uint8', scale=100, new_suffix = '_hcl_prob')
        fn_out_files += fn_hcl_prob_files

        if pred_uncer is not None and pred_uncer.ndim > 2:
          fn_hcl_uncer_file = self._write_layer(fn_base_layer, fn_output, input_data_shape, pred_argmax_uncer, nan_mask,
            separate_probs = False, spatial_win = spatial_win, dtype = 'uint8', scale=100, new_suffix = '_hcl_uncertainty')
          fn_out_files += fn_hcl_uncer_file

    return fn_out_files

  def train(self):
    """
    Train the ML/EML model according to the class arguments.
    """

    # Hyperparameter optization for all estimators
    for hyperpar_selection, estimator in zip(self.hyperpar_selection_list, self.estimator_list):
      if hyperpar_selection is not None:
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
    start = time.time()
    estimators_pred = []

    for estimator in self.estimator_list:

      estimator_name = type(estimator).__name__
      self._verbose(f'Executing {estimator_name}')

      if self._is_keras_model(estimator):

        n_elements, _ = input_data.shape
        pred_batch_size = int(n_elements/2)

        self._verbose(f'###batch_size={pred_batch_size}')
        estimator['estimator'].set_params(batch_size=pred_batch_size)

      if self._is_catboost_model(estimator):
        from catboost import Pool, FeaturesData
        start_featuresdata = time.time()
        input_data = FeaturesData(num_feature_data=input_data.astype(np.float32))
        #ttprint(f"creating FeaturesData took {(time.time() - start_featuresdata):.2f} seconds")
        start_pool = time.time()
        input_data = Pool(data=input_data)
        #ttprint(f"creating Pool from FeaturesData took {(time.time() - start_pool):.2f} seconds")
      
      start_pred = time.time()
      estimator_pred_method = getattr(estimator, self.pred_method)
      if self._is_keras_model(estimator):
        estimators_pred.append(estimator_pred_method(input_data, batch_size=pred_batch_size))
      else:
        estimators_pred.append(estimator_pred_method(input_data))
      self._verbose(f'{estimator_name} prediction time: {(time.time() - start_pred):.2f} seconds')
    self._verbose(f'Total time: {(time.time() - start):.2f} seconds')

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

      if self._is_classifier():
        input_meta_features = np.concatenate(estimators_pred, axis=1)
        std_meta_features = np.std(np.stack(estimators_pred, axis=2), axis=2)
      else:
        input_meta_features = np.stack(estimators_pred, axis=1)
        std_meta_features = np.std(input_meta_features, axis=1)

      meta_estimator_pred_method = getattr(self.meta_estimator, self.pred_method)
      meta_estimator_pred = meta_estimator_pred_method(input_meta_features)
      self._verbose(f'{meta_estimator_name} prediction time: {(time.time() - start):.2f} segs')

      return meta_estimator_pred.astype('float32'), self.corr_factor * std_meta_features.astype('float32')

  def predict_points(self,
    input_points:DataFrame
  ):
    """
    Predict point samples. It uses the ``feature_cols`` to retrieve the
    input feature/covariates.

    :param input_points: New set of point samples to be predicted.

    :returns: The prediction result and the prediction uncertainty (only for EML)
    :rtype: Tuple[Numpy.array, Numpy.array]
    """

    input_data = np.ascontiguousarray(input_points[self.feature_cols].to_numpy(), dtype=np.float32)

    n_points, _ = input_data.shape
    self._verbose(f'Predicting {n_points} points')

    return self._predict(input_data)

  def predict(self,
    dirs_layers:List = [],
    fn_layers:List = [],
    fn_output:str = None,
    spatial_win:Window = None,
    dtype = 'float32',
    fill_nodata:bool = False,
    separate_probs:bool = True,
    hard_class:bool = True,
    inmem_calc_func:Callable = None,
    dict_layers_newnames:set = {},
    allow_additional_layers:bool = False,
    n_jobs_io:int = 4,
    verbose_renaming:bool = True,
  ):

    """
    Predict raster data. It matches the raster filenames with the input feature/covariates
    used by training.

    :param dirs_layers: A list of folders where the raster files are located.
    :param fn_layers: A list with the raster paths. Provide it and the ``dirs_layers`` is ignored.
    :param fn_output: File path where the prediction result is saved. For multiple outputs (probabilities,
      uncertainty) the same location is used, adding specific suffixes in the provided file path.
    :param spatial_win: Read the data and predict according to the spatial window. By default is ``None``,
      which means all the data is read and predict.
    :param dtype: Convert the read data to specific ``dtype``. For ``Float*`` the ``nodata`` values are
      converted to ``np.nan``.
    :param fill_nodata: Use the ``nodata_imputer`` to fill all ``np.nan`` values. By default is ``False``
      because for almost all the cases it's preferable use the ``skmap.gapfiller module`` to perform this task.
    :param separate_probs: Use ``True`` to save the predict probabilities in a separate raster, otherwise it's
      write as multiple bands of a single raster file. For ``pred_method='predict'`` it's ignored.
    :param hard_class: When ``pred_method='predict_proba'`` use ``True`` to save the predict dominant
      class ``(*_hcl.tif)``, the probability ``(*_hcl_prob.tif)`` and uncertainty ``(*_hcl_uncertainty.tif)``
      values of each dominant class.
    :param inmem_calc_func: Function to be executed before the prediction. Use it to derive covariates/features
      on-the-fly, calculating in memory, for example, a NDVI from the red and NIR bands.
    :param dict_layers_newnames: A dictionary used to change the raster filenames on-the-fly. Use it to match
      the column names for the point samples with different raster filenames.
    :param allow_additional_layers: Use ``False`` to throw a ``Exception`` if a read raster is not present
      in ``feature_cols``.
    :param n_jobs_io: Number of parallel jobs to read the raster files.
    :param verbose_renaming: show which raster layers are renamed

    :returns: List with all the raster files produced as output.
    :rtype: List[Path]
    """

    input_data, fn_layers = self._read_layers(dirs_layers, fn_layers, spatial_win, \
      dtype=dtype, inmem_calc_func=inmem_calc_func, dict_layers_newnames=dict_layers_newnames, \
      allow_additional_layers=allow_additional_layers, n_jobs_io=n_jobs_io, \
      verbose_renaming=verbose_renaming)

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
      pred_uncer, nan_mask, spatial_win, separate_probs, hard_class, dtype = dtype)

    fn_out_files.sort()

    return fn_out_files

  def predict_multi(self,
    dirs_layers_list:List[List] = [],
    fn_layers_list:List[List] = [],
    fn_output_list:List[List] = [],
    spatial_win:Window = None,
    dtype:str = 'float32',
    fill_nodata:bool = False,
    separate_probs:bool = True,
    hard_class:bool = True,
    inmem_calc_func:Callable = None,
    dict_layers_newnames_list:List[set] = [],
    allow_additional_layers=False,
    prediction_strategy_type = PredictionStrategyType.Lazy
  ):
    """
    Predict multiple raster data. It matches the raster filenames with the input feature/covariates
    used by training.

    :param dirs_layers_list: A list of list containing the folders where the raster files are located.
    :param fn_layers_list: A list of list containing the raster paths. Provide it and the
      ``dirs_layers_list`` is ignored.
    :param fn_output_list: A list of file path where the prediction result is saved. For multiple outputs (probabilities,
      uncertainty) the same location is used, adding specific suffixes in the provided file path.
    :param spatial_win: Read the data and predict according to the spatial window. By default is ``None``,
      which means all the data is read and predict.
    :param dtype: Convert the read data to specific ``dtype``. For ``Float*`` the ``nodata`` values are
      converted to ``np.nan``.
    :param fill_nodata: Use the ``nodata_imputer`` to fill all ``np.nan`` values. By default is ``False``
      because for almost all the cases it's preferable use the ``skmap.gapfiller module`` to perform this task.
    :param separate_probs: Use ``True`` to save the predict probabilities in a separate raster, otherwise it's
      write as multiple bands of a single raster file. For ``pred_method='predict'`` it's ignored.
    :param hard_class: When ``pred_method='predict_proba'`` use ``True`` to save the predict dominant
      class ``(*_hcl.tif)``, the probability ``(*_hcl_prob.tif)`` and uncertainty ``(*_hcl_uncertainty.tif)``
      values of each dominant class.
    :param inmem_calc_func: Function to be executed before the prediction. Use it to derive covariates/features
      on-the-fly, calculating in memory, for example, a NDVI from the red and NIR bands.
    :param dict_layers_newnames: A list of dictionaries used to change the raster filenames on-the-fly. Use it to match
      the column names for the point samples with different raster filenames.
    :param allow_additional_layers: Use ``False`` to throw a ``Exception`` if a read raster is not present
      in ``feature_cols``.
    :param prediction_strategy_type: Which strategy is used to predict the multiple raster data. By default is ``Lazỳ``,
      loading one year while predict the other.

    :returns: List with all the raster files produced as output.
    :rtype: List[Path]
    """

    PredictionStrategyClass = _LazyLoadPrediction
    if PredictionStrategyType.Eager == prediction_strategy_type:
      PredictionStrategyClass = _EagerLoadPrediction

    prediction_strategy = PredictionStrategyClass(self, dirs_layers_list, fn_layers_list, fn_output_list,
        spatial_win, dtype, fill_nodata, separate_probs, hard_class, inmem_calc_func,
        dict_layers_newnames_list, allow_additional_layers)

    return prediction_strategy.run()

  @staticmethod
  def load_instance(
    fn_joblib
  ):
    """
    Load a class instance from disk.

    :param fn_joblib: Location of the saved instance.

    :returns: Class instance
    :rtype: LandMapper
    """

    if not isinstance(fn_joblib, Path):
      fn_joblib = Path(fn_joblib)

    landmapper = joblib.load(fn_joblib)
    for estimator in landmapper.estimator_list:
      if landmapper._is_keras_model(estimator):
        from tensorflow.keras.models import load_model

        fn_keras = fn_joblib.parent.joinpath(f'{fn_joblib.stem}_keras.h5')
        estimator['estimator'].model = load_model(fn_keras)

    return landmapper

  def save_instance(self,
    fn_joblib:Path,
    no_train_data:bool = False,
    compress:str = 'lz4'
  ):
    """
    Persist the class instance in disk using ``joblib.dump``. Use it to perform prediction
    over new raster/point data without retrain the models from scratch.

    :param fn_joblib: Location of the output file.
    :param no_train_data: Remove all the training data before persist it
      in disk.
    :param compress: Enable compression.

    """

    if not isinstance(fn_joblib, Path):
      fn_joblib = Path(fn_joblib)

    if no_train_data:
      prop_to_del = [ 'pts', 'features', 'target', 'samples_weight', 'cv_groups']
      for prop in prop_to_del:
        if hasattr(self, prop):
          if self.verbose:
            ttprint(f'Removing {prop} attribute')
          delattr(self, prop)

    for estimator in self.estimator_list:
      if self._is_keras_model(estimator):
        basedir = fn_joblib.parent
        fn_keras = fn_joblib.parent.joinpath(f'{fn_joblib.stem}_keras.h5')
        estimator['estimator'].model.save(fn_keras)
        estimator['estimator'].model = None

    result = joblib.dump(self, fn_joblib, compress=compress)

    for estimator in self.estimator_list:
      if self._is_keras_model(estimator):
        from tensorflow.keras.models import load_model
        
        fn_keras = fn_joblib.parent.joinpath(f'{fn_joblib.stem}_keras.h5')
        estimator['estimator'].model = load_model(fn_keras)

class _PredictionStrategy(ABC):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               dtype = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_additional_layers=False):

    self.landmapper = landmapper

    self.fn_layers_list = self._fn_layers_list(dirs_layers_list, fn_layers_list)
    self.fn_output_list = fn_output_list

    self.spatial_win = spatial_win
    self.dtype = dtype
    self.fill_nodata = fill_nodata
    self.separate_probs = separate_probs
    self.hard_class = hard_class
    self.inmem_calc_func = inmem_calc_func
    self.dict_layers_newnames_list = dict_layers_newnames_list
    self.allow_additional_layers = allow_additional_layers

    super().__init__()

  def _fn_layers_list(self, dirs_layers_list, fn_layers_list):
    if len(fn_layers_list) == 0:
      for dirs_layers in dirs_layers_list:
        fn_layers_list.append(find_files(dirs_layers, '*.tif'))

    return fn_layers_list

  @abstractmethod
  def run(self):
    pass

class _EagerLoadPrediction(_PredictionStrategy):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               dtype = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_additional_layers=False):

    super().__init__(landmapper, dirs_layers_list, fn_layers_list, fn_output_list,
      spatial_win, dtype, fill_nodata, separate_probs, hard_class, inmem_calc_func,
      dict_layers_newnames_list, allow_additional_layers)

    self.reading_futures = []
    self.writing_futures = []

    self.reading_pool = ThreadPoolExecutor(max_workers = 5)
    self.writing_pool = ThreadPoolExecutor(max_workers = 5)

  def reading_fn(self, i):

    fn_layers = self.fn_layers_list[i]

    dict_layers_newnames = {}
    if len(self.dict_layers_newnames_list) > 0:
      dict_layers_newnames = self.dict_layers_newnames_list[i]

    start = time.time()
    input_data, _ = self.landmapper._read_layers(fn_layers=fn_layers,
      spatial_win=self.spatial_win, inmem_calc_func = self.inmem_calc_func,
      dict_layers_newnames = dict_layers_newnames,
      allow_additional_layers=self.allow_additional_layers)

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

class _LazyLoadPrediction(_PredictionStrategy):

  def __init__(self,
               landmapper:LandMapper,
               dirs_layers_list:List = [],
               fn_layers_list:List = [],
               fn_output_list:List = [],
               spatial_win = None,
               dtype = 'float32',
               fill_nodata = False,
               separate_probs = True,
               hard_class = True,
               inmem_calc_func = None,
               dict_layers_newnames_list:list = [],
               allow_additional_layers=False,
               reading_pool_size = 1,
               processing_pool_size = 1,
               writing_pool_size = 1):

    super().__init__(landmapper, dirs_layers_list, fn_layers_list, fn_output_list,
      spatial_win, dtype, fill_nodata, separate_probs, hard_class, inmem_calc_func,
      dict_layers_newnames_list, allow_additional_layers)

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
      dict_layers_newnames = self.dict_layers_newnames_list[i]

    start = time.time()
    input_data, _ = self.landmapper._read_layers(fn_layers=fn_layers,
      spatial_win=self.spatial_win, inmem_calc_func = self.inmem_calc_func,
      dict_layers_newnames = dict_layers_newnames,
      allow_additional_layers=self.allow_additional_layers)

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

    reading_results = [ future.result() for future in as_completed(self.reading_futures) ]
    processing_results = [ future.result() for future in as_completed(self.processing_futures) ]
    output_fn_files = sum([ future.result() for future in as_completed(self.writing_futures) ], [])

    self.reading_pool.shutdown(wait=False)
    self.processing_pool.shutdown(wait=False)
    self.writing_pool.shutdown(wait=False)

    output_fn_files.sort()

    return output_fn_files

class _ParallelOverlay:
    # optimized for up to 200 points and about 50 layers
    # sampling only first band in every layer
    # assumption is that all layers have same blocks

    def __init__(self,
      points_x: np.ndarray,
      points_y:np.ndarray,
      fn_layers:List[str],
      max_workers:int = parallel.CPU_COUNT,
      verbose:bool = True
    ):

      self.error_val = -32768
      self.points_x = points_x
      self.points_y = points_y
      self.points_len = len(points_x)
      self.fn_layers = fn_layers
      self.max_workers = max_workers
      self.verbose = verbose

      self.layer_names = [fn_layer.with_suffix('').name for fn_layer in fn_layers]
      sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
      self.dimensions = [self._get_dimension(src) for src in sources]

      self.points_blocks = self.find_blocks()
      self.result = None

    @staticmethod
    def _get_dimension(src):
      return (src.height, src.width, *src.block_shapes[0], *src.transform.to_gdal())

    def _find_blocks_for_src_mt(self, ij, block, src, ptsx, ptsy):
      left, bottom, right, top = rasterio.windows.bounds(block, src.transform)
      ind = (ptsx>=left) & (ptsx<right) & (ptsy>bottom) & (ptsy<=top)

      if ind.any():
        inv_block_transform = ~rasterio.windows.transform(block, src.transform)
        col, row = inv_block_transform * (ptsx[ind], ptsy[ind])
        result = [block, np.nonzero(ind)[0], col.astype(int), row.astype(int)]
        return ij, result
      else:
        return None, None

    def _find_blocks_for_src(self, src, ptsx, ptsy):
      # find blocks for every point in given source
      blocks = {}

      args = src.block_windows(1)
      fixed_args = (src, ptsx, ptsy)

      chunk_size = math.ceil(len(list(src.block_windows(1))) / self.max_workers)

      for ij, result in parallel.ThreadGeneratorLazy(self._find_blocks_for_src_mt, args,
                                      self.max_workers, chunk=chunk_size, fixed_args = fixed_args):
        if ij is not None:
          blocks[ij] = result

      return blocks

    def find_blocks(self):
      # for every type of dimension find block for each point

      points_blocks = {}
      dimensions_set = set(self.dimensions)
      sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
      for dim in dimensions_set:
          src = sources[self.dimensions.index(dim)]
          points_blocks[dim] = self._find_blocks_for_src(src, self.points_x, self.points_y)
      return points_blocks

    def _sample_one_layer_sp(self, fn_layer):
      with rasterio.open(fn_layer) as src:
        #src=rasterio.open(fn_layer)
        dim = self._get_dimension(src)
        # out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])
        out_sample = np.full((self.points_len,), np.nan, np.float32)

        blocks = self.points_blocks[dim]
        for ij in blocks:
          # ij=next(iter(blocks)); (window, ind, col, row) = blocks[ij]
          (window, ind, col, row) = blocks[ij]
          try:
            data = src.read(1, window=window)
            mask = src.read_masks(1, window=window)
            sample = data[row,col].astype(np.float32)
            sample_mask = mask[row,col].astype(bool)
            sample[~sample_mask] = np.nan
            #sample = data[row.astype(int),col.astype(int)]
          except Exception as exception:
            traceback.print_exc()
            sample = self.error_val

          out_sample[ind] = sample

      return out_sample, fn_layer

    def _sample_one_block(self, args):
      out_sample, fn_layer, window, ind, col, row = args
      with rasterio.open(fn_layer) as src:

        try:
          data = src.read(1, window=window)
          sample = data[row,col]
        except Exception as exception:
          traceback.print_exc()
          sample = self.error_val

        out_sample[ind] = sample
          #return sample, ind

    def _sample_one_layer_mt(self, fn_layer):
      with rasterio.open(fn_layer) as src:
          dim = self._get_dimension(src)
          out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

      blocks = self.points_blocks[dim]
      args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)

      with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        for _ in executor.map(self._sample_one_block, args, chunksize=1000):
          pass #out_sample[ind] = sample

      return out_sample

    def _sample_one_layer_mp(self, fn_layer):
      with rasterio.open(fn_layer) as src:
          dim = self._get_dimension(src)
          out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

      blocks = self.points_blocks[dim]
      args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)

      with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        for _ in executor.map(self._sample_one_block, args, chunksize=1000):
            pass #out_sample[ind] = sample

      return out_sample

    def sample_v1(self):
      '''
      Serial sampling, block by block
      '''
      res={}
      n_layers = len(self.fn_layers)

      for i_layer, fn_layer in enumerate(self.fn_layers):
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
        # fn_layer=self.fn_layers[0]
        col = Path(fn_layer).with_suffix('').name
        sample,_ = self._sample_one_layer_sp(fn_layer)
        res[col]=sample

      return res

    def sample_v2(self):
      '''
      Serial layers, parallel blocks in layer
      '''

      res={}
      n_layers = len(self.fn_layers)

      for i_layer, fn_layer in enumerate(self.fn_layers):
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
        # fn_layer=self.fn_layers[0]
        col = Path(fn_layer).with_suffix('').name
        sample = self._sample_one_layer_mt(fn_layer)
        res[col]=sample

      return res

    def sample_v3(self):
      '''
      Parallel layers in threads, serial blocks in layer
      '''
      res={}
      i_layer=1
      n_layers=len(self.fn_layers)
      args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)

      for sample, fn_layer in parallel.ThreadGeneratorLazy(self._sample_one_layer_sp, args,
                          self.max_workers, self.max_workers*2):
        col = Path(fn_layer).with_suffix('').name
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {col}')
        res[col] = sample
        i_layer += 1

      return res

    def sample_v4(self):
      '''
      Parallel layers in processes, serial blocks in layer
      '''
      res={}
      i_layer=1
      n_layers=len(self.fn_layers)
      args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)
      #with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
          #for sample, fn_layer in executor.map(self._sample_one_layer_sp, args, chunksize=n_layers//self.max_workers):

      for sample, fn_layer in parallel.ProcessGeneratorLazy(self._sample_one_layer_sp, args, self.max_workers, 1):
        col = Path(fn_layer).with_suffix('').name
        if self.verbose:
          ttprint(f'{i_layer}/{n_layers} {col}')
        res[col] = sample
        i_layer += 1

      return res

    def run(self):
      # For now sample_v3 is the best method, because is parallel and use less memory than sample_v4
      if self.result == None:
        self.result = self.sample_v3()
      else:
        if self.verbose:
          ttprint('You already did run the overlay. Geting the cached result')

      return self.result

class SpaceOverlay():
  """
  Overlay a set of points over multiple raster files.
  The retrieved pixel values are organized in columns
  according to the filenames.

  :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
    the points.
  :param fn_layers: A list with the raster file paths. If it's not provided
    the ``dir_layers`` and ``regex_layers`` are used to retrieve the raster files.
  :param dir_layers: A list of folders where the raster files are located. The raster
    are selected according to the pattern specified in ``regex_layers``.
  :param regex_layers: Pattern to select the raster files in ``dir_layers``.
    By default all GeoTIFF files are selected.
  :param max_workers: Number of CPU cores to be used in parallel. By default all cores
    are used.
  :param verbose: Use ``True`` to print the overlay progress.

  Examples
  ========

  >>> from skmap.mapper import SpaceOverlay
  >>>
  >>> spc_overlay = SpaceOverlay('./my_points.gpkg', ['./raster_dir_1', './raster_dir_2'])
  >>> result = spc_overlay.run()
  >>>
  >>> print(result.shape)

  """

  def __init__(self,
    points,
    fn_layers:List[str] = [],
    dir_layers:List[str] = [],
    regex_layers = '*.tif',
    max_workers:int = parallel.CPU_COUNT,
    verbose:bool = True
  ):

    if len(fn_layers) == 0:
      fn_layers = find_files(dir_layers, regex_layers)

    self.fn_layers = [ Path(l) for l in fn_layers ]

    if not isinstance(points, gpd.GeoDataFrame):
      points = gpd.read_file(points)

    self.pts = points
    self.pts['overlay_id'] = range(1,len(self.pts)+1)

    self.parallelOverlay = _ParallelOverlay(self.pts.geometry.x.values, self.pts.geometry.y.values,
      fn_layers, max_workers, verbose)

  def run(self,
    dict_newnames:set = {}
  ):
    """
    Execute the space overlay.

    :param dict_newnames: A dictionary used to update the column names after the overlay.
      The ``key`` is the new name and the ``value`` is the raster file name (without extension).

    :returns: Data frame with the original columns plus the overlay result (one new
      column per raster).
    :rtype: geopandas.GeoDataFrame
    """
    result = self.parallelOverlay.run()

    for col in result:
      new_col = col
      for newname in dict_newnames.keys():
        new_col = re.sub(dict_newnames[newname], newname, new_col)
      self.pts.loc[:,new_col] = result[col]

    return self.pts

class SpaceTimeOverlay():
  """
  Overlay a set of points over multiple raster considering the year information.
  The retrieved pixel values are organized in columns according to the filenames.

  :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
    the points.
  :param col_date: Date column to retrieve the year information.
  :param fn_layers: A list with the raster file paths. The file path placeholders
    ``{year}``, ``{year_minus_1}``, ``{year_plus_1}`` are replaced considering the
    year information of each point.
  :param max_workers: Number of CPU cores to be used in parallel. By default all cores
    are used.
  :param verbose: Use ``True`` to print the overlay progress.

  Examples
  ========

  >>> from skmap.mapper import SpaceTimeOverlay
  >>>
  >>> fn_layers = [ 'raster_{year_minus_1}1202..{year}0320.tif' ] # raster_20101202..20110320.tif, ...
  >>> spt_overlay = SpaceTimeOverlay('./my_points.gpkg', 'survey_date' fn_layers)
  >>> result = spt_overlay.run()
  >>>
  >>> print(result.shape)

  """

  def __init__(self,
    points,
    col_date:str,
    fn_layers:List[str] = [],
    max_workers:int = parallel.CPU_COUNT,
    verbose:bool = False
  ):

    if not isinstance(points, gpd.GeoDataFrame):
      points = gpd.read_file(points)

    self.pts = points
    self.col_date = col_date
    self.overlay_objs = {}
    self.verbose = verbose
    self.year_placeholder = '{year}'

    self.pts.loc[:,self.col_date] = pd.to_datetime(self.pts[self.col_date])
    self.uniq_years = self.pts[self.col_date].dt.year.unique()

    self.fn_layers = [ Path(l) for l in fn_layers ]

    for year in self.uniq_years:

      year = int(year)
      year_points = self.pts[self.pts[self.col_date].dt.year == year]

      fn_layers_year = []
      for fn_layer in self.fn_layers:
        fn_layers_year.append(Path(self._replace_year(fn_layer, year)))

      if self.verbose:
        ttprint(f'Overlay {len(year_points)} points from {year} in {len(fn_layers_year)} raster layers')

      self.overlay_objs[year] = SpaceOverlay(points=year_points, fn_layers=fn_layers_year,
        max_workers=max_workers, verbose=verbose)

  def _replace_year(self, fn_layer, year = None):
    y, y_m1, y_p1 = ('', '', '')
    if year != None:
      y, y_m1, y_p1 = str(year), str((year - 1)), str((year + 1))

    fn_layer = str(fn_layer)

    return fn_layer \
        .replace('{year}', y) \
        .replace('{year_minus_1}', y_m1) \
        .replace('{year_plus_1}', y_p1) \

  def run(self):
    """
    Execute the spacetime overlay. It removes the year part from the column names.
    For example, the raster ``raster_20101202..20110320.tif`` results in the column
    name ``raster_1202..0320``.

    :returns: Data frame with the original columns plus the overlay result (one new
      column per raster).
    :rtype: geopandas.GeoDataFrame
    """
    self.result = None

    for year in self.uniq_years:

      year_newnames = {}
      for fn_layer in self.fn_layers:

        name = str(fn_layer.stem)
        curname = self._replace_year(name, year)
        newname = self._replace_year(name)

        year_newnames[newname] = curname

      if self.verbose:
        ttprint(f'Running the overlay for {year}')
      year_result = self.overlay_objs[year].run(year_newnames)

      if self.result is None:
        self.result = year_result
      else:
        self.result = self.result.append(year_result)

    return self.result
