'''
Miscellaneous utils
'''

import rasterio
import numpy as np
from pathlib import Path

def ttprint(*args, **kwargs):
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)

def _verbose(verbose, *args, **kwargs):
    if verbose:
      ttprint(*args, **kwargs)

def find_files(dir_list, dir_pattern = '*.*'):
  files = []

  glob_pattern = f'**/{dir_pattern}'

  for _dir in dir_list:
    for file in list(Path(_dir).glob(glob_pattern)):
      files.append(file)

  files = sorted(files)

  return files

def build_ann(input_shape, output_shape, n_layers = 3, n_neurons = 32,
              activation = 'relu', dropout_rate = 0.0, learning_rate = 0.0001,
              output_activation = 'softmax', loss = 'categorical_crossentropy'):

  from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.optimizers import Nadam

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

# See https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way
# Thanks Kersten :)
def nan_percentile(arr, q, keep_original_vals=False):
  
  if keep_original_vals:
    arr = np.copy(arr)

  nanall = np.all(np.isnan(arr), axis=0)
  res_shape = (len(q), arr.shape[1], arr.shape[2])
  nanall = np.broadcast_to(nanall, shape=res_shape)

  # valid (non NaN) observations along the first axis
  valid_obs = np.sum(np.isfinite(arr), axis=0)
  
  # replace NaN with maximum
  max_val = np.nanmax(arr)
  arr[np.isnan(arr)] = max_val
  
  # sort - former NaNs will move to the end
  arr = np.sort(arr, axis=0)

  # loop over requested quantiles
  if type(q) is list:
    qs = []
    qs.extend(q)
  else:
    qs = [q]
  if len(qs) <= 2:
    quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
  else:
    quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

  result = []
  for i in range(len(qs)):
    quant = qs[i]
    # desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * (quant / 100.0)
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    fc_equal_k_mask = f_arr == c_arr

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalueFromIndex(arr=arr, ind=f_arr) * (c_arr - k_arr)
    ceil_val = _zvalueFromIndex(arr=arr, ind=c_arr) * (k_arr - f_arr)

    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalueFromIndex(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

    result.append(quant_arr)

  result = np.stack(result, axis=0).astype('float16')
  result[nanall] = np.nan

  return result

def _zvalueFromIndex(arr, ind):
  """private helper function to work around the limitation of np.choose() by employing np.take()
  arr has to be a 3D array
  ind has to be a 2D array containing values for z-indicies to take from arr
  See: http://stackoverflow.com/a/32091712/4169585
  This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
  """
  # get number of columns and rows
  _,nC,nR = arr.shape

  # get linear indices and extract elements with np.take()
  idx = nC*nR*ind + nR*np.arange(nR)[:,None] + np.arange(nC)
  return np.take(arr, idx)