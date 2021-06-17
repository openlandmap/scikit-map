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
