'''
Miscellaneous utils
'''

import rasterio
import numpy as np

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam

def ttprint(*args, **kwargs):
	from datetime import datetime
	import sys

	print(f'[{datetime.now():%H:%M:%S}] ', end='')
	print(*args, **kwargs, flush=True)

def new_img(fn_base_img, fn_new_img, data, spatial_win = None, data_type = None, img_format = 'GTiff', nodata = 0):
		
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

def data_to_new_img(fn_base_img, fn_new_img, data, spatial_win = None, data_type = None, img_format = 'GTiff', nodata = 0):

	_, _, nbands = data.shape

	with new_img(fn_base_img, fn_new_img, data, spatial_win, data_type, img_format) as dst:
		dst.nodata = 0
		for band in range(0, nbands):
			
			dst.write(data[:,:,band].astype(dst.dtypes[band]), indexes=(band+1))

def data_from_img(fn_layer, indexes = 1, spatial_win = None):
	with rasterio.open(fn_layer) as ds:

		result = ds.read(indexes, window=spatial_win)
		
		if (isinstance(result, np.ndarray)):
			nodata = ds.nodatavals[0]
			result = result.astype('Float32')
			result[result == nodata] = np.nan
		elif (spatial_win is not None):
			result = np.empty((spatial_win.width, spatial_win.height))
			result[:] = np.nan

		return result

def build_ann(input_shape, output_shape, n_layers = 3, n_neurons = 32, 
							activation = 'relu', dropout_rate = 0.0, learning_rate = 0.0001,
							output_activation = 'softmax', loss = 'categorical_crossentropy'):
		
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