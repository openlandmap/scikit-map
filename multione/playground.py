# Just playground ...
# eval "$(micromamba shell hook --shell bash)"
# micromamba activate arcov2
#%%
# from pathlib import Path
# print(Path.cwd())
# import imports
from numpy.typing import NDArray
import numpy as np
import zarr

import utils
import time
#%%
# old crappy tiles = ['009E_04N', '009E_51N', '013E_61N', '050W_07S', '085W_52N', '091W_37N', '115E_03S', '127E_42N']
# tiles that Tom wants to check = ['055W_06S', '015E_43N', '090W_49N']
landsat_tile = '055W_06S'
# years = range(2000,2024)
years = range(2000, 2010)
#%%
start0 = time.time()
utils.ttprint(f"Start processing for tile {landsat_tile} and years {years} ...")

#%%
start = time.time()

utils.ttprint(f"Getting Landsat data for tile {landsat_tile} and years {years} ...")
landsat_files = utils.get_landsat_filenames_local(landsat_tile, years, '/mnt/nibble/gen_cog/arcov2')
landsat_data = utils.get_landsat_data(landsat_files, years)

utils.ttprint(f"Getting MODIS data for tile {landsat_tile} and years {years} ...")
modis_data = utils.get_modis_ndvi_data_rio(landsat_files, years)

end = time.time()
print(f"Time taken to get data: {end - start} seconds")
# Time taken to get data: 187.68489527702332 seconds
# Time taken to get data: 73.33880066871643 seconds

#%% Save landsat_data
import h5py
with h5py.File(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}.h5', 'w') as f:
    f.create_dataset('landsat_data', data=landsat_data)
    f.create_dataset('modis_data', data=modis_data)
    f.create_dataset('years', data=years)

#%% Load data
import h5py
import numpy as np

start = time.time()
utils.ttprint(f"Loading data from h5 for tile {landsat_tile} and years {years} ...")
with h5py.File(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}.h5', 'r') as f:
    landsat_data: NDArray[np.float32] = f['landsat_data'][:] # type: ignore
    modis_data: NDArray[np.float32] = f['modis_data'][:] # type: ignore
    years: NDArray[np.int32] = f['years'][:] # type: ignore

utils.ttprint(f"Loaded data from h5 in {time.time() - start} seconds")
# Loaded data from h5 in 70.32956600189209 seconds

#%% Masking Landsat data from QA:
start = time.time()
utils.ttprint(f"Masking Landsat data from QA...")
# utils.mask_from_qa(landsat_data, len(years))
utils.mask_from_qa(landsat_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")
# Finnished in 86.04056429862976 seconds
# Finnished in 85.93483519554138 seconds

# %% Masking Landsat data from MODIS:
start = time.time()
utils.ttprint(f"Masking Landsat data from MODIS...")
utils.mask_from_modis(landsat_data, modis_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")
# Finnished in 89.76117825508118 seconds
# Finnished in 93.40202569961548 seconds

#%% Save masked Landsat data
# import h5py
# with h5py.File(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.h5', 'w') as f:
#     f.create_dataset('landsat_data', data=landsat_data, compression='gzip', compression_opts=9)
#     f.create_dataset('modis_data', data=modis_data, compression='gzip', compression_opts=9)
#     f.create_dataset('years', data=years)

# np.savez(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.npz',
#          landsat_data=landsat_data,
#          modis_data=modis_data,
#          years=years,
#          allow_pickle=False)

start = time.time()
root = zarr.group(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr', overwrite=True)
root.create_array('landsat_data', data=landsat_data)
root.create_array('modis_data', data=modis_data)
root.create_array('years', data=years)
utils.ttprint(f"Saved masked Landsat data to zarr in {time.time() - start} seconds")
#  Saved masked Landsat data to zarr in 103.94907927513123 seconds

#%% Load masked Landsat data
# start = time.time()
# utils.ttprint(f"Loading masked data from npz for tile {landsat_tile} and years {years} ...")
# with np.load(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.npz') as data:
#     landsat_data: NDArray[np.float32] = data['landsat_data']  # type: ignore
#     modis_data: NDArray[np.float32] = data['modis_data']  # type: ignore
#     years: NDArray[np.int32] = data['years']  # type: ignore

# utils.ttprint(f"Loaded data from npz in {time.time() - start} seconds")
# #  Loaded data from npz in 265.75636100769043 seconds

start = time.time()
root = zarr.open(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr', mode='r')
utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile} and years {years} ...")
landsat_data: NDArray[np.float32] = root['landsat_data'][:]  # type: ignore
modis_data: NDArray[np.float32] = root['modis_data'][:]  # type: ignore
years: NDArray[np.int32] = root['years'][:]  # type: ignore 
utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")
# Loaded data from zarr in 191.95954155921936 seconds
#%% Inpainting Landsat data:
start = time.time()
utils.ttprint(f"Inpainting Landsat data...")
utils.inpaint_stripes(landsat_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")

#%%
utils.show_image_landsat(landsat_data, years, 2001, 10, 3)
utils.show_image_modis(modis_data, years, 2001, 10)
# %%
