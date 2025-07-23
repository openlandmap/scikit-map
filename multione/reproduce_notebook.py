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
years = range(2000, 2002)
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



#%% Mask Landsat outliers with MODIS NDVI:
# Maybe here the return is not needed because it should act on the pointer anyway
start = time.time()
utils.ttprint(f"Mask Landsat from MODIS NDVI...")
landsat_data = utils.mask_from_modis(landsat_data, modis_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")

#%% Inpainting Landsat data:
start = time.time()
utils.ttprint(f"Inpainting Landsat data...")
utils.inpaint_stripes(landsat_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")

#%% Aggregating to monmthly Landsat data:
start = time.time()
utils.ttprint(f"Bands aggregation Landsat data...")
landsat_bands_agg_t = utils.bands_aggregation(landsat_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")

#%% Gap filling with SWA Landsat data:
start = time.time()
utils.ttprint(f"Gap filling with SWA Landsat data...")
landsat_bands_rec_t = utils.swa_reconstructing(landsat_bands_agg_t, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")

#%% Write output data:
start = time.time()
utils.ttprint(f"Save data on s3 data...")
landsat_bands_rec_t = utils.save_landsat_bands(landsat_bands_rec_t, landsat_tile, years, landsat_files)
utils.ttprint(f"Finnished in {time.time() - start} seconds")
