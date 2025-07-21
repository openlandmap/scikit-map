# Just playground ...
# eval "$(micromamba shell hook --shell bash)"
# micromamba activate arcov2
#%%
# from pathlib import Path
# print(Path.cwd())
# import imports
import utils
import time
#%%
# old crappy tiles = ['009E_04N', '009E_51N', '013E_61N', '050W_07S', '085W_52N', '091W_37N', '115E_03S', '127E_42N']
# tiles that Tom wants to check = ['055W_06S', '015E_43N', '090W_49N']
landsat_tile = '055W_06S'
years = range(2000,2024)
years = range(2000, 2010)
#%%
utils.ttprint(f"Getting Landsat data for tile {landsat_tile} and years {years} ...")
start = time.time()
landsat_files = utils.get_landsat_filenames(landsat_tile, years)
landsat_data = utils.get_landsat_data(landsat_tile, years)
modis_data = utils.get_modis_ndvi_data(landsat_tile, years)
end = time.time()
print(f"Time taken to get data: {end - start} seconds")
# Time taken to get data: 187.68489527702332 seconds
#%% Masking Landsat data from QA:
start = time.time()
utils.ttprint(f"Masking Landsat data from QA...")
# utils.mask_from_qa(landsat_data, len(years))
utils.mask_from_qa(landsat_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")
# 93 sec
# %% Masking Landsat data from MODIS:
start = time.time()
utils.ttprint(f"Masking Landsat data from MODIS...")
utils.mask_from_modis(landsat_data, modis_data, len(years))
utils.ttprint(f"Finnished in {time.time() - start} seconds")
# 103 sec
#%%

utils.show_image(landsat_data, years, 2001, 10, 8)
# %%
