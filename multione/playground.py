# Just playground ...

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
#%%
start = time.time()
landsat_files = utils.get_landsat_filenames(landsat_tile, years)
landsat_data = utils.get_landsat_data(landsat_tile, years)
modis_data = utils.get_modis_ndvi_data(landsat_tile, years)
end = time.time()
print(f"Time taken to get data: {end - start} seconds")
#%%