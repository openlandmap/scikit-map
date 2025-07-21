#%%

import utils

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

#%%
def download_files():
    landsat_tile = '055W_06S'
    years = range(2000, 2010)
    fld_dst = Path(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}')
    fld_dst.mkdir(parents=True, exist_ok=True)
    landsat_files = utils.get_landsat_filenames(landsat_tile, years)

    def copy_file(file_path):       
        dest_path = fld_dst / Path(file_path).name
        if dest_path.exists():
            #print(f"File {dest_path} already exists, skipping download.")
            return dest_path
        try:
            dst, msg = urlretrieve(file_path, dest_path)
        except Exception as e:
            print(f"Error copying {file_path} to {dest_path}: {e}")
            dest_path.unlink(missing_ok=True)
            return None
        return dst
    
    executor = ProcessPoolExecutor(max_workers=utils.n_threads)
    futures = [executor.submit(copy_file, fn) for fn in landsat_files]

    n = 0
    errs= 0
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            print(f"Downloaded {n:4}: {result}")
            n += 1
        else:
            errs += 1
    print(f"Errors: {errs}, Downloaded: {n}")    
