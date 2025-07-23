#%%

import utils

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

#%%

def copy_file(src_path: str, dst_path: Path) -> str | None:               
        if dst_path.exists():
            #print(f"File {dest_path} already exists, skipping download.")
            return dst_path.as_posix()
        try:
            dst, msg = urlretrieve(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
            dst_path.unlink(missing_ok=True)
            return None
        return dst

#%%
def download_landsat_files():
    landsat_tile = '055W_06S'
    years = range(2000, 2010)
    fld_dst = Path(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}')
    fld_dst.mkdir(parents=True, exist_ok=True)
    landsat_files = utils.get_landsat_filenames_gaia(landsat_tile, years)

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

def download_modis_files():
    modis_tile = '055W_06S'
    years = range(2000, 2010)
    fld_dst = Path(f'/mnt/nibble/gen_cog/arcov2/modis/')
    fld_dst.mkdir(parents=True, exist_ok=True)

    from utils import doy_start, doy_end
    modis_files = []
    for year in years:
        for m in range(utils.n_imag_per_year):
            src = f'{utils.random.choice(utils.gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif'
            dst = f'{fld_dst}/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif'
            modis_files.append((src,dst))

    executor = ProcessPoolExecutor(max_workers=utils.n_threads)
    futures = [executor.submit(copy_file, src, Path(dst)) for src,dst in modis_files]

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


if __name__ == "__main__":
    #download_landsat_files()
    download_modis_files()
    print("All downloads completed.")