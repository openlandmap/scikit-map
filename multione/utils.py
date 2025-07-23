# Various utils 
#%%
from ast import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from numpy.typing import NDArray, ArrayLike
import gc
import rasterio as rio
import rasterio.vrt, rasterio.enums
from tqdm import tqdm
import os

import subprocess
from datetime import datetime
import sys
from pathlib import Path
import random
import numpy as np

from imports import skmap_bindings as sb
from imports import warp_tile

from settings import n_imag_per_year, n_imag_per_year_agg, doy_start, doy_end, n_pix, x_size, y_size, gdal_opts, x_off, y_off
from settings import TMP_DIR, gaia_addrs, bands_prefix, landsat_file_ending, n_threads, no_data
from settings import mask_result_scaling, mask_band_scaling, mask_result_offset
from settings import filter_params
from settings import att_env, att_seas, future_scaling
from settings import n_spect_bands, bands_prefix_out, file_ending_out, no_data_out, month_start, month_end
from settings import s3_aliases, s3_params, s3_setup
from settings import fft_th, gap_stripes_th, gap_general_th, inpaint_chunk_size, inpaint_radius, inpaint_padding

from processing_utils import get_SWA_weights
from skmap import data

#%%

# Function to set up S3 aliases using MinIO Client (mc)
# This function takes access key, secret key, and a list of Gaia addresses,
# def s3_setup(access_key, secret_key, gaia_addrs) -> List[str]:
#     s3_aliases = []
#     s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
#     commands = [
#         f'sudo mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
#         for i, addr in enumerate(gaia_addrs)
#     ]
#     for cmd in commands:
#         subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
#     return s3_aliases

def setup_gaia():
    
    from settings import gaia_s3_params

    s3_aliases = s3_setup(gaia_s3_params['s3_access_key'],
                gaia_s3_params['s3_secret_key'],
                gaia_s3_params['s3_addresses'])
    
    return s3_aliases

def ttprint(*args, **kwargs):    
    print(f'[{datetime.now():%H:%M:%S}] ', end='')
    print(*args, **kwargs, flush=True)

def make_tempdir(basedir='skmap', make_subdir = True) -> Path:
    import tempfile

    tempdir = Path(TMP_DIR).joinpath(basedir)
    if make_subdir: 
        name = Path(tempfile.NamedTemporaryFile().name).name
        tempdir = tempdir.joinpath(name)
    tempdir.mkdir(parents=True, exist_ok=True)
    return tempdir

# def get_modis_ndvi_filename(year) -> str:
#     """
#     Get the filename for MODIS NDVI data based on year, start and end DOY, and band.
#     """
#     #return f'MOD13Q1.A{year}{doy_start}.{doy_end}.{band}.hdf'
#     fn = f'/vsicurl/http://192.168.49.{random.randint(30,44)}:8333/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif'
#     return fn



def get_landsat_filenames_gaia(landsat_tile, years) -> List[str]:
    landsat_files = []
    for b in bands_prefix:
        for year in years:
            for m in range(n_imag_per_year):
                landsat_files.append(f'{random.choice(gaia_addrs)}/prod-landsat-ard2/{landsat_tile}/raw/{b}.ard2_m_30m_s_{year}{doy_start[m]}_{year}{doy_end[m]}{landsat_file_ending}')

    return landsat_files

def get_landsat_filenames_local(landsat_tile, years, fld_source) -> List[str]:
    landsat_files = []
    for b in bands_prefix:
        for year in years:
            for m in range(n_imag_per_year):
                # Path(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}')
                landsat_files.append(f'{fld_source}/landsat_{landsat_tile}/{b}.ard2_m_30m_s_{year}{doy_start[m]}_{year}{doy_end[m]}{landsat_file_ending}')
    return landsat_files

def get_landsat_data(landsat_files, years) -> NDArray[np.float32]:
    """
    Get the Landsat data based on year, start and end month, and band.
    """    

    n_years = len(years)
    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg
    landsat_data = np.empty((n_s*(n_spect_bands + 2), n_pix), dtype=np.float32)
    sb.readData(landsat_data, n_threads, landsat_files, range(len(landsat_files)), x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    # sb.readData(landsat_data, n_threads, landsat_files, [2], x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    # sb.readData(landsat_data, n_threads, ld, [0], x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    return landsat_data

def get_modis_ndvi_rio(ref_file, modis_file, resampling_strategy=rasterio.enums.Resampling.bilinear):
    '''
    ref_file = landsat_files[11]
    modis_file = modis_files[11][0]
    resampling_strategy=rasterio.enums.Resampling.bilinear
    '''
    with rio.open(ref_file) as ref:
        profile = ref.profile
        dst_crs = ref.crs
        bounds = ref.bounds
        dd = ref.read(1)

    try:
        with rio.open(modis_file) as src:        
            warp_options = {
                'crs': dst_crs,            
                'resampling': resampling_strategy
            }
            with rasterio.vrt.WarpedVRT(src, **warp_options) as vrt:
                window = vrt.window(*bounds)            
                data = vrt.read(1,window=window, out_shape=(profile['height'], profile['width']), 
                                out_dtype=np.float32, resampling=resampling_strategy)

            data[data == src.nodata] = np.nan  # Set nodata values to NaN
    except:
        return null, ref_file, modis_file, resampling_strategy # type: ignore
    
    return data, ref_file, modis_file, resampling_strategy # type: ignore

def get_modis_ndvi_data_rio(landsat_files, years, resampling_strategy=rasterio.enums.Resampling.bilinear) -> NDArray[np.float32]:
    
    modis_files = []
    for year in years:
        for m in range(n_imag_per_year):
            modis_files.append(f'/vsicurl/{random.choice(gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif')


    n_years = len(years)
    n_s = n_years*n_imag_per_year
    modis_data = np.empty((n_s, n_pix), dtype=np.float32)
    executor = ProcessPoolExecutor(max_workers=n_threads)
    futures = [executor.submit(get_modis_ndvi_rio, landsat_files[i], modis_files[i], resampling_strategy)
               for i in range(len(modis_files))]
    for i, future in tqdm(enumerate(futures), total=len(modis_files), desc='Processing MODIS NDVI data'):
        data, ref_file, modis_file, resampling_strategy = future.result()
        if data is None:
            #print(f"Failed to process {modis_file}")
            futures.append(executor.submit(get_modis_ndvi_rio, ref_file, modis_file, resampling_strategy))
        else:
            modis_data[i, :] = data.ravel()

    executor.shutdown()
    return modis_data
    

def get_modis_ndvi_data(landsat_files, years, resampling_strategy='GRA_Bilinear') -> NDArray[np.float32]:

    #landsat_files = get_landsat_filenames(landsat_tile, years)

    modis_files = []
    for year in years:
        for m in range(n_imag_per_year):
            modis_files.append(f'/vsicurl/{random.choice(gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif')

    n_years = len(years)
    n_s = n_years*n_imag_per_year
    modis_data = np.empty((n_s, n_pix), dtype=np.float32)
    executor = ProcessPoolExecutor(max_workers=n_threads)
    futures = [executor.submit(warp_tile, landsat_files[i], modis_files[i], n_pix, resampling_strategy)
        for i in range(len(modis_files))]
    for i, future in enumerate(futures):
        modis_data[i, :] = future.result()

    executor.shutdown()
    return modis_data


    # modis_files = []
    # for year in years:
    #     for m in range(n_imag_per_year):
    #         modis_files.append(f'/vsicurl/{random.choice(gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif')

    # landsat_files = get_landsat_filenames(landsat_tile, years)
    # n_years = len(years)
    # n_s = n_years*n_imag_per_year
    # modis_data = np.empty((n_s, n_pix), dtype=np.float32)
    # executor = ProcessPoolExecutor(max_workers=n_threads)
    # futures = {executor.submit(warp_tile, i, landsat_files[i], modis_files[i], n_threads, 
    #                             n_pix, resampling_strategy, gdal_opts): i for i in range(len(modis_files))}
    # for future in as_completed(futures):
    #     i = futures[future]
    #     try:
    #         modis_data[i, :] = future.result()
    #     except Exception as e:
    #         print(f"Task {i} generated an exception: {e}")
    # executor.shutdown()
    # return modis_data

def mask_from_qa(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:

    n_s = n_years*n_imag_per_year
    range_qa = range(n_s*(n_spect_bands), n_s*(n_spect_bands+1))
    
    ''' This is a workaround for the above commented code, which is not parallel
    landsat_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.extractArrayRows(landsat_data, n_threads, landsat_mask, range_qa)
    '''
    #landsat_mask = landsat_data[range_qa, :]
    # Try removing snow, check 16d_intervals.xlsx for the QA info and scaling
    # 14 = additional cloud buffer over land
    # 3 = cloud
    # 6 = snow
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17   
    gap_mask_keep_buffer   = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    # @FIXME this thing is basically not parallel
    ind_start = n_s*n_spect_bands
    landsat_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.fillArray(landsat_mask, n_threads, 1.)
    for i in range(n_s):
        n_cloud_pix = np.sum((landsat_data[ind_start + i,:] == 3))# + 1
        n_buff_pix = np.sum((landsat_data[ind_start + i,:] == 14))# + 1 # avoid division by 0
        gap_mask = gap_mask_remove_buffer if (n_cloud_pix>n_buff_pix) else gap_mask_keep_buffer
        '''
        for k in range(0,18):
            sb.swapRowsValues(landsat_mask, n_threads, [i], k, gap_mask[k])
        '''
        # This is a workaround for the above commented code, which is not parallel
        #mask_ones = np.nonzero(gap_mask)[0]
        mask_zeros = np.nonzero(np.logical_not(gap_mask))[0]
        ind = np.isin(landsat_data[ind_start + i,:], mask_zeros) # kind='table' is faster but only for integer arrays
        landsat_mask[i,ind] = 0.
        

    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), landsat_mask, 1., np.nan)

    del landsat_mask
    gc.collect()

    return landsat_data

def mask_from_qa_parallel(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    # Use parallel processing to mask Landsat data from QA

    from concurrent.futures import ThreadPoolExecutor

    n_s = n_years*n_imag_per_year

    #landsat_mask = landsat_data[range_qa, :]
    # Try removing snow, check 16d_intervals.xlsx for the QA info and scaling
    # 14 = additional cloud buffer over land
    # 3 = cloud
    # 6 = snow
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17   
    gap_mask_keep_buffer   = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    # @FIXME this thing is basically not parallel
    ind_start = n_s*n_spect_bands
    landsat_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.fillArray(landsat_mask, n_threads, 1.)

    def process_mask_row(i: int) -> None:
        n_cloud_pix = np.sum((landsat_data[ind_start + i,:] == 3))
        n_buff_pix = np.sum((landsat_data[ind_start + i,:] == 14))  # avoid division by 0
        gap_mask = gap_mask_remove_buffer if (n_cloud_pix > n_buff_pix) else gap_mask_keep_buffer

        mask_zeros = np.nonzero(np.logical_not(gap_mask))[0]
        ind = np.isin(landsat_data[ind_start + i,:], mask_zeros) # kind='table' is faster but only for integer arrays
        landsat_mask[i,ind] = 0.
        
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_mask_row, i) for i in range(n_s)]
        for future in futures:
            future.result()
        
    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), landsat_mask, 1., np.nan)

    del landsat_mask
    gc.collect()

    return landsat_data

def mask_from_modis(landsat_data: NDArray[np.float32], modis_data:NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Create a mask from the QA band of Landsat data.
    """
    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg

    range_nir = range(n_s*1, n_s*2)
    range_red = range(n_s*0, n_s*1)
    range_ndvi = range(n_s*(n_spect_bands+1), n_s*(n_spect_bands+2))
    range_qa = range(n_s*(n_spect_bands), n_s*(n_spect_bands+1))

    diff_th, count_th = filter_params(n_years)
    
    sb.computeNormalizedDifference(landsat_data, n_threads,
                                range_nir, range_red, range_ndvi,
                                mask_band_scaling, mask_band_scaling, mask_result_scaling, 
                                mask_result_offset, [-mask_result_scaling, mask_result_scaling])
    landsat_NDVI_masked = np.empty((n_s, n_pix), dtype=np.float32)
    sb.extractArrayRows(landsat_data, n_threads, landsat_NDVI_masked, range_ndvi)
    landsat_NDVI_masked_t = np.empty((n_pix, n_s), dtype=np.float32)
    sb.transposeArray(landsat_NDVI_masked, n_threads, landsat_NDVI_masked_t)

    modis_data_t = np.empty((n_pix, n_s), dtype=np.float32)
    sb.transposeArray(modis_data, n_threads, modis_data_t)

    modis_ndvi_mask_t = np.empty((n_pix, n_s), dtype=np.float32)
    modis_ndvi_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.maskDifference(landsat_NDVI_masked_t, n_threads, diff_th, count_th, modis_data_t, modis_ndvi_mask_t)
    sb.transposeArray(modis_ndvi_mask_t, n_threads, modis_ndvi_mask)

    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), modis_ndvi_mask, 1., np.nan)

    del modis_data
    del modis_data_t
    del landsat_NDVI_masked
    del landsat_NDVI_masked_t
    del modis_ndvi_mask
    del modis_ndvi_mask_t
    gc.collect()

    return landsat_data

def inpaint_stripes(landsat_data: NDArray[np.float32], n_years:int) -> None:
    """
    Inpaint stripes in Landsat data.
    """
    
    n_s = n_years*n_imag_per_year

    sample_idxs_band0, row_starts_band0, row_ends_band0, col_starts_band0, col_ends_band0, fill_true_erase_false_band0 = [], [], [], [], [], []
    sample_idxs, row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], [], []
    for i in range(n_s):
        tmp_image = landsat_data[i].reshape(x_size, y_size)
        row_starts_tmp, row_ends_tmp, col_starts_tmp, col_ends_tmp, fill_true_erase_false_tmp, _, _, _ = process_image_in_chunks(tmp_image, inpaint_chunk_size, gap_stripes_th, gap_general_th, fft_th)
        sample_idxs_band0 += [i for x in row_starts_tmp]
        row_starts_band0 += row_starts_tmp
        row_ends_band0 += row_ends_tmp
        col_starts_band0 += col_starts_tmp
        col_ends_band0 += col_ends_tmp
        fill_true_erase_false_band0 += fill_true_erase_false_tmp

    for i in range(n_spect_bands):
        sample_idxs_band = [x + n_s*i for x in sample_idxs_band0]
        sample_idxs += sample_idxs_band
        row_starts += row_starts_band0
        row_ends += row_ends_band0
        col_starts += col_starts_band0
        col_ends += col_ends_band0
        fill_true_erase_false += fill_true_erase_false_band0
        
    row_starts_fill, row_starts_erase = [], []
    row_ends_fill, row_ends_erase = [], []
    col_starts_fill, col_starts_erase = [], []
    col_ends_fill, col_ends_erase = [], []
    sample_idxs_fill, sample_idxs_erase = [], []
    for si, rs, re, cs, ce, fe in zip(sample_idxs, row_starts, row_ends, col_starts, col_ends, fill_true_erase_false):
        if fe:
            row_starts_fill.append(rs)
            row_ends_fill.append(re)
            col_starts_fill.append(cs)
            col_ends_fill.append(ce)
            sample_idxs_fill.append(si)
        else:
            row_starts_erase.append(rs)
            row_ends_erase.append(re)
            col_starts_erase.append(cs)
            col_ends_erase.append(ce)
            sample_idxs_erase.append(si)
            

    sb.inpaintChunks(landsat_data, n_threads, inpaint_radius, inpaint_padding, x_size, y_size, sample_idxs_fill, row_starts_fill, row_ends_fill, col_starts_fill, col_ends_fill)
    sb.eraseChunks(landsat_data, n_threads, x_size, y_size, sample_idxs_erase, row_starts_erase, row_ends_erase, col_starts_erase, col_ends_erase)


def bands_aggregation(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Aggregate Landsat data bands.
    """
    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg

    landsat_bands = np.empty((n_s*n_spect_bands, n_pix), dtype=np.float32)
    sb.extractArrayRows(landsat_data, n_threads, landsat_bands, range(0, n_s*n_spect_bands))
    del landsat_data
    gc.collect()

    landsat_bands_t = np.empty((n_pix, n_s*n_spect_bands), dtype=np.float32)
    sb.transposeArray(landsat_bands, n_threads, landsat_bands_t)
    del landsat_bands
    gc.collect()

    landsat_bands_agg_t = np.empty((n_pix, n_s_agg*n_spect_bands), dtype=np.float32)
    agg_pattern = []
    for i in range(n_spect_bands):
        for j in range(n_years):
            base_idx = n_s*i + j*n_imag_per_year
            agg_pattern.append([base_idx+0,base_idx+1])
            agg_pattern.append([base_idx+2,base_idx+3])
            agg_pattern.append([base_idx+4,base_idx+5])
            agg_pattern.append([base_idx+6,base_idx+7])
            agg_pattern.append([base_idx+8,base_idx+9])
            agg_pattern.append([base_idx+10,base_idx+11])
            agg_pattern.append([base_idx+11,base_idx+12])
            agg_pattern.append([base_idx+13,base_idx+14])
            agg_pattern.append([base_idx+15,base_idx+16])
            agg_pattern.append([base_idx+17,base_idx+18])
            agg_pattern.append([base_idx+19,base_idx+20])
            agg_pattern.append([base_idx+21,base_idx+22])
            
    sb.nanMeanAggregatePattern(landsat_bands_t, n_threads, landsat_bands_agg_t, agg_pattern)
    del landsat_bands_t
    gc.collect()

    return landsat_bands_agg_t

def swa_reconstructing(landsat_bands_agg_t: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Reconstruct SWA from aggregated Landsat bands.
    """
    w_0_agg = 1.0
    n_s_agg = n_years*n_imag_per_year_agg

    w_p_agg = (get_SWA_weights(att_env, att_seas, n_imag_per_year_agg, n_s_agg)[1:][::-1]).astype(np.float32)
    w_f_agg = (get_SWA_weights(att_env, att_seas, n_imag_per_year_agg, n_s_agg)[1:]).astype(np.float32)*future_scaling
    
    landsat_bands_rec_t = np.empty((n_pix, n_s_agg*n_spect_bands), dtype=np.float32)

    for b in range(n_spect_bands):
        sb.applyTsirf(landsat_bands_agg_t, n_threads, landsat_bands_rec_t, n_s_agg, b*n_s_agg, b*n_s_agg, w_0_agg, w_p_agg, w_f_agg, True)
    del landsat_bands_agg_t
    gc.collect()

    return landsat_bands_rec_t

def process_image_in_chunks(image, chunk_size, gap_stripes_th, gap_general_th, fft_th):
    mask = np.isnan(image)
    height, width = image.shape
    n_chunk_height = int(np.floor(height/chunk_size))
    n_chunk_width = int(np.floor(width/chunk_size))
    gap_fraq = np.zeros((n_chunk_height, n_chunk_width))
    fft_score = np.zeros((n_chunk_height, n_chunk_width))
    rec_flag = np.zeros((n_chunk_height, n_chunk_width))
    #output_image = image.copy()
    row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], []
    # Loop through the image by chunks
    for i in range(0, n_chunk_height):
        for j in range(0, n_chunk_width):
            # @FIXME check is also theretically the location of patial frequencies in different share chunks is the same 
            if i != (n_chunk_height-1):
                row_start, row_end = (i * chunk_size, (i+1) * chunk_size)
            else:
                row_start, row_end = (i * chunk_size, height)
            if j != (n_chunk_width-1):
                col_start, col_end = (j * chunk_size, (j+1) * chunk_size)
            else:
                col_start, col_end = (j * chunk_size, width)
            image_chunk = image[row_start:row_end, col_start:col_end]
            mask_chunk = mask[row_start:row_end, col_start:col_end]
            gap_count_chunk = np.sum(mask_chunk)
            gap_fraq[i, j] = gap_count_chunk/(row_end-row_start)/(col_end-col_start)
            if gap_fraq[i, j] < gap_general_th:
                row_starts += [row_start]
                row_ends += [row_end]
                col_starts += [col_start]
                col_ends += [col_end]
                fill_true_erase_false += [True]
                rec_flag[i,j] = 1
            else:
                image_filled = np.nan_to_num(image_chunk, nan=0)
                image_filled = image_filled[0:chunk_size,0:chunk_size].copy()
                # image_filled /= max(np.max(image_filled),1)
                image_filled[image_filled!=0] = 1
                ft = np.fft.ifftshift(image_filled)
                ft = np.fft.fft2(ft, norm='ortho')
                ft = np.fft.fftshift(ft)
                ft[48:80,48:80] = 0
                fft_score[i, j] = np.max(np.abs(ft))
                if fft_score[i, j] > fft_th:
                    row_starts += [row_start]
                    row_ends += [row_end]
                    col_starts += [col_start]
                    col_ends += [col_end]
                    if gap_fraq[i, j] < gap_stripes_th:
                        fill_true_erase_false += [True]
                        rec_flag[i,j] = 1
                    else:
                        fill_true_erase_false += [False]
                        rec_flag[i,j] = -1
                    
    return row_starts, row_ends, col_starts, col_ends, fill_true_erase_false, gap_fraq, fft_score, rec_flag


def save_landsat_bands(landsat_bands_rec_t: NDArray[np.float32], landsat_tile: str, years: List[int], landsat_files: List[str]) -> None:    
    """
    Save reconstructed Landsat bands to disk.
    """
    n_years = len(years)
    n_s_agg = n_years*n_imag_per_year_agg

    out_data = np.empty((n_s_agg*n_spect_bands, n_pix), dtype=np.float32)
    sb.transposeArray(landsat_bands_rec_t, n_threads, out_data)
    del landsat_bands_rec_t
    gc.collect()

    
    out_dir = f'/tmp/{landsat_tile}'
    os.makedirs(out_dir, exist_ok = True)
    
    compression_command = f"gdal_translate -a_nodata {no_data_out} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
    out_files = []
    for band in bands_prefix_out:
        for year in years:
            for m in range(n_imag_per_year_agg):
                out_files.append(f'{band}.ard2_m_30m_s_{year}{month_start[m]}_{year}{month_end[m]}{file_ending_out}')

    s3_out = [f'{random.choice(s3_aliases)}/{s3_params["s3_prefix"]}/{landsat_tile}' for _ in range(len(out_files))]
    sb.writeUInt16Data(out_data, n_threads, gdal_opts, landsat_files[0:len(out_files)], out_dir, out_files, range(len(out_files)),
                x_off, y_off, x_size, y_size, no_data_out, compression_command, s3_out)
    os.rmdir(out_dir)
    print(f"Check gaia at {s3_out[0]}")

    
# %%
def show_image_landsat(landsat_data: NDArray[np.float32], years:List, year:int, img_in_year:int, band:int) -> None:
    """
    Show Landsat image for a specific band.
    """
    # year=2001; img_in_year=10; band=7
    # band 7=qa; band 8=ndvi
    import matplotlib.pyplot as plt
    
    n_years = len(years)
    n_s = n_years*n_imag_per_year    
    ind_band = n_s*band + (year - years[0])*n_s + img_in_year

    data = landsat_data[ind_band, :].reshape((y_size, x_size))
    plt.imshow(data)
    plt.title(f'Band {band} for year {year} image {img_in_year}')
    plt.colorbar()
    plt.show()  

def show_image_modis(modis_data: NDArray[np.float32], years:List, year:int, img_in_year:int) -> None:
    """
    Show MODIS image for a specific year and image in year.
    """
    # year=2001; img_in_year=10
    import matplotlib.pyplot as plt
            
    ind_band = (year - years[0])*n_imag_per_year + img_in_year

    data = modis_data[ind_band, :].reshape((y_size, x_size))
    plt.imshow(data)
    plt.title(f'MODIS NDVI for year {year} image {img_in_year}')
    plt.colorbar()
    plt.show()