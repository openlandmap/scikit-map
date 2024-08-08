from predict_api import s3_setup, s3_list_files, read_json, print_catalog_statistics,\
    DataCatalog, DataLoader, RFRegressorDepths, RFClassifierProbs, PredictedProbs, PredictedDepths, TimeTracker, _create_image_template
import os
import sys
import geopandas as gpd
import numpy as np
import warnings
import skmap_bindings as smb
import time
import random
warnings.filterwarnings("ignore", module="sklearn")
#
# global constants
#
YEARS = ['2000', '2004', '2008', '2012', '2016', '2020', '2022']
DEPTHS = [0, 20, 50, 100, 200]
QUANTILES = [0.025, 0.975]
THREADS = 96
COVS_PATH = '../ai4sh_robert/ai4sh_vrt.json'
TILES_PATH = '../ai4sh_robert/eu_tiles_epsg.3035.gpkg'
TILES_SHUF = '../ai4sh_robert/shuf.txt'
MODEL_PATH = '../ai4sh_robert'
MASK_PATH = '/vsicurl/http://192.168.49.30:8333/ai4sh/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719.tif'
VALID_MASK_VALUE = 1
GDAL_OPTS = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}
TILES_ID = [1025, 1052, 1083] # Norway, Germany, Italy
# register S3 aliases
# S3_PREFIX = '/tmp-eumap-ai4sh/v4'
S3_PREFIX = None
ACCESS_KEY = 'iwum9G1fEQ920lYV4ol9'
SECRET_KEY = 'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0'
#
# main function
#
def main():
    # read tiles
    tiles = gpd.read_file(TILES_PATH)
    # check script arguments
    check_otf = False
    tiles_id = [TILES_ID[0]]
    if (len(sys.argv) == 2):
        base_dir = f'{os.getcwd()}/tmp'
        if sys.argv[1] == "--check-otf":
            check_otf = True
        elif sys.argv[1] == "--check-tile1":
            pass
        elif sys.argv[1] == "--check-tiles":
            tiles_id = TILES_ID
        else:
            raise ValueError("Expecting one argument: --check-otf | --check-tile1 | --check-tiles")
    elif len(sys.argv) == 4:
        start_tile=max(int(sys.argv[1]), 0)
        end_tile=min(int(sys.argv[2]), len(tiles))
        with open(TILES_SHUF, 'r') as file:
            shuf = [int(line.strip()) for line in file]
        tiles_id = tiles['id'][shuf[start_tile:end_tile]]
        server_name=sys.argv[3]
        base_dir = f'/mnt/{server_name}/ai4sh_pred/tmp'
    else:
        raise ValueError("Usage: run_models.py ([--check-otf | --check_tile1 | --check_tiles] | <start_tile> <end_tile> <server_name)>")
    #
    # load models
    
    # texture1
    texture1_params = {
        'model':RFRegressorDepths(
            model_name='texture1',
            model_path=f'{MODEL_PATH}/propduction.model_rf.texture1_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1
    }
    # texture2
    texture2_params = {
        'model':RFRegressorDepths(
            model_name='texture2',
            model_path=f'{MODEL_PATH}/propduction.model_rf.texture2_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1
    }
    
    #
    # catalogs
    
    catalog = DataCatalog.read_catalog('ai4sh', COVS_PATH)
    textures_model_params = [texture1_params, texture2_params]
    textures_features = {f for params in textures_model_params for f in params['model'].model_features}
    textures_catalog = catalog.query('soil.textures', YEARS, textures_features)
    print_catalog_statistics(textures_catalog)
    
    if check_otf:
        return
    
    # register s3 aliases
    s3_aliases = s3_setup(S3_PREFIX is not None, ACCESS_KEY, SECRET_KEY)
    
    # get data loader
    textures_data = DataLoader(textures_catalog, tiles, MASK_PATH, VALID_MASK_VALUE)
        
    # get models
    texture1:RFRegressorDepths = texture1_params['model']
    texture2:RFRegressorDepths = texture2_params['model']
    
    # NOTE must have the same numbers of trees
    assert(texture1.num_trees == texture2.num_trees)
    num_trees = texture1.num_trees
    
    # tile loop
    for tile_id in tiles_id:
        with TimeTracker(f"Tile {tile_id}", False):
            
            # resume
            if len(s3_list_files(s3_aliases, S3_PREFIX, tile_id, 'texture.')) == \
                3 * (len(QUANTILES) + 1) * (len(DEPTHS) - 1) * (len(YEARS) - 1):
                print(f"Tile {tile_id} already computed, skipping...")
                continue
            
            with TimeTracker(f" - Reading data", False):
                # x_size, y_size = (1000, 1000) # To debug
                # textures_data.load_tile_data(tile_id, THREADS, GDAL_OPTS, x_size, y_size)
                textures_data.load_tile_data(tile_id, THREADS, GDAL_OPTS)
                n_years = len(YEARS)
                n_depths = len(DEPTHS)
                n_quant = len(QUANTILES)
                n_years_avg = n_years - 1
                n_depths_avg = n_depths - 1
                n_pix = textures_data.x_size * textures_data.y_size
                n_pix_val = textures_data.num_pixels_valid
                n_trees = texture1.num_trees
                n_textures = 3
                n_files = n_depths_avg * n_years_avg * n_textures * (n_quant + 1)
                # This order is also representing the ordering of the array            
                n_files_dephts = n_years_avg * n_textures * (n_quant + 1) # Offset of the dephts
                n_files_years = n_textures * (n_quant + 1) # Offset of the years
                # Textures order: clay, sand, silt
                out_data_t = np.empty((n_pix_val, n_files), dtype=np.float32)
                write_data_t = np.empty((n_pix, n_files), dtype=np.float32)
                write_data = np.empty((n_files, n_pix), dtype=np.float32)
                nodata = 255

            with TimeTracker(f" - Getting raw tree predictions", False):
                # Get raw trees predictions            
                # [n_depths](n_trees, n_samples)
                pred1_trees = [texture1.predictDepth(textures_data, i) for i in range(n_depths)]
                pred2_trees = [texture2.predictDepth(textures_data, i) for i in range(n_depths)]

            with TimeTracker(f" - Deriving statistics", False):
                # Compute derived statistics
                out_files = []
                for d in range(n_depths_avg):
                    for y in range(n_years_avg):
                        trees1_avg = np.empty((n_trees, n_pix_val), dtype=np.float32)
                        trees2_avg = np.empty((n_trees, n_pix_val), dtype=np.float32)                    
                        smb.averageAi4sh(trees1_avg, THREADS, pred1_trees[d], pred1_trees[d+1], n_pix_val, y)
                        smb.averageAi4sh(trees2_avg, THREADS, pred2_trees[d], pred2_trees[d+1], n_pix_val, y)      
                        trees1_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
                        trees2_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
                        mean1 = np.empty((n_pix_val,), dtype=np.float32)
                        mean2 = np.empty((n_pix_val,), dtype=np.float32)                    
                        smb.transposeArray(trees1_avg, THREADS, trees1_avg_t)
                        smb.transposeArray(trees2_avg, THREADS, trees2_avg_t)
                        smb.nanMean(trees1_avg_t, THREADS, mean1)
                        smb.nanMean(trees2_avg_t, THREADS, mean2)
                        if texture1_params['expm1']:
                            np.expm1(mean1, out=mean1)
                            np.expm1(trees1_avg_t, out=trees1_avg_t)
                        if texture2_params['expm1']:
                            np.expm1(mean2, out=mean2)                        
                            np.expm1(trees2_avg_t, out=trees2_avg_t)
                        clay_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                        sand_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                        silt_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                        smb.fitPercentage(clay_trees, THREADS, trees1_avg_t, trees2_avg_t)
                        smb.hadamardProduct(sand_trees, THREADS, trees1_avg_t, clay_trees)
                        smb.hadamardProduct(silt_trees, THREADS, trees2_avg_t, clay_trees)                
                        PERCENTILES = [q*100. for q in QUANTILES]                    
                        offset_caly = d * n_files_dephts + y * n_files_years
                        offset_sand = d * n_files_dephts + y * n_files_years + (n_quant + 1)
                        offset_silt = d * n_files_dephts + y * n_files_years + 2 * (n_quant + 1)
                        smb.computePercentiles(clay_trees, THREADS, out_data_t, offset_caly + 1, PERCENTILES)
                        smb.computePercentiles(sand_trees, THREADS, out_data_t, offset_sand + 1, PERCENTILES)
                        smb.computePercentiles(silt_trees, THREADS, out_data_t, offset_silt + 1, PERCENTILES)
                        clay_mean = np.empty((n_pix_val,), dtype=np.float32)
                        sand_mean = np.empty((n_pix_val,), dtype=np.float32)
                        silt_mean = np.empty((n_pix_val,), dtype=np.float32)                    
                        smb.fitPercentage(clay_mean, THREADS, mean1, mean2)
                        smb.hadamardProduct(sand_mean, THREADS, mean1, clay_mean)
                        smb.hadamardProduct(silt_mean, THREADS, mean2, clay_mean)
                        out_data_t[:,offset_caly] = clay_mean
                        out_data_t[:,offset_sand] = sand_mean
                        out_data_t[:,offset_silt] = silt_mean
                        for t in ['clay.tot_iso.11277.2020.wpct', 'sand.tot_iso.11277.2020.wpct', 'silt.tot_iso.11277.2020.wpct']:
                            out_files.append(f'{t}_m_30m_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_eu_epsg.3035_v20240804')
                            for q in QUANTILES:
                                formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                                out_files.append(f'{t}_{formatted_p}_30m_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_eu_epsg.3035_v20240804')

            with TimeTracker(f" - Saving results", False):
                smb.offsetAndScale(out_data_t, THREADS, 0.5, 1.)
                smb.fillArray(write_data_t, THREADS, nodata)
                smb.expandArrayRows(out_data_t, THREADS, write_data_t, textures_data._pixels_valid_idx)
                smb.transposeArray(write_data_t, THREADS, write_data)
                tile_dir = base_dir + f'/{tile_id}'
                os.makedirs(tile_dir, exist_ok=True)
                temp_tif = _create_image_template(textures_data.mask_path, textures_data.tiles, textures_data.tile_id, textures_data.x_size, textures_data.y_size, 'uint8', nodata, tile_dir)
                write_idx = range(len(out_files))
                compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
                s3_out = None
                if S3_PREFIX is not None:
                    s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{S3_PREFIX}/{textures_data.tile_id}' for _ in range(len(out_files))]
                smb.writeByteData(write_data, THREADS, GDAL_OPTS, [temp_tif for _ in range(len(out_files))], tile_dir, out_files, 
                                                 write_idx, 0, 0, textures_data.x_size, textures_data.y_size, int(nodata), compress_cmd, s3_out)
                 
               
            
#
if __name__ == "__main__":
    main()
