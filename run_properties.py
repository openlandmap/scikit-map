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
import gc

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
MODEL_PATH = '../ai4sh_robert'
TILES_SHUF = '../ai4sh_robert/shuf.txt'
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
    #
    
    # bulk density
    bulk_density_params = {
        'model':RFRegressorDepths(
            model_name='bulk.density',
            model_path=f'{MODEL_PATH}/propduction.model_rf.bulk_density_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':False,
        'scale':100,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'bd.core_iso.11272.2017.g.cm3',
        'out_files_prefix':['bulk.density_ai4sh_m_30m', 'bulk.density_ai4sh_p025_30m', 'bulk.density_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # soc
    soc_params = {
        'model':RFRegressorDepths(
            model_name='soc', 
            model_path=f'{MODEL_PATH}/propduction.model_rf.oc_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767, 
        'dtype':'int16',
        'prop_file_name':'oc_iso.10694.1995.wpct',
        'out_files_prefix':['soc_ai4sh_m_30m', 'soc_ai4sh_p025_30m', 'soc_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # ocd
    ocd_params = {
        'model':RFRegressorDepths(
            model_name='ocd', 
            model_path=f'{MODEL_PATH}/propduction.model_rf.ocd_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767, 
        'dtype':'int16',
        'prop_file_name':'oc_iso.10694.1995.mg.cm3',
        'out_files_prefix':['ocd_ai4sh_m_30m', 'ocd_ai4sh_p025_30m', 'ocd_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # ph h2o
    ph_h2o_params = {
        'model':RFRegressorDepths(
            model_name='ph.h2o',
            model_path=f'{MODEL_PATH}/propduction.model_rf.ph_h2o_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':False,
        'scale':10,
        'nodata':255,
        'dtype':'uint8',
        'prop_file_name':'ph.h2o_iso.10390.2021.index',
        'out_files_prefix':['ph.h2o_ai4sh_m_30m', 'ph.h2o_ai4sh_p025_30m', 'ph.h2o_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # ph cacl2
    ph_cacl2_params = {
        'model':RFRegressorDepths(
            model_name='ph.cacl2',
            model_path=f'{MODEL_PATH}/propduction.model_rf.ph_cacl2_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':False,
        'scale':10,
        'nodata':255,
        'dtype':'uint8',
        'prop_file_name':'ph.cacl2_iso.10390.2021.index',
        'out_files_prefix':['ph.cacl2_ai4sh_m_30m', 'ph.cacl2_ai4sh_p025_30m', 'ph.cacl2_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # N
    nitrogen_params = {
        'model':RFRegressorDepths(
            model_name='total.n',
            model_path=f'{MODEL_PATH}/propduction.model_rf.N_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'n.tot_iso.13878.1998.wpct',
        'out_files_prefix':['total.n_ai4sh_m_30m', 'total.n_ai4sh_p025_30m', 'total.n_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # P
    phosphorus_params = {
        'model':RFRegressorDepths(
            model_name='available.p',
            model_path=f'{MODEL_PATH}/propduction.model_rf.P_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'p.ext_iso.11263.1994.mg.kg',
        'out_files_prefix':['available.p_ai4sh_m_30m', 'available.p_ai4sh_p025_30m', 'available.p_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # K
    potassium_params = {
        'model':RFRegressorDepths(
            model_name='available.k',
            model_path=f'{MODEL_PATH}/propduction.model_rf.K_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'k.ext_usda.nrcs.mg.kg',
        'out_files_prefix':['available.k_ai4sh_m_30m', 'available.k_ai4sh_p025_30m', 'available.k_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # CEC
    cec_params = {
        'model':RFRegressorDepths(
            model_name='cec',
            model_path=f'{MODEL_PATH}/propduction.model_rf.CEC_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'cec.ext_iso.11260.1994.cmol.kg',
        'out_files_prefix':['cec_ai4sh_m_30m', 'cec_ai4sh_p025_30m', 'cec_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # EC
    ec_params = {
        'model':RFRegressorDepths(
            model_name='ec',
            model_path=f'{MODEL_PATH}/propduction.model_rf.EC_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'ec_iso.11265.1994.ms.m',
        'out_files_prefix':['ec_ai4sh_m_30m', 'ec_ai4sh_p025_30m', 'ec_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'], 
        's3_prefix':S3_PREFIX
    }
    # CaCO3
    carbonates_params = {
        'model':RFRegressorDepths(
            model_name='carbonates',
            model_path=f'{MODEL_PATH}/propduction.model_rf.caco3_ccc.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':10,
        'nodata':32767,
        'dtype':'int16',
        'prop_file_name':'caco3_iso.10693.1995.wpct',
        'out_files_prefix':['carbonates_ai4sh_m_30m', 'carbonates_ai4sh_p025_30m', 'carbonates_ai4sh_p975_30m'], 
        'out_files_suffix':['epsg.3035_v20240731', 'epsg.3035_v20240731', 'epsg.3035_v20240731'],
        's3_prefix':S3_PREFIX
    }
    
    #
    # catalogs
    catalog = DataCatalog.read_catalog('ai4sh', COVS_PATH)
    
    
    
    # properties_model_params = [
    #     cec_params, 
    #     ec_params, 
    #     carbonates_params,
    #     bulk_density_params, 
    #     soc_params, 
    #     ocd_params, 
    #     ph_h2o_params, 
    #     ph_cacl2_params, 
    #     nitrogen_params, 
    #     phosphorus_params, 
    #     potassium_params
    # ]
    
    properties_model_params = [
        ph_h2o_params, 
        ph_cacl2_params, 
        nitrogen_params, 
        phosphorus_params, 
        potassium_params
    ]
    
    # properties_model_params = [
    #     cec_params, 
    #     ec_params, 
    #     carbonates_params,
    #     bulk_density_params, 
    #     soc_params, 
    #     ocd_params
    # ]
    properties_features = {f for params in properties_model_params for f in params['model'].model_features}
    properties_catalog = catalog.query('soil.properties', YEARS, properties_features)
    print_catalog_statistics(properties_catalog)
    
    if check_otf:
        return
    
    # register s3 aliases
    s3_aliases = s3_setup(S3_PREFIX is not None, ACCESS_KEY, SECRET_KEY)

    # get data loader
    properties_data = DataLoader(properties_catalog, tiles, MASK_PATH, VALID_MASK_VALUE)
 
    # tile loop
    for tile_id in tiles_id:
        with TimeTracker(f"tile {tile_id}", True):
            
            with TimeTracker(f" - Reading data", False):
                # x_size, y_size = (200, 200) # To debug
                # properties_data.load_tile_data(tile_id, THREADS, GDAL_OPTS, x_size, y_size)
                properties_data.load_tile_data(tile_id, THREADS, GDAL_OPTS)
            
            # traverse soil properties models' params
            for params in properties_model_params:
                with TimeTracker(f" # Modeling {params['prop_file_name']}", False):
                    # resume
                    if len(s3_list_files(s3_aliases, params['s3_prefix'], tile_id, params['out_files_prefix'][0][:-9])) == \
                        (len(QUANTILES) + 1) * (len(DEPTHS) - 1) * (len(YEARS) - 1):
                        print(f"tile {tile_id} already computed, skipping...")
                        continue

                    properties_model:RFRegressorDepths = params['model']
                    n_years = len(YEARS)
                    n_depths = len(DEPTHS)
                    n_quant = len(QUANTILES)
                    n_years_avg = n_years - 1
                    n_depths_avg = n_depths - 1
                    n_pix = properties_data.x_size * properties_data.y_size
                    n_pix_val = properties_data.num_pixels_valid
                    n_trees = properties_model.num_trees
                    n_files = n_depths_avg * n_years_avg * (n_quant + 1)
                    # This order is also representing the ordering of the array            
                    n_files_dephts = n_years_avg  * (n_quant + 1) # Offset of the dephts
                    n_files_years = n_quant + 1 # Offset of the years
                    # Textures order: clay, sand, silt
                    out_data_t = np.empty((n_pix_val, n_files), dtype=np.float32)
                    nodata = params['nodata']

                    with TimeTracker(f" - Getting raw tree predictions", False):
                        # Get raw trees predictions            
                        # [n_depths](n_trees, n_samples)
                        pred_trees = [properties_model.predictDepth(properties_data, i) for i in range(n_depths)]
                        

                    with TimeTracker(f" - Deriving statistics", False):
                        # Compute derived statistics
                        out_files = []
                        for d in range(n_depths_avg):
                            for y in range(n_years_avg):
                                trees_avg = np.empty((n_trees, n_pix_val), dtype=np.float32)
                                smb.averageAi4sh(trees_avg, THREADS, pred_trees[d], pred_trees[d+1], n_pix_val, y)
                                trees_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
                                prop_mean = np.empty((n_pix_val,), dtype=np.float32)
                                smb.transposeArray(trees_avg, THREADS, trees_avg_t)
                                smb.nanMean(trees_avg_t, THREADS, prop_mean)
                                if params['expm1']:
                                    np.expm1(prop_mean, out=prop_mean)
                                    np.expm1(trees_avg_t, out=trees_avg_t)
                                PERCENTILES = [q*100. for q in QUANTILES]                    
                                offset_prop = d * n_files_dephts + y * n_files_years
                                smb.computePercentiles(trees_avg_t, THREADS, out_data_t, offset_prop + 1, PERCENTILES)
                                out_data_t[:,offset_prop] = prop_mean
                                out_files.append(f"{params['prop_file_name']}_m_30m_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_eu_epsg.3035_v20240804")
                                for q in QUANTILES:
                                    formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                                    out_files.append(f"{params['prop_file_name']}_{formatted_p}_30m_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_eu_epsg.3035_v20240804")
                                                            
                    del pred_trees
                    gc.collect()

                    with TimeTracker(f" - Saving results", False):
                        write_data_t = np.empty((n_pix, n_files), dtype=np.float32)
                        write_data = np.empty((n_files, n_pix), dtype=np.float32)
                        smb.scaleAndOffset(out_data_t, THREADS, 0.5, params['scale'])
                        smb.fillArray(write_data_t, THREADS, nodata)
                        smb.expandArrayRows(out_data_t, THREADS, write_data_t, properties_data._pixels_valid_idx)
                        smb.transposeArray(write_data_t, THREADS, write_data)
                        tile_dir = base_dir + f'/{tile_id}'
                        os.makedirs(tile_dir, exist_ok=True)
                        temp_tif = _create_image_template(properties_data.mask_path, properties_data.tiles, properties_data.tile_id, properties_data.x_size, properties_data.y_size, params['dtype'], nodata, tile_dir)
                        write_idx = range(len(out_files))
                        compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
                        s3_out = None
                        if S3_PREFIX is not None:
                            s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{S3_PREFIX}/{properties_data.tile_id}' for _ in range(len(out_files))]
                        if params['dtype'] == 'int16':
                            smb.writeInt16Data(write_data, THREADS, GDAL_OPTS, [temp_tif for _ in range(len(out_files))], tile_dir, out_files, 
                                                             write_idx, 0, 0, properties_data.x_size, properties_data.y_size, int(nodata), compress_cmd, s3_out)
                        elif params['dtype'] == 'uint8':
                            smb.writeByteData(write_data, THREADS, GDAL_OPTS, [temp_tif for _ in range(len(out_files))], tile_dir, out_files, 
                                                             write_idx, 0, 0, properties_data.x_size, properties_data.y_size, int(nodata), compress_cmd, s3_out)
                        else:
                            assert(False, 'Not available save data type')
                            
                    del write_data_t
                    del write_data
                    gc.collect()

                # except:
                #     trace = traceback.format_exc()
                #     print(f"Tile {tile_id}/Model {model.model_name} - Prediction error")
                #     print(trace)
                #     continue
#
if __name__ == "__main__":
    main()
