from predict_api import s3_setup, s3_list_files, print_catalog_statistics, \
    DataCatalog, DataLoader, RFClassifierProbs, PredictedProbs, TimeTracker
import os
import sys
import geopandas as gpd
import numpy as np
import treelite_runtime
import warnings
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="treelite_runtime")
#
# global constants
#
THREADS = 96
# COVS_PATH = '/mnt/slurm/jobs/ai4sh_pred/ai4sh_vrt.json'
# TILES_PATH = '/mnt/slurm/jobs/ai4sh_pred/eu_tiles_epsg.3035.gpkg'
# MODELS_PATH = '/mnt/slurm/jobs/ai4sh_pred'
COVS_PATH = '/mnt/barron/ai4sh/ai4sh_robert/prep/ai4sh_vrt.json'
TILES_PATH = '/mnt/barron/ai4sh/ai4sh_robert/prep/tiles/eu_tiles_epsg.3035.gpkg'
MODELS_PATH = '/mnt/barron/ai4sh/ai4sh_robert/models'
MASK_PATH = '/vsicurl/http://192.168.49.30:8333/ai4sh/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719.tif'
VALID_MASK_VALUE = 1
S3_PREFIX = '/tmp-eumap-ai4sh/v3'
GDAL_OPTS = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}
TILES_ID = [1094, 1095, 1017, 714, 759, 944, 975, 1082, 1095, 1163, 1194, 1248, 1341, ]
# register S3 aliases
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
        tiles_id = tiles['id'][start_tile:end_tile]
        server_name=sys.argv[3]
        base_dir = f'/mnt/{server_name}/ai4sh_pred/tmp'
    else:
        raise ValueError("Usage: run_models.py ([--check-otf | --check_tile1 | --check_tiles] | <start_tile> <end_tile> <server_name)>")
    #
    # load models
    
    # TODO get number of classes from the model!
    num_class = 185
    # soil types lgbm
    soil_types_lgbm_params = {
        'model':RFClassifierProbs(
            model_name='soil.types.lgbm',
            model_path=f'{MODELS_PATH}/compiled/PQ185_lgbm_multiclass.so',
            model_covs_path=f'{MODELS_PATH}/soil_types/PQ_features_20240725.txt',
            num_class=num_class,
            predict_fn=lambda predictor, data: predictor.predict(treelite_runtime.DMatrix(data, dtype='float32'))
        ),
        'scale':100
    }
    # soil types scikit-learn
    soil_types_sklearn_params = {
        'model':RFClassifierProbs(
            model_name='soil.types.sklearn',
            model_path=f'{MODELS_PATH}/compiled/PQ185_rf_multiclass.so',
            model_covs_path=f'{MODELS_PATH}/soil_types/PQ_features_20240725.txt',
            num_class=num_class,
            predict_fn=lambda predictor, data: predictor.predict(treelite_runtime.DMatrix(data, dtype='float32'))
        ),
        'scale':100
    }
    
    #
    # catalogs
    
    catalog = DataCatalog.read_catalog('ai4sh', COVS_PATH)
    soil_types_model_params = [soil_types_lgbm_params, soil_types_sklearn_params]
    soil_types_features = {f for params in soil_types_model_params for f in params['model'].model_features}
    soil_types_catalog = catalog.query('soil.types', ['2022'], soil_types_features)
    print_catalog_statistics(soil_types_catalog)
    
    if check_otf:
        return
    #
    # register s3 aliases
    s3_aliases = s3_setup(S3_PREFIX is not None, ACCESS_KEY, SECRET_KEY)
    
    #
    # prediction
    
    # get data loader
    soil_types_data = DataLoader(soil_types_catalog, tiles, MASK_PATH, VALID_MASK_VALUE)
    
    # get models
    soil_types_lgbm:RFClassifierProbs = soil_types_lgbm_params['model']
    soil_types_sklearn:RFClassifierProbs = soil_types_sklearn_params['model']
    
    # tile loop
    for tile_id in tiles_id:
        with TimeTracker(f"tile {tile_id}", True):
            
            # resume
            if len(s3_list_files(s3_aliases, S3_PREFIX, tile_id, 'soil.types_ai4sh.ensemble')) == num_class + 1:
                print(f"tile {tile_id} already computed, skipping...")
                continue
            
            # load features
            soil_types_data.load_tile_data(tile_id, THREADS, GDAL_OPTS)
            
            # predict
            with soil_types_lgbm.predict(soil_types_data) as pred1, soil_types_sklearn.predict(soil_types_data) as pred2:
                # ensemble
                pred_ensemble = PredictedProbs(soil_types_data, 'soil.types.ensemble', num_class)
                with TimeTracker(f'tile {tile_id}/model {pred_ensemble.model_name} - ensemble'):
                    pred_ensemble.predicted_probs[:] = np.round((pred1.predicted_probs[:] + pred2.predicted_probs[:]) / 2)
            
            # save hard classes
            pred_ensemble.save_class_layer(
                base_dir=base_dir,
                nodata=255,
                dtype='uint8',
                out_files_prefix=[f'soil.types_ai4sh.ensemble_c_30m'],
                out_files_suffix=['epsg.3035_v20240731'],
                s3_prefix=S3_PREFIX,
                s3_aliases=s3_aliases,
                gdal_opts=GDAL_OPTS,
                threads=THREADS
            )
            
            # save probabilities
            pred_ensemble.save_probs_layers(
                base_dir=base_dir,
                nodata=255,
                dtype='uint8',
                out_files_prefix=[f'soil.types_ai4sh.ensemble.cl{i:03}_p_30m' for i in range(num_class)],
                out_files_suffix=['epsg.3035_v20240731' for _ in range(num_class)],
                s3_prefix=S3_PREFIX,
                s3_aliases=s3_aliases,
                gdal_opts=GDAL_OPTS,
                threads=THREADS
            )
            
            # free memory
            del pred_ensemble
#
if __name__ == "__main__":
    main()
