from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.windows import Window, bounds

from skmap.raster import read_rasters, save_rasters
from skmap.mapper import SpaceTimeOverlay
from skmap.misc import ttprint

DEFAULT = {
    'ndvi_rasters': [
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2014.12.02..2015.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2015.03.21..2015.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2015.06.25..2015.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2015.09.13..2015.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2015.12.02..2016.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2016.03.21..2016.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2016.06.25..2016.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2016.09.13..2016.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2016.12.02..2017.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2017.03.21..2017.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2017.06.25..2017.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2017.09.13..2017.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2017.12.02..2018.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2018.03.21..2018.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2018.06.25..2018.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2018.09.13..2018.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2018.12.02..2019.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2019.03.21..2019.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2019.06.25..2019.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2019.09.13..2019.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2019.12.02..2020.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2020.03.21..2020.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2020.06.25..2020.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_2020.09.13..2020.12.01_eumap_epsg3035_v1.1.tif'
    ],
    'qa_ndvi_rasters': [
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2014.12.02..2015.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2015.03.21..2015.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2015.06.25..2015.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2015.09.13..2015.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2015.12.02..2016.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2016.03.21..2016.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2016.06.25..2016.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2016.09.13..2016.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2016.12.02..2017.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2017.03.21..2017.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2017.06.25..2017.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2017.09.13..2017.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2017.12.02..2018.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2018.03.21..2018.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2018.06.25..2018.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2018.09.13..2018.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2018.12.02..2019.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2019.03.21..2019.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2019.06.25..2019.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2019.09.13..2019.12.01_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2019.12.02..2020.03.20_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2020.03.21..2020.06.24_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2020.06.25..2020.09.12_eumap_epsg3035_v1.1.tif',
        'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_qa_landsat.glad.ard_f_30m_0..0cm_2020.09.13..2020.12.01_eumap_epsg3035_v1.1.tif'
    ],
    'window': Window(col_off=104020, row_off=74740, width=256, height=256),
    'n_samples': 256,
    'start_year': 2015,
    'end_year': 2020,
    'lc_raster': 'https://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_landcover.hcl_lucas.corine.eml_f_30m_0..0cm_{year}_eumap_epsg3035_v0.2.tif',
    'lc_label_remap': {
       '1': 'Continuous urban fabric',
        '2': 'Discontinuous urban fabric',
        '3': 'Industrial or commercial units',
        '4': 'Road and rail networks and associated land',
        '5': 'Port areas',
        '6': 'Airports',
        '7': 'Mineral extraction sites',
        '8': 'Dump sites',
        '9': 'Construction sites',
        '10': 'Green urban areas',
        '11': 'Sport and leisure facilities',
        '12': 'Non-irrigated arable land',
        '13': 'Permanently irrigated land',
        '14': 'Rice fields',
        '15': 'Vineyards',
        '16': 'Fruit trees and berry plantations',
        '17': 'Olive groves',
        '18': 'Pastures',
        '19': 'Annual crops associated with permanent crops',
        '20': 'Complex cultivation patterns',
        '21': 'Land principally occupied by agriculture, with significant areas of natural vegetation',
        '22': 'Agro-forestry areas',
        '23': 'Broad-leaved forest',
        '24': 'Coniferous forest',
        '25': 'Mixed forest',
        '26': 'Natural grasslands',
        '27': 'Moors and heathland',
        '28': 'Sclerophyllous vegetation',
        '29': 'Transitional woodland-shrub',
        '30': 'Beaches, dunes, sands',
        '31': 'Bare rocks',
        '32': 'Sparsely vegetated areas',
        '33': 'Burnt areas',
        '34': 'Glaciers and perpetual snow',
        '35': 'Inland marshes',
        '36': 'Peat bogs',
        '37': 'Salt marshes',
        '38': 'Salines',
        '39': 'Intertidal flats',
        '40': 'Water courses',
        '41': 'Water bodies',
        '42': 'Coastal lagoons',
        '43': 'Estuaries',
        '48': 'Sea and ocean'
    },
    'lc_code_remap': {
        '1': '111',
        '2': '112',
        '3': '121',
        '4': '122',
        '5': '123',
        '6': '124',
        '7': '131',
        '8': '132',
        '9': '133',
        '10': '141',
        '11': '142',
        '12': '211',
        '13': '212',
        '14': '213',
        '15': '221',
        '16': '222',
        '17': '223',
        '18': '231',
        '19': '241',
        '20': '242',
        '21': '243',
        '22': '244',
        '23': '311',
        '24': '312',
        '25': '313',
        '26': '321',
        '27': '322',
        '28': '323',
        '29': '324',
        '30': '331',
        '31': '332',
        '32': '333',
        '33': '334',
        '34': '335',
        '35': '411',
        '36': '412',
        '37': '421',
        '38': '422',
        '39': '423',
        '40': '511',
        '41': '512',
        '42': '521',
        '43': '522',
        '48': '523'
    },
    'ndvi_outdir': Path('skmap/datasets/data/ndvi'),
    'lc_outdir': Path('skmap/datasets/data/lc')
}

def _out_rasters(raster_files, basedir = 'data'):
    
    out_files = []

    for f in [ 
        Path(basedir).joinpath(Path(f).name.replace('lcv_','')
                              .replace('0..0cm','s')
                              .replace('..','_')
                              .replace('eumap','nl')
                              .replace('v1.1','v20230622')
                              .replace('epsg','epsg.')
                              .replace('glad.','')
                              .replace('ard','ard1')
                             ) 
        for f in raster_files 
    ]:
        dt1 = str(f).split('_')[5]
        dt2 = str(f).split('_')[6]
        out_files.append((str(f).replace(dt1,dt1.replace('.','')).replace(dt2,dt2.replace('.',''))))

    return out_files

def gen_dataset(start_year, end_year, ndvi_rasters, qa_ndvi_rasters, window, n_samples, 
                 lc_raster, lc_label_remap, lc_code_remap, ndvi_outdir, lc_outdir):
    
    ds = rasterio.open(ndvi_rasters[0])
    
    ttprint("Reading NDVI time series")

    ndvi_data, _ = read_rasters(raster_files=ndvi_rasters, spatial_win=window)
    ndvi_data = np.where(ndvi_data < 100, 100, ndvi_data) - 100

    qa_data, _ = read_rasters(raster_files=qa_ndvi_rasters, spatial_win=window)
    qa_mask = ~np.isnan(qa_data)

    ttprint("Saving NDVI time series")
    save_rasters(ndvi_rasters[0], _out_rasters(ndvi_rasters, ndvi_outdir.joinpath('filled')), ndvi_data,  spatial_win=window, nodata = 255)
    ndvi_data[qa_mask] = np.nan
    save_rasters(ndvi_rasters[0], _out_rasters(ndvi_rasters,  ndvi_outdir.joinpath('gappy')), ndvi_data,  spatial_win=window, nodata = 255)

    ttprint("Generating random LC samples")
    x_min, y_min, x_max, y_max = bounds(window, ds.transform)

    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    years = np.random.randint(start_year, end_year+1, n_samples)

    samples = gpd.GeoDataFrame(
        data={'date': pd.to_datetime(years, format='%Y')},
        geometry=gpd.GeoSeries(gpd.points_from_xy(x, y, crs=ds.crs))
    )

    ttprint("Running spacetime overlay")
    spt_overlay = SpaceTimeOverlay(samples, 'date', [ lc_raster ])
    samples_overlaid = spt_overlay.run()

    rename_col = {}
    rename_col[samples_overlaid.columns[-1]] = 'target'
    samples_overlaid = samples_overlaid.drop(columns=['overlay_id']) \
                    .rename(columns=rename_col)
    samples_overlaid['target'] = samples_overlaid['target'].astype('int')
    samples_overlaid['label'] = samples_overlaid['target'].astype('int').astype('str').replace(lc_label_remap)
    samples_overlaid['code'] = samples_overlaid['target'].astype('int').astype('str').replace(lc_code_remap)

    ttprint("Saving land-cover samples")
    lc_outdir.mkdir(parents=True, exist_ok=True)
    samples_overlaid[['date','label','code','target','geometry']].to_file(lc_outdir.joinpath('samples.gpkg'))

gen_dataset(**DEFAULT)