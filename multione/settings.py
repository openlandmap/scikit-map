
n_threads = 96

TMP_DIR = '/mnt/silva/tmp'

gaia_addrs = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
gaia_s3_params = {
        's3_addresses':gaia_addrs,
        's3_access_key':'iwum9G1fEQ920lYV4ol9',
        's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
        's3_prefix':'tmp-landsat-arco-v2',
    }

gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

gdal_co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']

bands_prefix = ['red_glad',
                'nir_glad',
                'blue_glad',
                'green_glad',
                'swir1_glad',
                'swir2_glad',
                'thermal_glad',
                'qa_mask']

doy_start = ['0101', '0117', '0202', '0218', '0305', '0321', '0406', '0422', '0508', '0524', '0609',
             '0625', '0711', '0727', '0812', '0828', '0913', '0929', '1015', '1031', '1116', '1202', '1218']

doy_end = ['0116', '0201', '0217', '0304', '0320', '0405', '0421', '0507', '0523', '0608', '0624',
           '0710', '0726', '0811', '0827', '0912', '0928', '1014', '1030', '1115', '1201', '1217', '1231']

month_start = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201']
month_end = ['0131', '0228', '0331', '0430', '0531', '0630', '0731', '0831', '0930', '1031', '1130', '1231']

# SWA time-series reconstruction parameters
att_env, att_seas, future_scaling = (20.0, 40.0, 0.1)

# MODIS NDVI filtering parameters
# diff_th, count_th = (3000, int(0.3*n_s))
# diff_th, count_th = (1000, 12*n_years)
#diff_th, count_th = (2500, 2*n_years)

resampling_strategy = "GRA_Bilinear"

x_off = y_off = 0  # GLAD and MODIS images are aligned to the top-left corner
x_size = y_size = 4004  # GLAD and MODIS images are 4004x4004 pixels
n_pix = x_size * y_size
n_imag_per_year = 23
n_imag_per_year_agg = 12
no_data = 0

landsat_file_ending = '_go_epsg.4326_v20240521.tif'