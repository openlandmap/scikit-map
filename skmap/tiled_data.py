from typing import Dict
from osgeo import gdal
import numpy as np
import subprocess, os, re
import skmap_bindings as sb
from skmap.misc import TimeTracker, ttprint, sb_arr, sb_vec
from concurrent.futures import ProcessPoolExecutor
from skmap.catalog import DataCatalog, run_whales
import warnings, random
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
n_threads = os.cpu_count()
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_self.n_threads'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_self.n_threads'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_self.n_threads"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_self.n_threads"] = f'{n_threads}' 
os.environ["MKL_NUM_self.n_threads"] = f'{n_threads}'
os.environ["VECLIB_MAXIMUM_self.n_threads"] = f'{n_threads}'
os.environ["OMP_DYNAMIC"] = f'TRUE'


mask_aggregation_bash_script = '''#!/bin/bash
    if [ -z "$1" ]; then
        echo "Usage: $0 <input_tif_file>"
        exit 1
    fi
    tif_file="$1"
    basename_tif=$(basename "$tif_file" .tif)
    output_file="comp_${basename_tif}.tif"

    if [ -f "$output_file" ]; then
        rm "$output_file"
    fi
    xmin=$(gdalinfo $tif_file | grep "Upper Left" | awk -F'[(), ]+' '{print $3}')
    ymax=$(gdalinfo $tif_file | grep "Upper Left" | awk -F'[(), ]+' '{print $4}')
    xmax=$(gdalinfo $tif_file | grep "Lower Right" | awk -F'[(), ]+' '{print $3}')
    ymin=$(gdalinfo $tif_file | grep "Lower Right" | awk -F'[(), ]+' '{print $4}')
    tmp_out=$(gdalinfo "$tif_file" | grep "Pixel Size" | awk -F'[(),]' '{print $2, $3}')
    read -r pixel_size_x pixel_size_y <<< "$tmp_out"
    pixel_size_y=$(awk "BEGIN {print -1 * $pixel_size_y}")
    pixel_size_x_new=$(awk "BEGIN {print $pixel_size_x * <AGG_FACTOR>}")
    pixel_size_y_new=$(awk "BEGIN {print $pixel_size_y * <AGG_FACTOR>}")
    xmin_new=$(awk "BEGIN {print $xmin + 2 * $pixel_size_x}")
    ymin_new=$(awk "BEGIN {print $ymin + 2 * $pixel_size_y}")
    xmax_new=$(awk "BEGIN {print $xmax - 2 * $pixel_size_x}")
    ymax_new=$(awk "BEGIN {print $ymax - 2 * $pixel_size_y}")
    gdalwarp -q -te "$xmin_new" "$ymin_new" "$xmax_new" "$ymax_new" -tr "$pixel_size_x_new" "$pixel_size_y_new" -r max "$tif_file" "$output_file"
    '''

def warp_tile(tile_file, mosaic_paths, n_pix, resample):
    warp_data = sb_vec(n_pix,)
    try:
        sb.warpTile(warp_data, 1, {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}, tile_file, mosaic_paths, resample)
    except:
        sb.fillArray(warp_data, 1, 0.0)
        print(f"Mosaic {mosaic_paths} has no data in {tile_file}, filling with 0.0")
    return warp_data

def s3_list_files(s3_aliases, s3_prefix, tile_id, file_pattern=None):
    if len(s3_aliases) == 0: return []
    bash_cmd = f"mc find {s3_aliases[0]}/{s3_prefix}/{tile_id}"
    process = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    # stderr = stderr.decode('utf-8')
    # assert stderr == '', f"Error listing S3 `{s3_aliases[0]}{s3_prefix}/{tile_id}`. \nError: {stderr}"
    stdout = stdout.decode('utf-8')
    lines = stdout.splitlines()
    if file_pattern is not None:
        pattern = re.compile(file_pattern)
        lines = [line for line in lines if pattern.search(line)]
    return lines
#
def s3_setup(access_key, secret_key, gaia_addrs):
    s3_aliases = []
    s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
    commands = [
        f'mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
        for i, addr in enumerate(gaia_addrs)
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
    return s3_aliases
#


class TiledData():
    def __init__(self,
                 n_layers:int = None,
                 n_pixels:int = None,
                 tile_id:str = None,
                 n_threads:int = os.cpu_count()) -> None:
        self.tile_id = tile_id
        self.n_threads = n_threads
        if (n_layers != None) & (n_pixels != None):
            self.array = sb_arr(n_layers, n_pixels)
        else:
            self.array = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.array = None
    
    def convert_nan_to_value(self, value):
        sb.maskNan(self.array, self.n_threads, range(self.array.shape[0]), value)
        
    def convert_nan_to_median(self):
        medians = sb_arr(self.array.shape[0],1)
        sb.computePercentiles(self.array, self.n_threads, range(self.array.shape[1]), medians, [0], [50.])
        nan_indices = np.argwhere(np.isnan(medians[:,0])).flatten()
        if len(nan_indices) > 0:
            for nan_idx in nan_indices:
                print(f"scikit-map ERROR 101: index {nan_idx} has all NaN for tile {self.tile_id}")
            # raise Exception("scikit-map ERROR 101")
        sb.maskNanRows(self.array, self.n_threads, range(self.array.shape[0]), medians)
    

class TiledDataLoader(TiledData):
    def __init__(self,
                 catalog:DataCatalog, 
                 mask_template_path:str,
                 spatial_aggregation:bool = None,
                 resampling_strategy:str = "GRA_NearestNeighbour",
                 gdal_opts:Dict[str,str] = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}, 
                 n_threads:int = os.cpu_count(),
                 verbose:bool = True) -> None:
        self.catalog = catalog
        self.mask_template_path = mask_template_path
        self.spatial_aggregation = spatial_aggregation
        self.resampling_strategy = resampling_strategy
        self.gdal_opts = gdal_opts
        self.n_threads = n_threads
        self.executor = ProcessPoolExecutor(max_workers=self.n_threads)
        self.mask_path = None
        self.array = None
        self.array_valid = None
        self.tile_id = None
        self.x_off = None
        self.y_off = None
        self.x_size = None
        self.y_size = None
        self.n_pixels = None
        self.mask = None
        self.verbose = verbose
        self.pixels_valid_idx = None
        self.n_pixels_valid = None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.mask_path) & (self.spatial_aggregation != None):
            os.remove(self.mask_path)
            ttprint(f"Temporary mask data {self.mask_path} has been deleted.")
    
    def load_tile_data(self, tile_id):
        self.tile_id = tile_id
        self.mask_path = self.mask_template_path.format(tile_id=tile_id)
        # @FIXME: this only work with our setting of Landsat data
        if self.spatial_aggregation:
            if self.verbose:
                warnings.warn("Spatial aggregation only works with 4004x4004 maks tiles")
            assert 4000%self.spatial_aggregation == 0, "Aggregation factor must be a integer dividend of 4000"
            mask_aggregation_bash_script_parsed = mask_aggregation_bash_script.replace('<AGG_FACTOR>', str(self.spatial_aggregation))
            with open('mask_aggregation_bash_script.sh', 'w') as file:
                file.write(mask_aggregation_bash_script_parsed)
            subprocess.run(['chmod', '+x', 'mask_aggregation_bash_script.sh'])
            input_file = self.mask_path
            subprocess.run(['./mask_aggregation_bash_script.sh', input_file])
            self.mask_path = f"comp_{input_file.split('/')[-1].split('.')[0]}.tif"
        # check for valid pixels
        gdal_img = gdal.Open(self.mask_path)
        self.x_size = gdal_img.RasterXSize
        self.y_size = gdal_img.RasterYSize
        gdal_img = None
        self.x_off, self.y_off = (0, 0)
        self.n_pixels = self.x_size * self.y_size
        # prepare mask
        with TimeTracker(f"   Prepare mask for tile {self.tile_id}", False):
            self.mask = np.zeros((1, self.n_pixels), dtype=np.float32)
            sb.readData(self.mask, 1, [self.mask_path], [0], self.x_off, self.y_off, self.x_size, self.y_size, [1], self.gdal_opts)
            self.pixels_valid_idx = np.arange(0, self.n_pixels)[self.mask[0, :].astype(int).astype(bool)]
            self.n_pixels_valid = int(np.sum(self.mask))
            if self.n_pixels_valid == 0:
                return self
        # read rasters
        self.array = sb_arr(self.catalog.data_size, self.n_pixels)
        with TimeTracker(f"   Read rasters and compute whales for tile {self.tile_id}", False):
            paths, paths_idx, _ = self.catalog.get_paths()
            tile_paths, tile_idxs, mosaic_paths, mosaic_idxs = [], [], [], []
            for path, idx in zip(paths, paths_idx):
                if '{tile_id}' in path:
                    tile_paths += [path.format(tile_id=self.tile_id)]
                    tile_idxs += [idx]
                else:
                    mosaic_paths += [path]
                    mosaic_idxs += [idx]
            # Reading resampled data
            if mosaic_paths:
                # @FIXME we could copy locally the mask and then delete to reduce number of requests
                tile_template_paths = [self.mask_path for i in range(len(mosaic_paths))]
                futures = [self.executor.submit(warp_tile, tile_template_paths[i], mosaic_paths[i], self.n_pixels, self.resampling_strategy)
                    for i in range(len(mosaic_paths))]
                for i, future in enumerate(futures):
                    self.array[mosaic_idxs[i], :] = future.result()

            # Reading tiled data
            if tile_paths:
                if self.spatial_aggregation:
                    tmp_data = sb_arr(len(tile_paths), 4000 * 4000)
                    sb.readData(tmp_data, self.n_threads, tile_paths, range(len(tile_paths)), 2, 2, 4000, 4000, [1], self.gdal_opts, None, np.nan)
                    arr_reshaped = tmp_data.reshape(len(tile_paths), 4000, 4000)
                    arr_aggregated = arr_reshaped.reshape(len(tile_paths), self.y_size, self.spatial_aggregation, self.x_size, self.spatial_aggregation).mean(axis=(2, 4))
                    arr_final = arr_aggregated.reshape(len(tile_paths), self.x_size * self.y_size)
                    self.array[tile_idxs,:] = arr_final[:,:]
                else:
                    sb.readData(self.array, self.n_threads, tile_paths, tile_idxs, self.x_off, self.y_off, self.x_size, 
                                self.y_size, [1], self.gdal_opts, None, np.nan)
            # Go whales, go!!
            run_whales(self.catalog, self.array, self.n_threads)
        return self
                    
    def convert_nan_to_median(self):
        medians = sb_arr(self.array.shape[0],1)
        sb.computePercentiles(self.array, self.n_threads, range(self.array.shape[1]), medians, [0], [50.])
        nan_indices = np.argwhere(np.isnan(medians[:,0])).flatten()
        if len(nan_indices) > 0:
            for nan_idx in nan_indices:
                group_name, feature_name = self.catalog.find_group_and_feature_by_index(nan_idx)
                print(f"scikit-map ERROR 101: index {nan_idx} corresponding to group {group_name} and feature {feature_name} has all NaN for tile {self.tile_id}")
            # raise Exception("scikit-map ERROR 101")
        sb.maskNanRows(self.array, self.n_threads, range(self.array.shape[0]), medians)
    
    def filter_valid_pixels(self):
        self.array_valid = sb_arr(self.catalog.data_size, self.n_pixels_valid)
        sb.selArrayCols(self.array, self.n_threads, self.array_valid, self.pixels_valid_idx)
    
    def expand_valid_pixels(self, array_valid, array_expanded):
        sb.expandArrayCols(array_valid, self.n_threads, array_expanded, self.pixels_valid_idx)
    
    def get_pixels_valid_idx(self, n_groups):
        return np.concatenate([self.pixels_valid_idx + self.n_pixels * i for i in range(n_groups)]).tolist()
    
    def fill_otf_constant(self, otf_name, otf_const):
        otf_idx = self.catalog.get_otf_idx()
        assert(otf_name in otf_idx)
        otf_name_idx = otf_idx[otf_name]
        self.array[otf_name_idx] = otf_const

        
class TiledDataExporter(TiledData):
    def __init__(self,
                 n_pixels:int = None,
                 tile_id:str = None,
                 mode:str = None,
                 spatial_res:str = None,
                 s3_params = None,
                 verbose = False,
                 years = None,
                 depths = None,
                 quantiles = None,
                 n_threads:int = os.cpu_count()) -> None:
        if s3_params:
            self.s3_aliases = s3_setup(s3_params['s3_access_key'],
                                       s3_params['s3_secret_key'],
                                       s3_params['s3_addresses'])
            self.s3_prefix = s3_params['s3_prefix']
        else:
            self.s3_aliases = None
            self.s3_prefix = None
        self.spatial_res = spatial_res
        self.n_pixels = n_pixels
        self.mode = mode
        self.verbose = verbose
        self.tile_id = tile_id
        self.n_threads = n_threads
        self.years = years
        self.depths = depths
        self.quantiles = quantiles
        if self.mode == 'depths_years_quantiles_textures':
            assert (years != None) & (depths != None) & (quantiles != None), "Need to provide years, depths, quantiles"
        elif self.mode == 'depths_years_quantiles':
            assert (years != None) & (depths != None) & (quantiles != None), "Need to provide years, depths, quantiles"
        elif self.mode == 'static_depths_quantiles':
            assert (depths != None) & (quantiles != None), "Need to provide depths, quantiles"
        elif self.mode == 'static_quantiles':
            assert (quantiles != None), "Need to provide quantiles"
        elif self.mode == 'depths_years':
            assert (years != None) & (depths != None), "Need to provide years, depths"
        elif self.mode == 'years':
            assert (years != None), "Need to provide years"
        elif self.mode == 'static':
            pass
        else:
            raise Exception("Available modes: depths_years_quantiles_textures, depths_years_quantiles, depths_years, years, static_depths_quantiles, static_quantiles, static")
        self.n_layers = len(self._get_out_names("",""))
        if (self.n_layers != None) & (self.n_pixels != None):
            self.array = sb_arr(self.n_layers, n_pixels)
        else:
            self.array = None
                
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.array = None
    
    def _get_out_names(self, prefix, sufix, time_frame = None):
        if self.mode == 'depths_years_quantiles_textures':
            return self._get_out_names_depths_years_quantiles_textures(prefix, sufix)
        if self.mode == 'depths_years_quantiles':
            return self._get_out_names_depths_years_quantiles(prefix, sufix)
        elif self.mode == 'depths_years':
            return self._get_out_names_depths_years(prefix, sufix)
        elif self.mode == 'years':
            return self._get_out_names_years(prefix, sufix)
        else:
            raise Exception("Available modes: depths_years_quantiles_textures, depths_years_quantiles, depths_years, years")
    
    def _get_out_names_years(self, prefix, sufix):
        out_files = []
        for y in self.years:
            out_files.append(f"{prefix}_m_{self.spatial_res}_s_{y}0101_{y}1231_{sufix}")
        return out_files
    
    def _get_out_names_static(self, prefix, sufix, time_frame):
        out_files = [out_files.append(f"{prefix}_m_{self.spatial_res}_s_{time_frame}_{sufix}")]
        return out_files
    
    def _get_out_names_static_quantiles(self, prefix, sufix, time_frame):
        out_files = []
        out_files.append(f"{prefix}_m_{self.spatial_res}_s_{time_frame}_{sufix}")
        for q in self.quantiles:
            formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
            out_files.append(f"{prefix}_{formatted_p}_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{time_frame}_{sufix}")
        return out_files
    
    def _get_out_names_depths_years(self, prefix, sufix):
        out_files = []
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                out_files.append(f"{prefix}_m_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{self.years[y]}0101_{self.years[y+1]}1231_{sufix}")
        return out_files
    
    def _get_out_names_static_depths_quantiles(self, prefix, sufix, time_frame):
        out_files = []
        for d in range(len(self.depths) - 1):
            out_files.append(f"{prefix}_m_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{time_frame}_{sufix}")
            for q in self.quantiles:
                formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                out_files.append(f"{prefix}_{formatted_p}_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{time_frame}_{sufix}")
        return out_files
    
    def _get_out_names_depths_years_quantiles(self, prefix, sufix):
        out_files = []
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                out_files.append(f"{prefix}_m_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{self.years[y]}0101_{self.years[y+1]}1231_{sufix}")
                for q in self.quantiles:
                    formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                    out_files.append(f"{prefix}_{formatted_p}_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{self.years[y]}0101_{self.years[y+1]}1231_{sufix}")
        return out_files
    
    def _get_out_names_depths_years_quantiles_textures(self, prefixes, sufix):
        if prefixes == "":
            prefixes = ["","",""]
        # order: prefixes = [prefix_caly, prefix_sand, prefix_silt]
        out_files = []
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                for prefix in prefixes :
                    out_files.append(f"{prefix}_m_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{self.years[y]}0101_{self.years[y+1]}1231_{sufix}")
                    for q in self.quantiles:
                        formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                        out_files.append(f"{prefix}_{formatted_p}_{self.spatial_res}_b{self.depths[d]}cm..{self.depths[d+1]}cm_{self.years[y]}0101_{self.years[y+1]}1231_{sufix}")
        return out_files
                  
    
    def check_all_exported(self, prefix, sufix):
        assert (self.s3_aliases != None) & (self.s3_prefix != None), "The check requires that S3 is properly set"
        out_files = self._get_out_names(prefix, sufix)
        files_in_s3 = s3_list_files(self.s3_aliases, self.s3_prefix, self.tile_id)
        basename_files_in_s3 = [os.path.basename(f) for f in files_in_s3]
        flag = True
        for s in out_files:
            if f"{s}.tif" not in basename_files_in_s3:
                flag = False
                if self.verbose:
                    print(f"Missing file {s}")
        return flag

        
    def derive_block_quantiles_and_mean(self, depths_trees_pred, expm1):
        assert self.mode == 'depths_years_quantiles', "Mode must be 'depths_years_quantiles'"
        self.n_pixels = int(depths_trees_pred[0].array.shape[1]/len(self.years))
        self.array = sb_arr(self.n_layers, self.n_pixels)
        array_t = sb_arr(self.n_pixels, self.n_layers)
        n_trees = depths_trees_pred[0].array.shape[0]
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                trees_avg = sb_arr(n_trees, self.n_pixels)
                sb.blocksAverage(trees_avg, self.n_threads,
                                 depths_trees_pred[d].array, depths_trees_pred[d+1].array, self.n_pixels, y)
                trees_avg_t = sb_arr(self.n_pixels, n_trees)
                prop_mean = sb_vec(self.n_pixels)
                sb.transposeArray(trees_avg, self.n_threads, trees_avg_t)
                sb.nanMean(trees_avg_t, self.n_threads, prop_mean)
                if expm1:
                    np.expm1(prop_mean, out=prop_mean)
                    np.expm1(trees_avg_t, out=trees_avg_t)
                percentiles = [q*100. for q in self.quantiles]
                offset_prop = d * ((len(self.years) - 1)  * (len(self.quantiles) + 1)) + y * (len(self.quantiles) + 1)
                sb.computePercentiles(trees_avg_t, self.n_threads, range(trees_avg_t.shape[1]), array_t,
                                      range(offset_prop+1,offset_prop+1+len(percentiles)), percentiles)
                array_t[:,offset_prop] = prop_mean
        sb.transposeArray(array_t, self.n_threads, self.array)
            
      
    def derive_static_depths_quantiles_and_mean(self, depths_trees_pred, expm1):
        assert self.mode == 'static_depths_quantiles', "Mode must be static_depths_quantiles"
        self.n_pixels = int(depths_trees_pred[0].array.shape[1])
        self.array = sb_arr(self.n_layers, self.n_pixels)
        array_t = sb_arr(self.n_pixels, self.n_layers)
        n_trees = depths_trees_pred[0].array.shape[0]
        for d in range(len(self.depths) - 1):
            trees_avg = sb_arr(n_trees, self.n_pixels)
            sb.elementwiseAverage(trees_avg, self.n_threads,
                                  depths_trees_pred[d].array, depths_trees_pred[d+1].array)
            trees_avg_t = sb_arr(self.n_pixels, n_trees)
            prop_mean = sb_vec(self.n_pixels)
            sb.transposeArray(trees_avg, self.n_threads, trees_avg_t)
            sb.nanMean(trees_avg_t, self.n_threads, prop_mean)
            if expm1:
                np.expm1(prop_mean, out=prop_mean)
                np.expm1(trees_avg_t, out=trees_avg_t)
            percentiles = [q*100. for q in self.quantiles]
            offset_prop = d * ((len(self.quantiles) + 1))
            sb.computePercentiles(trees_avg_t, self.n_threads, range(trees_avg_t.shape[1]), array_t,
                                  range(offset_prop+1,offset_prop+1+len(percentiles)), percentiles)
            array_t[:,offset_prop] = prop_mean
        sb.transposeArray(array_t, self.n_threads, self.array)
            
        
    def derive_block_quantiles_and_mean_textures(self, pred_depths_texture1, pred_depths_texture2, k=1., a=100.):
        assert self.mode == 'depths_years_quantiles_textures', "Mode must be 'depths_years_quantiles'"
        self.n_pixels = int(pred_depths_texture1[0].array.shape[1]/len(self.years))
        n_quant = len(self.quantiles)
        self.array = sb_arr(self.n_layers, self.n_pixels)
        array_t = sb_arr(self.n_pixels, self.n_layers)
        n_trees = pred_depths_texture1[0].array.shape[0]
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                offset_caly = d * (len(self.years) - 1) * 3 * (n_quant + 1) + y * 3 * (n_quant + 1)
                offset_sand = d * (len(self.years) - 1) * 3 * (n_quant + 1) + y * 3 * (n_quant + 1) + (n_quant + 1)
                offset_silt = d * (len(self.years) - 1) * 3 * (n_quant + 1) + y * 3 * (n_quant + 1) + 2 * (n_quant + 1)
                trees_avg_texture1 = sb_arr(n_trees, self.n_pixels)
                trees_avg_texture2 = sb_arr(n_trees, self.n_pixels)
                sb.blocksAverage(trees_avg_texture1, self.n_threads,
                                 pred_depths_texture1[d].array, pred_depths_texture1[d+1].array, self.n_pixels, y)
                sb.blocksAverage(trees_avg_texture2, self.n_threads,
                                 pred_depths_texture2[d].array, pred_depths_texture2[d+1].array, self.n_pixels, y)
                trees_avg_texture1_t = sb_arr(self.n_pixels, n_trees)
                trees_avg_texture2_t = sb_arr(self.n_pixels, n_trees)
                mean_texture1 = sb_arr(self.n_pixels, 1)
                mean_texture2 = sb_arr(self.n_pixels, 1)
                sb.transposeArray(trees_avg_texture1, self.n_threads, trees_avg_texture1_t)
                sb.transposeArray(trees_avg_texture2, self.n_threads, trees_avg_texture2_t)
                sb.nanMean(trees_avg_texture1_t, self.n_threads, mean_texture1)
                sb.nanMean(trees_avg_texture2_t, self.n_threads, mean_texture2)
                clay_trees = sb_arr(self.n_pixels, n_trees)
                sand_trees = sb_arr(self.n_pixels, n_trees)
                silt_trees = sb_arr(self.n_pixels, n_trees)
                clay_mean = sb_arr(self.n_pixels, 1)
                sand_mean = sb_arr(self.n_pixels, 1)
                silt_mean = sb_arr(self.n_pixels, 1)
                
                sb.texturesBwTransform(trees_avg_texture1_t, self.n_threads, trees_avg_texture2_t, k, a, sand_trees, silt_trees, clay_trees)
                sb.texturesBwTransform(mean_texture1, self.n_threads, mean_texture2, k, a, sand_mean, silt_mean, clay_mean)
                
                percentiles = [q*100. for q in self.quantiles]
                sb.computePercentiles(clay_trees, self.n_threads, range(clay_trees.shape[1]), array_t,
                                      range(offset_caly+1,offset_caly+1+len(percentiles)), percentiles)
                sb.computePercentiles(sand_trees, self.n_threads, range(sand_trees.shape[1]), array_t,
                                      range(offset_sand+1,offset_sand+1+len(percentiles)), percentiles)
                sb.computePercentiles(silt_trees, self.n_threads, range(silt_trees.shape[1]), array_t,
                                      range(offset_silt+1,offset_silt+1+len(percentiles)), percentiles)
                array_t[:,offset_caly] = clay_mean[:,0]
                array_t[:,offset_sand] = sand_mean[:,0]
                array_t[:,offset_silt] = silt_mean[:,0]
        sb.transposeArray(array_t, self.n_threads, self.array)
        
    def derive_block_mean(self, depths_pred, expm1):
        assert self.mode == 'depths_years', "Mode must be 'depths_years'"
        self.n_pixels = int(depths_pred[0].array.shape[1]/len(self.years))
        self.array = sb_arr(self.n_layers, self.n_pixels)
        for d in range(len(self.depths) - 1):
            for y in range(len(self.years) - 1):
                offset_prop = d * (len(self.years) - 1)  + y
                sb.blocksAverageVecs(self.array, self.n_threads,
                                 depths_pred[d].array, depths_pred[d+1].array, self.n_pixels, y, offset_prop)
        if expm1:
            np.expm1(self.array, out=self.array)
        
    def export_files(self, prefix, sufix, nodata, 
                     template_file, save_type, valid_idx,
                     write_folder='.',
                     scaling = 1,
                     gdal_opts:Dict[str,str] = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'},):
        out_files = self._get_out_names(prefix, sufix)
        n_files = len(out_files)
        gdal_img = gdal.Open(template_file)
        x_size = gdal_img.RasterXSize
        y_size = gdal_img.RasterYSize
        n_pixels = x_size * y_size
        write_data = sb_arr(n_files, n_pixels)
        write_data_t = sb_arr(n_pixels, n_files)
        out_data_t = sb_arr(self.array.shape[1], self.array.shape[0])
        sb.transposeArray(self.array, self.n_threads, out_data_t)
        offset = 0.5 if save_type in {'byte', 'uint16', 'int16', 'uint32', 'int32'} else 0.0
        sb.scaleAndOffset(out_data_t, self.n_threads, offset, scaling)
        sb.fillArray(write_data_t, self.n_threads, nodata)
        sb.expandArrayRows(out_data_t, self.n_threads, write_data_t, valid_idx)
        sb.transposeArray(write_data_t, self.n_threads, write_data)
        tile_dir = write_folder + f'/{self.tile_id}'
        os.makedirs(write_folder, exist_ok=True)
        os.makedirs(tile_dir, exist_ok=True)
        compress_cmd = f"gdal_translate -a_nodata {nodata} -a_scale {1./scaling} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
        s3_out = ([f'{random.choice(self.s3_aliases)}/{self.s3_prefix}/{self.tile_id}' for _ in range(len(out_files))])
        sb.writeData(write_data, n_threads, gdal_opts, [template_file for _ in range(n_files)], tile_dir, out_files,
               range(n_files), 0, 0, x_size, y_size, nodata,
               save_type, compress_cmd, s3_out)
        if s3_out:
            ttprint(f'Export complete, check mc ls {s3_out[0]}/{out_files[0]}')
        