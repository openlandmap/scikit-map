#%%
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from typing import List
from munch import Munch
import rasterio as rio
import rasterio.warp
import rasterio.transform
import rasterio.windows
from rasterio.enums import Resampling

import numpy as np
from pathlib import Path
import pandas, geopandas
import boto3, botocore
import multiprocessing as mp
import time
import traceback
import concurrent.futures
import pygeos
import tqdm
import itertools as it

from .mosaic_helper import GridSystem, tprint, ttprint, ThreadGeneratorLazy, ProcessGeneratorLazy
from .speedups import distance_to_lines, cr2xy, cr2xy_v2, cr2xy_v4
from .speedups import orbit_average, mosaic_final_weight, mosaic_final_weight_v1, _update_mask_or
from . import _dist_data

fld = Path(__file__).resolve().parent

#%%
def _read_tiles_lazy(args, chunk_size):
    chunks = map(lambda x: it.islice(args, x, x+chunk_size), range(0, len(args), chunk_size))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = zip(args, 
                      it.chain.from_iterable(map(lambda x: executor.map(_read_tile_worker, x), 
                                                 chunks)))
        return results

def _read_tile_worker(args):
    mm, row, landmask = args

    infile = mm.get_input_file(row.fn)
    print(infile.name)

    with rio.open(infile) as src:
        #print(row.fn)
        vrt_options = { 'resampling': mm.resampling,
            'crs': mm.dst_crs,
            'transform': row.dst_transform,
            'height': row.dst_height,
            'width': row.dst_width,
            #'src_nodata': src.nodata,
            'add_alpha': False
            }
    
        with rio.vrt.WarpedVRT(src, **vrt_options) as vrt:
            # vrt = rio.vrt.WarpedVRT(src, **vrt_options)
            # add 10% of pixel around becouse of rounding errors
            round_err = 0.0 #self.dxy/4.0
            xl, yl, xu, yu = row.dst_bounds
            dst_window = vrt.window(xl-round_err, yl-round_err, xu+round_err, yu+round_err)
            dst_window = dst_window.round_offsets(pixel_precision=1).round_lengths(pixel_precision=1)

            #tprint('    read data and mask')
            sdata = vrt.read(1,window=dst_window)
            mask = vrt.read_masks(1, window=dst_window)!=0

    if landmask:            
        bounds = rasterio.windows.bounds(dst_window, row.dst_transform)
        with rio.open(mm.landmask) as src:
            sub_window_lc = rio.windows.from_bounds(*bounds, transform=src.transform)
            sub_window_lc = sub_window_lc.round_lengths(pixel_precision=1).round_offsets(pixel_precision=1)
            
            if abs(sub_window_lc.width - dst_window.width) == 1:
                sub_window_lc.width = dst_window.width
            if abs(sub_window_lc.height - dst_window.height) == 1:
                sub_window_lc.height = dst_window.height
            
            lc_mask = src.read_masks(1, window = sub_window_lc, boundless=True) != 0
            lc_data = src.read(1, window = sub_window_lc, boundless=True)
            mask = mask & lc_mask & (lc_data>0)

    return sdata, mask

#%%
class MosaicMaker():
    max_distance_from_orbit = 143000
    landmask = None

    def __init__(self, nworkers = 1, dst_crs: str = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs', 
            dxy: int = 30, align_xy: tuple = (900000, 5460010), 
            tmp_folder:str=None, bucket:str=None, debug:bool =False, tiles:list=None):
 
        self.tmp_folder = tmp_folder
        self.nworkers = nworkers
        self.debug = debug
        self.tiles=tiles
        
        if bucket is not None:
            self.file_mode='s3'
            self.bucket = bucket
        else:
            self.file_mode='fs'

        self.dst_crs = dst_crs
        self.dxy = dxy
        self.align_xy = align_xy

        self.input_files = []
        
    def align_transform(self, dst_transform):
        if self.align_xy is None:
            return dst_transform
        else: 
            axoff, ayoff = self.align_xy
            ddxy = dst_transform.a

            xoff = round((dst_transform.xoff - axoff) / ddxy) * ddxy + axoff
            yoff = round((dst_transform.yoff - ayoff) / ddxy) * ddxy + ayoff

            dst_transform = rio.transform.from_origin(
                xoff, yoff, ddxy, ddxy
            )
        return dst_transform
            
    def save_tif(self, data, out_file_name, out_profile, dst_type):
        if (self.file_mode=='fs') or self.debug:
            with rio.open(out_file_name,'w',**out_profile) as dst:
                dst.write(data.astype(dst_type), 1)        
        else:
            tmp_file_name = Path(self.tmp_folder)/(Path(out_file_name).as_posix().replace('/','-'))
            print(tmp_file_name)
            with rio.open(tmp_file_name,'w',**out_profile) as dst:
                if np.dtype(dst_type)!=data.dtype:
                    data = data.astype(dst_type)
                dst.write(data, 1)
            s3 = boto3.client('s3')
            s3.upload_file(tmp_file_name.as_posix(), self.bucket, Path(out_file_name).as_posix())
            tmp_file_name.unlink()

    def save_tif_init(self, out_file_name, out_profile):
        self.out_file_name = out_file_name
        if (self.file_mode=='fs') or self.debug:
            self.dst_file = rio.open(out_file_name,'w',**out_profile)
        else:
            self.tmp_file_name = Path(self.tmp_folder)/(Path(out_file_name).as_posix().replace('/','-'))
            print(self.tmp_file_name)
            self.dst_file = rio.open(self.tmp_file_name,'w',**out_profile)

    def save_tif_update(self, data, window):
        self.dst_file.write(data, 1, window=window)
    
    def save_tif_finish(self):
        self.dst_file.close()
        if (self.file_mode != 'fs') and (not self.debug):
            s3 = boto3.client('s3')
            tprint(f'Uploading to {self.out_file_name} ... ',end='')
            s3.upload_file(self.tmp_file_name.as_posix(), self.bucket, Path(self.out_file_name).as_posix())
            self.tmp_file_name.unlink()
            tprint(' Done.')


    def get_output_folder(self, output_folder):
        if self.file_mode=='fs':
            output_folder=Path(output_folder)
            Path.mkdir(output_folder, parents=True, exist_ok=True)            
        return output_folder

    def check_output_file(self, out_file_name):
        '''
        Checks if file exists
        '''
        if self.file_mode=='fs':
            return out_file_name.exists()
        else:            
            try:
                #self.s3bucket.Object(out_file_name.as_posix()).load()
                boto3.resource('s3').Object(self.bucket, out_file_name.as_posix()).load()
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
            else:
                return True

    def get_input_file(self, input_file):
        if self.file_mode=='fs':
            return input_file
        else:
            tmp_file_name = Path(self.tmp_folder)/input_file.replace('/','-')
            if not tmp_file_name.exists():
                s3bucket = boto3.resource('s3').Bucket(self.bucket)
                s3bucket.download_file(input_file, tmp_file_name.as_posix())    
            if tmp_file_name.as_posix() not in self.input_files:
                self.input_files.append(tmp_file_name.as_posix())   
            return tmp_file_name

    def get_all_orbits(self, input_folder:str):
        if self.file_mode=='fs':
            input_folder=Path(input_folder)            
            all_files = list(input_folder.glob(f'*.tif'))
        else:
            s3bucket = boto3.resource('s3').Bucket(self.bucket)
            all_files = (o.key for o in s3bucket.objects.filter(Prefix=input_folder))

        orbits = {}
        for f in filter(lambda x: Path(x).suffix == '.tif', all_files):
            #print(f)            
            tile, orbit = Path(f).with_suffix('').name.split('_')
            if self.tiles is None:
                orbits[orbit] = orbits.get(orbit,[]) + [f]
            elif tile in self.tiles:
                orbits[orbit] = orbits.get(orbit,[]) + [f]

        return orbits

    def get_final_orbits(self, input_folder):   
        '''
        Finds all orbits and files for making final mosaic

        param: input_folder: Folder with orbits

        returns: Dict of names and files of orbits
        '''     
        if self.file_mode=='fs': 
            input_folder = Path(input_folder)           
            all_files = list(input_folder.glob(f'*.tif'))            
        else:
            s3bucket = boto3.resource('s3').Bucket(self.bucket)
            all_files = (o.key for o in s3bucket.objects.filter(Prefix=input_folder))
           
        orbits = dict([(Path(f).with_suffix('').name, Path(f).as_posix()) for f in all_files])

        return orbits

    def get_bbox_nodata(self, filename):
        '''
        Finds bounding box, transform, width and height of tile in destination crs. Not depending on real data values.

        param: filename: Filename of tile image

        returns: dst_bbox: bounding box
        returns: dst_transform: transform
        returns: dst_height
        returns: dst_width
        '''
        infile = self.get_input_file(filename)
        with rio.open(infile) as src:
            width = src.profile['width']
            height = src.profile['height']
 
            dst_transform, dst_width, dst_height = rio.warp.calculate_default_transform(src.crs, self.dst_crs, 
                            width, height, *src.bounds, resolution=self.dxy)                
            
            dst_width=int(dst_width)
            dst_height=int(dst_height)

            # align transform to global 
            dst_transform = self.align_transform(dst_transform)

            dst_bbox = rio.transform.array_bounds(dst_height, dst_width, dst_transform) 

            return dst_bbox, dst_transform, dst_height, dst_width

    def get_bbox(self, filename):
        '''
        Finds bounding box, transform, width and height of tile in destination crs. Depends on real data values, i.e. ignores nodata value for calculating bounding box. Slow.

        param: filename: Filename of tile image

        returns: dst_bbox: bounding box
        returns: dst_transform: transform
        returns: dst_height
        returns: dst_width
        
        '''
        infile = self.get_input_file(filename)
        with rio.open(infile) as src:
            mask = src.read_masks(1)
            data_window = rio.windows.get_data_window(mask, nodata=0) # Window cropped to DATA 

            if (data_window.width==0) or (data_window.height==0):
                return None
            else:
                data_bounds = rio.windows.bounds(data_window, src.transform)
                try:
                    dst_transform, dst_width, dst_height = rio.warp.calculate_default_transform(src.crs, self.dst_crs, 
                                data_window.width, data_window.height, *data_bounds, resolution=self.dxy)                
                    
                    dst_width=int(dst_width)
                    dst_height=int(dst_height)

                    # align transform to global 
                    dst_transform = self.align_transform(dst_transform)

                    dst_bbox = rio.transform.array_bounds(dst_height, dst_width, dst_transform) 
                except:
                    return None
                    #dst_width = 0
                    #dst_height = 0
                    #dst_transform = None
                    #dst_bbox = None

                return dst_bbox, dst_transform, dst_height, dst_width

    def get_extent(self, input_file_list:list):
        ''' 
        Finds extent, transform and size of output mosaic 

        params: input_file_list: List of input files

        returns: df: DataFrame with all files and their transforms
        returns: dst_left: left boundary of mosaic
        returns: dst_top: top boundary of mosaic
        returns: dst_right: right boundary of transform
        returns: dst_bottom: bottom boundary of mosdsaic
        '''
        df = pandas.DataFrame({'fn':input_file_list})
        df['dst_transform'] = pandas.Series(dtype=object)
        df['dst_bounds'] = pandas.Series(dtype=object)        

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nworkers) as executor:  #ThreadPoolExecutor #ProcessPoolExecutor
            bboxes = executor.map(lambda x: self.get_bbox_nodata(x[1].fn), df.iterrows())                

        dst_left=None
        for (irow, row), bbox_result in zip(df.iterrows(), bboxes):
            # irow=0; row=df.loc[irow]
            
            if bbox_result is None:
                (dst_bbox, dst_transform, dst_height, dst_width) = None, None, 0, 0
            else:
                (dst_bbox, dst_transform, dst_height, dst_width) = bbox_result

                if dst_left is None:    # first bbox
                    dst_left, dst_bottom, dst_right, dst_top = dst_bbox
                else:
                    dst_left = min(dst_left,dst_bbox[0])
                    dst_bottom = min(dst_bottom, dst_bbox[1])
                    dst_right =  max(dst_right, dst_bbox[2])
                    dst_top = max(dst_top, dst_bbox[3])

            df.loc[irow, 'dst_width'] = dst_width
            df.loc[irow, 'dst_height'] = dst_height
            df.loc[irow, 'dst_transform'] = dst_transform
            df._set_value(irow, 'dst_bounds', dst_bbox)

        if dst_left is None:    #No data at all
            return None

        df.dst_width = df.dst_width.astype(int)
        df.dst_height = df.dst_height.astype(int)

        return df, dst_left, dst_top, dst_right, dst_bottom

    def read_tile(self, row, landmask=False):
        '''
        Reads data and mask of one tile into memory
        '''
        infile = self.get_input_file(row.fn)
        try:        
            with rio.open(infile) as src:            
                vrt_options = { 'resampling': self.resampling,
                    'crs': self.dst_crs,
                    'transform': row.dst_transform,
                    'height': row.dst_height,
                    'width': row.dst_width,
                    'src_nodata': src.nodata,
                    'add_alpha': False
                    }
            
                with rio.vrt.WarpedVRT(src, **vrt_options) as vrt:
                    round_err = 0.0 #self.dxy/4.0
                    xl, yl, xu, yu = row.dst_bounds
                    dst_window = vrt.window(xl-round_err, yl-round_err, xu+round_err, yu+round_err)
                    dst_window = dst_window.round_offsets(pixel_precision=1).round_lengths(pixel_precision=1)

                    #tprint('    read data and mask')
                    #tprint(' .. reading sdata')
                    sdata = vrt.read(1,window=dst_window)
                    #tprint(' .. reading mask')
                    mask = (sdata>0) & (vrt.read_masks(1, window=dst_window)!=0)

            if landmask:            
                bounds = rasterio.windows.bounds(dst_window, row.dst_transform)
                with rio.open(self.landmask) as src:
                    sub_window_lc = rio.windows.from_bounds(*bounds, transform=src.transform)
                    sub_window_lc = sub_window_lc.round_lengths(pixel_precision=1).round_offsets(pixel_precision=1)
                    
                    #if abs(sub_window_lc.width - dst_window.width) == 1:
                    #    sub_window_lc.width = dst_window.width
                    #if abs(sub_window_lc.height - dst_window.height) == 1:
                    #    sub_window_lc.height = dst_window.height
                    
                    #tprint(' .. reading lc mask')
                    lc_mask = src.read_masks(1, window = sub_window_lc, boundless=True, out_shape=sdata.shape, resampling=Resampling.nearest) != 0
                    #tprint(' .. reading lc data')
                    lc_data = src.read(1, window = sub_window_lc, boundless=True, out_shape=sdata.shape, resampling=Resampling.nearest)
                    #tprint(' .. masking lc')
                    mask = mask & lc_mask & (lc_data>0)
        except Exception as e:
            print(f'Error reading file {infile}')
            error_message = traceback.format_exc()
            print(error_message)
            return row, None, None

        return row, sdata, mask       

#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%  mozaici pojedinih orbita orbite %%%%%%%%%%%%%
    def make_mosaic_tiled_mp_worker(self, j, args):
        '''
        Worker procedure to make one subtile
        '''
        dst_nodata=args.dst_nodata
        ntiles_cols=args.ntiles_cols; ntiles_rows=args.ntiles_rows
        tiles_cols=args.tiles_cols; tiles_rows=args.tiles_rows
        dst_transform = args.dst_transform; landmask = args.landmask
        df = args.df

        tc = j%ntiles_cols
        tr = j//ntiles_cols

        dst_window = rio.windows.Window(col_off=tc*tiles_cols, row_off=tr*tiles_rows, width=tiles_cols, height=tiles_rows)
        # NAđem sve ulazne tileove koji se preklapaju sa radnim
        ind = df.window.apply(lambda x: rio.windows.intersect(x,dst_window))
        ntiles_input = ind.sum()
        #tprint(f'tile {j+1} has {ntiles_input} s2tiles')
        if ntiles_input==0:
            return None
        
        grid = GridSystem(int(dst_window.height), int(dst_window.width), rio.windows.transform(dst_window, dst_transform))
        #work_bounds = rio.windows.bounds(dst_window, dst_transform)

        # working array, 
        gdata = np.zeros((grid.nrows, grid.ncols), dtype='int32')
        # count array
        cdata = np.zeros((grid.nrows, grid.ncols), dtype='uint8')

        dff = df.loc[ind].copy()
        for ri, r in dff.iterrows():
            if r.dst_width==0:
                continue
            row, sdata, mask = self.read_tile(r, landmask)
            #i=i+1
            if sdata is None:
                continue
            dst_height, dst_width = mask.shape
            grid.set_subgrid(dst_height, dst_width, row.dst_transform)

            grid.set_subgrid_mask(mask)
            grid.add_to_subgrid_inc1_masked(gdata, cdata, sdata)
 
        gdata=orbit_average(gdata, cdata, dst_nodata)

        return j, dst_window, gdata

    def make_mosaic_tiled_mp(self, input_file_list, out_file_name, dst_type, dst_nodata, resampling_method, output_profile,
                    dst_bounds=None, max_cols=1024*10, max_rows=1024*10, landmask=False):
        '''
        Makes mosaic of one relative orbit using multiprocessing.

        param: input_file_list: List of all input tile images
        param: out_file_name: Name of output file
        param: dst_type: Type of data in output file
        param: dst_nodata: Nodata value in output file
        param: resampling_method: Method of resampling
        param: output_profile: Profile of output file
        param: dst_bounds: Not used, default is None
        param: max_cols: Maximum number of columns in one subtile, default is 10240
        param: max_rows: Maximum number of rows in one subtile, default is 10240
        param: landmask: If True then output data is masked with landmask, default is False
        '''
        dxy = self.dxy
        self.resampling = rio.enums.Resampling[resampling_method]
        #tprint(f'  get_extent')
        extent = self.get_extent(input_file_list)
        if extent is None:
            tprint(f'   NO EXTENT - SKIPPING !')
            return
        
        # Nađen transform i bounds svih tile-ova, i ukupni extent
        df, dst_left, dst_top, dst_right, dst_bottom = extent
        df = df.loc[~df.dst_bounds.isnull()] 
        
        # naštimama sve to na dx, dy
        dst_transform = self.align_transform(rasterio.transform.from_origin(dst_left, dst_top, dxy, dxy))
        dst_bounds = rasterio.transform.array_bounds((dst_top-dst_bottom-1)//dxy+1, (dst_right-dst_left-1)//dxy+1, dst_transform)
        df['window'] = [rio.windows.from_bounds(*r.dst_bounds, transform = dst_transform) for i,r in df.iterrows()]

        ncols = int((dst_bounds[2]-dst_bounds[0])//dxy)
        nrows = int((dst_bounds[3]-dst_bounds[1])//dxy)
        # Izračun broja radnih tile-ova
        ntiles_cols = int(np.ceil(ncols/max_cols))
        ntiles_rows = int(np.ceil(nrows/max_rows))
        # Broj rows i coils u jednom working tile-u
        tiles_cols = int(np.ceil(ncols/ntiles_cols))
        tiles_rows = int(np.ceil(nrows/ntiles_rows))

        ncols = tiles_cols*ntiles_cols
        nrows = tiles_rows*ntiles_rows

        output_profile.update({'width':ncols, 'height':nrows, 'transform': dst_transform})
        self.save_tif_init(out_file_name, output_profile)
        # Petlja po working tiles
        n=ntiles_cols*ntiles_rows
        pbar = tqdm.tqdm(total=n)
        mp_args_fixed = Munch(ntiles_cols=ntiles_cols, ntiles_rows=ntiles_rows, tiles_cols=tiles_cols, tiles_rows=tiles_rows,
            dst_transform=dst_transform, df=df, dst_nodata=dst_nodata, landmask=landmask)
        mp_args = ((j,mp_args_fixed) for j in range(n))
        for ret in ThreadGeneratorLazy(self.make_mosaic_tiled_mp_worker, mp_args, self.nworkers, self.nworkers):
            pbar.update(1)
            if ret is not None:
                j, dst_window, gdata=ret
                pbar.set_description(f'Window {j}')
                #tprint(f'  saving window {j+1}/{ntiles_cols*ntiles_rows} ... ',end='')            
                self.save_tif_update(gdata.astype(dst_type), dst_window)
                #tprint(' done')

        self.save_tif_finish()


    def mosaic_orbits_tiled(self, input_folder: str, output_folder:str, dst_nodata=2**15-1, dst_dtype='int16', 
        resampling_method='nearest', landmask=False):
        '''
        Main procedure for making mosaic of all relative orbits

        param: input_folder: Folder with all tiles
        param: outpout_folder: Folder where to write output orbits
        param: dst_nodata: nodata value for output files, default 2**15-1
        param: dst_dtype: type of data in output orbit mosaics, default is 'int16'
        param: resampling_methos: method of resampling, default is 'nearest'
        param: landmask: If true then output is masked with landmask, default is False
        '''
        # mosaic_name = 's2l2a_B04_2018_q2_P25'; preklop='cnt_min'; dst_dtype='uint8'; resampling_method='average'; dst_nodata=0

        out_profile = dict(driver='GTiff', crs=self.dst_crs, dtype=dst_dtype, count=1, nodata=dst_nodata, 
            compress='deflate', tiled=True, blockxsize=1024, blockysize=1024, bigtiff='YES',num_threads='all_cpus')

        tprint(f'mosaic: {input_folder}')
        orbits = self.get_all_orbits(input_folder)
        tprint(f'number of orbits: {len(orbits)}')

        output_folder = self.get_output_folder(output_folder)

        i=0
        pbar = tqdm.tqdm(total=len(orbits))
        for orbit, files in sorted(orbits.items(), key =lambda x: len(x[1])):
            pbar.update(1)
            i=i+1                    
           
            out_file_name = (Path(output_folder)/orbit).with_suffix('.tif')
            if self.check_output_file(out_file_name):
                print(f'Already exists: {out_file_name}')
                continue

            input_file_list = sorted(files, key = lambda f: Path(f).with_suffix('').name.split('_')[-2], reverse=True)

            #tprint(f'{i}/{len(orbits)} {out_file_name}, {len(input_file_list)} files')
            pbar.set_description(f'{i}/{len(orbits)} {out_file_name}, {len(input_file_list)} files')

            self.make_mosaic_tiled_mp(input_file_list=input_file_list, out_file_name=out_file_name,
                dst_type=dst_dtype, dst_nodata=dst_nodata, resampling_method=resampling_method,
                output_profile = out_profile, landmask=landmask)  
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def make_mosaic_final_tiled_worker(self, j, args):
        '''
        Worker procedure for final mosaic. Calculates one subtile.
        '''
        dst_nodata=args.dst_nodata
        ntiles_cols=args.ntiles_cols; ntiles_rows=args.ntiles_rows
        tiles_cols=args.tiles_cols; tiles_rows=args.tiles_rows
        dst_transform = args.dst_transform;
        orbits = args.orbits; preklop = args.preklop

        #tprint(f'    {j}/{ntiles_rows*ntiles_cols}')
        tc = j%ntiles_cols
        tr = j//ntiles_cols

        dst_window = rio.windows.Window(tc*tiles_cols, tr*tiles_rows, tiles_cols, tiles_rows)
        grid = GridSystem(int(dst_window.height), int(dst_window.width), rio.windows.transform(dst_window, dst_transform))
        gdata = np.full((grid.nrows, grid.ncols), dst_nodata, dtype='int16')
        gmask = np.zeros_like(gdata, dtype=bool)    # global mask (true==value is valid)
        lmask = gmask.copy()
        gmask_clip = gmask.copy()

        lrow = None
        n = len(orbits)
        for i,row in enumerate(orbits):
            #i=27; row=orbits[i]                                        

            infile = self.get_input_file(row['file'])
            with rio.open(infile) as src:
                # src=rio.open(infile)
                orbit_window = rio.windows.from_bounds(*src.bounds,dst_transform)
                if rio.windows.intersect(orbit_window, dst_window):
                    #print(f'tile {j+1}/{ntiles_rows*ntiles_cols}, orbit - {i}-{row["orbit"]}')
                    grid.set_subgrid(src.height, src.width, src.transform)
                    sub_window = rio.windows.from_bounds(*grid.get_subgrid_bounds(), transform=src.transform)
                    sub_window = sub_window.round_lengths(pixel_precision=1).round_offsets(pixel_precision=1)
                    sdata = src.read(1, window=sub_window)
                    mask = src.read_masks(1, window=sub_window)!=0

                    #TODO: if self.landmaskis not None: ...

                    grid.update_lmask_clip(mask, lmask, gmask, gmask_clip)

                    if not gmask_clip.any():       
                        gdata[lmask] = sdata[mask]
                    else:                                   
                        if preklop=='weight':
                            #tprint('gmask2lmask ...')
                            lmask_clip = grid.gmask2lmask(gmask_clip)
                            gdata_clip = gdata[gmask_clip] #gdata.flat[ginds_clip]        
                            sdata_clip = sdata[lmask_clip]

                            #tprint(f'clipping ... {sdata_clip.sum()} pixels')

                            # x,y coordinates of pixels in clip area
                            x, y = cr2xy_v4(gmask_clip, tuple(grid.transform)[:6])

                            #tprint('distance ...')

                            # distance of pixels in clip form left and right orbit
                            dstg, dstl = distance_to_lines(x, y, pygeos.get_coordinates(lrow['geom']), pygeos.get_coordinates(row['geom']))

                            #tprint('distance finished ...')
                    
                            mosaic_final_weight_v1(dstl, dstg, sdata_clip, gdata_clip, self.max_distance_from_orbit) #, dst_dtype=np.dtype(dst_dtype)) #, nthreads=mp.cpu_count())

                            gdata[gmask_clip] = gdata_clip
                            gdata[lmask & (~gmask_clip)] = sdata[mask & (~lmask_clip)]

                            #tprint('calculation finished ...')
                            
                        elif preklop =='cnt_min':
                            lmask_clip = grid.gmask2lmask(gmask_clip)
                            gdata_clip = gdata[gmask_clip] #gdata.flat[ginds_clip]        
                            sdata_clip = sdata[lmask_clip]
                            #gdata[gmask_clip] = np.fmin(np.where(gdata_clip==0, np.nan, gdata_clip), 
                            #                            np.where(sdata_clip==0, np.nan, sdata_clip))
                            gdata[gmask_clip] = np.fmin(gdata_clip, sdata_clip)
                            gdata[lmask & (~gmask_clip)] = sdata[mask & (~lmask_clip)]

                    _update_mask_or(gmask, lmask)
                    lrow = row
        return j, dst_window, gdata                  
    
    def make_mosaic_final_tiled_mp(self, out_file_name:str, preklop:str, orbits:list, extent: list, output_profile:dict,
            tile_size=1024*10):    
        '''
        Makes final mosaic using tiled multiprocessing.

        param: out_file_name: Output file name
        param: preklop: Method for overlap calculating
        param: orbits: List of all orbits and files
        param: extent: Extent of final mosaic
        param: output_profile: Profile of output image
        '''   
        
        dxy=self.dxy
        dst_left, dst_bottom, dst_right, dst_top = extent
        dst_transform = self.align_transform(rasterio.transform.from_origin(dst_left, dst_top, dxy, dxy))
        dst_bounds = rasterio.transform.array_bounds((dst_top-dst_bottom-1)//dxy+1, (dst_right-dst_left-1)//dxy+1, dst_transform)
        dst_nodata = output_profile['nodata']
        dst_dtype = output_profile['dtype']

        # download all orbirts
        tprint('Downloading orbit files ...')
        pbar = tqdm.tqdm()
        for o in orbits:
            pbar.set_description(o['file'])
            #tprint(o['file'])
            self.get_input_file(o['file'])
            pbar.update(1)
        tprint('\n\nDone.')
        
        dst_window = rio.windows.from_bounds(*dst_bounds, transform=dst_transform)
        
        ncols = int(dst_window.width)
        nrows = int(dst_window.height)
        # Izračun broja radnih tile-ova
        ntiles_cols = int(np.ceil(ncols/tile_size))
        ntiles_rows = int(np.ceil(nrows/tile_size))
        # Broj rows i coils u jednom working tile-u
        tiles_cols = int(np.ceil(ncols/ntiles_cols))
        tiles_rows = int(np.ceil(nrows/ntiles_rows))

        ncols = tiles_cols*ntiles_cols
        nrows = tiles_rows*ntiles_rows

        grid = GridSystem(nrows, ncols, dst_transform)
        output_profile.update({'width':ncols, 'height':nrows, 'transform': dst_transform})
        self.save_tif_init(out_file_name, output_profile)
        tprint('  mosaic')

        mp_args_fixed = Munch(ntiles_cols=ntiles_cols, ntiles_rows=ntiles_rows, tiles_cols=tiles_cols, tiles_rows=tiles_rows,
            dst_transform=dst_transform, orbits=orbits, dst_nodata=dst_nodata, preklop=preklop)
        mp_args = ((j,mp_args_fixed) for j in range(ntiles_cols*ntiles_rows))

        pbar=tqdm.tqdm(total=ntiles_cols*ntiles_rows)
        for ret in ThreadGeneratorLazy(self.make_mosaic_final_tiled_worker, mp_args, self.nworkers, self.nworkers):
            pbar.update(1)
            if ret is not None:
                j, tile_window, gdata=ret
                pbar.set_description(f'Window {j+1}')                
                #tprint(f'  Saving window {j+1}/{ntiles_cols*ntiles_rows} ... ')            
                self.save_tif_update(gdata, tile_window)
                #tprint(f' Window {j+1} saved.')

        self.save_tif_finish()

    
    def mosaic_final_tiled(self, orbits_folder: str, out_filename:str, extent:list=[900000, 900010, 6540000, 5460010], preklop='weight'):
        '''
        Makes final mosaic from all orbit files

        param: orbits_folder: Folder with all input orbit files
        param: out_filename: Output folder for final mosaic
        param: extent: Extent of output mosaic
        param: preklop: Method to use for overlap, default is 'weight'
        '''
        tprint(out_filename, end='')
        files = self.get_final_orbits(orbits_folder)
        if len(files)==0:
            print(' ... NO INPUTS!')
            return
        elif self.check_output_file(Path(out_filename)):
            print(' ... ALREADY EXISTS!')
            return

        print()
        
        with pkg_resources.path(_dist_data, 's2a_rel_orbit_gh.geojson') as orbits_file:
            df = geopandas.read_file(orbits_file)
        
        df['geom'] = pygeos.from_wkb(df.geometry.apply(lambda x: x.wkb))   
        df['file'] = df.orbit.apply(lambda x: files.get(x, None))
        df['order'] = np.where(df.ANX_lon.values<0, df.ANX_lon.values+360, df.ANX_lon.values)
        df.sort_values('order', ascending = False, inplace=True)
        
        orbits = df.loc[~df['file'].isnull(),['orbit','geom','file']].to_dict('records')
        
        with rio.open(self.get_input_file(orbits[0]['file'])) as src:

            p = src.profile
            out_profile = dict(driver='GTiff', crs=p['crs'], dtype=p['dtype'], count=1, nodata=p['nodata'], 
                compress='deflate', predictor=2, zlevel=8, tiled=True, blockxsize=1024, blockysize=1024, bigtiff='YES', num_threads='all_cpus')

        ret = self.make_mosaic_final_tiled_mp(out_filename, preklop, orbits, extent, out_profile)
        tprint(ret)
# %%

with pkg_resources.path(_dist_data, 's2a_rel_orbit_gh.geojson') as orbits_file:
    df = geopandas.read_file(orbits_file)