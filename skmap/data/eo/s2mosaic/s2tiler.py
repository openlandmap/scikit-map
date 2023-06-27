
import warnings
import sys, os, shutil
import boto3
import geopandas, pandas
import sentinelhub
from sentinelhub import AwsTile, AwsTileRequest, DataSource

from pathlib import Path
import rasterio as rio
import numpy as np
from datetime import datetime
import traceback
#from filelock import Timeout, FileLock
import json

from .speedups import nan_percentile

fld = Path(__file__).resolve().parent

warnings.filterwarnings("ignore", category=UserWarning)
tile_nodata=0

#%%

class S2Tiler():
    '''
    Class that calculates one s2 tile of mosaic
    '''
    LOG=[]
    scl_masked_values = np.array([0,1,2,3,8,9,10],dtype='uint8')
    bd = dict(B01='R60m',B02='R10m',B03='R10m',B04='R10m',B05='R20m',B06='R20m',B07='R20m',B08='R10m',B8A='R20m',B09='R60m',B11='R20m',B12='R20m')
    tile_nodata = 2**15-1
    verbose = False

    @staticmethod
    def get_relative_orbit_tileInfo(fn_tileInfo):
        '''
        Read information on relative orbit from "tileInfo"
        '''
        with open(fn_tileInfo) as fid:
            return json.load(fid)['productName'].split('_')[4]

    def mask_scl_s2l2a(self, fn_scl, vrt_options):
        '''
        Read mask raster from SCL file

        param: fn_scl: File name of SCL layer
        param: vrt_options: Options for WarpedVRT

        returns: numy array with mask according to 'scl_masked_values'
        '''
        with rio.open(fn_scl) as src:
            vrt_options.update({'resampling': rio.enums.Resampling.nearest})
            if vrt_options['crs'] is None:
                '''
                The metadata.xml file in the AWS root directory contains among other things the EPSG and the Tile_Geocoding for each of the resolutions. You can parse these from the metadata.xml and set them using the GDAL functions: ds.SetProjection and ds.SetGeoTransform. GDAL will give a few warnings as it is not able to save these values, but it will generally work fine.
                '''
                print("CRS is None!!")
                print(fn_scl)
                del vrt_options['crs']
            with rio.vrt.WarpedVRT(src, **vrt_options) as vrt:
                scl = vrt.read(1)
        mask = np.isin(scl, self.scl_masked_values)
        return mask

    def __init__(self, source:str, band:str, tile_name:str, satimgs: pandas.DataFrame, bucket:str, 
                tmp_folder:str, data_folder:str, out_parent_folder:str, out_folder_prefix:str, 
                debug:bool=False, verbose=False, **kwargs):
        '''
        param: source: Source of images 
        param: band: Band name ('B02','B03', ...)
        param: tile_name: Name of tile thats going to be mosaicked
        param: satimgs: Pandas DataFrame with all images to be mosaicked
        param: bucket: name of the bucket where result will be saved
        param: tmp_folder: temporary folder 
        param: data_folder: folder for downloading source images
        param: out_arent_folder:
        param: out_folder_prefix:
        param: debug: If true then there is no uploading to AWS S3
        param: verbose:
        param: kwargs: Additional arguments

        returns: 

        '''
        self.perc= [25,50,75] if 'perc' not in kwargs else kwargs['perc']
        if 'group_by_orbits' in kwargs:
            self.group_by_orbits=kwargs['group_by_orbits']
        else:
            self.group_by_orbits = True
        self.debug = debug
        self.source = source
        self.band = band
        self.band_folder = self.bd[band]
        self.resolution = self.band_folder[1:-1]
        self.tile_name = tile_name
        self.data_folder = data_folder
        self.satimgs = satimgs
        self.verbose = verbose
        
        self.bucket = bucket
        self.out_parent_folder = Path(out_parent_folder)

        self.out_folder_name = []
        self.out_folder = []
        for p in self.perc:
            out_folder_name = f'{out_folder_prefix}/P{p:2d}'
            out_folder = self.out_parent_folder/out_folder_name
            self.out_folder_name.append(out_folder_name)
            self.out_folder.append(out_folder)

        self.out_folder_name_cnt = f'{out_folder_prefix}/CNT'
        self.out_folder_cnt = self.out_parent_folder/self.out_folder_name_cnt

        self.tmp_folder = Path(tmp_folder)
        if debug:
            pp = Path(bucket)
            self.tmp_fullpath=[]
            for p in self.perc:
                tmp_fullpath_pp = pp/self.out_parent_folder/f'{out_folder_prefix}/P{p:02d}'
                if not tmp_fullpath_pp.exists():
                    Path.mkdir(tmp_fullpath_pp,parents=True, exist_ok=True)
                self.tmp_fullpath.append(tmp_fullpath_pp)
            
            self.tmp_fullpath_cnt = pp/self.out_folder_cnt
            if not self.tmp_fullpath_cnt.exists():
                Path.mkdir(self.tmp_fullpath_cnt, parents=True, exist_ok=True)

        #s2l2a        
        
        self.resampling = rio.enums.Resampling.nearest  #this is not important any more !!!
        
        self.data_source = DataSource.SENTINEL2_L2A        
        self.metafiles = ['tileInfo','metadata.xml']
        self.bands = [f'{self.band_folder}/{band}', 'R60m/SCL.jp2']

    @property 
    def output_filename(self):
        return f'{self.tile_name}_{self.orbit}.tif'

    #@property
    def tmp_path(self, iperc):
        if self.debug:
            return self.tmp_fullpath[iperc]/self.output_filename
        else:
            fn = f"{self.out_folder[iperc].as_posix().replace('/','_')}_{self.output_filename}"
            return self.tmp_folder/fn
    
    @property
    def tmp_path_cnt(self):
        if self.debug:
            return self.tmp_fullpath_cnt/self.output_filename
        else:
            fn = f"{self.out_folder_cnt.as_posix().replace('/','_')}_{self.output_filename}"
            return self.tmp_folder/fn

    #@property
    def s3_path(self, iperc):
        return self.out_folder[iperc]/self.output_filename
  
    @property
    def s3_path_cnt(self):
        return self.out_folder_cnt/self.output_filename

    def move_tmp_to_s3(self): 
        '''
        Move result image from temporary folder to S3
        '''       
        s3 = boto3.resource('s3')

        for tmp_fn, s3_fn in zip(
            [self.tmp_path(i) for i in range(len(self.perc))] + [self.tmp_path_cnt],
            [self.s3_path(i) for i in range(len(self.perc))] + [self.s3_path_cnt]):   
            
            s3.meta.client.upload_file(tmp_fn.as_posix(), self.bucket, s3_fn.as_posix())
            os.remove(tmp_fn)


    def log(self, message):
        self.LOG.append(message)

    def geolocation_from_metadata(self, fn_metadata, resolution=None):
        '''
        Read geolocation of source image from metadata.xml
        '''        
        import xml.etree.ElementTree as ET
        
        if resolution is None:
            resolution = self.resolution

        try: 
            ns={'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd"}
            tree = ET.parse(fn_metadata)
            root = tree.getroot()

            crs = root.find(f"n1:Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE",ns).text

            # esize = root.find(f"n1:Geometric_Info/Tile_Geocoding/Size[@resolution='{self.resolution}']",ns)
            # nrows = int(esize.find('NROWS').text)
            # ncols = int(esize.find('NCOLS').text)

            egp = root.find(f"n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='{resolution}']",ns)
            ulx = float(egp.find('ULX').text)
            uly = float(egp.find('ULY').text)
            xdim = float(egp.find('XDIM').text)
            ydim = float(egp.find('YDIM').text)

            transform = rio.transform.from_origin(ulx,uly,abs(xdim),abs(ydim))
        except:
            crs, transform = None, None

        return crs, transform

    def make_tile(self):
        '''
        Main procedure to make tile
        '''
        dff = pandas.DataFrame(self.satimgs)
        # download all imgs
        for i,r in dff.iterrows():
            request = AwsTileRequest(tile=r.scene_tile_name, time=r.scene_date, aws_index=r.scene_aws_index,
                                        bands=self.bands, metafiles=self.metafiles, data_folder=self.data_folder, #self.data_folder,
                                        data_collection=self.data_source)
            if self.data_folder is None:
                fn_band, fn_scl, fn_tileInfo, fn_metadata = request.get_filename_list()
            else:
                fn_band, fn_scl, fn_tileInfo, fn_metadata = request.get_filename_list()
                fn_band=f'{self.data_folder}/{fn_band}'
                fn_scl = f'{self.data_folder}/{fn_scl}'
                fn_tileInfo = f'{self.data_folder}/{fn_tileInfo}'
                fn_metadata = f'{self.data_folder}/{fn_metadata}'

                request.save_data()
                        
            orbit = self.get_relative_orbit_tileInfo(fn_tileInfo)  #get_relative_orbit(r.scene_tile_name, r.scene_date, r.scene_aws_index)
            dff.loc[i,'orbit']=orbit
            dff.loc[i,'fn_band']=fn_band 
            dff.loc[i,'fn_scl']=fn_scl
            dff.loc[i,'fn_metadata'] = fn_metadata

        # Group by orbit
        if self.group_by_orbits:
            dfg = dff.groupby('orbit')
        else:
            dfg = [('',dff)]

        for orbit, dfo in dfg:
            self.orbit = orbit
            if self.verbose:
                print ('  ',orbit)
            # orbit = list(dfg.groups.keys())[1]; dfo=dfg.get_group(orbit)
            ndates=len(dfo)
            idate=0
            imgcube=None
            for ir, row in dfo.iterrows(): 
                try:
                    # ir=dfo.index[0]; row=dfo.loc[i]
                    if self.verbose:
                        print (idate, row.scene_date)

                    with rio.open(row.fn_band) as src:
                        #print(src.width, src.height)
                        #if src.crs is None:
                        #    print(row,fn_band)
                        if imgcube is None:
                            imgcube = np.full((src.height, src.width, ndates), fill_value=np.nan, dtype='float32')
                            out_profile = src.profile 
                                                                    
                            if src.crs is None:                            
                                src_crs, src_transform = self.geolocation_from_metadata(row.fn_metadata)
                                if src_crs is not None:
                                    out_profile.update({'crs':src_crs, 'transform':src_transform})
                            
                                                
                        data = src.read(1).astype('float32')
                        mmask = src.read_masks(1)==0
                        '''
                        Datasets older than May 2018 have broken crs and transfrom.
                        This is workaround:
                        The metadata.xml file in the AWS root directory contains among other things the EPSG and the Tile_Geocoding for each of the resolutions. You can parse these from the metadata.xml and set them using the GDAL functions: ds.SetProjection and ds.SetGeoTransform. GDAL will give a few warnings as it is not able to save these values, but it will generally work fine.
                        '''
                        scl_crs, scl_transform = self.geolocation_from_metadata(row.fn_metadata,resolution='60')
                        if src.crs is None:
                            src_crs, src_transform = self.geolocation_from_metadata(row.fn_metadata)
                            if src_crs is None:
                                continue
                            out_profile.update({'crs':src_crs, 'transform':src_transform})
                        else:
                            src_crs = src.crs
                            src_transform = src.transform
                        vrt_options= {'resampling': rio.enums.Resampling.nearest, 
                                    'src_crs': scl_crs, 'src_transform': scl_transform,
                                    'crs': src_crs, 'transform': src_transform, 'height': src.height, 'width': src.width}

                    dmask0 = data==0
                    mask_cloud = self.mask_scl_s2l2a(row.fn_scl, vrt_options)
                    mask =  mmask | mask_cloud | dmask0
                    data[mask] = np.nan 

                    imgcube[:,:,idate]=data
                except Exception as e:
                    error_message = traceback.format_exc()
                    self.log(f'While processing row {row}, this happened:\n{error_message}')
                    
                idate = idate + 1 

            prc, mosaic_cnt = nan_percentile(imgcube, self.perc)
            del imgcube

            for i in range(len(self.perc)):
                mosaic = np.nan_to_num(prc[i,:,:], nan=self.tile_nodata).astype(rio.uint16)
                out_profile.update(dict(driver='GTiff', dtype=rio.uint16, count=1, nodata=self.tile_nodata, compress='deflate'))
                with rio.open(self.tmp_path(i), 'w', **out_profile) as dst:
                    dst.write(mosaic, 1)
            
            mosaic_cnt = mosaic_cnt.astype(rio.uint8) 
            out_profile.update(dtype=rio.uint8, nodata=None)
            with rio.open(self.tmp_path_cnt, 'w', **out_profile) as dst:
                dst.write(mosaic_cnt, 1)

            del prc

            if not self.debug:
                self.move_tmp_to_s3()

    def __call__(self):
        '''
        Entry point for procedure
        '''
        try:
            self.make_tile()
            return 'OK'
        except Exception as e:
            error_message = traceback.format_exc()
            print(error_message)
            self.log(error_message)
            #self.save_log_to_s3()
            if isinstance(e, MemoryError):
                return 'MEMORY_ERROR' #, self.tile_id, start_time, self.LOG
            return 'OTHER_ERROR'


        