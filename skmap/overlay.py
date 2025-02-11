from typing import List, Union
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from shapely import box
from pathlib import Path
from skmap import parallel
from skmap.misc import ttprint
from skmap.catalog import DataCatalog, run_whales
import skmap_bindings as sb
import hashlib
import itertools
n_threads = os.cpu_count()
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}'
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'


class _ParallelOverlay:
    # sampling only first band in every layer

    TILE_PLACEHOLDER = '{tile_id}'

    def __init__(self,
        points_x: np.ndarray,
        points_y:np.ndarray,
        raster_files:List[str],
        points_crs: None,
        raster_tiles:Union[gpd.GeoDataFrame, str] = None,
        tile_id_col:Union[str] = 'tile_id',
        n_threads:int = parallel.CPU_COUNT,
        verbose:bool = True
    ):

        self.verbose = verbose

        self.default_tile_id = ''
        self.tile_id_col = tile_id_col

        if raster_tiles is not None:
          if not isinstance(raster_tiles, gpd.GeoDataFrame):
            if self.verbose:
              ttprint(f"Reading {raster_tiles}")
            raster_tiles = gpd.read_file(raster_tiles)

          self.default_tile_id = raster_tiles[self.tile_id_col].iloc[0]

        self.raster_tiles = raster_tiles
        
        if points_crs is None:
          points_crs = rasterio.open(
            raster_files[0].replace(_ParallelOverlay.TILE_PLACEHOLDER, self.default_tile_id)
          ).crs

        samples = gpd.GeoDataFrame( 
                geometry=gpd.points_from_xy(points_x, points_y), 
                crs=points_crs
        ).reset_index(drop=True)
        
        self.raster_files = raster_files


        self.layers = pd.DataFrame({
            'name': [ str(Path(raster_file).with_suffix('').name) for raster_file in raster_files ],
            'path': self.raster_files
        }).reset_index(drop=True
        ).apply(_ParallelOverlay._layer_metadata, default_tile_id=self.default_tile_id, axis=1)

        self.query_pixels = self._find_blocks(samples)

    
    @staticmethod
    def _layer_metadata(row, default_tile_id):
        
        path = row['path']
        if _ParallelOverlay._is_tiled(path):
          path = path.replace(_ParallelOverlay.TILE_PLACEHOLDER, default_tile_id)        
        src = rasterio.open(path)

        row['nodata'] = src.nodata
        _, window = next(src.block_windows(1))
        row['block_height'] = window.height
        row['block_width'] = window.width

        key = ''.join([
          str(default_tile_id), 
          str(src.height), str(src.width),
          str(src.block_shapes[0]), 
          str(src.transform.to_gdal())
        ])
        row['group'] = str(
            hashlib.md5(key.encode('utf-8')).hexdigest()
        )

        return row

    @staticmethod
    def _is_tiled(path):
      return (_ParallelOverlay.TILE_PLACEHOLDER in str(path))

    @staticmethod
    def _tiled_blocks(tile_path, tile_id):
      tile_blocks = []

      with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_VERSION='1.1'):
        src = rasterio.open(tile_path)

        for (i,j), window in src.block_windows(1):
          tile_blocks.append({
              'tile_id': tile_id,
              'window': window,    
              'geometry': box(*rasterio.windows.bounds(window, src.transform))
          })

      return tile_blocks

    def _find_blocks_for_src(self, path, samples):

        gdf_blocks = []

        if _ParallelOverlay._is_tiled(path):
          

          samp_tiles = self.raster_tiles.sjoin(samples.to_crs(self.raster_tiles.crs), how='inner')[self.tile_id_col].unique()
          raster_tiles = self.raster_tiles[self.raster_tiles[self.tile_id_col].isin(samp_tiles)]
          if self.verbose:
            ttprint(f"Retrieving block information for {raster_tiles.shape[0]} tiles.")

          for _, tile in raster_tiles.iterrows():
            tile_id = tile[self.tile_id_col]
            tile_path = path.replace(_ParallelOverlay.TILE_PLACEHOLDER, tile_id)
            src = rasterio.open(tile_path)
            for (i,j), window in src.block_windows(1):
              gdf_blocks.append({
                  'tile_id': tile_id,
                  'window': window,  
                  'inv_transform': ~rasterio.windows.transform(window, src.transform),
                  'geometry': box(*rasterio.windows.bounds(window, src.transform))
              })

        else:
          src = rasterio.open(path)

          for (i,j), window in src.block_windows(1):
              gdf_blocks.append({
                  'window': window,
                  'inv_transform': ~rasterio.windows.transform(window, src.transform),
                  'geometry': box(*rasterio.windows.bounds(window, src.transform))
              })

        gdf_blocks = gpd.GeoDataFrame(gdf_blocks, crs=src.crs).reset_index(drop=True)
        assert gdf_blocks.crs is not None, f'The layer {path} has not crs, need fix'
        gdf_blocks = samples.to_crs(gdf_blocks.crs).sjoin(gdf_blocks, how='inner').rename(columns={ 'index_right': 'block_id' })

        query_pixels = []

        for (ij, block) in gdf_blocks.groupby('block_id'):
            inv_block_transform = block['inv_transform'].iloc[0]
            window = block['window'].iloc[0]
            block['x'] = block.geometry.x
            block['y'] = block.geometry.y
            block['block_col_off'] = window.col_off
            block['block_row_off'] = window.row_off
            block['block_width'] = window.width
            block['block_height'] = window.height

            block.loc[:,'sample_col'], block.loc[:,'sample_row'] = inv_block_transform * (block['x'], block['y'])
            block['sample_col'] = block['sample_col'].astype('int')
            block['sample_row'] = block['sample_row'].astype('int')
            block = block.drop(columns='geometry')

            query_pixels.append(block)
        
        
        assert query_pixels, f"query_pixels is empty for path: {path}"
        res = pd.concat(query_pixels).drop(columns=['window', 'inv_transform'])
        if len(set(res.index)) < int(samples.shape[0] * 0.5):
            print(f"Less then 50% of points queryed for path: {path}, this is suspicious")
        
        return res

    def _find_blocks(self, samples):

        if self.verbose:
            ttprint(f'Scanning blocks of {len(self.raster_files)} layers')
        
        query_pixels = {}
        
        for _, layers in self.layers.groupby('group'):
            group = layers.iloc[0]['group']
            path = layers.iloc[0]['path']

            if self.verbose:
              ttprint(f"Finding query pixels for {group} ({layers.shape[0]} layers)")
            
            query_pixels[group] = self._find_blocks_for_src(path, samples)

        if self.verbose:
            ttprint(f'End')
        
        return query_pixels



class SpaceOverlay():
    """
    Overlay a set of points over multiple raster files.
    The retrieved pixel values are organized in columns
    according to the filenames.

    :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
        the points.
    :param catalog: scikit-map data catalog.
    :param n_threads: Number of CPU cores to be used in parallel. By default all cores
        are used.
    :param verbose: Use ``True`` to print the overlay progress.


    """


    def __init__(self,
        points:Union[gpd.GeoDataFrame, str, pd.DataFrame],
        catalog:DataCatalog = [],
        raster_tiles:Union[gpd.GeoDataFrame, str] = None,
        tile_id_col:Union[str] = 'tile_id',
        n_threads:int = parallel.CPU_COUNT,
        verbose:bool = True
    ):

        self.catalog = catalog
        self.layer_paths, self.layer_idxs, self.layer_names = self.catalog.get_paths()

        if not isinstance(points, gpd.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_parquet(points)                
            points['geometry'] = points.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
            points = gpd.GeoDataFrame(points, geometry='geometry')
        self.pts = points.reset_index(drop=True)
        self.n_threads = n_threads
        self.verbose = verbose

        self.parallelOverlay = _ParallelOverlay(self.pts.geometry.x.values, self.pts.geometry.y.values,
            self.layer_paths, points_crs=self.pts.crs, raster_tiles=raster_tiles, tile_id_col=tile_id_col, 
            n_threads=self.n_threads, verbose=verbose)
        
        # Drop duplicates (from overlapping tiles), find union of missing points (out of extent) to drop them and sorting the dataframes
        keys = list(self.parallelOverlay.query_pixels.keys())
        missing_indices = []
        for key in keys:
            self.parallelOverlay.query_pixels[key] = \
                self.parallelOverlay.query_pixels[key][~self.parallelOverlay.query_pixels[key].index.duplicated(keep='first')]
            missing_indices_key_set = set(self.pts.index) - set(self.parallelOverlay.query_pixels[key].index)
            missing_indices += list(missing_indices_key_set)
        for key in keys:
            key_indices_to_drop = set(missing_indices) & set(self.parallelOverlay.query_pixels[key].index)
            self.parallelOverlay.query_pixels[key] = self.parallelOverlay.query_pixels[key].drop(index=key_indices_to_drop)
            self.parallelOverlay.query_pixels[key] = self.parallelOverlay.query_pixels[key].sort_index(axis=0, ascending=True)
            
        print(f"Dropping {len(list(set(missing_indices)))} points out of {self.pts.shape[0]} because out of extent")
        self.pts = self.pts.drop(index=set(missing_indices)).sort_index(axis=0, ascending=True)
        
        

    def run(self,
        max_ram_mb:int,
        out_file_name:str,
        gdal_opts = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'},
    ):
        """
        Execute the space overlay.

        :param gdal_opts: Options for GDAL in a dictionary.

        :param max_ram_mb: Maximum RAM that can be used for the overlay expressed in MB.

        :returns: Data frame with the original columns plus the overlay result (one new
            column per raster).
        :rtype: geopandas.GeoDataFrame
        """
        feats_names, _, feats_idx = self.catalog.get_unrolled_catalog()
        assert (self.catalog.data_size == len(self.catalog.get_feature_names())) & (self.catalog.data_size == len(feats_names)), \
            "Catalog data size should coincide wiht the number of features, something went wrong"
        
        self.ordered_feats_names = [s for _, s in sorted(zip(feats_idx, feats_names))]
    
        self.data_overlay = self.read_data(gdal_opts, max_ram_mb)
        self.data_array = np.empty((self.catalog.data_size, self.data_overlay.shape[1]), dtype=np.float32)
        assert self.pts.shape[0] == self.data_overlay.shape[1], "Not matching size between input points and the overalied data"
        
        self.data_array[self.layer_idxs,:] = self.data_overlay[:,:]
        run_whales(self.catalog, self.data_array, self.n_threads)
        # @FIXME check that all the filled flages are True or assert at this point
        df = pd.DataFrame(self.data_array.T, columns=self.ordered_feats_names)
        self.pts_out = pd.concat([self.pts.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

        self.pts_out['lon'] = self.pts_out['geometry'].x
        self.pts_out['lat'] = self.pts_out['geometry'].y
        self.pts_out = self.pts_out.drop(columns=['geometry'])
        if out_file_name is not None:
            self.pts_out.to_parquet(out_file_name)

        return self.pts_out
    
    def read_data(self, gdal_opts, max_ram_mb):
        layers = self.parallelOverlay.layers
        layers['layer_id'] = layers.index
        query_pixels = self.parallelOverlay.query_pixels
        keys = list(query_pixels.keys())
        data_overlay = np.empty((layers.shape[0], query_pixels[keys[0]].shape[0]), dtype=np.float32)
        sb.fillArray(data_overlay, self.n_threads, np.nan)
        for i, key in enumerate(keys):
            key_layer_ids = [int(l) for l in list(layers[layers['group'] == key]['layer_id'])]
            key_query_pixels = query_pixels[key]
            assert key_query_pixels.shape[0] == query_pixels[keys[0]].shape[0], "Query pixel size is inconsisten between keys, something went wrong"
            unique_blocks = key_query_pixels[['block_id', 'block_col_off', 'block_row_off', 'block_height', 'block_width']].drop_duplicates('block_id')
            unique_blocks_ids = unique_blocks['block_id'].tolist()
            unique_blocks_dict = unique_blocks.set_index('block_id')[['block_row_off', 'block_col_off', 'block_height', 'block_width']].to_dict('index')
            layer_path_dict = layers.set_index('layer_id')['path'].to_dict()
            if self.verbose:
                ttprint(f'Loading and sampling {len(layer_path_dict)} raster layers for group {key}')
            layer_nodata_dict = layers.set_index('layer_id')['nodata'].to_dict()
            key_layer_ids_comb, unique_blocks_ids_comb = map(list, zip(*itertools.product(key_layer_ids, unique_blocks_ids)))
            block_row_off_comb = [unique_blocks_dict[ubid]['block_row_off'] for ubid in unique_blocks_ids_comb]
            block_col_off_comb = [unique_blocks_dict[ubid]['block_col_off'] for ubid in unique_blocks_ids_comb]
            block_height_comb = [unique_blocks_dict[ubid]['block_height'] for ubid in unique_blocks_ids_comb]
            block_width_comb = [unique_blocks_dict[ubid]['block_width'] for ubid in unique_blocks_ids_comb]
            key_layer_paths_comb = [layer_path_dict[ulid] for ulid in key_layer_ids_comb]
            key_layer_nodatas_comb = [np.nan if layer_nodata_dict[ulid] is None else layer_nodata_dict[ulid] for ulid in key_layer_ids_comb]
            n_comb = len(key_layer_paths_comb)
            bands_list = [1]
            if 'tile_id' in query_pixels[key].columns:
                block_tile_id_dict = key_query_pixels.set_index('block_id')['tile_id'].to_dict()
                key_layer_paths_comb = [key_layer_paths_comb[j].format(tile_id=block_tile_id_dict[unique_blocks_ids_comb[j]])
                    for j in range(len(key_layer_paths_comb))]
            block_height = np.max(block_height_comb)
            block_width = np.max(block_width_comb)
            n_block_pix = block_height * block_width            
            # Factor 2 is heuristic to keep margin for temporary variables
            max_comb_chunk = int(np.ceil((max_ram_mb - data_overlay.size*4/1024/1024)*1024*1024/4/n_block_pix/2))
            assert max_comb_chunk > 1, "skmap-error 42: max_ram_mb too small, can not chunk"
            for chunk_start in range(0, n_comb, max_comb_chunk):
                key_layer_paths_chunk = key_layer_paths_comb[chunk_start:chunk_start + max_comb_chunk]
                block_col_off_chunk = block_col_off_comb[chunk_start:chunk_start + max_comb_chunk]
                block_row_off_chunk = block_row_off_comb[chunk_start:chunk_start + max_comb_chunk]
                block_height_chunk = block_height_comb[chunk_start:chunk_start + max_comb_chunk]
                block_width_chunk = block_width_comb[chunk_start:chunk_start + max_comb_chunk]
                key_layer_nodatas_chunk = key_layer_nodatas_comb[chunk_start:chunk_start + max_comb_chunk]
                unique_blocks_ids_chunk = unique_blocks_ids_comb[chunk_start:chunk_start + max_comb_chunk]
                key_layer_ids_chunk = key_layer_ids_comb[chunk_start:chunk_start + max_comb_chunk]
                chunk_size = len(key_layer_paths_chunk)
                data_array = np.empty((chunk_size, n_block_pix), dtype=np.float32)
                perm_vec = range(chunk_size)
                sb.readDataBlocks(data_array, self.n_threads, key_layer_paths_chunk, perm_vec, block_col_off_chunk, block_row_off_chunk,
                                  block_width_chunk, block_height_chunk, bands_list, gdal_opts, None, np.nan)
                pix_blok_ids = key_query_pixels['block_id'].tolist()
                sample_rows = key_query_pixels['sample_row'].tolist()
                sample_cols = key_query_pixels['sample_col'].tolist()
                block_width_dict = unique_blocks.set_index('block_id')['block_width'].to_dict()
                pix_inblock_idxs = [sample_rows[k] * block_width_dict[ubid] + sample_cols[k]
                    for k, ubid in enumerate(pix_blok_ids)]
                sb.extractOverlay(data_array, self.n_threads, pix_blok_ids, pix_inblock_idxs, unique_blocks_ids_chunk, key_layer_ids_chunk, data_overlay)
        return data_overlay
    
    
    
class SpaceTimeOverlay():
    """
    Overlay a set of points over multiple raster considering the year information.
    The retrieved pixel values are organized in columns according to the filenames.

    :param points: The path for vector file or ``geopandas.GeoDataFrame`` with
        the points.
    :param col_date: Date column to retrieve the year information.
    :param  catalog.
    :param n_threads: Number of CPU cores to be used in parallel. By default all cores
        are used.
    :param verbose: Use ``True`` to print the overlay progress.

    Examples
    ========

    >>> from skmap.mapper import SpaceTimeOverlay
    >>>

    """

    def __init__(self,
        points:Union[gpd.GeoDataFrame, str, pd.DataFrame],
        col_date:str,
        catalog:DataCatalog = [],
        raster_tiles:Union[gpd.GeoDataFrame, str] = None,
        tile_id_col:Union[str] = 'tile_id',
        n_threads:int = parallel.CPU_COUNT,
        verbose:bool = False
    ):
        
        if not isinstance(points, gpd.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_parquet(points)                
            points['geometry'] = points.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
            points = gpd.GeoDataFrame(points, geometry='geometry')
        self.pts = points.reset_index(drop=True)
        self.n_threads = n_threads

        self.col_date = col_date
        self.overlay_objs = {}
        self.year_catalogs = {}
        self.verbose = verbose
        self.catalog = catalog

        self.pts[self.col_date] = self.pts[self.col_date].astype(int)
        self.year_points = {}

        for year in self.catalog.get_groups():
            
            self.year_points[year] = self.pts[self.pts[self.col_date] == int(year)]
            if len(self.year_points[year]) > 0:
                year_catalog = catalog.copy()
                year_catalog.query(catalog.get_feature_names(), [year]) # 'common' group is retrieved by default
                self.year_catalogs[year] = year_catalog

                if self.verbose:
                    ttprint(f'Overlay {len(self.year_points[year])} points from {year} in {year_catalog.data_size} raster layers')

                self.overlay_objs[year] = SpaceOverlay(points=self.year_points[year], catalog=self.year_catalogs[year],
                    raster_tiles=raster_tiles, tile_id_col=tile_id_col, n_threads=n_threads, verbose=verbose)
            else:
                print(f"No points to overlay for year {year}, removing it from the catalog")
                del catalog.data[year]

    
    def run(self,
        max_ram_mb:int,
        out_file_name:str,
        gdal_opts = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'},
    ):
        """
        Execute the spacetime overlay. It removes the year part from the column names.
        For example, the raster ``raster_20101202..20110320.tif`` results in the column
        name ``raster_1202..0320``.

        :returns: Data frame with the original columns plus the overlay result (one new
            column per raster).
        :rtype: geopandas.GeoDataFrame
        """
        self.result = None

        for year in self.catalog.get_groups():

            if self.verbose:
                ttprint(f'Running the overlay for {year}')
            year_result = self.overlay_objs[year].run(max_ram_mb, out_file_name, gdal_opts)

            if self.result is None:
                self.result = year_result
            else:
                self.result = pd.concat([self.result, year_result], ignore_index=True)

        return self.result
