'''
Overlay and spatial prediction fully compatible with ``scikit-learn``.
'''
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
from skmap.catalog import DataCatalog
import skmap_bindings as sb
import hashlib
import itertools
n_threads = os.cpu_count() * 2
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
        # optimized for up to 200 points and about 50 layers
        # sampling only first band in every layer
        # assumption is that all layers have same blocks

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

        return pd.concat(query_pixels).drop(columns=['window', 'inv_transform'])

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
    :param catalog.
    :param dir_layers: A list of folders where the raster files are located. The raster
        are selected according to the pattern specified in ``regex_layers``.
    :param regex_layers: Pattern to select the raster files in ``dir_layers``.
        By default all GeoTIFF files are selected.
    :param n_threads: Number of CPU cores to be used in parallel. By default all cores
        are used.
    :param verbose: Use ``True`` to print the overlay progress.

    Examples
    ========

    >>> from skmap.mapper import SpaceOverlay
    >>>
    >>> spc_overlay = SpaceOverlay('./my_points.gpkg', ['./raster_dir_1', './raster_dir_2'])
    >>> result = spc_overlay.run()
    >>>
    >>> print(result.shape)

    """


    def __init__(self,
        points:Union[gpd.GeoDataFrame, str],
        catalog:DataCatalog = [],
        dir_layers:List[str] = [],
        regex_layers = '*.tif',
        raster_tiles:Union[gpd.GeoDataFrame, str] = None,
        tile_id_col:Union[str] = 'tile_id',
        n_threads:int = parallel.CPU_COUNT,
        verbose:bool = True
    ):

        self.catalog = catalog
        layer_paths, self.layer_idxs, self.layer_names = self.catalog.get_paths()

        if not isinstance(points, gpd.GeoDataFrame):
            pq_points = pd.read_parquet(points)
            pq_points['geometry'] = pq_points.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
            points = gpd.GeoDataFrame(pq_points, geometry='geometry')
        self.pts = points
        self.n_threads = n_threads

        self.parallelOverlay = _ParallelOverlay(self.pts.geometry.x.values, self.pts.geometry.y.values,
            layer_paths, points_crs=self.pts.crs, raster_tiles=raster_tiles, tile_id_col=tile_id_col, 
            n_threads=self.n_threads, verbose=verbose)

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
        assert self.catalog.data_size == len(self.catalog.get_features()), \
            "Catalog data size should coincide wiht the number of features, something went wrong"
        data_overlay = self.read_data(gdal_opts, max_ram_mb)
        data_array = np.empty((len(self.catalog.data_size), data_overlay.shape[1]), dtype=np.float32)
        data_array[self.layer_idxs,:] = data_overlay[:,:]
        run_whales(self.catalog, data_array, self.n_threads)
        # @FIXME check that all the filled flages are True or assert at this point
        df = pd.DataFrame(data_array.T, columns=self.catalog.get_features())
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
            unique_blocks = key_query_pixels[['block_id', 'block_col_off', 'block_row_off', 'block_height', 'block_width']].drop_duplicates('block_id')
            unique_blocks_ids = unique_blocks['block_id'].tolist()
            key_layer_ids_comb, unique_blocks_ids_comb = map(list, zip(*itertools.product(key_layer_ids, unique_blocks_ids)))
            block_row_off_comb = [unique_blocks.query(f"block_id == @ubid")['block_row_off'].iloc[0] for ubid in unique_blocks_ids_comb]
            block_col_off_comb = [unique_blocks.query(f"block_id == @ubid")['block_col_off'].iloc[0] for ubid in unique_blocks_ids_comb]
            block_height_comb = [unique_blocks.query(f"block_id == @ubid")['block_height'].iloc[0] for ubid in unique_blocks_ids_comb]
            block_width_comb = [unique_blocks.query(f"block_id == @ubid")['block_width'].iloc[0] for ubid in unique_blocks_ids_comb]
            key_layer_paths_comb = [f'/vsicurl/{path}' if path.startswith("http") and path.endswith(".tif") else path
                    for path in (str(layers.query(f"layer_id == @ulid")['path'].iloc[0]) for ulid in key_layer_ids_comb)]            
            key_layer_nodatas_comb = [np.nan if layers.query(f"layer_id == @ulid")['nodata'].iloc[0] is None 
                                        else layers.query(f"layer_id == @ulid")['nodata'].iloc[0] 
                                        for ulid in key_layer_ids_comb]
            n_comb = len(key_layer_paths_comb)
            if 'tile_id' in query_pixels[key].columns:
                block_tile_id_comb = [key_query_pixels.query(f"block_id == @ubid")['tile_id'].iloc[0] for ubid in unique_blocks_ids_comb]
                for j in range(n_comb):            
                    key_layer_paths_comb[j] = key_layer_paths_comb[j].format(tile_id=block_tile_id_comb[j])
            block_height = max(block_height_comb)
            block_width = max(block_width_comb)
            n_block_pix = block_height * block_width
            bands_list = [1]
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
                                  block_width_chunk, block_height_chunk, bands_list, gdal_opts, key_layer_nodatas_chunk, np.nan)
                pix_blok_ids = key_query_pixels['block_id'].tolist()
                sample_rows = key_query_pixels['sample_row'].tolist()
                sample_cols = key_query_pixels['sample_col'].tolist()
                pix_inblock_idxs = [sample_rows[k] * unique_blocks.query(f"block_id == @ubid")['block_width'].iloc[0] + sample_cols[k] \
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
        points:Union[gpd.GeoDataFrame, str],
        col_date:str,
        catalog:DataCatalog = [],
        raster_tiles:Union[gpd.GeoDataFrame, str] = None,
        tile_id_col:Union[str] = 'tile_id',
        n_threads:int = parallel.CPU_COUNT,
        verbose:bool = False
    ):


        if not isinstance(points, gpd.GeoDataFrame):
            pq_points = pd.read_parquet(points)
            pq_points['geometry'] = pq_points.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
            points = gpd.GeoDataFrame(pq_points, geometry='geometry')
        self.pts = points
        self.n_threads = n_threads

        self.col_date = col_date
        self.overlay_objs = {}
        self.year_catalogs = {}
        self.verbose = verbose
        self.catalog = catalog

        self.pts.loc[:,self.col_date] = pd.to_datetime(self.pts[self.col_date])
        self.uniq_years = self.pts[self.col_date].dt.year.unique()
        self.year_points = {}

        for year in self.uniq_years:

            year = int(year)
            self.year_points[str(year)] = self.pts[self.pts[self.col_date].dt.year == year]
            year_catalog = catalog.copy()
            year_catalog.query(['common', str(year)], catalog.get_features())
            self.year_catalogs[str(year)] = year_catalog

            if self.verbose:
                ttprint(f'Overlay {len(self.year_points[str(year)])} points from {year} in {len(year_catalog.get_features())} raster layers')

            self.overlay_objs[str(year)] = SpaceOverlay(points=self.year_points[str(year)], catalog=self.year_catalogs[str(year)],
                raster_tiles=raster_tiles, tile_id_col=tile_id_col, n_threads=n_threads, verbose=verbose)

    

    
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

        for year in self.uniq_years:

            if self.verbose:
                ttprint(f'Running the overlay for {year}')
            year_result = self.overlay_objs[str(year)].run(max_ram_mb, out_file_name, gdal_opts)

            if self.result is None:
                self.result = year_result
            else:
                self.result = pd.concat([self.result, year_result], ignore_index=True)

        return self.result
