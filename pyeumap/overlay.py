'''
Script for overlaying points (sampling)
Only first band from each file
'''
#%%
from typing import List
import rasterio
import rasterio.windows
from pathlib import Path
import geopandas, pandas
import numpy as np
import concurrent.futures
import multiprocessing

from . import parallel
from .misc import ttprint

#%%

class SamplePointsParallel:
    # optimized for up to 200 points and about 50 layers
    # sampling only first band in every layer
    # assumption is that all layers have same blocks

    def __init__(self, points_x: np.ndarray, points_y:np.ndarray, fn_layers:List[str], fn_outfile:str, max_workers:int):
        self.points_x = points_x
        self.points_y = points_y
        self.points_len = len(points_x)
        self.fn_layers = fn_layers
        self.fn_outfile = fn_outfile
        self.max_workers = max_workers

        self.layer_names = [fn_layer.with_suffix('').name for fn_layer in fn_layers]
        sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
        self.dimensions = [self._get_dimension(src) for src in sources]

        self.points_blocks = self.find_blocks()

    @staticmethod
    def _get_dimension(src):
        return (src.height, src.width, *src.block_shapes[0], *src.transform.to_gdal())

    def _find_blocks_for_src(self, src, ptsx, ptsy):
        # find blocks for every point in given source
        blocks = {}
        for ij, block in src.block_windows(1):
            left, bottom, right, top = rasterio.windows.bounds(block, src.transform)
            ind = (ptsx>=left) & (ptsx<right) & (ptsy>=bottom) & (ptsy<top)
            if ind.any():
                inv_block_transform = ~rasterio.windows.transform(block, src.transform)
                col, row = inv_block_transform * (ptsx[ind], ptsy[ind])
                blocks[ij]=[block, np.nonzero(ind)[0], col.astype(int), row.astype(int)]
        return blocks

    def find_blocks(self):
        # for every type of dimension find block for each point

        points_blocks = {}
        dimensions_set = set(self.dimensions)
        sources = [rasterio.open(fn_layer) for fn_layer in self.fn_layers]
        for dim in dimensions_set:
            src = sources[self.dimensions.index(dim)]
            points_blocks[dim] = self._find_blocks_for_src(src, self.points_x, self.points_y)
        return points_blocks

    def _sample_one_layer_sp(self, fn_layer):
        with rasterio.open(fn_layer) as src:
            #src=rasterio.open(fn_layer)
            dim = self._get_dimension(src)
            # out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])
            out_sample = np.full((self.points_len,), np.nan, np.float64)

            blocks = self.points_blocks[dim]
            for ij in blocks:
                # ij=next(iter(blocks)); (window, ind, col, row) = blocks[ij]
                (window, ind, col, row) = blocks[ij]
                data = src.read(1, window=window)
                mask = src.read_masks(1, window=window)
                sample = data[row,col].astype(np.float64)
                sample_mask = mask[row,col].astype(bool)
                sample[~sample_mask] = np.nan
                #sample = data[row.astype(int),col.astype(int)]
                out_sample[ind] = sample
        return out_sample, fn_layer

    def _sample_one_block(self, args):
        out_sample, fn_layer, window, ind, col, row = args
        with rasterio.open(fn_layer) as src:
            data = src.read(1, window=window)
            sample = data[row,col]
            out_sample[ind] = sample
        #return sample, ind

    def _sample_one_layer_mt(self, fn_layer):
        with rasterio.open(fn_layer) as src:
            dim = self._get_dimension(src)
            out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

        blocks = self.points_blocks[dim]
        args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for _ in executor.map(self._sample_one_block, args, chunksize=1000):
                pass #out_sample[ind] = sample
        return out_sample

    def _sample_one_layer_mp(self, fn_layer):
        with rasterio.open(fn_layer) as src:
            dim = self._get_dimension(src)
            out_sample = np.full((self.points_len,), src.nodata, src.dtypes[0])

        blocks = self.points_blocks[dim]
        args = ((out_sample, fn_layer, *blocks[ij]) for ij in blocks)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for _ in executor.map(self._sample_one_block, args, chunksize=1000):
                pass #out_sample[ind] = sample
        return out_sample

    def sample_v1(self):
        '''
        Serial sampling, block by block
        '''
        res={}
        n_layers = len(self.fn_layers)
        for i_layer, fn_layer in enumerate(self.fn_layers):
            ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
            # fn_layer=self.fn_layers[0]
            col = Path(fn_layer).with_suffix('').name
            sample,_ = self._sample_one_layer_sp(fn_layer)
            res[col]=sample
        return res

    def sample_v2(self):
        '''
        Serial layers, parallel blocks in layer
        '''
        res={}
        n_layers = len(self.fn_layers)
        for i_layer, fn_layer in enumerate(self.fn_layers):
            ttprint(f'{i_layer}/{n_layers} {Path(fn_layer).name}')
            # fn_layer=self.fn_layers[0]
            col = Path(fn_layer).with_suffix('').name
            sample = self._sample_one_layer_mt(fn_layer)
            res[col]=sample
        return res

    def sample_v3(self):
        '''
        Parallel layers in threads, serial blocks in layer
        '''
        res={}
        i_layer=1
        n_layers=len(self.fn_layers)
        args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)
        for sample, fn_layer in parallel.ThreadsGeneratorLazy(self._sample_one_layer_sp, args,
                            self.max_workers, self.max_workers*2):
            col = Path(fn_layer.with_suffix('').name)
            ttprint(f'{i_layer}/{n_layers} {col}')
            res[col] = sample
            i_layer += 1
        return res

    def sample_v4(self):
        '''
        Parallel layers in processes, serial blocks in layer
        '''
        res={}
        i_layer=1
        n_layers=len(self.fn_layers)
        args = ((fn_layer.as_posix(),) for fn_layer in self.fn_layers)
        #with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            #for sample, fn_layer in executor.map(self._sample_one_layer_sp, args, chunksize=n_layers//self.max_workers):
        for sample, fn_layer in parallel.ProcessGeneratorLazy(self._sample_one_layer_sp, args, self.max_workers, self.max_workers*2):
            col = Path(fn_layer).with_suffix('').name
            ttprint(f'{i_layer}/{n_layers} {col}')
            res[col] = sample
            i_layer += 1
        return res


#%%
'''
Have to collect outputs in memory because it's most efficient to do loop by image and then by blocks.
And outputs have to be by points amd then by images.
Only if we have all dimensions same ...
'''

def test_SamplePointsParallel():
    fn_points = '/data/wbsoilhr/HR_soil_nutritiens_30cm_3035.gpkg'
    fn_layers = list(Path(r'/data/wbsoilhr/mosaics').glob('*.tif'))
    fn_outfile = '/data/wbsoilhr/test_overlay.csv'

    max_workers = multiprocessing.cpu_count()


    pts = geopandas.read_file(fn_points)

    ptsx = pts.geometry.x.values
    ptsy = pts.geometry.y.values

    ttprint('Prepare ...')
    spp = SamplePointsParallel(ptsx, ptsy, fn_layers, fn_outfile, max_workers)
    ttprint('Sample ...')
    res = spp.sample_v4()
    ttprint('... Done.')

    for col in res:
        pts[col] = res[col]

    ttprint('Saving csv ...')
    pts.to_csv(fn_outfile, sep='\t', index=False)
    ttprint('... done.')


if __name__=='__main__':
    test_SamplePointsParallel()
