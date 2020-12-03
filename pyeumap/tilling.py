from pyeumap import datasets
from pyeumap.parallel import ThreadGeneratorLazy, ProcessGeneratorLazy
import geopandas as gpd
import multiprocessing
from rasterio.windows import Window
import os.path

from .misc import ttprint

EUMAP_TILLING_SYSTEM_FN = 'eu_tilling system_30km.gpkg'

class TillingProcessing():
  
  def __init__(self, tilling_system_fn = None, 
         col_xoff = None,
         col_yoff = None,
         xsize = 1000,
         ysize = 1000,
         verbose:bool = True):
    
    if tilling_system_fn is None:
      
      if verbose:
        ttprint('Using default eumap tilling system')

      tilling_system_fn = f'eumap_data/{EUMAP_TILLING_SYSTEM_FN}'
      if not os.path.isfile(tilling_system_fn):
        datasets.get_data(EUMAP_TILLING_SYSTEM_FN)
        
      col_xoff = 'offst_x'
      col_yoff = 'offst_y'

    self.col_xoff = col_xoff
    self.col_yoff = col_yoff
    self.xsize = xsize
    self.ysize = ysize
    
    self.tiles = gpd.read_file(tilling_system_fn)
    self.num_tiles = self.tiles.shape[0]

    if verbose:
      ttprint(f'{self.num_tiles} tiles available')
      
  def process_one(self, idx, func, func_args = ()):
    
    tile = self.tiles.iloc[idx]
    xoff = tile[self.col_xoff]
    yoff = tile[self.col_yoff]
    
    window = Window(xoff, yoff, self.xsize, self.ysize)
        
    return func(tile, window, *func_args)
  
  def process_multiple(self, idx_list, func, func_args = (), 
    max_workers:int = multiprocessing.cpu_count(),
    use_threads:bool = True):
    
    args = []
    for idx in idx_list:
      tile = self.tiles.iloc[idx]
      
      xoff = tile[self.col_xoff]
      yoff = tile[self.col_yoff]
      
      window = Window(xoff, yoff, self.xsize, self.ysize)
      
      args.append((tile, window))
    
    WorkerPool = (ThreadGeneratorLazy if use_threads else ProcessGeneratorLazy)

    results = []
    for r in WorkerPool(func, iter(args), fixed_args=func_args, max_workers=max_workers, chunk=max_workers*2):
      results.append(r)

    return results

  def process_all(self, func, func_args = (), 
    max_workers:int = multiprocessing.cpu_count(),
    use_threads:bool = True):
  
    idx_list = range(0, self.num_tiles)
    return self.process_multiple(idx_list, func, func_args, max_workers, use_threads)

class MultidimensionalData(TillingProcessing):

  def __init__(self, tilling_system_fn = None, 
         col_xoff = None,
         col_yoff = None,
         xsize = 1000,
         ysize = 1000,
         verbose:bool = True):

        super().__init__(tilling_system_fn=tilling_system_fn, col_xoff=col_xoff, col_yoff=col_yoff, 
          xsize=xsize, ysize=ysize, verbose=verbose)