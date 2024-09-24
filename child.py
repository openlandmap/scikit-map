import os
import numpy as np
import sys


def child_process(rank, tile_files, modis_mosaics, n_threads, n_pix, resample):
    os.environ['USE_PYGEOS'] = '0'
    os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
    os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
    os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
    os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
    os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
    os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}'
    os.environ["MKL_NUM_THREADS"] = f'{n_threads}'
    os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'
    import skmap_bindings
    gdal_opts = {
     'GDAL_HTTP_VERSION': '1.0',
     'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    }
    
    warp_data = np.empty((n_pix,), dtype=np.float32)
    skmap_bindings.warpTile(warp_data, n_threads, gdal_opts, tile_files[rank], modis_mosaics[rank], resample)
    
    return warp_data

def main():
    os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
    from mpi4py import MPI
    
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    
    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = f.read().strip().split('\n')
    tile_files = args[0].split(',')
    modis_mosaics = args[1].split(',')
    n_threads = int(args[2])
    n_pix = int(args[3])
    resample = args[4]
        
    
    # tile_files = sys.argv[1].split(',')
    # modis_mosaics = sys.argv[2].split(',')
    # n_threads = int(sys.argv[3])
    # n_pix = int(sys.argv[4])
    # resample = sys.argv[5]
    array = child_process(rank, tile_files, modis_mosaics, n_threads, n_pix, resample)
    comm.Send(array, dest=0, tag=rank)
    comm.Disconnect()

if __name__ == "__main__":
    main()