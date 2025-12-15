import os

from pyproj import datadir

n_threads = os.cpu_count()
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
os.environ["PROJ_LIB"] = datadir.get_data_dir()

os.environ["NUMEXPR_MAX_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"
os.environ["OMP_THREAD_LIMIT"] = f"{n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"

os.environ["OMP_DYNAMIC"] = "TRUE"

os.environ["TREELITE_BIND_THREADS"] = "0"
