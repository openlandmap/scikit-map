import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

'''
Parallelization helpers based in thread and process pools
'''
def ThreadGeneratorLazy(worker, args, max_workers = CPU_COUNT, chunk = CPU_COUNT*2):
  import concurrent.futures
  from itertools import islice

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    group = islice(args, chunk)
    futures = {executor.submit(worker, *arg) for arg in group}

    while futures:
      done, futures = concurrent.futures.wait(
        futures, return_when=concurrent.futures.FIRST_COMPLETED
      )

      for future in done:
        yield future.result()

      group = islice(args, chunk)

      for arg in group:
        futures.add(executor.submit(worker,*arg))

def ProcessGeneratorLazy(worker, args, max_workers = CPU_COUNT, chunk = CPU_COUNT*2):
  import concurrent.futures
  from itertools import islice

  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    group = islice(args, chunk)
    futures = {executor.submit(worker, *arg) for arg in group}

    while futures:
      done, futures = concurrent.futures.wait(
        futures, return_when=concurrent.futures.FIRST_COMPLETED
      )

      for future in done:
        yield future.result()

      group = islice(args, chunk)

      for arg in group:
        futures.add(executor.submit(worker, *arg))

def job(worker, worker_args, n_jobs = -1, joblib_args = {}):
  from joblib import Parallel, delayed

  joblib_args['n_jobs'] = -1

  for worker_result in Parallel(**joblib_args)(delayed(worker)(*args) for args in worker_args):
    yield worker_result

# Source code based on https://stackoverflow.com/a/45555516
# Thanks Eric :)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
  import numpy as np

  def run(func1d, axis, arr, args, kwargs):
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

  """
  Like numpy.apply_along_axis(), but takes advantage of multiple
  cores.
  """        
  # Effective axis where apply_along_axis() will be applied by each
  # worker (any non-zero axis number would work, so as to allow the use
  # of `np.array_split()`, which is only done on axis 0):
  effective_axis = 1 if axis == 0 else axis
  if effective_axis != axis:
      arr = arr.swapaxes(axis, effective_axis)

  # Chunks for the mapping (only a few chunks):
  chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
            for sub_arr in np.array_split(arr, CPU_COUNT)]
  
  result = []
  for r in job(run, chunks):
    result.append(r)
    
  return np.concatenate(result)