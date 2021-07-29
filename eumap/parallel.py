'''
Parallelization helpers based in thread and process pools
'''

def ThreadGeneratorLazy(worker, args, max_workers, chunk):
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

def ProcessGeneratorLazy(worker, args, max_workers, chunk):
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

def unpacking_apply_along_axis(all_args):
  import numpy as np
  
  (func1d, axis, arr, args, kwargs) = all_args
  return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

# Source code from https://stackoverflow.com/a/45555516
# Thanks Eric :)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
  import multiprocessing
  import numpy as np

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
            for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]
  
  
  pool = multiprocessing.Pool()
  individual_results = pool.map(unpacking_apply_along_axis, chunks)
  
  # Freeing the workers:
  pool.close()
  pool.join()

  return np.concatenate(individual_results)