'''
Miscellaneous utils
'''

from typing import List, Dict, Union, Iterable
from datetime import datetime, timedelta
from functools import reduce

import rasterio
import geopandas as gp
import numpy as np
from pathlib import Path

def _warn_deps(e, module_name):
    import warnings
    warnings.warn(
        f'ERROR: {e}\n\n' \
        f'Encountered because {module_name} has additional dependencies, please try:\n\n' \
        '\tpip install eumap[full]\n'
    )

def ttprint(*args, **kwargs):
  """
  A print function that displays the date and time.

  Examples
  ========

  >>> from eumap.misc import ttprint
  >>> ttprint('eumap rocks!')

  """
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)


def find_files(
  dir_list:List,
  pattern:str = '*.*'
):
  """
  Recursively find files in multiple directories according to the
  specified pattern. It's basically a wrapper for
  glob module [1]

  :param dir_list: List with multiple directory paths.
  :param pattern: Pattern to match with the desired files.

  Examples
  ========

  >>> from eumap.misc import find_files
  >>> libs_so = find_files(['/lib', '/usr/lib64/'], f'*.so')
  >>> print(f'{len(libs_so)} files found')

  References
  ==========

  [1] `Python glob module <https://docs.python.org/3/library/glob.html>`_

  """
  files = []

  if not isinstance(dir_list, list):
    dir_list = [dir_list]

  glob_pattern = f'**/{pattern}'

  for _dir in dir_list:
    for file in list(Path(_dir).glob(glob_pattern)):
      files.append(Path(file))

  files = sorted(files)

  return files

def nan_percentile(
  arr:np.array,
  q:List = [25, 50, 75],
  keep_original_vals=False
):
  """
  Optimized function to calculate percentiles ignoring ``np.nan``
  in a 3D Numpy array [1].

  :param arr: 3D Numpy array where the first dimension is used to
    derive the percentiles.
  :param q: Percentiles values between 0 and 100.
  :param keep_original_vals: If ``True`` it does a copy of ``arr``
    to preserve the structure and values.

  Examples
  ========

  >>> import numpy as np
  >>> from eumap.misc import nan_percentile
  >>>
  >>> data = np.random.rand(10, 10, 10)
  >>> data[2:5,0:10,0] = np.nan
  >>> data_perc = nan_percentile(data, q=[25, 50, 75])
  >>> print(f'Shape: data={data.shape} data_perc={data_perc.shape}')

  References
  ==========

  [1] `Kersten's blog <https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way>`_

  """

  if keep_original_vals:
    arr = np.copy(arr)

  nanall = np.all(np.isnan(arr), axis=0)
  res_shape = (len(q), arr.shape[1], arr.shape[2])
  nanall = np.broadcast_to(nanall, shape=res_shape)

  # valid (non NaN) observations along the first axis
  valid_obs = np.sum(np.isfinite(arr), axis=0)

  # replace NaN with maximum
  max_val = np.nanmax(arr)
  arr[np.isnan(arr)] = max_val

  # sort - former NaNs will move to the end
  arr = np.sort(arr, axis=0)

  # loop over requested quantiles
  if type(q) is list:
    qs = []
    qs.extend(q)
  else:
    qs = [q]
  if len(qs) <= 2:
    quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
  else:
    quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

  result = []
  for i in range(len(qs)):
    quant = qs[i]
    # desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * (quant / 100.0)
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    fc_equal_k_mask = f_arr == c_arr

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalueFromIndex(arr=arr, ind=f_arr) * (c_arr - k_arr)
    ceil_val = _zvalueFromIndex(arr=arr, ind=c_arr) * (k_arr - f_arr)

    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalueFromIndex(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

    result.append(quant_arr)

  result = np.stack(result, axis=0).astype('float16')
  result[nanall] = np.nan

  return result

def _zvalueFromIndex(arr, ind):
  """private helper function to work around the limitation of np.choose() by employing np.take()
  arr has to be a 3D array
  ind has to be a 2D array containing values for z-indicies to take from arr
  See: http://stackoverflow.com/a/32091712/4169585
  This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
  """
  # get number of columns and rows
  _,nC,nR = arr.shape

  # get linear indices and extract elements with np.take()
  idx = nC*nR*ind + nR*np.arange(nR)[:,None] + np.arange(nC)
  return np.take(arr, idx)

def _stringify(arr):
    if isinstance(arr, np.ndarray):
        try:
            arr = arr.astype(int)
        except ValueError:
            pass
        arr = arr.astype(str)
    return arr

def _add_group_elements(a1, a2):
    return np.core.defchararray.add(a1, _stringify(a2))

def sample_groups(
    points: gp.GeoDataFrame,
    *group_element_columns: Iterable[str],
    spatial_resolution: Union[int, float]=None,
    temporal_resolution: timedelta=None,
    date_column: str='date',
) -> np.ndarray:
    """
    Construct group IDs for spatial and temporal cross-validation.

    Groups point samples into tiles of `spatial_resolution` width and height
    and/or intervals of `temporal_resolution` size.
    `group_element_columns` are also concatenated into the final group ID of each sample.

    :param points: GeoDataFrame containing point samples.
    :param *group_element_columns: Names of additional columns to be concatenated into the final group IDs.
    :param spatial_resolution: Tile size (both x and y) for grouping, in sample CRS units.
    :param temporal_resolution: Interval size for grouping.
    :param date_column: Name of the column containing sample timestamps (as datetime objects).

    :returns: 1D string array containing the group id of each sample.

    Examples
    ========

    >>> import geopandas as gp
    >>> import pygeos as pg
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score, GroupKFold
    >>>
    >>> from eumap.misc import sample_groups
    >>>
    >>> # construct some synthetic point data
    >>> coords = np.random.random((1000, 2)) * 4000
    >>> dates = datetime.now() + np.array([*map(
    >>>     timedelta,
    >>>     range(1000),
    >>> )])
    >>>
    >>> points = gp.GeoDataFrame({
    >>>     'geometry': pg.points(coords),
    >>>     'date': dates,
    >>>     'group': np.random.choice(['a', 'b'], size=1000),
    >>>     'predictor': np.random.random(1000),
    >>>     'target': np.random.randint(2, size=1000),
    >>> })
    >>>
    >>> # get the point groups
    >>> groups = sample_groups(
    >>>     points,
    >>>     'group',
    >>>     spatial_resolution=1000,
    >>>     temporal_resolution=timedelta(days=365),
    >>> )
    >>>
    >>> print(np.unique(groups))
    >>>
    >>> kfold = GroupKFold(n_splits=5)
    >>>
    >>> # cross validate a classifier
    >>> print(cross_val_score(
    >>>     estimator=LogisticRegression(),
    >>>     X=points.predictor.values.reshape(-1, 1),
    >>>     y=points.target,
    >>>     scoring='f1',
    >>>     groups=groups, # our groups go here
    >>> ))
    """

    group_elements = [
        points[col].values.astype(str)
        for col in group_element_columns
    ]

    if all((
        not group_elements,
        spatial_resolution is None,
        temporal_resolution is None,
    )):
        raise ValueError(
            'no group elemements, specify one or more of the following:\n' \
            '\tgroup_element_columns\n' \
            '\tspatial_resolution\n' \
            '\ttemporal_resolution\n' \
        )

    if spatial_resolution is not None:

        x_el = points.geometry.x.values // spatial_resolution
        x_el -= x_el.min()
        group_elements += ['x', x_el]

        y_el = points.geometry.y.values // spatial_resolution
        y_el -= y_el.min()
        group_elements += ['y', y_el]

    if temporal_resolution is not None:

        _tres = np.timedelta64(temporal_resolution)
        times = points[date_column].values
        t_el = (times - times.min()) // _tres
        group_elements += ['t', t_el]

    return reduce(
        _add_group_elements,
        group_elements,
    )
