'''
Miscellaneous utils
'''

from typing import List, Dict, Union, Iterable
from datetime import datetime, timedelta
from functools import reduce

import rasterio
import geopandas as gp
import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta

def _warn_deps(e, module_name):
    import warnings
    warnings.warn(
        f'ERROR: {e}\n\n' \
        f'Encountered because {module_name} has additional dependencies, please try:\n\n' \
        '\tpip install skmap[full]\n'
    )

def ttprint(*args, **kwargs):
  """
  A print function that displays the date and time.

  Examples
  ========

  >>> from skmap.misc import ttprint
  >>> ttprint('skmap rocks!')

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

  >>> from skmap.misc import find_files
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
  >>> from skmap.misc import nan_percentile
  >>>
  >>> data = np.random.rand(10, 10, 10)
  >>> data[2:5,0:10,0] = np.nan
  >>> data_perc = nan_percentile(data, q=[25, 50, 75])
  >>> print(f'Shape: data={data.shape} data_perc={data_perc.shape}')

  References
  ==========

  [1] `Kersten's blog <https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way>`_

  """
  # loop over requested quantiles
  if type(q) is list:
    qs = []
    qs.extend(q)
  else:
      qs = [q]
  # eliminate duplicate percentile
  qs = list(set(qs))
  if len(qs) != len(q):
    print('duplicate percentile is eliminated')
  if keep_original_vals:
    arr = np.copy(arr)
  nanall = np.all(np.isnan(arr), axis=0)
  single_value = ~(np.sum(~np.isnan(arr),axis=0)-1).astype(bool)
  res_shape = (len(qs), arr.shape[1], arr.shape[2])
  nanall = np.broadcast_to(nanall, shape=res_shape)
  # valid (non NaN) observations along the first axis
  valid_obs = np.sum(np.isfinite(arr), axis=0)
  # replace NaN with maximum
  max_val = np.nanmax(arr)
  arr[np.isnan(arr)] = max_val
  # sort - former NaNs will move to the end
  arr = np.sort(arr, axis=0)

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
  result = np.stack(result, axis=0)
  result[nanall] = np.nan

  md = [i==50 for i in qs]
  if sum(md)==1:
    md_value = np.copy(result[md]) 
    result[:,single_value] = np.nan
    result[md] = md_value
  else: 
    result[:,single_value] = np.nan
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
  idx = nC*nR*ind + np.arange(nC*nR).reshape((nC,nR))
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
    >>> from skmap.misc import sample_groups
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

def _eval(str_val, args):
  return eval("f'"+str_val+"'", args)

def _date_step( 
  date_step, 
  i = 0
):
  if isinstance(date_step, list):
    if i >= len(date_step):
      i = 0
    date_step_cur = int(date_step[i])
    i += 1
    return i, date_step_cur
  else:
    return i, int(date_step)

def gen_dates(
  start_date:str, 
  end_date:str, 
  date_unit:str, 
  date_step:int,
  date_offset:int = 0,
  date_format = '%Y-%m-%d',
  ignore_29feb = False,
  return_str = False
):
  
  start_date = datetime.strptime(start_date, date_format)
  end_date = datetime.strptime(end_date, date_format)

  result = []

  dt1 = start_date
  date_step_i = 0

  watchdog = 0

  while(dt1 <= end_date):
    delta_args = {}
    date_step_i, date_step_cur = _date_step(date_step, date_step_i)
    delta_args[date_unit] = date_step_cur # TODO: Threat the value "month"
    
    dt1n = dt1 + relativedelta(**delta_args)
    dt_feb = (datetime.strptime(f'{dt1n.year}0228', '%Y%m%d'))

    ## FIXME: bug on ignore_29feb=True ending in feb.
    if (dt_feb > dt1 and dt_feb <= dt1n and ignore_29feb):
      dt1n = dt1n + relativedelta(leapdays=+1)

    dt2 = dt1n + relativedelta(days=-1)
  
    if ignore_29feb:
      if dt2.month == 2 and dt2.day == 29:
        dt2 = dt2 + relativedelta(days=-1)
    
    if return_str:
      result.append((dt1.strftime(date_format), dt2.strftime(date_format)))
    else:
      result.append((dt1, dt2))
    
    if date_offset > 0:
      delta_args = {}
      delta_args[date_unit] = date_offset
      dt1n = dt1n + relativedelta(**delta_args)

    dt1 = dt1n

    if watchdog >= 365000:
      raise Exception('Infinite loop avoided.'
        + ' Check the date arguments.'
      ) 

    watchdog += 1
  
  return result

def update_by_separator(
  text:str, 
  separator:str, 
  position:int, 
  new_text:str, 
  suffix=False
):
  split = text.split(separator)
  if suffix:
    new_text = split[position] + new_text
  split[position] = new_text
  return separator.join(split)

try:
  
  import gspread
  import pytz

  class GoogleSheet():
    """
    Utility class able to convert a remote Google Spreadsheet file into a pandas.DataFrame.
    Each sheet is converted to a separate pandas.DataFrame accessible by class attribute.

    :param key_file: Authentication key to access spreadsheets via Google Sheets API
    :param url: Complete URL referring to a Google Spreadsheet file (public accessible).
    :param col_list_suffix: All the columns with this suffix are converted to a list of strings.
    :param col_list_delim: Text delimiter used to separate the list elements.
    :param col_date_suffix: All the columns with this suffix are converted to a date object.
    :param col_date_format: Date format used to convert string values in date object.
    :param verbose: Use ``True`` to print the progress of all steps.

    Examples
    ========

    >>> # Generate your key follow the instructions in https://docs.gspread.org/en/latest/oauth2.html
    >>> key_file = '<GDRIVE_KEY>'
    >>> # Public accessible Google Spreadsheet (Anyone on the internet with this link can view)
    >>> url = 'https://docs.google.com/spreadsheets/d/1O3n5O6MQ3OPX--ZbJEREC5fu-bLKK2AaTYDqeAPMRQY/edit?usp=sharing'
    >>> 
    >>> gsheet = GoogleSheet(key_file, url)
    >>> print('Sheet points_nl: ', gsheet.points_nl.shape)
    >>> print('Sheet tiles: ', gsheet.tiles.shape)

    References
    ==========

      [1] `Authentication - gspread <https://docs.gspread.org/en/latest/oauth2.html>`_


    """

    def __init__(self,
      key_file:str,
      url:str,
      col_list_suffix:str = '_list',
      col_list_delim:str = ',',
      col_date_suffix:str = '_date',
      col_date_format:str = '%Y-%m-%d',
      verbose:bool = False,
    ):

      self.key_file = key_file
      self.url = url
      self.verbose = verbose
      
      self.col_list_suffix = col_list_suffix
      self.col_list_delim = col_list_delim
      self.col_date_suffix = col_date_suffix
      self.col_date_format = col_date_format

      self._read_gsheet()

    def _verbose(self, *args, **kwargs):
      if self.verbose:
        ttprint(*args, **kwargs)

    def _read_gsheet(self):

      gc = gspread.service_account(filename=self.key_file)
      self._verbose(f"Accessing {self.url}")
      sht = gc.open_by_url(self.url)

      for wsht in sht.worksheets():
        self._verbose(f"Retrieving the data from {wsht.title}")
        rows = wsht.get_values()
        title = wsht.title

        try:
          setattr(self, title, self._parse_df(rows))
        except:
          ttprint(f'ERROR: unable to parse {title}, ignoring it.')

    def _parse_df(self, rows):

      pytz.timezone("UTC")

      df = pd.DataFrame(rows[1:], columns=rows[0])
      to_drop = []

      for column in df.columns:
        
        self._verbose(f" Parsing column {column}")

        if column.endswith(self.col_list_suffix):
          new_column = column.replace(self.col_list_suffix, '')
          df[new_column] = df[column].str.split(self.col_list_delim)
          to_drop.append(column)

        if column.endswith(self.col_date_suffix):
          df[column] = pd.to_datetime(df[column], 
            format=self.col_date_format, errors='coerce')

      return df.drop(columns=to_drop)

except ImportError as e:
  from .misc import _warn_deps
  _warn_deps(e, 'misc.GoogleSheet')
