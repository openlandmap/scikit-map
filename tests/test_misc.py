import pytest
from skmap.misc import date_range
from datetime import datetime

class TestDataRange:

  def pluck_year(self, dates, sep = '-'):
    
    result = []
    
    for dt1, dt2 in dates:
      if isinstance(dt1, datetime):
        result.append(f'{dt1.month}{dt1.day}{dt2.month}{dt2.day}')
      else:
        result.append(f'{dt1[4:]}{dt2[4:]}')
    
    return sep.join(result)

  def test_001(self):
    assert self.pluck_year(
      date_range('2013-01-01', '2016-01-01', 'months', 1, 
        ignore_29feb=True)
    ) == self.pluck_year(
      date_range('2016-01-01', '2019-01-01', 'months', 1, 
        ignore_29feb=True)
    )

  def test_002(self):
    assert self.pluck_year(
      date_range('2013-01-01', '2016-01-01', 'months', 1, 
        ignore_29feb=True)
    ) != self.pluck_year(
      date_range('2016-01-01', '2019-01-01', 'months', 1,
       ignore_29feb=False)
    )

  def test_003(self):
    assert self.pluck_year(
      date_range('2013001', '2016001', 'months', 1, 
        date_format="%Y%j", ignore_29feb=True)
    ) == self.pluck_year(
      date_range('2016001', '2019001', 'months', 1, 
        date_format="%Y%j", ignore_29feb=True)
    )

  def test_004(self):
    assert self.pluck_year(
      date_range('2013001', '2016001', 'months', 1, 
        date_format="%Y%j", ignore_29feb=True)
    ) != self.pluck_year(
      date_range('2016001', '2019001', 'months', 1, 
        date_format="%Y%j", ignore_29feb=False)
    )

  def test_005(self):
    date_step = ([16] * 22) + [13]
    assert self.pluck_year(
      date_range('2013001', '2016001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True)
    ) == self.pluck_year(
      date_range('2016001', '2019001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True)
    )

  def test_006(self):
    date_step = ([16] * 22) + [13]
    assert self.pluck_year(
      date_range('2013001', '2016001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True)
    ) != self.pluck_year(
      date_range('2016001', '2019001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=False)
    )

  def test_007(self):
    date_step = ([16] * 22) + [13]
    assert self.pluck_year(
      date_range('2013001', '2016001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True, return_str=True)
    ) == self.pluck_year(
      date_range('2016001', '2019001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True, return_str=True)
    )

  def test_008(self):
    date_step = ([16] * 22) + [13]
    assert self.pluck_year(
      date_range('2013001', '2016001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=True, return_str=True)
    ) != self.pluck_year(
      date_range('2016001', '2019001', 'days', date_step, 
        date_format="%Y%j", ignore_29feb=False, return_str=True)
    )