__version__ = '0.7.0'

from abc import ABC, abstractmethod
from skmap.misc import ttprint

class SKMapBase(ABC):

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)

class SKMapRunner(SKMapBase, ABC):

  def __init__(self,
    verbose:bool = True,
    temporal:bool = False
  ):
    self.verbose = verbose

  @abstractmethod
  def run(self, 
    data, 
    outname:str
  ):
    pass

class SKMapGroupRunner(SKMapRunner, ABC):

  def __init__(self,
    verbose:bool = True,
    temporal:bool = False
  ):
    self.verbose = verbose
    self.temporal = temporal
