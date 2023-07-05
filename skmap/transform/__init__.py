from skmap import SKMapBase
from abc import ABC, abstractmethod

class SKMapTransformer(SKMapBase, ABC):

  def __init__(self,
    verbose:bool = True
  ):
    self.verbose = verbose

  @abstractmethod
  def run(self, data):
    pass