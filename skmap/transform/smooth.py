'''
Gapfilling approaches using temporal and spatial neighbor pixels
'''
import time
import os

try:
    from typing import List, Dict, Union
    from abc import ABC, abstractmethod

    import numpy as np
    import scipy.sparse as sparse
    from scipy.sparse.linalg import splu

    from skmap.transform import SKMapTransformer
    from skmap import parallel

    class Smoother(SKMapTransformer, ABC):
      
      def __init__(self,
        verbose:bool = True
      ):
        self.verbose = verbose

      def run(self, data):
        """
        Execute the gapfilling approach.

        """

        start = time.time()
        smoothed = self._run(data)

        return smoothed

      @abstractmethod
      def _run(self, data):
        pass

    class Whittaker(Smoother):    
      """
      https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py
      """
      
      def __init__(self,
        lmbd = 1, 
        d = 2,
        n_jobs:int = os.cpu_count(),
        verbose = False
      ):

        super().__init__(verbose=verbose)

        self.lmbd = lmbd
        self.d = d
        self.n_jobs = n_jobs
      
      def _speyediff(self, N, d, format='csc'):
        """
        (utility function)
        Construct a d-th order sparse difference matrix based on 
        an initial N x N identity matrix
        
        Final matrix (N-d) x N
        """
        
        assert not (d < 0), "d must be non negative"
        shape     = (N-d, N)
        diagonals = np.zeros(2*d + 1)
        diagonals[d] = 1.
        for i in range(d):
          diff = diagonals[:-1] - diagonals[1:]
          diagonals = diff
        offsets = np.arange(d+1)
        spmat = sparse.diags(diagonals, offsets, shape, format=format)
        return spmat
      
      def _process_ts(self, data):
        y = data.reshape(-1).copy()
        n_gaps = np.sum((np.isnan(y)).astype('int'))
        
        if n_gaps == 0:
          r = splu(self.coefmat).solve(y)
          return r
        else:
          return y

      def _run(self, data):
        
        m = data.shape[-1]
        E = sparse.eye(m, format='csc')
        D = self._speyediff(m, self.d, format='csc')
        self.coefmat = E + self.lmbd * D.conj().T.dot(D)

        return parallel.apply_along_axis(self._process_ts, 2, data, n_jobs=self.n_jobs)

except ImportError as e:
    from skmap.misc import _warn_deps
    _warn_deps(e, 'gapfiller')
