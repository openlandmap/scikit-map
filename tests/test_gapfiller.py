from pathlib import Path
import numpy as np
import pytest

from skmap import gapfiller

DATA_SHAPE = (128, 128)
TIMESERIES_LEN = 80
THRESHOLD = .05

class TestGapfiller:
    data = np.random.uniform(size=(*DATA_SHAPE, TIMESERIES_LEN))

    def _test(self, gf_algorithm, **kwargs):
        """
        Generic gapfiller test

        Tests if the number of nodata pixels decreased after running the algorithm.
        """

        _data = self.data.copy()
        _data[_data<THRESHOLD] = np.nan
        n_missing = np.isnan(_data).sum()

        gf = gf_algorithm(data=_data, **kwargs)
        data_gapfilled = gf.run()
        assert np.isnan(data_gapfilled).sum() < n_missing

    def test_TMWM(self):
        """
        Test Temporal Moving Window Median gapfiller
        """

        self._test(gapfiller.TMWM, season_size=4)

    def test_TLI(self):
        """
        Test Temporal Linear Interpolation gapfiller
        """

        self._test(gapfiller.TLI)

    def test_InPainting(self):
        """
        Test InPainting gapfiller
        """

        self._test(gapfiller.InPainting)
