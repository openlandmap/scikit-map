import pytest
import requests
from eumap.datasets import Catalogue
from owslib.wms import WebMapService

class TestCatalogue:
    use_csw = True

    def _lc(self, **kwargs):
        cat = Catalogue(use_csw=self.use_csw)
        return cat.search('land cover', exclude=['corine'], **kwargs)
        
    def test_001(self):
        """Get all records."""
        cat = Catalogue(use_csw=self.use_csw)
        results = cat.search('')

        assert len(results) > 0

    #def test_002(self):
    #    """Land Cover product is expected. Check basic metadata."""
    #    results = self._lc(years=[2015, 2018])
    #    
    #    assert len(results) > 0
    #
    #    # basic metadata
    #    for key in ('title', 'abstract', 'theme'):
    #        assert key in results[0].meta.keys()
    #    assert results[0].meta['theme'] == 'Land cover, land use and administrative data'

    def test_003(self):
        """Land Cover product is expected. Check download link."""
        results = self._lc(years=[2015, 2018])
        request = requests.head(results[0])
        
        assert request.status_code == 200

    def test_004(self):
        """Land Cover product is expected. Check WMS link."""
        results = self._lc(frmt=["OGC WMS"])

        wms = WebMapService(results[0])
        img = wms.getmap(
            layers=[results[0].meta.layer],
            size=[800, 600],
            srs="EPSG:3035",
            bbox=[900000.0, 930010.0, 6540000.0, 5460010.0],
            format="image/png"
        )

        assert len(img.read()) > 0
