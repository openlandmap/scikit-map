'''
Access to the Geoharmonizer data product catalogue.
'''

import re
from functools import reduce
from operator import add

# hide future warnings (owslib)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import Iterable

def _match(*args, strategy=all):
    def wrapped(layer_name):
        return strategy((
            str(arg).lower() in layer_name.lower()
            for arg in args
        ))
    return wrapped

def _meta_repr(meta, **kwargs):
    _meta = {
        **meta,
        **kwargs,
    }
    key_width = max(map(len, _meta)) + 2
    return '\n'.join([
        (k+':').ljust(key_width) + str(v)
        for k, v in _meta.items()
    ])

class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        _self = _DotDict(self)
        if 'resources' in _self:
            _self.resources = [*map(str, _self.resources)]
        return _meta_repr(_self)

class _Resource(str):
    def __init__(self,
        content: str,
    ):
        super().__init__()

    def __repr__(self):
        return _meta_repr(self.meta, url=self.__str__())

    def set_meta(self, meta):
        # need to call this after calling constructor
        # because of str subclassing issues
        self.meta = _DotDict({
            k: v
            for k, v in meta.items()
            if k in ( # metadata cleanup
                'title',
                'abstract',
                'theme',
                'year',
                'layer',
                'authors',
            )
        })
        return self

class _ResourceCollection(list):
    def __init__(self,
        content: Iterable[_Resource],
    ):
        super().__init__(content)

        self.meta = []
        for resource in content:
            self._add_resource(resource)

    def _add_resource(self, resource):
        for metaset in self.meta:
            if resource.meta == metaset.resources[0].meta:
                metaset.resources.append(resource)
                return

        self.meta.append(_DotDict(resource.meta))
        self.meta[-1]['resources'] = [resource]

    def __repr__(self):
        return f'ResourceCollection of {len(self)} total assets:\n\n' + \
            '\n\n'.join([
                _meta_repr(metaset, resources=[*map(str, metaset.resources)], N_resources=len(metaset.resources))
                for metaset in self.meta
            ])

class Catalogue:
    """
    Quick access to all Geoharmonizer data products catalogued on the project's GeoNetwork server.

    Provides resources with metadata either live from GeoNetwork [1, 2] (with included access to the raw OWSLib API [3] through ``Catalogue.csw``) or from a local copy of the record included in the package (which might not always be up to date).

    :param use_csw: Indicates wheather to use live GeoNetwork access via CSW (the default) or to fall back to the local copy of the record (faster but might not be up to date).

    For full usage examples please refer to the catalogue tutorial notebook [4].

    Examples
    ========

    >>> from skmap.datasets import Catalogue
    >>>
    >>> cat = Catalogue()
    >>> results = cat.search('land-cover')
    >>> print(results)

    References
    ==========

    [1] `Geoharmonizer GeoNetwork server <https://data.opendatascience.eu>`_

    [2] `GeoNetwork <https://www.geonetwork-opensource.org/>`_

    [3] `OWSLib CSW API <https://geopython.github.io/OWSLib/usage.html#csw>`_

    [4] `Catalogue tutorial <../notebooks/07_catalogue.html>`_

    """

    GEONETWORK_URL = 'https://data.opendatascience.eu/geonetwork/srv/eng/csw?service=csw&version=2.0.2'
    """
    Geoharmonizer GeoNetwork CSW API endpoint.

    """

    KEYWORD_SEPARATORS = ' _-,;\t\n'
    """
    Default separators used to tokenize keywords.

    The ``Catalogue.search`` method splits both e.g. ``'land-cover'`` and ``'land cover'`` into ``['land', 'cover']`` and maches the tokens individually (if ``split_keywords=True``).

    """

    csw = None
    """
    Access to the raw `OWSLib CSW API <https://geopython.github.io/OWSLib/usage.html#csw>`_.

    If ``Catalogue`` is initialized with ``use_csw=True`` (the default) this becomes an ``owslib.csw.CatalogueServiceWeb`` instance, and is otherwise ``None``.

    """

    def __init__(self,
        use_csw: bool=True,
    ):
        self.use_csw = use_csw
        if use_csw:
            try:
                self._init_csw()
            except Exception as e:
                warnings.warn(f'Unable to connect to GeoNetwork: {e}')
                self.__init__(use_csw=False)
        else:
            self._init_fallback()

        self._themes = None
        self._ci_protocol = []

    def _init_csw(self):
        from owslib.csw import CatalogueServiceWeb

        self.csw = CatalogueServiceWeb(self.GEONETWORK_URL)

    def _init_fallback(self):
        import pandas as pd
        from io import BytesIO
        from base64 import b64decode
        from ._fallback import CATALOGUE_CSV

        self.layers = pd.read_csv(
            BytesIO(b64decode(CATALOGUE_CSV)),
            sep=',',
            compression='gzip',
        )

    def search(self,
        *args: Iterable[str],
        years: Iterable[int]=[],
        exclude: Iterable[str]=[],
        frmt: Iterable[str]=[],
        split_keywords: bool=True,
        key: str='title',
    ):
        """
        Search the catalogue for resources by matching metadata with keywords.

        Returns a list of resources with ``key`` matching ``*args``, from ``years``, and excluding those matchin keywords in ``exclude``.

        For full usage examples please refer to the `catalogue tutorial notebook <../notebooks/07_catalogue.html>`_.

        :param *args: Keywords to match.
        :param years: Years to match.
        :param exclude: Keywords to match for exclusion of resources.
        :param frmt: Formats to match (default: ["GeoTIFF"]). Supported formats: "GeoTIFF", "OGC WMS", "OGC WFS". Only supported by CSW implementation.
        :param split_keywords: Wheather or not to split keywords by ``Catalogue.KEYWORD_SEPARATORS`` and match tokens individually.
        :param key: Which metadata field to match with ``*args``. Can be ``'title'``, ``'theme'``, ``'abstract'``, ``'url'`` (case insensitive).

        :returns: List of resources (URL strings) with attached metadata (see `tutorial <../notebooks/07_catalogue.html>`_)
        :rtype: _ResourceCollection

        """

        _args = args
        if split_keywords:
            _args = reduce(add, [
                re.split('|'.join(self.KEYWORD_SEPARATORS), arg)
                for arg in args
            ])

        if self.use_csw:
            _search = self._search_csw
            if len(frmt) < 1:
                frmt = ["GeoTIFF"]
            self._ci_protocol = []
            for f in [*frmt]:
                if f == "GeoTIFF":
                    self._ci_protocol.append("WWW:DOWNLOAD-1.0-http--download")
                elif f == "OGC WMS":
                    self._ci_protocol.append("OGC:WMS")
                elif f == "OGC WFS":
                    self._ci_protocol.append("OGC:WFS")
                else:
                    warnings.warn(f'Format {f} is not supported by CSW implementation')
        else:
            _search = self._search_fallback
            if len(frmt) > 0:
                warnings.warn(f'Filtering by the format is only supported by CSW implementation')

        results = map(_DotDict, _search(
            *_args,
            key=key,
        ))

        exclude = [*exclude]
        if len(exclude) > 0:
            results = [
                res for res in results
                if not _match(*exclude, strategy=any)(res[key])
            ]

        years = sorted(years)
        if len(years) > 0:
            for year in years:
                for res in results:
                    if _match(str(year))(str(res.url)):
                        try:
                            res.year.append(year)
                        except AttributeError:
                            res['year'] = [year]
            results = sorted([
                res for res in results
                if 'year' in res
            ], key=lambda res: res.year[0])

        results = _ResourceCollection([
            _Resource(res.url).set_meta(res)
            for res in results
        ])
        return results

    def _search_csw(self,
        *args: Iterable[str],
        key: str='title',
        full: bool=True,
    ):
        from owslib.fes import PropertyIsEqualTo, And

        def getrecords(query, start_pos):
            self.csw.getrecords2(
                [And(query)] if len(query) > 1 else query,
                esn='full', typenames='gmd:MD_Metadata',
                outputschema="http://www.isotc211.org/2005/gmd",
                startposition=start_pos
            )

        def key2csw(key):
            return key if key == 'title' else 'AnyText'
            # TODO: add support for more keys
            # elif key == 'theme':
            #     return 'gmd:MD_Keywords'

        layers = []
        results = []

        if key not in ('title', 'theme', 'abstract'):
            warnings.warn(f'Key {key} is not supported by CSW implementation')
            return results

        query = [PropertyIsEqualTo('csw:Subject', 'Geoharmonizer')]
        for arg in args:
            query.append(PropertyIsEqualTo(f'csw:{key2csw(key)}', arg))

        step = 10
        start_pos = 1
        getrecords(query, start_pos)
        while start_pos <= self.csw.results['matches']:
            getrecords(query, start_pos)
            start_pos += step

            self._records(layers, limit=(key, args))

        for layer in layers:
            for url in layer['urls']:
                results.append({
                    'url': url,
                    **layer,
                })

        return results

    def _records(self, layers, limit):
        for rec in self.csw.records:
            items = {
                "title": self.csw.records[rec].identificationinfo[0].title,
                "abstract": self.csw.records[rec].identificationinfo[0].abstract,
                "urls": [],
                "authors": [*map(
                    lambda con: {
                        k: con.__dict__[k]
                        for k in (
                            'name',
                            'email',
                        )
                    },
                    self.csw.records[rec].contact,
                )],
            }
            # get theme (first keyword)
            for i in self.csw.records[rec].identificationinfo[0].keywords2:
                if i.type == "theme":
                    items["theme"] = i.keywords[0]
                    break
            # get urls
            for ci in self.csw.records[rec].distribution.online:
                if ci.protocol in self._ci_protocol:
                    if ci.protocol in ("OGC:WMS", "OGC:WFS"):
                        items['layer'] = ci.name
                    items['urls'].append(ci.url)

            if (limit[0] == 'theme' and items['theme'] not in limit[1]) or \
               (limit[0] == 'abstract' and limit[1][0].lower() not in items['abstract'].lower()):
                continue

            layers.append(items)

        return layers

    @property
    def themes(self):
        """
        Provides a list of all themes in the catalogue.

        Accessing for the first time might take up to several seconds if ``use_csw=True``.

        """

        if self._themes is None:
            results = self.search('')
            self._themes = [*set((
                m.theme
                for m in results.meta
            ))]
        return self._themes

    def _search_fallback(self,
        *args: Iterable[str],
        key: str='title',
    ):
        matches = self.layers[key].apply(_match(*args))
        results = self.layers[matches]

        return [res for __, res in results.iterrows()]
