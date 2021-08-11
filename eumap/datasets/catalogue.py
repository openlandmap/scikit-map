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
    GEONETWORK_URL = 'https://data.opendatascience.eu/geonetwork/srv/eng/csw?service=csw&version=2.0.2'
    KEYWORD_SEPARATORS = ' _-,;\t\n'

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
        split_keywords: bool=True,
        key: str='title',
    ):
        _args = args
        if split_keywords:
            _args = reduce(add, [
                re.split('|'.join(self.KEYWORD_SEPARATORS), arg)
                for arg in args
            ])

        if self.use_csw:
            _search = self._search_csw
        else:
            _search = self._search_fallback

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

        layers = []

        query = [PropertyIsEqualTo('csw:Subject', 'Geoharmonizer')]
        for arg in args:
            query.append(PropertyIsEqualTo(f'csw:{key}', arg))

        step = 10
        start_pos = 1
        getrecords(query, start_pos)
        while start_pos < self.csw.results['matches']:
            getrecords(query, start_pos)
            start_pos += step

            self._records(layers)

        results = []
        for layer in layers:
            for url in layer['urls']:
                results.append({
                    'url': url,
                    **layer,
                })

        return results

    def _records(self, layers):
        for rec in self.csw.records:
            items = {
                "title": self.csw.records[rec].identificationinfo[0].title,
                "abstract": self.csw.records[rec].identificationinfo[0].abstract,
                "urls": []
            }
            # get theme (first keyword)
            for i in self.csw.records[rec].identificationinfo[0].keywords2:
                if i.type == "theme":
                    items["theme"] = i.keywords[0]
                    break
            # get urls
            for ci in self.csw.records[rec].distribution.online:
                if ci.protocol == "WWW:DOWNLOAD-1.0-http--download":
                    items['urls'].append(ci.url)

            layers.append(items)

        return layers

    @property
    def themes(self):
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
