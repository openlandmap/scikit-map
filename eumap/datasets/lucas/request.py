'''
Build request for quering LUCAS dataset.
'''

from owslib.etree import etree
from owslib.fes import Or, And, PropertyIsEqualTo, BBox, PropertyIsNull, Not

from .exceptions import LucasRequestError

class LucasRequest:
    """Define a request.
    """
    bbox = None
    operator = None
    propertyname = None
    literal = None
    logical = None
    years = None
    aoi_polygon = None
    st_aggregated = False
    group = None

    gh_workspace = 'lucas'
    gh_typename = 'lucas_points'
    gh_typename_st = 'lucas_st_points'

    @property
    def typename(self):
        typename = self.gh_typename_st if self.st_aggregated else self.gh_typename
        if self.group is not None:
            typename += f'_{self.group}'
        return f'{self.gh_workspace}:{typename}'

    def __setattr__(self, name, value):
        if name not in (
                'bbox',
                'operator',
                'propertyname',
                'literal',
                'logical',
                'years',
                'aoi_polygon',
                'group',
                'st_aggregated'):
            raise LucasRequestError(f"{name} not allowed")

        super().__setattr__(name, value)

    def _check_args(self):
        """Check for required arguments.

        Raise LucasRequestError if requirements are not defined.
        """
        if self.bbox is not None and self.operator is not None:
            raise LucasRequestError("BBox and operator are mutually exclusive.")
        if self.bbox is not None and self.aoi_polygon is not None:
            raise LucasRequestError("BBox and aoi_polygon are mutually exclusive.")
        if self.operator is not None and self.aoi_polygon is not None:
            raise LucasRequestError("Operator and aoi_polygon are mutually exclusive.")

    def build(self):
        """
        Return a request arguments (bbox, filter) for the getfeature() method.
        
        :return dict:
        """
        self._check_args()

        req = {
            'typename': self.typename
        }

        # filter by years
        filter_years = []
        if self.years is not None:
            for value in self.years:
                if self.st_aggregated:
                    filter_years.append(
                        Not([PropertyIsNull(propertyname=f'survey_date_{str(value)}')])
                    )
                else:
                    filter_years.append(
                        PropertyIsEqualTo(propertyname='survey_year',
                                          literal=str(value))
                    )

        # filter by bbox
        if self.bbox is not None:
            bbox_query = BBox(self.bbox)
            # merge with year filter if defined
            if filter_years:
                if len(filter_years) > 1:
                    filter_ = And([
                        bbox_query,
                        Or(filter_years)
                    ])
                else:
                    filter_ = And([
                        bbox_query,
                        filter_years[0]
                    ])
            else:
                filter_ = bbox_query

            # TODO: workaround for owslib bug
            tree = filter_.toXML()
            namespaces = {'gml311': 'http://www.opengis.net/gml'}
            envelopes = tree.findall('gml311:Envelope', namespaces)
            for envelope in envelopes:
                envelope.set('srsName', 'http://www.opengis.net/gml/srs/epsg.xml#3035')

            filter_xml = etree.tostring(tree).decode("utf-8")
            # TODO maybe change this in XML way as well
            filter_xml = filter_xml.replace('ows:BoundingBox', 'geom')

            req.update({'filter': filter_xml})

        #
        # OR
        #
        # filter by polygon
        elif self.aoi_polygon is not None:

            # merge with year filter if defined
            if filter_years:
                if len(filter_years) > 1:
                    filter_ = Or(filter_years)
                else:
                    filter_ = filter_years[0]
                filter_xml = etree.tostring(filter_.toXML()).decode("utf-8")
                filter_xml = '<ogc:And xmlns:ogc="http://www.opengis.net/ogc">' + filter_xml + self.aoi_polygon + '</ogc:And>'
            else:
                filter_xml = self.aoi_polygon

            req.update({'filter': filter_xml})

        #
        # OR
        #
        # filter by property
        elif self.operator is not None:
            # check requirements
            if self.propertyname is None:
                raise LucasRequestError(
                    "Property name is not defined"
                )
            if self.literal is None:
                raise LucasRequestError(
                    "Literal property is not defined"
                )

            # collect list of literals
            if isinstance(self.literal, str):
                literal = [self.literal]
            elif isinstance(self.literal, list):
                literal = self.literal
            else:
                raise LucasRequestError(
                    "Literal property must be provided as a string or a list."
                )

            # check if logical operator is required
            if len(literal) > 1 and self.logical is None:
                raise LucasRequestError(
                    "Logical property is not defined."
                )

            # get list of property filters
            filter_list = []
            for value in literal:
                if self.propertyname == 'nuts0' or self.st_aggregated is False:
                    filter_list.append(
                        self.operator(propertyname=self.propertyname, literal=value)
                    )
                else:
                    st_filter = []
                    for year in ['_2006', '_2009', '_2012', '_2015', '_2018']:
                        st_filter.append(
                            self.operator(propertyname=self.propertyname + year, literal=value)
                        )
                    filter_list.append(Or(st_filter))


            # get combined property filter
            if len(literal) > 1:
                filter_property = self.logical(filter_list)
            else:
                filter_property = filter_list[0]

            # merge with year filter if defined
            if filter_years:
                if len(filter_years) > 1:
                    filter_ = And([
                        filter_property,
                        Or(filter_years)
                    ])
                else:
                    filter_ = And([
                        filter_property,
                        filter_years[0]
                    ])
            else:
                filter_ = filter_property

            req.update({
                'filter': etree.tostring(filter_.toXML()).decode("utf-8")
            })

        return req
