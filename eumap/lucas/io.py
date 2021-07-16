import os
import logging
import tempfile
from pathlib import Path
from shutil import copy
from requests.exceptions import HTTPError, ConnectionError, InvalidSchema, InvalidURL, MissingSchema
from osgeo import gdal, ogr

gdal.UseExceptions()

from owslib.wfs import WebFeatureService
from owslib.util import ServiceException
from owslib import __version__ as owslib_version

from .logger import Logger
from .exceptions import LucasDownloadError, LucasDataError

class LucasIO:
    """
    LUCAS features input / output class.

    :param str url: WFS endpoint
    :param str version: WFS version to be used
    """

    def __init__(self, url='http://lincalc-02.fsv.cvut.cz:8080/geoserver/wfs',
                 version='1.1.0'):
        Logger.info(f"Using owslib version {owslib_version}")
        self._wfs_url = url
        self._wfs_version = version
        self._years = None
        self._st_aggregated = None

        self._gml = None

    @staticmethod
    def __get_tempfile_name(extension):
        return os.path.join(tempfile.gettempdir(),
                            "eumap_lucas_{n}.{e}".format(
                                n=next(tempfile._get_candidate_names()),
                                e=extension)
                            )

    def download(self, request):
        """
        Download LUCAS features from dedicated WFS server based on
        specified request :class:`.request.LucasRequest`.

        :param LucasRequest: request
        """
        try:
            wfs = WebFeatureService(url=self._wfs_url, version=self._wfs_version)
        except (HTTPError, ConnectionError, InvalidSchema, InvalidURL, MissingSchema, AttributeError, UnicodeError) as e:
            raise LucasDownloadError(f"Cannot connect to server: {e}")


        Logger.info(f"Connected to {self._wfs_url}")

        # collect getfeature() arguments
        args = {
            'srsname': "http://www.opengis.net/gml/srs/epsg.xml#3035"
        }
        args.update(request.build())
        Logger.info(f"Request: {args}")

        try:
            response = wfs.getfeature(**args)
        except (ServiceException, AttributeError, TypeError) as e:
            raise LucasDownloadError(f"Unable to get features from WFS server: {e}")


        gml = response.read()
        Logger.info(
            "Download process successfuly finished. Size of downloaded data: {}kb".format(
                int(len(gml) / 1000)
            ))

        self._years = request.years
        self._st_aggregated = request.st_aggregated

        self._path = self._load(gml)
        self._postprocessing(self._path)

    def _load(self, gml):
        """Load features from GML string and creates temporary GPKG file.

        :param str gml: GML string to be loaded
        
        :return str: path to temporary GPKG file
        """

        # 1. store GML string into temporary file
        path_gml = self.__get_tempfile_name("gml")
        with open(path_gml, 'wb') as f:
            f.write(gml)

        # 2. convert GML to GPKG
        path_gpkg = self.__get_tempfile_name("gpkg")
        try:
            gdal.VectorTranslate(path_gpkg, path_gml)
        except RuntimeError as e:
            raise LucasDataError(f"Unable to translate into GPKG: {e}")

        return path_gpkg

    def _postprocessing(self, path):
        """Delete gml_id column. If data are space time aggregated, delete columns which aren't required

        :param str path: path to GPKG file
        """

        try:
            ds = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_UPDATE)
            layer = ds.GetLayer()
            defn = layer.GetLayerDefn()
            layer.DeleteField(defn.GetFieldIndex('gml_id'))
            if self._st_aggregated and self._years is not None:
                # TBD: to be reviewed
                driver = ogr.GetDriverByName("GPKG")
                gpkg = driver.Open(path)
                layer1 = gpkg.GetLayer()
                layer_definition = layer1.GetLayerDefn()
                for i in range(layer_definition.GetFieldCount()):
                    attr = layer_definition.GetFieldDefn(i).GetName()
                    if attr not in ['point_id', 'nuts0', 'geom'] and int(attr[-4:]) not in self._years:
                        layer.DeleteField(defn.GetFieldIndex(attr))
            del ds
        except RuntimeError as e:
            raise LucasDataError(f"Postprocessing failed: {e}")

    def __check_data(self):
        """Check whether LUCAS features are downloaded.

        Raise LucasDownloadError of failure
        """
        if self._path is None:
            raise LucasDownloadError("No LUCAS features downloaded")

    def to_gml(self):
        """Get downloaded LUCAS features as `OGC GML
        <https://www.ogc.org/standards/gml>`__ string.

        :return str: GML string
        """
        self.__check_data()

        path_gml = self.__get_tempfile_name("gml")

        try:
            gdal.VectorTranslate(path_gml, self._path)
        except RuntimeError as e:
            raise LucasDataError(f"Unable to translate into GML: {e}")

        with open(path_gml) as fd:
            data = fd.read()

        return data

    def to_gpkg(self, output_path):
        """Save downloaded LUCAS features into `OGC GeoPackage
        <https://www.ogc.org/standards/gml>`__ file.

        Raises LucasDataError on failure.

        :param str output_path: path to the output OGC GeoPackage file
        """
        self.__check_data()

        out_path = Path(output_path)

        # Delete file if exists
        if out_path.exists():
            try:
                out_path.unlink()
            except PermissionError as e:
                raise LucasDataError(f"Unable to overwrite existing file: {e}")

        # Copy file from temporary directory
        try:
            copy(self._path, out_path.parent)
        except PermissionError as e:
            raise LucasDataError(f"Permission denied: {e}")

        # Rename file
        path = Path(out_path.parent) / Path(self._path).name
        path.rename(out_path)

    def to_geopandas(self):
        """Get downloaded LUCAS features as GeoPandas `GeoDataFrame
        <https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.html>`__
        structure.

        :return GeoDataFrame:

        """
        from geopandas import read_file

        self.__check_data()

        return read_file(self._path)

    def num_of_features(self):
        """Get number of downloaded LUCAS features.

        :return int: number of downloaded features
        """
        self.__check_data()
        
        try:
            ds = gdal.OpenEx(self._path, gdal.OF_VECTOR | gdal.OF_READONLY)
            layer = ds.GetLayer()
            nop = layer.GetFeatureCount()
            del ds
        except RuntimeError as e:
            raise LucasDataError(f"Postprocessing failed: {e}")

        return nop

    def is_empty(self):
        """Check whether downloaded LUCAS feature collection is empty.

        :return bool: True for empty collection otherwise False
        """
        return self.num_of_features() < 1

    # def get_images(self, point_id):
    #     """Get images of point and its surrounding from ftp server
    #     :param point_id:
    #     :return images: dictionary of images
    #     """
    #     images = {}
    #     url = f'https://gisco-services.ec.europa.eu/lucas/photos/2015/BE/{str(point_id)[0:3]}/{str(point_id)[3:6]}/{str(point_id)}'
    #     for i in ("Point", "South", "North", "East", "West"):
    #         images[i] = url+i[0:1]+".jpg"
    #     return images
