#include "io/IoArray.h"
using namespace skmap;

void main()
{
    "https://s3.openlandmap.org/arco/wv_mcd19a2v061.seasconv.sd.yearly_p50_1km_s_20000101_20001231_go_epsg.4326_v20230619.tif"
    "WORKDIR/skmap/data/toy/swir1/swir1_landsat.ard1_p50_30m_s_20141202_20150320_nl_epsg.3035_v20230720.tif"
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.warpTile(tilePath, mosaicPath, resample);
}
