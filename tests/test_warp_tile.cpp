#include "io/IoArray.h"
using namespace skmap;

void main()
{
    "https://s3.openlandmap.org/arco/wv_mcd19a2v061.seasconv.sd.yearly_p50_1km_s_20000101_20001231_go_epsg.4326_v20230619.tif"
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.warpTile(tilePath, mosaicPath, resample);
}
