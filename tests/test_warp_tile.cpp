#include "io/IoArray.h"
using namespace skmap;

void main()
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.warpTile(tilePath, mosaicPath, resample);
}
