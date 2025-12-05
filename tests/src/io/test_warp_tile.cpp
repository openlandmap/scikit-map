#include "io/IoArray.h"
#include <filesystem>
#include <gtest/gtest.h>
using namespace skmap;

TEST(LegacyIo, DISABLED_WarpTile) {
  // mosaic file to load data from
  std::string mosaic_path = "https://s3.openlandmap.org/arco/"
                            "wv_mcd19a2v061.seasconv.sd.yearly_p50_1km_s_"
                            "20000101_20001231_go_epsg.4326_v20230619.tif";
  // Compute repository root relative to this source file so the test is
  // portable.
  std::filesystem::path src_path = __FILE__;
  // tests/<this file> -> parent() == tests, parent().parent() == repo root
  std::filesystem::path repo_root =
      src_path.parent_path().parent_path().parent_path();

  // file to match layout of
  std::filesystem::path ref_rel = repo_root / "skmap" / "data" / "toy" /
                                  "swir1" /
                                  "swir1_landsat.ard1_p50_30m_s_20141202_"
                                  "20150320_nl_epsg.3035_v20230720.tif";
  std::string ref_tile_path = ref_rel.string();

  // resampling parameters
  std::string resample = "GRA_CubicSpline";
  uint_t n_threads = 1;
  dict_t conf_GDAL;
  conf_GDAL["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif";
  conf_GDAL["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR";
  conf_GDAL["GDAL_HTTP_MULTIPLEX"] = "YES";
  conf_GDAL["GDAL_HTTP_VERSION"] = "2";
  conf_GDAL["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES";
  uint_t x_size = 256;
  uint_t y_size = 256;
  uint_t x_offset = 0;
  uint_t y_offset = 0;
  uint_t n_pix = x_size * y_size;

  // create out array and warp
  MatFloat data(1, n_pix);
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(conf_GDAL);
  ioArray.warpTile(ref_tile_path, mosaic_path, resample);

  // test some values
  ASSERT_FLOAT_EQ(data(0, 0), 314.435);
  ASSERT_FLOAT_EQ(data(0, 255), 241.68356);
  ASSERT_FLOAT_EQ(data(0, 100 * 256), 297.15668);
}
