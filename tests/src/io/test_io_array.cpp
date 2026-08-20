#include "io/IoArray.h"

#include <filesystem>
#include <gtest/gtest.h>

using namespace skmap;

TEST(Io, test_write_data) {
  std::vector<std::string> file_names = {"test"};
  std::string base_folder = std::filesystem::temp_directory_path();
  int16_t nodata = -1;
  uint_t x_size = 256;
  uint_t y_size = 256;
  uint_t x_offset = 0;
  uint_t y_offset = 0;
  uint_t n_pix = x_size * y_size;
  MatFloat data(1, n_pix);
  dict_t conf_GDAL;
  std::filesystem::path src_path = __FILE__;
  // tests/<this file> -> parent() == tests, parent().parent() == repo root
  std::filesystem::path repo_root =
      src_path.parent_path().parent_path().parent_path().parent_path();
  std::filesystem::path ref_rel = repo_root / "skmap" / "data" / "toy" /
                                  "swir1" /
                                  "swir1_landsat.ard1_p50_30m_s_20141202_"
                                  "20150320_nl_epsg.3035_v20230720.tif";
  std::string ref_tile_path = ref_rel.string();
  std::vector<std::string> base_files = {ref_tile_path};

  IoArray ioArray(data, 1);
  ioArray.setupGdal(conf_GDAL);
  ioArray.writeData(base_files, base_folder, file_names, {0}, x_offset,
                    y_offset, x_size, y_size, GDT_Int16, -1, {}, 1.0);

  ASSERT_TRUE(
      std::filesystem::exists(base_folder + "/" + file_names[0] + ".tif"));
}

TEST(Io, test_get_lat_lon_array_non_square) {
  std::filesystem::path src_path = __FILE__;
  std::filesystem::path repo_root =
      src_path.parent_path().parent_path().parent_path().parent_path();
  std::filesystem::path ref_rel = repo_root / "skmap" / "data" / "toy" /
                                  "swir1" /
                                  "swir1_landsat.ard1_p50_30m_s_20141202_"
                                  "20150320_nl_epsg.3035_v20230720.tif";
  std::string ref_tile_path = ref_rel.string();

  // Non-square window: the old code used `i * y_size` for the latitude row,
  // which scrambled/overflowed whenever x_size != y_size.
  uint_t x_off = 10, y_off = 20, x_size = 100, y_size = 50;
  MatFloat data(2, x_size * y_size);
  dict_t conf_GDAL;
  IoArray ioArray(data, 1);
  ioArray.setupGdal(conf_GDAL);
  ioArray.getLatLonArray(ref_tile_path, x_off, y_off, x_size, y_size);

  GDALDataset *ds = (GDALDataset *)GDALOpen(ref_tile_path.c_str(), GA_ReadOnly);
  ASSERT_NE(ds, nullptr);
  double gt[6];
  ASSERT_EQ(ds->GetGeoTransform(gt), CE_None);
  GDALClose(ds);

  for (uint_t i = 0; i < y_size; ++i) {
    for (uint_t j = 0; j < x_size; ++j) {
      double x = gt[0] + (x_off + j) * gt[1] + (y_off + i) * gt[2];
      double y = gt[3] + (x_off + j) * gt[4] + (y_off + i) * gt[5];
      EXPECT_NEAR(data(0, i * x_size + j), (float_t)x, 1e-3);
      EXPECT_NEAR(data(1, i * x_size + j), (float_t)y, 1e-3);
    }
  }
}
