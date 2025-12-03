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
                    y_offset, x_size, y_size, GDT_Int16, -1, std::nullopt,
                    std::nullopt);

  ASSERT_TRUE(std::filesystem::exists(base_folder + "/" + file_names[0] + ".tif"));
}