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

  IoArray ioArray(data, std::thread::hardware_concurrency());
  ioArray.setupGdal(conf_GDAL);
  ioArray.writeData(base_files, base_folder, file_names, {0}, x_offset,
                    y_offset, x_size, y_size, GDT_Int16, -1, std::nullopt,
                    std::nullopt);

  ASSERT_TRUE(
      std::filesystem::exists(base_folder + "/" + file_names[0] + ".tif"));
}

TEST(Io, test_read_data_parallel) {
  std::filesystem::path src_path = __FILE__;

  std::filesystem::path repo_root =
      src_path.parent_path().parent_path().parent_path().parent_path();

  std::filesystem::path swir1_dir =
      repo_root / "skmap" / "data" / "toy" / "swir1";

  std::vector<std::string> file_locs;

  for (const auto &entry : std::filesystem::directory_iterator(swir1_dir)) {
    if (!entry.is_regular_file())
      continue;
    if (entry.path().extension() == ".tif") {
      file_locs.push_back(entry.path().string());
    }
  }

  std::sort(file_locs.begin(), file_locs.end());

  ASSERT_FALSE(file_locs.empty());

  uint_t x_size = 256;
  uint_t y_size = 256;
  uint_t n_pix = x_size * y_size;

  MatFloat data(file_locs.size(), n_pix);

  std::vector<uint_t> perm_vec(file_locs.size());
  std::iota(perm_vec.begin(), perm_vec.end(), 0);

  std::vector<int> bands_list(file_locs.size(), 1);

  dict_t conf_GDAL;

  IoArray ioArray(data, std::thread::hardware_concurrency());
  ioArray.setupGdal(conf_GDAL);

  EXPECT_NO_THROW({
    ioArray.readData(file_locs, perm_vec, 0, 0, x_size, y_size, GDT_Float32,
                     bands_list, std::nullopt, std::nullopt);
  });

  // minimal sanity check: ensure data is not entirely zero
  double sum = 0.0;
  for (int i = 0; i < data.rows(); ++i) {
    for (int j = 0; j < data.cols(); ++j) {
      sum += data(i, j);
    }
  }

  EXPECT_NE(sum, 0.0);
}
