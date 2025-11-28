#include "io/IoArray.h"
#include <filesystem>
#include <gtest/gtest.h>
using namespace skmap;

TEST(IoLegacy, WarpTile)
{
    std::string mosaic_path = "https://s3.openlandmap.org/arco/wv_mcd19a2v061.seasconv.sd.yearly_p50_1km_s_20000101_20001231_go_epsg.4326_v20230619.tif";
    // Compute repository root relative to this source file so the test is portable.
    std::filesystem::path src_path = __FILE__;
    // tests/<this file> -> parent() == tests, parent().parent() == repo root
    std::filesystem::path repo_root = src_path.parent_path().parent_path().parent_path();
    std::filesystem::path ref_rel = repo_root / "skmap" / "data" / "toy" / "swir1" / "swir1_landsat.ard1_p50_30m_s_20141202_20150320_nl_epsg.3035_v20230720.tif";
    std::string ref_tile_path = ref_rel.string();
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
    MatFloat data(1, n_pix);
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(conf_GDAL);
    ioArray.warpTile(ref_tile_path, mosaic_path, resample);
    std::cout << "Element (0,0): " << data(0, 0) << std::endl;
    std::cout << "Element (0,255): " << data(0, 255) << std::endl;
    std::cout << "Element (0,100*256): " << data(0, 100*256) << std::endl;
    
    std::vector<std::string> base_files;
    std::vector<std::string> file_names;
    std::vector<uint_t> data_indices;
    std::string base_folder = ".";
    base_files.push_back(ref_tile_path);
    int16_t nodata = -1;
    file_names.push_back("test");
    data_indices.push_back(0);
    ioArray.writeData(base_files, base_folder, file_names, data_indices, x_offset, y_offset, x_size, y_size, GDT_Int16, nodata, std::nullopt, std::nullopt);
}
