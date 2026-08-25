#include "io/IoArray.h"
#include "transform/TransArray.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
namespace py = pybind11;
using namespace skmap;

using NodataVariant = std::variant<byte_t, uint16_t, int16_t, uint32_t, int32_t,
                                   float32_t, float64_t>;

GDALDataType GetGDALDataTypeFromString(const std::string &gdal_data_type_str) {
  if (gdal_data_type_str == "byte") {
    return GDT_Byte;
  } else if (gdal_data_type_str == "uint16") {
    return GDT_UInt16;
  } else if (gdal_data_type_str == "int16") {
    return GDT_Int16;
  } else if (gdal_data_type_str == "uint32") {
    return GDT_UInt32;
  } else if (gdal_data_type_str == "int32") {
    return GDT_Int32;
  } else if (gdal_data_type_str == "float32") {
    return GDT_Float32;
  } else if (gdal_data_type_str == "float64") {
    return GDT_Float64;
    // } else if (gdal_data_type_str == "GDT_CInt16") {
    //     return GDT_CInt16;
    // } else if (gdal_data_type_str == "GDT_CInt32") {
    //     return GDT_CInt32;
    // } else if (gdal_data_type_str == "GDT_CFloat32") {
    //     return GDT_CFloat32;
    // } else if (gdal_data_type_str == "GDT_CFloat64") {
    //     return GDT_CFloat64;
  } else {
    // Default case if the string does not match any known GDALDataType
    throw std::invalid_argument(
        "Unknown GDALDataType string: " + gdal_data_type_str +
        "format should be one of 'byte', 'uint16', 'int16', 'uint32', 'int32', "
        "'float32', 'float64'");
  }
}

std::optional<NodataVariant>
getNodataVariant(const std::string &gdal_data_type_str, float_t no_data_value) {
  if (gdal_data_type_str == "byte") {
    return static_cast<byte_t>(no_data_value);
  } else if (gdal_data_type_str == "uint16") {
    return static_cast<uint16_t>(no_data_value);
  } else if (gdal_data_type_str == "int16") {
    return static_cast<int16_t>(no_data_value);
  } else if (gdal_data_type_str == "uint32") {
    return static_cast<uint32_t>(no_data_value);
  } else if (gdal_data_type_str == "int32") {
    return static_cast<int32_t>(no_data_value);
  } else if (gdal_data_type_str == "float32") {
    return static_cast<float32_t>(no_data_value);
  } else if (gdal_data_type_str == "float64") {
    return static_cast<float64_t>(no_data_value);
    // } else if (gdal_data_type_str == "GDT_CInt16") {
    //     return static_cast<cint16_t>(no_data_value);
    // } else if (gdal_data_type_str == "GDT_CInt32") {
    //     return static_cast<cint32_t>(no_data_value);
    // } else if (gdal_data_type_str == "GDT_CFloat32") {
    //     return static_cast<cfloat32_t>(no_data_value);
    // } else if (gdal_data_type_str == "GDT_CFloat64") {
    //     return static_cast<cfloat64_t>(no_data_value);
  }
  return std::nullopt; // Unknown type
}

dict_t convPyDict(py::dict in_dict) {
  dict_t cpp_dict;
  for (auto item : in_dict) {
    cpp_dict[py::str(item.first)] = py::str(item.second);
  }
  return cpp_dict;
}

map_t convPyMap(py::dict in_map) {
  map_t cpp_map;
  for (auto item : in_map) {
    cpp_map[py::str(item.first)] = item.second.cast<std::vector<uint_t>>();
  }
  return cpp_map;
}

/**
 * @defgroup io IO / Storage
 * @brief Functions for reading/writing raster or matrix data to/from disk.
 *
 * This group contains all I/O related functions, including GDAL-backed
 * reading/writing and Python-friendly wrapper functions.
 * @{
 */

/**
 * @brief see `IoArray::readData`
 *
 * Currently only supports Float32 data
 */
void readData(Eigen::Ref<MatFloat> data, const uint_t n_threads,
              const std::vector<std::string> &file_locs,
              const std::vector<uint_t> perm_vec, const uint_t x_off,
              const uint_t y_off, const uint_t x_size, const uint_t y_size,
              const std::vector<int> bands_list, py::dict conf_GDAL,
              std::optional<float_t> value_to_mask,
              std::optional<float_t> value_to_set, const int overview) {
  // Bug fix: convPyDict calls py::str() (Python C-API) and must run while the
  // GIL is still held. Convert first, then release before GDAL/OMP work.
  dict_t cpp_conf = convPyDict(conf_GDAL);
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  ioArray.readData(file_locs, perm_vec, x_off, y_off, x_size, y_size,
                   GDALDataType::GDT_Float32, bands_list, value_to_mask,
                   value_to_set, overview);
}

/**
 * @brief Python-friendly wrapper for writing raster data using GDAL and Eigen.
 *
 * This function wraps the `IoArray::writeData` method, handling GDAL setup,
 * nodata type dispatching, and optional post-processing (compression or remote
 * storage).
 *
 * @note For full details on the writing behavior, memory handling, and
 * parallelization, see `IoArray::writeData`:
 *       @ref IoArray::writeData
 *
 * @param data      Eigen::matrix reference containing the data to write.
 * @param n_threads Number of threads for parallel I/O operations.
 * @param conf_GDAL Python dictionary of GDAL configuration options.
 * @param creation_options Vector of "KEY=VALUE" GDAL driver creation options.
 * @param scale Band scale metadata (equivalent to gdal_translate -a_scale).
 *
 * Other parameters are passed to `IoArray::WriteData`: @ref IoArray::WriteData
 */
void writeData(Eigen::Ref<MatFloat> data, const uint_t n_threads,
               py::dict conf_GDAL, std::vector<std::string> base_files,
               std::string base_folder, std::vector<std::string> file_names,
               std::vector<uint_t> data_indices, uint_t x_off, uint_t y_off,
               uint_t x_size, uint_t y_size, float_t no_data_value,
               std::string gdal_data_type_str,
               std::vector<std::string> creation_options, double scale) {
  dict_t cpp_conf = convPyDict(conf_GDAL);
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  GDALDataType gdal_data_type = GetGDALDataTypeFromString(gdal_data_type_str);
  auto no_data_variant = getNodataVariant(gdal_data_type_str, no_data_value);
  if (!no_data_variant)
    throw std::invalid_argument(
        "scikit-map ERROR 61: Unknown data type for no_data_value: " +
        gdal_data_type_str);
  std::visit(
      [&](auto &&casted_nodata) {
        ioArray.writeData(base_files, base_folder, file_names, data_indices,
                          x_off, y_off, x_size, y_size, gdal_data_type,
                          casted_nodata, creation_options, scale);
      },
      *no_data_variant);
}

/**
 * @brief wrapper for `IoArray::extractOverlay`
 */
void extractOverlay(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    const std::vector<uint_t> pix_block_ids,
                    const std::vector<uint_t> pix_inblock_idxs,
                    const std::vector<uint_t> unique_blocks_ids_comb,
                    const std::vector<uint_t> key_layer_ids_comb,
                    Eigen::Ref<MatFloat> data_overlay) {
  IoArray ioArray(data, n_threads);
  py::gil_scoped_release release;
  ioArray.extractOverlay(pix_block_ids, pix_inblock_idxs,
                         unique_blocks_ids_comb, key_layer_ids_comb,
                         data_overlay);
}

/**
 * @brief wrapper for `IoArray::warpTile`
 * @deprecated Please use gdal VRTs
 */
void warpTile(Eigen::Ref<MatFloat> data, const uint_t n_threads,
              py::dict conf_GDAL, std::string tile_path, std::string mosaic_path,
              std::string resample) {
  dict_t cpp_conf = convPyDict(conf_GDAL);
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  ioArray.warpTile(tile_path, mosaic_path, resample);
}

void getLatLonArray(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    py::dict conf_GDAL, std::string file_loc, uint_t x_off,
                    uint_t y_off, uint_t x_size, uint_t y_size) {
  dict_t cpp_conf = convPyDict(conf_GDAL);
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  ioArray.getLatLonArray(file_loc, x_off, y_off, x_size, y_size);
}
/** @} */ // end of io group

void readDataBlocks(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    const std::vector<std::string> &file_locs,
                    const std::vector<uint_t> perm_vec,
                    const std::vector<uint_t> x_off_vec,
                    const std::vector<uint_t> y_off_vec,
                    const std::vector<uint_t> x_size_vec,
                    const std::vector<uint_t> y_size_vec,
                    const std::vector<int> bands_list, py::dict conf_GDAL,
                    std::optional<std::vector<float_t>> value_to_mask_vec,
                    std::optional<float_t> value_to_set) {
  dict_t cpp_conf = convPyDict(conf_GDAL); // must run before GIL release
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  ioArray.readDataBlocks(file_locs, perm_vec, x_off_vec, y_off_vec, x_size_vec,
                         y_size_vec, GDALDataType::GDT_Float32, bands_list,
                         value_to_mask_vec, value_to_set);
}

/** @brief core implementation, see `IoArray::readDataCore` */
void readDataCore(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  const std::string file_loc, const uint_t x_off,
                  const uint_t y_off, const uint_t x_size, const uint_t y_size,
                  const std::vector<int> bands_list, py::dict conf_GDAL,
                  std::optional<float_t> value_to_mask,
                  std::optional<float_t> value_to_set, const int overview) {
  dict_t cpp_conf = convPyDict(conf_GDAL); // must run before GIL release
  py::gil_scoped_release release;
  IoArray ioArray(data, n_threads);
  ioArray.setupGdal(cpp_conf);
  // Bug fix: pass raw pointer + size to match the updated readDataCore signature.
  ioArray.readDataCore(data.row(0).data(), static_cast<uint_t>(data.cols()),
                       file_loc, x_off, y_off, x_size, y_size,
                       GDALDataType::GDT_Float32, bands_list, value_to_mask,
                       value_to_set, overview);
}

/**
 * @deprecated `IoArray::writeData`
 */
void writeInt16Data(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    py::dict conf_GDAL, std::vector<std::string> base_files,
                    std::string base_folder,
                    std::vector<std::string> file_names,
                    std::vector<uint_t> data_indices, uint_t x_off,
                    uint_t y_off, uint_t x_size, uint_t y_size,
                    int16_t no_data_value,
                    std::vector<std::string> creation_options, double scale) {
  writeData(data, n_threads, conf_GDAL, base_files, base_folder, file_names,
            data_indices, x_off, y_off, x_size, y_size, no_data_value, "int16",
            creation_options, scale);
}

/**
 * @deprecated use `IoArray::writeData`
 */
void writeUInt16Data(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                     py::dict conf_GDAL, std::vector<std::string> base_files,
                     std::string base_folder,
                     std::vector<std::string> file_names,
                     std::vector<uint_t> data_indices, uint_t x_off,
                     uint_t y_off, uint_t x_size, uint_t y_size,
                     uint16_t no_data_value,
                     std::vector<std::string> creation_options, double scale) {
  writeData(data, n_threads, conf_GDAL, base_files, base_folder, file_names,
            data_indices, x_off, y_off, x_size, y_size, no_data_value, "uint16",
            creation_options, scale);
}

/**
 * @deprecated use `IoArray::writeData`
 */
void writeByteData(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                   py::dict conf_GDAL, std::vector<std::string> base_files,
                   std::string base_folder, std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices, uint_t x_off, uint_t y_off,
                   uint_t x_size, uint_t y_size, byte_t no_data_value,
                   std::vector<std::string> creation_options, double scale) {
  writeData(data, n_threads, conf_GDAL, base_files, base_folder, file_names,
            data_indices, x_off, y_off, x_size, y_size, no_data_value, "byte",
            creation_options, scale);
}

/**
 * @defgroup mangling Data mangling
 *
 * Functions for changing the shape of data for efficient parallel processing
 *
 * For canonical usage examples, see [unit
 * tests](https://github.com/openlandmap/scikit-map/blob/ci/tests/src/transform/test_mangling.cpp)
 * @{
 */

/** @brief see `TransArray::copyVecInMatrixRow` */
void copyVecInMatrixRow(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                        Eigen::Ref<VecFloat> in_vec, uint_t row_idx) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.copyVecInMatrixRow(in_vec, row_idx);
}

/** @brief see `TransArray::selArrayRows` */
void selArrayRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<uint_t> row_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.selArrayRows(out_data, row_select);
}

/** @brief see `TransArray::selArrayCols` */
void selArrayCols(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<uint_t> col_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.selArrayCols(out_data, col_select);
}

/** @brief see `TransArray::expandArrayRows` */
void expandArrayRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> row_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.expandArrayRows(out_data, row_select);
}

/** @brief see `TransArray::expandArrayCols` */
void expandArrayCols(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> col_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.expandArrayCols(out_data, col_select);
}

/** @brief see `TransArray::reorderArray` */
void reorderArray(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<std::vector<uint_t>> indices_matrix) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.reorderArray(out_data, indices_matrix);
}

/** @brief simple transpose, see `TransArray::transposeArray` for details */
void transposeArray(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    Eigen::Ref<MatFloat> out_data) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.transposeArray(out_data);
}

/** @brief see `TransArray::transposeReorderArray` */
void transposeReorderArray(
    Eigen::Ref<MatFloat> data, const uint_t n_threads,
    Eigen::Ref<MatFloat> out_data,
    std::vector<std::vector<uint_t>> permutation_matrix) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.transposeReorderArray(out_data, permutation_matrix);
}

/** @brief see `TransArray::inverseReorderArray` */
void inverseReorderArray(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                         Eigen::Ref<MatFloat> out_data,
                         std::vector<std::vector<uint_t>> indices_matrix) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.inverseReorderArray(out_data, indices_matrix);
}
/** @} */ // endgroup mangling

/**
 * @deprecated use selArrayRows in stead
 */
void extractArrayRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data,
                      std::vector<uint_t> row_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.selArrayRows(out_data, row_select);
}

/**
 * @deprecated use selArrayCols in stead
 */
void extractArrayCols(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data,
                      std::vector<uint_t> col_select) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.selArrayCols(out_data, col_select);
}

/**
 * @defgroup manipulation Data manipulation
 * parallel operations on data arrays
 * @{
 */

/** @brief see `TransArray::fillArray` */
void fillArray(Eigen::Ref<MatFloat> data, const uint_t n_threads, float_t val) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.fillArray(val);
}

/** @brief see `TransArray::maskNan` */
void maskNan(Eigen::Ref<MatFloat> data, const uint_t n_threads,
             std::vector<uint_t> row_select, float_t new_value_in_data) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.maskNan(row_select, new_value_in_data);
}

/** @brief see `TransArray::maskNanRows` */
void maskNanRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                 std::vector<uint_t> row_select,
                 Eigen::Ref<VecFloat> new_value_vec) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.maskNanRows(row_select, new_value_vec);
}

/** @brief see `TransArray::maskData` */
void maskData(Eigen::Ref<MatFloat> data, const uint_t n_threads,
              std::vector<uint_t> row_select, Eigen::Ref<MatFloat> mask,
              float_t value_of_mask_to_mask, float_t new_value_in_data) {
  py::gil_scoped_release release; // release GIL before OMP work
  TransArray transArray(data, n_threads);
  transArray.maskData(row_select, mask, value_of_mask_to_mask,
                      new_value_in_data);
}

/** @brief see `TransArray::maskDataRows` */
void maskDataRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  std::vector<uint_t> row_select, Eigen::Ref<MatFloat> mask,
                  float_t value_of_mask_to_mask, float_t new_value_in_data) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.maskDataRows(row_select, mask, value_of_mask_to_mask,
                          new_value_in_data);
}

/** @brief see `TransArray::swapRowsValues` */
void swapRowsValues(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    std::vector<uint_t> row_select, float_t value_to_mask,
                    float_t new_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.swapRowsValues(row_select, value_to_mask, new_value);
}

/** @brief see `TransArray::hadamardProduct` */
void hadamardProduct(Eigen::Ref<MatFloat> out, const uint_t n_threads,
                     Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2) {
  TransArray transArray(out, n_threads);
  py::gil_scoped_release release;
  transArray.hadamardProduct(in1, in2);
}

/** @brief see `TransArray::offsetAndScale` */
void offsetAndScale(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    float_t offset, float_t scaling) {
  py::gil_scoped_release release; // release GIL before OMP work
  TransArray transArray(data, n_threads);
  transArray.offsetAndScale(offset, scaling);
}

/** @brief see `TransArray::scaleAndOffset` */
void scaleAndOffset(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    float_t offset, float_t scaling) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.scaleAndOffset(offset, scaling);
}

/** @brief see `TransArray::offsetsAndScales` */
void offsetsAndScales(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      std::vector<uint_t> row_select,
                      Eigen::Ref<VecFloat> offsets,
                      Eigen::Ref<VecFloat> scalings) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.offsetsAndScales(row_select, offsets, scalings);
}

/**
 * @brief Cast a `float` array to a `double` array
 *
 * Will error on a size mismatch
 *
 * @param data Input data
 * @param n_threads number of threads to use
 * @param out_data output array
 */
void castFloat32ToFloat64(
    Eigen::Ref<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data,
    const uint_t n_threads,
    Eigen::Ref<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        out_data) {
  skmapAssertIfTrue(((uint_t)data.rows() != (uint_t)out_data.rows()),
                    "scikit-map ERROR 52: rows of the new array does not match "
                    "the size of selected");
  skmapAssertIfTrue(((uint_t)data.cols() != (uint_t)out_data.cols()),
                    "scikit-map ERROR 53: cols of the new array does not match "
                    "the size of selected");

  py::gil_scoped_release release;
  omp_set_num_threads(n_threads);
  Eigen::initParallel();
  Eigen::setNbThreads(n_threads);
  out_data = data.cast<double>();
}

/**
 * @brief Cast a `double` array to a `float` array
 *
 * Will error on a size mismatch
 *
 * @param data Input data
 * @param n_threads number of threads to use
 * @param out_data output array
 */
void castFloat64ToFloat32(
    Eigen::Ref<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data,
    const uint_t n_threads,
    Eigen::Ref<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        out_data) {
  skmapAssertIfTrue(((uint_t)data.rows() != (uint_t)out_data.rows()),
                    "scikit-map ERROR 52: rows of the new array does not match "
                    "the size of selected");
  skmapAssertIfTrue(((uint_t)data.cols() != (uint_t)out_data.cols()),
                    "scikit-map ERROR 53: cols of the new array does not match "
                    "the size of selected");

  py::gil_scoped_release release;
  omp_set_num_threads(n_threads);
  Eigen::initParallel();
  Eigen::setNbThreads(n_threads);
  out_data = data.cast<float>();
}

/** @} */ // endgroup manipulation

/**
 * @defgroup processing Data processing
 * data processing and statistics
 * @{
 */

/** @brief see `TransArray::nanMean` */
void nanMean(Eigen::Ref<MatFloat> data, const uint_t n_threads,
             Eigen::Ref<VecFloat> out_data) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.nanMean(out_data);
}

/** @brief see `TransArray::blocksAverage` */
void blocksAverage(Eigen::Ref<MatFloat> out, const uint_t n_threads,
                   Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2,
                   uint_t n_pix, uint_t y) {
  TransArray transArray(out, n_threads);
  py::gil_scoped_release release;
  transArray.blocksAverage(in1, in2, n_pix, y);
}

void blocksAverageVecs(Eigen::Ref<MatFloat> out, const uint_t n_threads,
                       Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2,
                       uint_t n_pix, uint_t y, uint_t row_offset) {
  TransArray transArray(out, n_threads);
  py::gil_scoped_release release;
  transArray.blocksAverageVecs(in1, in2, n_pix, y, row_offset);
}

void elementwiseAverage(Eigen::Ref<MatFloat> out, const uint_t n_threads,
                        Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2) {
  TransArray transArray(out, n_threads);
  py::gil_scoped_release release;
  transArray.elementwiseAverage(in1, in2);
}

void extractIndicators(Eigen::Ref<MatFloat> data_in, const uint_t n_threads,
                       Eigen::Ref<MatFloat> data_out, uint_t col_in_select,
                       std::vector<uint_t> col_out_select,
                       std::vector<uint_t> classes) {
  TransArray transArray(data_in, n_threads);
  py::gil_scoped_release release;
  transArray.extractIndicators(data_out, col_in_select, col_out_select,
                               classes);
}

void fitPercentage(Eigen::Ref<MatFloat> out, const uint_t n_threads,
                   Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2) {
  TransArray transArray(out, n_threads);
  py::gil_scoped_release release;
  transArray.fitPercentage(in1, in2);
}

void texturesBwTransform(Eigen::Ref<MatFloat> texture_1, const uint_t n_threads,
                         Eigen::Ref<MatFloat> texture_2, float_t k, float_t a,
                         Eigen::Ref<MatFloat> sand, Eigen::Ref<MatFloat> silt,
                         Eigen::Ref<MatFloat> clay) {
  TransArray transArray(texture_1, n_threads);
  py::gil_scoped_release release;
  transArray.texturesBwTransform(texture_2, k, a, sand, silt, clay);
}

void linearRegression(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      Eigen::Ref<VecFloat> x, Eigen::Ref<VecFloat> beta_0,
                      Eigen::Ref<VecFloat> beta_1) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.linearRegression(x, beta_0, beta_1);
}

void computeMannKendallPValues(Eigen::Ref<MatFloat> data,
                               const uint_t n_threads,
                               Eigen::Ref<VecFloat> out_data) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeMannKendallPValues(out_data);
}

void averageAggregate(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data, uint_t agg_factor) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.averageAggregate(out_data, agg_factor);
}

void maskDifference(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                    float_t diff_th, uint_t count_th,
                    Eigen::Ref<MatFloat> ref_data,
                    Eigen::Ref<MatFloat> mask_out) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.maskDifference(diff_th, count_th, ref_data, mask_out);
}

void computeNormalizedDifference(
    Eigen::Ref<MatFloat> data, const uint_t n_threads,
    std::vector<uint_t> positive_indices, std::vector<uint_t> negative_indices,
    std::vector<uint_t> result_indices, float_t positive_scaling,
    float_t negative_scaling, float_t result_scaling, float_t result_offset,
    std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeNormalizedDifference(
      positive_indices, negative_indices, result_indices, positive_scaling,
      negative_scaling, result_scaling, result_offset, clip_value);
}

void computeNirv(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                 std::vector<uint_t> nir_indices,
                 std::vector<uint_t> red_indices,
                 std::vector<uint_t> result_indices, float_t nir_scaling,
                 float_t red_scaling, float_t result_scaling,
                 float_t result_offset, std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeNirv(nir_indices, red_indices, result_indices, nir_scaling,
                         red_scaling, result_scaling, result_offset,
                         clip_value);
}

void computeBsi(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                std::vector<uint_t> swir1_indices,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices, float_t swir1_scaling,
                float_t red_scaling, float_t nir_scaling, float_t blue_scaling,
                float_t result_scaling, float_t result_offset,
                std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeBsi(swir1_indices, red_indices, nir_indices, blue_indices,
                        result_indices, swir1_scaling, red_scaling, nir_scaling,
                        blue_scaling, result_scaling, result_offset,
                        clip_value);
}

void computeEvi(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices, float_t red_scaling,
                float_t nir_scaling, float_t blue_scaling,
                float_t result_scaling, float_t result_offset,
                std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeEvi(red_indices, nir_indices, blue_indices, result_indices,
                        red_scaling, nir_scaling, blue_scaling, result_scaling,
                        result_offset, clip_value);
}

void computeFapar(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  std::vector<uint_t> red_indices,
                  std::vector<uint_t> nir_indices,
                  std::vector<uint_t> result_indices, float_t red_scaling,
                  float_t nir_scaling, float_t result_scaling,
                  float_t result_offset, std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeFapar(red_indices, nir_indices, result_indices, red_scaling,
                          nir_scaling, result_scaling, result_offset,
                          clip_value);
}

void computeSavi(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                 std::vector<uint_t> red_indices,
                 std::vector<uint_t> nir_indices,
                 std::vector<uint_t> result_indices, float_t red_scaling,
                 float_t nir_scaling, float_t result_scaling,
                 float_t result_offset, std::vector<float_t> clip_value) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeSavi(red_indices, nir_indices, result_indices, red_scaling,
                         nir_scaling, result_scaling, result_offset,
                         clip_value);
}

void computeGeometricTemperature(
    Eigen::Ref<MatFloat> data, const uint_t n_threads,
    Eigen::Ref<MatFloat> latitude, Eigen::Ref<MatFloat> elevation,
    float_t elevation_scaling, float_t a, float_t b, float_t result_scaling,
    std::vector<uint_t> result_indices, std::vector<float_t> days_of_year) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computeGeometricTemperature(latitude, elevation, elevation_scaling,
                                         a, b, result_scaling, result_indices,
                                         days_of_year);
}

/** @brief see `TransArray::computePercentiles` */
void computePercentiles(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                        std::vector<uint_t> col_in_select,
                        Eigen::Ref<MatFloat> out_data,
                        std::vector<uint_t> col_out_select,
                        std::vector<float_t> percentiles) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.computePercentiles(col_in_select, out_data, col_out_select,
                                percentiles);
}

void fitProbabilities(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data, float_t input_scaling,
                      uint_t target_scaling,
                      Eigen::Ref<MatFloat> best_classes_data,
                      uint_t n_best_classes) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.fitProbabilities(out_data, input_scaling, target_scaling,
                              best_classes_data, n_best_classes);
}

void applyTsirf(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                Eigen::Ref<MatFloat> out_data, uint_t out_index_offset,
                float_t w_0, Eigen::Ref<VecFloat> w_p, Eigen::Ref<VecFloat> w_f,
                bool keep_original_values, const std::string &version,
                const std::string &backend)

{
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.applyTsirf(out_data, out_index_offset, w_0, w_p, w_f,
                        keep_original_values, version, backend);
}

void nanMeanAggregatePattern(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                             Eigen::Ref<MatFloat> out_data,
                             std::vector<std::vector<uint_t>> &agg_pattern) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.nanMeanAggregatePattern(out_data, agg_pattern);
}

void convolveRows(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data, float_t w_0,
                  Eigen::Ref<VecFloat> w_p, Eigen::Ref<VecFloat> w_f) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.convolveRows(out_data, w_0, w_p, w_f);
}

void slidingWindowClassMode(Eigen::Ref<MatFloat> data, const uint_t n_threads,
                            Eigen::Ref<MatFloat> out_data, size_t window_size) {
  TransArray transArray(data, n_threads);
  py::gil_scoped_release release;
  transArray.slidingWindowClassMode(out_data, window_size);
}

/** @} */ // endgroup processing

void checkSimdInstructionSetsInUse() {
  auto simd_instructions = Eigen::SimdInstructionSetsInUse();
  std::cout << "SimdInstructionSetsInUse: " << simd_instructions << std::endl;
  return;
}

PYBIND11_MODULE(skmap_bindings, m) {
  m.def("readDataCore", &readDataCore, py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt, py::arg("overview") = 0,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
  m.def("readData", &readData, py::arg("data"), py::arg("n_threads"), py::arg("file_locs"), py::arg("perm_vec"),
        py::arg("x_off"), py::arg("y_off"), py::arg("x_size"), py::arg("y_size"), py::arg("bands_list"), py::arg("conf_GDAL"),
        py::arg("value_to_mask") = std::nullopt, py::arg("value_to_set") = std::nullopt, py::arg("overview") = 0,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
  m.def("readDataBlocks", &readDataBlocks, py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
  m.def("copyVecInMatrixRow", &copyVecInMatrixRow,
        "Copy a vector in a matrix row");
  m.def("fillArray", &fillArray, "Fill array");
  m.def("selArrayRows", &selArrayRows, "Mask array rows");
  m.def("selArrayCols", &selArrayCols, "Mask array cols");
  m.def("averageAggregate", &averageAggregate, "Average aggregate");
  m.def("maskData", &maskData, "Mask data");
  m.def("maskDataRows", &maskDataRows, "Mask data rows");
  m.def("maskNan", &maskNan, "Mask NaN");
  m.def("maskNanRows", &maskNanRows, "Mask NaN Rows");
  m.def("swapRowsValues", &swapRowsValues, "Swap array values");
  m.def("expandArrayRows", &expandArrayRows, "Expand array rows");
  m.def("expandArrayCols", &expandArrayCols, "Expand array cols");
  m.def("extractArrayRows", &extractArrayRows, "Extract array rows");
  m.def("extractArrayCols", &extractArrayCols, "Extract array cols");
  m.def("transposeArray", &transposeArray, "Transpose an array into a new one");
  m.def("reorderArray", &reorderArray, "Reorder an array into a new one");
  m.def("offsetsAndScales", &offsetsAndScales,
        "Add offsets and muplitply by scalings each array row selected");
  m.def("offsetAndScale", &offsetAndScale,
        "Add an offset and muplitply by a scaling each array element");
  m.def("inverseReorderArray", &inverseReorderArray,
        "Reorder and transpose an array into a new one");
  m.def("writeByteData", &writeByteData, py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(),
        py::arg() = std::vector<std::string>(), py::arg() = 1.0,
        "Write data in Byte format");
  m.def("writeInt16Data", &writeInt16Data, py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(),
        py::arg() = std::vector<std::string>(), py::arg() = 1.0,
        "Write data in Int16 format");
  m.def("writeUInt16Data", &writeUInt16Data, py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(),
        py::arg() = std::vector<std::string>(), py::arg() = 1.0,
        "Write data in Int16 format");
  m.def("writeData", &writeData, py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(),
        py::arg() = std::vector<std::string>(), py::arg() = 1.0, "Write data");
  m.def("getLatLonArray", &getLatLonArray,
        "Compute latitude and longitude for each pixel of a GeoTIFF");
  m.def("computeNormalizedDifference", &computeNormalizedDifference,
        "Compute normalized difference indices");
  m.def("computeBsi", &computeBsi, "Compute BSI");
  m.def("computeEvi", &computeEvi, "Compute EVI");
  m.def("computeNirv", &computeNirv, "Compute NIRv");
  m.def("scaleAndOffset", &scaleAndOffset,
        "Muplitply by a scaling and add an offset each array element");
  m.def("computeFapar", &computeFapar, "Compute FAPAR");
  m.def("nanMeanAggregatePattern", &nanMeanAggregatePattern,
        "Nan mean agg pattern");
  m.def("computeSavi", &computeSavi, "Compute SAVI");
  m.def("nanMean", &nanMean, "Compute average between available values");
  m.def("computeMannKendallPValues", &computeMannKendallPValues,
        "Compute Mann-Kendall p-values");
  m.def("warpTile", &warpTile, "Warp tile, deprecated in favor of VRTs");
  m.def("linearRegression", &linearRegression,
        "Compute linear regression slope and intercept");
  m.def("transposeReorderArray", &transposeReorderArray,
        "Transpose and reorder an array into a new one");
  m.def("computeGeometricTemperature", &computeGeometricTemperature,
        "Compute geometric temperautre");
  m.def("computePercentiles", &computePercentiles, "Compute percentile");
  m.def("applyTsirf", &applyTsirf, "Apply TSIRF");
  m.def("convolveRows", &convolveRows, "Convolve rows");
  m.def("fitPercentage", &fitPercentage,
        "Fit a three percages to 100 starting from 2");
  m.def("hadamardProduct", &hadamardProduct, "Elemennt wise product");
  m.def("maskDifference", &maskDifference,
        "Mask outliers by difference from a reference");
  m.def("extractIndicators", &extractIndicators, "Extract classes indicators");
  m.def("blocksAverage", &blocksAverage,
        "Vecorized average of 4 neighbor elements");
  m.def("texturesBwTransform", &texturesBwTransform, "Texture transformation");
  m.def("blocksAverageVecs", &blocksAverageVecs,
        "Vecorized average of 4 neighbor elements");
  m.def("elementwiseAverage", &elementwiseAverage,
        "Vecorized average between two arrays elements");
  m.def("extractOverlay", &extractOverlay, "Extract overlay data");
  m.def("slidingWindowClassMode", &slidingWindowClassMode, "A weird stuff");
  m.def("checkSimdInstructionSetsInUse", checkSimdInstructionSetsInUse);
  m.def("castFloat64ToFloat32", castFloat64ToFloat32);
  m.def("castFloat32ToFloat64", castFloat32ToFloat64);
  m.def("fitProbabilities", fitProbabilities);
}
