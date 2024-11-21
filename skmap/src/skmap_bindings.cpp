#include "io/IoArray.h"
#include "transform/TransArray.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace skmap;


GDALDataType GetGDALDataTypeFromString(const std::string& gdal_data_type_str) {
    if (gdal_data_type_str == "GDT_Byte") {
        return GDT_Byte;
    } else if (gdal_data_type_str == "GDT_UInt16") {
        return GDT_UInt16;
    } else if (gdal_data_type_str == "GDT_Int16") {
        return GDT_Int16;
    } else if (gdal_data_type_str == "GDT_UInt32") {
        return GDT_UInt32;
    } else if (gdal_data_type_str == "GDT_Int32") {
        return GDT_Int32;
    } else if (gdal_data_type_str == "GDT_Float32") {
        return GDT_Float32;
    } else if (gdal_data_type_str == "GDT_Float64") {
        return GDT_Float64;
    } else if (gdal_data_type_str == "GDT_CInt16") {
        return GDT_CInt16;
    } else if (gdal_data_type_str == "GDT_CInt32") {
        return GDT_CInt32;
    } else if (gdal_data_type_str == "GDT_CFloat32") {
        return GDT_CFloat32;
    } else if (gdal_data_type_str == "GDT_CFloat64") {
        return GDT_CFloat64;
    } else {
        // Default case if the string does not match any known GDALDataType
        throw std::invalid_argument("Unknown GDALDataType string: " + gdal_data_type_str);
    }
}

dict_t convPyDict(py::dict in_dict)
{
    dict_t cpp_dict;
    for (auto item : in_dict) {
        cpp_dict[py::str(item.first)] = py::str(item.second);
    }
    return cpp_dict;
}


map_t convPyMap(py::dict in_map)
{
    map_t cpp_map;
    for (auto item : in_map) {
        cpp_map[py::str(item.first)] = item.second.cast<std::vector<uint_t>>();
    }
    return cpp_map;
}


void readDataCore(Eigen::Ref<MatFloat> data,
              const uint_t n_threads,
              const std::string file_loc,
              const uint_t x_off,
              const uint_t y_off,
              const uint_t x_size,
              const uint_t y_size,
              const std::vector<int> bands_list,
              py::dict conf_GDAL,
              std::optional<float_t> value_to_mask,
              std::optional<float_t> value_to_set) 
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.readDataCore(data.row(0), file_loc, x_off, y_off, x_size, y_size, GDALDataType::GDT_Float32,
                     bands_list, value_to_mask, value_to_set);
}



void extractOverlay(Eigen::Ref<MatFloat> data,
              const uint_t n_threads,
              const std::vector<uint_t> pix_blok_ids,
              const std::vector<uint_t> pix_inblock_idxs,
              const std::vector<uint_t> unique_blocks_ids_comb,
              const std::vector<uint_t> key_layer_ids_comb,
              Eigen::Ref<MatFloat> data_overlay)
{
    IoArray ioArray(data, n_threads);
    ioArray.extractOverlay(pix_blok_ids, pix_inblock_idxs, unique_blocks_ids_comb, key_layer_ids_comb, data_overlay);
}


void readData(Eigen::Ref<MatFloat> data,
              const uint_t n_threads,
              const std::vector<std::string>& file_locs,
              const std::vector<uint_t> perm_vec,
              const uint_t x_off,
              const uint_t y_off,
              const uint_t x_size,
              const uint_t y_size,
              const std::vector<int> bands_list,
              py::dict conf_GDAL,
              std::optional<float_t> value_to_mask,
              std::optional<float_t> value_to_set) 
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.readData(file_locs, perm_vec, x_off, y_off, x_size, y_size, GDALDataType::GDT_Float32,
                     bands_list, value_to_mask, value_to_set);
}


void readDataBlocks(Eigen::Ref<MatFloat> data,
              const uint_t n_threads,
              const std::vector<std::string>& file_locs,
              const std::vector<uint_t> perm_vec,
              const std::vector<uint_t> x_off_vec,
              const std::vector<uint_t> y_off_vec,
              const std::vector<uint_t> x_size_vec,
              const std::vector<uint_t> y_size_vec,
              const std::vector<int> bands_list,
              py::dict conf_GDAL,
              std::optional<std::vector<float_t>> value_to_mask_vec,
              std::optional<float_t> value_to_set) 
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.readDataBlocks(file_locs, perm_vec, x_off_vec, y_off_vec, x_size_vec, y_size_vec, GDALDataType::GDT_Float32,
                     bands_list, value_to_mask_vec, value_to_set);
}



void getLatLonArray(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    py::dict conf_GDAL,
                    std::string file_loc,
                    uint_t x_off,
                    uint_t y_off,
                    uint_t x_size,
                    uint_t y_size)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.getLatLonArray(file_loc, x_off, y_off, x_size, y_size);
}

void blocksAverage(Eigen::Ref<MatFloat> out,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> in1,
                  Eigen::Ref<MatFloat> in2,
                  uint_t n_pix,
                  uint_t y)
{
    TransArray transArray(out, n_threads);
    transArray.blocksAverage(in1, in2, n_pix, y);
}


void reorderArray(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.reorderArray(out_data, indices_matrix);
}



void selArrayRows(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.selArrayRows(out_data, row_select);
}



void selArrayCols(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<uint_t> col_select)
{
    TransArray transArray(data, n_threads);
    transArray.selArrayCols(out_data, col_select);
}


void expandArrayRows(Eigen::Ref<MatFloat> data,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.expandArrayRows(out_data, row_select);
}

void extractArrayRows(Eigen::Ref<MatFloat> data,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.extractArrayRows(out_data, row_select);
}

void extractArrayCols(Eigen::Ref<MatFloat> data,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> col_select)
{
    TransArray transArray(data, n_threads);
    transArray.extractArrayCols(out_data, col_select);
}

void swapRowsValues(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    float_t value_to_mask,
                    float_t new_value)
{
    TransArray transArray(data, n_threads);
    transArray.swapRowsValues(row_select, value_to_mask, new_value);
}


void extractIndicators(Eigen::Ref<MatFloat> data_in,
                            const uint_t n_threads,
                            Eigen::Ref<MatFloat> data_out,
                            uint_t col_in_select,
                            std::vector<uint_t> col_out_select,
                            std::vector<uint_t> classes)
{
    TransArray transArray(data_in, n_threads);
    transArray.extractIndicators(data_out, col_in_select, col_out_select, classes);
}


void maskNan(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    float_t new_value_in_data)
{
    TransArray transArray(data, n_threads);
    transArray.maskNan(row_select, new_value_in_data);
}

void maskData(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    Eigen::Ref<MatFloat> mask,
                    float_t value_of_mask_to_mask,
                    float_t new_value_in_data)
{
    TransArray transArray(data, n_threads);
    transArray.maskData(row_select, mask, value_of_mask_to_mask, new_value_in_data);
}

void fitPercentage(Eigen::Ref<MatFloat> out,
                   const uint_t n_threads,
                   Eigen::Ref<MatFloat> in1,
                   Eigen::Ref<MatFloat> in2)
{
    TransArray transArray(out, n_threads);
    transArray.fitPercentage(in1, in2);
}

void hadamardProduct(Eigen::Ref<MatFloat> out,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> in1,
                     Eigen::Ref<MatFloat> in2)
{
    TransArray transArray(out, n_threads);
    transArray.hadamardProduct(in1, in2);
}

void maskDataRows(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    Eigen::Ref<MatFloat> mask,
                    float_t value_of_mask_to_mask,
                    float_t new_value_in_data)
{
    TransArray transArray(data, n_threads);
    transArray.maskDataRows(row_select, mask, value_of_mask_to_mask, new_value_in_data);
}

void fillArray(Eigen::Ref<MatFloat> data,
               const uint_t n_threads,
               float_t val)
{
    TransArray transArray(data, n_threads);
    transArray.fillArray(val);
}

void copyVecInMatrixRow(Eigen::Ref<MatFloat> data,
               const uint_t n_threads,
               Eigen::Ref<VecFloat> in_vec,
               uint_t row_idx)
{
    TransArray transArray(data, n_threads);
    transArray.copyVecInMatrixRow(in_vec, row_idx);
}


void inverseReorderArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data,
                          std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.inverseReorderArray(out_data, indices_matrix);
}

void transposeReorderArray(Eigen::Ref<MatFloat> data,
                         const uint_t n_threads,
                         Eigen::Ref<MatFloat> out_data,
                         std::vector<std::vector<uint_t>> permutation_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.transposeReorderArray(out_data, permutation_matrix);
}

void transposeArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.transposeArray(out_data);
}

void offsetAndScale(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t offset,
                    float_t scaling)
{
    TransArray transArray(data, n_threads);
    transArray.offsetAndScale(offset, scaling);
}


void offsetsAndScales(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    Eigen::Ref<VecFloat> offsets,
                    Eigen::Ref<VecFloat> scalings)
{
    TransArray transArray(data, n_threads);
    transArray.offsetsAndScales(row_select, offsets, scalings);
}

void nanMean(Eigen::Ref<MatFloat> data,
             const uint_t n_threads,
             Eigen::Ref<VecFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.nanMean(out_data);
}

void linearRegression(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<VecFloat> x,
                      Eigen::Ref<VecFloat> beta_0,
                      Eigen::Ref<VecFloat> beta_1)
{
    TransArray transArray(data, n_threads);
    transArray.linearRegression(x, beta_0, beta_1);
}


void computeMannKendallPValues(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<VecFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.computeMannKendallPValues(out_data);
}

void averageAggregate(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data,
                      uint_t agg_factor)
{
    TransArray transArray(data, n_threads);
    transArray.averageAggregate(out_data, agg_factor);
}

void maskDifference(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t diff_th,
                    uint_t count_th,
                    Eigen::Ref<MatFloat> ref_data,
                    Eigen::Ref<MatFloat> mask_out)
{
    TransArray transArray(data, n_threads);
    transArray.maskDifference(diff_th, count_th, ref_data, mask_out);
}

void computeNormalizedDifference(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 std::vector<uint_t> positive_indices,
                                 std::vector<uint_t> negative_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t positive_scaling,
                                 float_t negative_scaling,
                                 float_t result_scaling,
                                 float_t result_offset,
                                 std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeNormalizedDifference(positive_indices, negative_indices, result_indices,
                                           positive_scaling, negative_scaling, result_scaling, result_offset, clip_value);
}


void computeNirv(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 std::vector<uint_t> nir_indices,
                                 std::vector<uint_t> red_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t nir_scaling,
                                 float_t red_scaling,
                                 float_t result_scaling,
                                 float_t result_offset,
                                 std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeNirv(nir_indices, red_indices, result_indices,
                          nir_scaling, red_scaling, result_scaling, result_offset, clip_value);
}

void computeBsi(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> swir1_indices,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices,
                float_t swir1_scaling,
                float_t red_scaling,
                float_t nir_scaling,
                float_t blue_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeBsi(swir1_indices, red_indices, nir_indices, blue_indices, result_indices,
                          swir1_scaling, red_scaling, nir_scaling, blue_scaling, result_scaling, result_offset, clip_value);

}


void computeEvi(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices,
                float_t red_scaling,
                float_t nir_scaling,
                float_t blue_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeEvi(red_indices, nir_indices, blue_indices, result_indices,
                          red_scaling, nir_scaling, blue_scaling, result_scaling, result_offset, clip_value);

}

void computeFapar(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> result_indices,
                float_t red_scaling,
                float_t nir_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeFapar(red_indices, nir_indices, result_indices,
                           red_scaling, nir_scaling, result_scaling, result_offset, clip_value);
}


void computeSavi(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> result_indices,
                float_t red_scaling,
                float_t nir_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeSavi(red_indices, nir_indices, result_indices,
                           red_scaling, nir_scaling, result_scaling, result_offset, clip_value);
}

void computeGeometricTemperature(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 Eigen::Ref<MatFloat> latitude,
                                 Eigen::Ref<MatFloat> elevation,
                                 float_t elevation_scaling,
                                 float_t a,
                                 float_t b,
                                 float_t result_scaling,
                                 std::vector<uint_t> result_indices,
                                 std::vector<float_t> days_of_year)
{
    TransArray transArray(data, n_threads);
    transArray.computeGeometricTemperature(latitude, elevation, elevation_scaling, a, b, result_scaling, result_indices, days_of_year);
}

void writeByteData(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   byte_t no_data_value,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_Byte, no_data_value, bash_compression_command, seaweed_path);

}

void writeData(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   float_t no_data_value,
                   std::string gdal_data_type_str,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    GDALDataType gdal_data_type = GetGDALDataTypeFromString(gdal_data_type_str);
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, gdal_data_type, no_data_value, bash_compression_command, seaweed_path);
}


void writeInt16Data(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   int16_t no_data_value,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_Int16, no_data_value, bash_compression_command, seaweed_path);

}



void writeUInt16Data(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   uint16_t no_data_value,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_UInt16, no_data_value, bash_compression_command, seaweed_path);

}


void warpTile(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    py::dict conf_GDAL,
                    std::string tilePath,
                    std::string mosaicPath,
                    std::string resample)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.warpTile(tilePath, mosaicPath, resample);
}

void computePercentiles(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          std::vector<uint_t> col_in_select,
                          Eigen::Ref<MatFloat> out_data,
                          std::vector<uint_t> col_out_select,
                          std::vector<float_t> percentiles)
{
    TransArray transArray(data, n_threads);
    transArray.computePercentiles(col_in_select, out_data, col_out_select, percentiles);
}

void applyTsirf(Eigen::Ref<MatFloat> data,
                 const uint_t n_threads,
                 Eigen::Ref<MatFloat> out_data,
                 uint_t out_index_offset,
                 float_t w_0,
                 Eigen::Ref<VecFloat> w_p,
                 Eigen::Ref<VecFloat> w_f,
                 bool keep_original_values,
                 const std::string& version,
                 const std::string& backend)

{
    TransArray transArray(data, n_threads);
    transArray.applyTsirf(out_data, out_index_offset,
                           w_0, w_p, w_f, keep_original_values, version, backend);
}

void scaleAndOffset(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t offset,
                    float_t scaling)
{
    TransArray transArray(data, n_threads);
    transArray.scaleAndOffset(offset, scaling);
}

void convolveRows(Eigen::Ref<MatFloat> data,
                 const uint_t n_threads,
                 Eigen::Ref<MatFloat> out_data,
                 float_t w_0,
                 Eigen::Ref<VecFloat> w_p,
                 Eigen::Ref<VecFloat> w_f)

{
    TransArray transArray(data, n_threads);
    transArray.convolveRows(out_data, w_0, w_p, w_f);
}


PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("readDataCore", &readDataCore,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("readData", &readData,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("readDataBlocks", &readDataBlocks,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("copyVecInMatrixRow", &copyVecInMatrixRow, "Copy a vector in a matrix row");
    m.def("fillArray", &fillArray, "Fill array");
    m.def("selArrayRows", &selArrayRows, "Mask array rows");
    m.def("averageAggregate", &averageAggregate, "Average aggregate");
    m.def("maskData", &maskData, "Mask data");
    m.def("maskDataRows", &maskDataRows, "Mask data rows");
    m.def("maskNan", &maskNan, "Mask NaN");
    m.def("swapRowsValues", &swapRowsValues, "Swap array values");
    m.def("expandArrayRows", &expandArrayRows, "Expand array rows");
    m.def("extractArrayRows", &extractArrayRows, "Extract array rows");
    m.def("extractArrayCols", &extractArrayCols, "Extract array cols");
    m.def("transposeArray", &transposeArray, "Transpose an array into a new one");
    m.def("reorderArray", &reorderArray, "Reorder an array into a new one");
    m.def("offsetsAndScales", &offsetsAndScales, "Add offsets and muplitply by scalings each array row selected");
    m.def("offsetAndScale", &offsetAndScale, "Add an offset and muplitply by a scaling each array element");
    m.def("inverseReorderArray", &inverseReorderArray, "Reorder and transpose an array into a new one");
    m.def("writeByteData", &writeByteData, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Byte format");
    m.def("writeInt16Data", &writeInt16Data, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Int16 format");
    m.def("writeUInt16Data", &writeUInt16Data, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Int16 format");
    m.def("writeData", &writeData, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Int16 format");
    m.def("getLatLonArray", &getLatLonArray, "Compute latitude and longitude for each pixel of a GeoTIFF");
    m.def("computeNormalizedDifference", &computeNormalizedDifference, "Compute normalized difference indices");
    m.def("computeBsi", &computeBsi, "Compute BSI");
    m.def("computeEvi", &computeEvi, "Compute EVI");
    m.def("computeNirv", &computeNirv, "Compute NIRv");
    m.def("scaleAndOffset", &scaleAndOffset, "Muplitply by a scaling and add an offset each array element");
    m.def("computeFapar", &computeFapar, "Compute FAPAR");
    m.def("computeSavi", &computeSavi, "Compute SAVI");
    m.def("nanMean", &nanMean, "Compute average between available values");
    m.def("computeMannKendallPValues", &computeMannKendallPValues, "Compute Mann-Kendall p-values");
    m.def("warpTile", &warpTile, "Compute FAPAR");
    m.def("linearRegression", &linearRegression, "Compute linear regression slope and intercept");
    m.def("transposeReorderArray", &transposeReorderArray, "Transpose and reorder an array into a new one");
    m.def("computeGeometricTemperature", &computeGeometricTemperature, "Compute geometric temperautre");
    m.def("computePercentiles", &computePercentiles, "Compute percentile");
    m.def("applyTsirf", &applyTsirf, "Apply TSIRF");
    m.def("convolveRows", &convolveRows, "Convolve rows");
    m.def("fitPercentage", &fitPercentage, "Fit a three percages to 100 starting from 2");
    m.def("hadamardProduct", &hadamardProduct, "Elemennt wise product");
    m.def("maskDifference", &maskDifference, "Mask outliers by difference from a reference");
    m.def("extractIndicators", &extractIndicators, "Extract classes indicators");
    m.def("blocksAverage", &blocksAverage, "Vecorized average of 4 neighbor elemnts");
    m.def("extractOverlay", &extractOverlay, "Extract overlay data");

}

