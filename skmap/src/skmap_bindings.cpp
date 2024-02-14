#include "io/IoArray.h"
#include "transform/TransArray.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace skmap;

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

// void rearrangeChunks(Eigen::Ref<MatFloat> data,
//                      const uint_t n_threads,
//                      uint_t x_chunk,
//                      uint_t y_chunk,
//                      map_t input_map,
//                      Eigen::Ref<MatFloat> out_data,
//                      map_t output_map)
// {
//     TransArray transArray(data, n_threads);
//     transArray.rearrangeChunks(x_chunk, y_chunk, input_map, out_data, output_map);

// }


void reorderArray(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.reorderArray(out_data, indices_matrix);

}


void reorderTransposeArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data,
                          std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.reorderTransposeArray(out_data, indices_matrix);

}

void transposeArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.transposeArray(out_data);

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
                   std::string base_file,
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
    ioArray.writeData(base_file, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_Byte, no_data_value, bash_compression_command, seaweed_path);

}

PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("readData", &readData,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("transposeArray", &transposeArray, "Transpose an array into a new one");
    m.def("reorderArray", &reorderArray, "Reorder an array into a new one");
    // m.def("rearrangeChunks", &rearrangeChunks, "Rearrange chunks of data in a new array");
    m.def("writeByteData", &writeByteData, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Byte format");
    m.def("getLatLonArray", &getLatLonArray, "Compute latitude and longitude for each pixel of a GeoTIFF");
    m.def("reorderTransposeArray", &reorderTransposeArray, "Reorder and transpose an array into a new one");
    m.def("computeNormalizedDifference", &computeNormalizedDifference, "Compute normalized difference indices");
    m.def("computeBsi", &computeBsi, "Compute BSI");
    m.def("computeFapar", &computeFapar, "Compute FAPAR");
    m.def("computeGeometricTemperature", &computeGeometricTemperature, "Compute geometric temperautre");
}


