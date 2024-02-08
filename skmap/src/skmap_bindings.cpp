#include "io/IoArray.h"
#include "transform/TransArray.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace skmap;

dict_t convPyDict(py::dict py_dict)
{
    dict_t cpp_dict;
    for (auto item : py_dict) {
        cpp_dict[py::str(item.first)] = py::str(item.second);
    }
    return cpp_dict;
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


void computeNormalizedDifference(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 std::vector<uint_t> positive_indices,
                                 std::vector<uint_t> negative_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t positive_scaling,
                                 float_t negative_scaling,
                                 float_t result_scaling,
                                 float_t result_offset)
{
    TransArray transArray(data, n_threads);
    transArray.computeNormalizedDifference(positive_indices, negative_indices, result_indices,
                                           positive_scaling, negative_scaling, result_scaling, result_offset);

}



PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("readData", &readData, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg("value_to_mask") = std::nullopt, py::arg("value_to_set") = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("computeNormalizedDifference", &computeNormalizedDifference, "Compute normalized difference indices");
}