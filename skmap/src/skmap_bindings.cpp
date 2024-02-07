#include "io/IoArray.h"
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


void parReadFiles(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  const std::vector<std::string>& file_locs,
                  const std::vector<uint_t> perm_vec,
                  const uint_t x_off,
                  const uint_t y_off,
                  const uint_t x_size,
                  const uint_t y_size,
                  const std::vector<int> bands_list,
                  py::dict conf_GDAL) 
{

    IoArray testIoArray(data, n_threads);
    testIoArray.setupGdal(convPyDict(conf_GDAL));
    testIoArray.readData(file_locs, perm_vec, x_off, y_off, x_size, y_size, GDALDataType::GDT_Float32, bands_list);

}



PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("parReadFiles", &parReadFiles, "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
}