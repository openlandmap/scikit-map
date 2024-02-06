#include "io/IoArray.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace skmap;
namespace py = pybind11;


void parReadFiles(Eigen::Ref<MatFloat> data,
                  const uint_t n_pix,
                  const uint_t n_feat,
                  const uint_t n_threads,
                  const std::vector<std::string>& file_urls,
                  const std::vector<int> perm_vec) 
{
    IoArray testIoArray(data, n_pix, n_feat, n_threads);
    std::cout << "Hello!" << std::endl;
    
    testIoArray.readData(4004, 4004, file_urls, perm_vec, GDALDataType::GDT_Float32);


}



PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("parReadFiles", &parReadFiles, "Process Eigen Matrix");
}