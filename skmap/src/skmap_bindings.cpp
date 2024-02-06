#include "io/IoArray.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace skmap;
namespace py = pybind11;


void parReadFiles(Eigen::Ref<MatFloat> data,
                  const uint_t n_pix,
                  const uint_t n_feat,
                  const uint_t n_threads) 
{
    IoArray testIoArray(data, n_pix, n_feat, n_threads);
    std::cout << "Hello 3!" << std::endl;
    std::vector<std::string> outDates = {
        "20090101_20090228",
        "20090301_20090430",
        "20090501_20090630",
        "20090701_20090831",
        "20090901_20091031",
        "20091101_20091231",
    };

    std::vector<std::string> file_urls;
    std::vector<int> perm_vec;
    for (int i = 0; i < 6; i++)
    {
        std::string str =  "http://192.168.1.30:8333/prod-landsat-ard2/120E_46N/agg/swir2_glad.ard2_m_30m_s_" + outDates[i] + "_go_epsg.4326_v20230908.tif";
        file_urls.push_back(str);
        perm_vec.push_back(i);
    }
    testIoArray.readData(4004, 4004, file_urls, perm_vec, GDT_Float32);


}



PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("parReadFiles", &parReadFiles, "Process Eigen Matrix");
}