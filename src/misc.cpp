#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <chrono>

std::vector<std::string> outDatesStart = {
    "0101",
    "0301",
    "0501",
    "0701",
    "0901",
    "1101",
};
std::vector<std::string> outDatesEnd = {
    "0228",
    "0430",
    "0630",
    "0831",
    "1031",
    "1231",
};

// RowMajor to match the WrapFFT format
using MatrixFloat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixComplexFloat = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// ColMajor to match the GDAL format
using MatrixUI16 = Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixUI8 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixBool = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using size_t = std::size_t;


auto tic(void) {
    return std::chrono::high_resolution_clock::now();
}

template<class TimeType>
auto toc(TimeType t_tic) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_tic).count()/1000.;
}
