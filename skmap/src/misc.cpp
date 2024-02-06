#ifndef MISC_CPP
#define MISC_CPP

#include <iostream>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <functional>

namespace skmap {

using uint_t = unsigned int;
using float_t = float;
// RowMajor -> C order ,  ColMajor -> F order 
// C order is default in Numpy and Eigen pybind11 require it to get this input
using MatFloat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}


#endif
