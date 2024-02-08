#ifndef MISC_CPP
#define MISC_CPP

#include <iostream>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <functional>
#include <cassert>
#include <unordered_map>
#include <cmath>
#include <optional>

namespace skmap {


inline void skmapAssertIfTrue(bool cond, std::string message)
{
    if (cond)
    {
        std::cerr << message << std::endl;
        std::cout << message << std::endl;
        assert(false);
    }
}

using uint_t = unsigned int;
using float_t = float;
inline float_t nan_v = std::numeric_limits<float_t>::quiet_NaN();
inline float_t inf_v = std::numeric_limits<float_t>::infinity();
using dict_t = std::unordered_map<std::string, std::string>;
// RowMajor -> C order ,  ColMajor -> F order 
// C order is default in Numpy and Eigen pybind11 require it to get this input
using MatFloat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;



}


#endif
