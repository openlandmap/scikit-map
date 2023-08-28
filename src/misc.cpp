#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>

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

using ReadMatrix = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using WriteMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;