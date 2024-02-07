#include "io/IoArray.h"

namespace skmap {

IoArray::IoArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
: ParArray(data, n_threads) 
{
}

void IoArray::setupGdal(dict_t dict)
{
    for (auto& pair : dict) {
        CPLSetConfigOption(pair.first.c_str(), pair.second.c_str());
    }
    GDALAllRegister(); // Initialize GDAL
}



void IoArray::readData(std::vector<std::string> file_locs,
                       std::vector<uint_t> perm_vec,
                       uint_t x_off,
                       uint_t y_off,
                       uint_t x_size,
                       uint_t y_size,
                       GDALDataType read_type,                       
                       std::vector<int> bands_list)
{
	skmapAssertIfTrue(m_data.cols() < x_size * y_size, "scikit-map ERROR 0: reading region size smaller then the number of columns");
    auto readTiff = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
    {
        std::string file_loc = file_locs[i];
        GDALDataset *readDataset = (GDALDataset *)GDALOpen(file_loc.c_str(), GA_ReadOnly);
        skmapAssertIfTrue(readDataset == nullptr, "scikit-map ERROR 1: issues in opening the file with URL " + file_loc);
    	// It is assumed that the X/Y buffers size is equevalent to the portion of data to read
        CPLErr outRead = readDataset->RasterIO(GF_Read, x_off, y_off, x_size, y_size, row.data(),
                       x_size, y_size, read_type, bands_list.size(), &bands_list[0], 0, 0, 0);
        skmapAssertIfTrue(outRead != CE_None, "Error 2: issues in reading the file with URL " + file_loc);
        GDALClose(readDataset);
        
    };
    this->parRowPerm(readTiff, perm_vec);
}



}