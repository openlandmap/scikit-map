#include "io/IoArray.h"

namespace skmap {

IoArray::IoArray(Eigen::Ref<MatFloat> data, const uint_t n_feat, const uint_t n_pix, const uint_t n_threads)
: ParArray(data, n_feat, n_pix, n_threads) 
{
}

void IoArray::readData(uint_t n_row,
	                   uint_t n_col,
	                   std::vector<std::string> file_urls,
	                   std::vector<int> perm_vec,
	                   GDALDataType read_type)
{
	CPLSetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif");
    CPLSetConfigOption("GDAL_HTTP_MULTIRANGE", "SINGLE_GET");
    CPLSetConfigOption("GDAL_HTTP_CONNECTTIMEOUT", "320");
    CPLSetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO");
    CPLSetConfigOption("GDAL_HTTP_VERSION", "1.0");
    CPLSetConfigOption("GDAL_HTTP_TIMEOUT", "320");
    CPLSetConfigOption("CPL_CURL_GZIP", "NO");
    GDALAllRegister();  // Initialize GDAL

	auto lambda = [&] (int i)
    {
    	int perm_id = perm_vec[i];
    	std::cout << "Perm id = " << perm_id << ", i = " << i << std::endl;
    	if(perm_id != -1)
    	{
    		int bandList[1];
	        bandList[0] = 1;
	    	std::string file_url = file_urls[perm_id];
    		std::cout << "I'm inside, perm id = " << perm_id << ", i = " << i << ", file_url = " << file_url << std::endl;
	    	GDALDataset *readDataset = (GDALDataset *)GDALOpen(file_url.c_str(), GA_ReadOnly);
	    	if (readDataset == nullptr) 
	    	{
				m_data.block(perm_id, 0, 1, m_n_pix).setZero();
				std::cerr << "scikit-map ERROR 1: issues in opening the file with URL " << file_url << std::endl;
				std::cout << "Issues in opening the file with URL " << file_url << ", considering as gap." << std::endl;
				GDALClose(readDataset);
			} else
			{
				CPLErr outRead = readDataset->RasterIO(GF_Read, 0, 0, n_row, n_col, m_data.data() + perm_id * m_n_pix,
				               n_row, n_col, read_type, 1, bandList, 0, 0, 0);
				if (outRead != CE_None)
				{
					m_data.block(perm_id, 0, 1, m_n_pix).setZero();
					std::cerr << "Error 2: issues in reading the file with URL " << file_url << std::endl;
					std::cout << "Issues in reading the file with URL " << file_url << ", considering as gap." << std::endl;
				}
				GDALClose(readDataset);
			}
    	}
    	
    };
    this->parFeat(lambda);
}



}