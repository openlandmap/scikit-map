#include "misc.cpp"

class LandsatPipeline {
private:    
    unsigned int m_N_row = 4004;
    unsigned int m_N_col = 4004;
    unsigned int m_id = 4004;


public:
    LandsatPipeline(unsigned int id) {
        m_id = id;
        CPLSetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif");
        CPLSetConfigOption("GDAL_HTTP_MULTIRANGE", "SINGLE_GET");
        CPLSetConfigOption("GDAL_HTTP_CONNECTTIMEOUT", "320");
        CPLSetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO");
        CPLSetConfigOption("GDAL_HTTP_VERSION", "1.0");
        CPLSetConfigOption("GDAL_HTTP_TIMEOUT", "320");
        CPLSetConfigOption("CPL_CURL_GZIP", "NO");
        GDALAllRegister();  // Initialize GDAL
    }

    void readInputFiles(std::vector<std::string>& fileUrlsRead,
                        unsigned int nSlice,
                        unsigned int sliceStep,
                        unsigned int band_id,
                        ReadMatrix& evenReadMatrix) {
        std::vector<std::string> fileUrlsReadSlice;
        int img_offset;
        if (m_id < sliceStep) {
            img_offset = m_id * nSlice;
            fileUrlsReadSlice.assign(fileUrlsRead.begin() + img_offset, fileUrlsRead.begin() + img_offset + nSlice);
        } else {
            img_offset = nSlice * sliceStep + (nSlice - 1) * (m_id - sliceStep);
            fileUrlsReadSlice.assign(fileUrlsRead.begin() + img_offset, fileUrlsRead.begin() + img_offset + nSlice - 1);
        }
        for (unsigned int i = 0; i < fileUrlsReadSlice.size(); ++i) {
            GDALDataset *readDataset = (GDALDataset *)GDALOpen(fileUrlsReadSlice[i].c_str(), GA_ReadOnly);
            if (readDataset == nullptr) {
                // @TODO: Handle error
                continue;
            }
            // Read the data for one band
            GDALRasterBand *band = readDataset->GetRasterBand(band_id);
            auto pixelsRead = band->RasterIO(GF_Read, 0, 0, m_N_row, m_N_col,
                           evenReadMatrix.data() + (img_offset + i) * m_N_row * m_N_col, m_N_row, m_N_col, GDT_UInt16, 0, 0);
            if (pixelsRead != m_N_row * m_N_col) {
                // @TODO: Reading was not successful for this file, handle it
            }
            GDALClose(readDataset);
        }
    }

    
};
