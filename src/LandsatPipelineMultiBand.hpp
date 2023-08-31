#include "misc.cpp"
#include "seasconv.cpp"

class LandsatPipelineMultiBand {
private:    
    size_t m_id;
    size_t m_N_row;
    size_t m_N_col;
    size_t m_N_pix;
    size_t m_N_img;
    size_t m_N_aimg;
    size_t m_N_ipy;
    size_t m_N_aipy;
    size_t m_N_aggr;
    size_t m_N_years;
    size_t m_N_slice;
    size_t m_N_slice_read;
    size_t m_slice_read_step;
    size_t m_N_slice_proc;
    size_t m_slice_proc_step;
    size_t m_N_slice_procMB;
    size_t m_slice_proc_stepMB;

public:
    LandsatPipelineMultiBand(size_t id, 
                            size_t N_row, 
                            size_t N_col, 
                            size_t N_img, 
                            size_t N_aimg, 
                            size_t N_ipy, 
                            size_t N_aipy, 
                            size_t N_aggr, 
                            size_t N_years, 
                            size_t N_slice,
                            size_t N_bands) {
        m_id = id; 
        m_N_row = N_row; 
        m_N_col = N_col; 
        m_N_pix = N_row*N_col;
        m_N_img = N_img; 
        m_N_aimg = N_aimg; 
        m_N_ipy = N_ipy; 
        m_N_aipy = N_aipy; 
        m_N_aggr = N_aggr; 
        m_N_years = N_years;
        m_N_slice = N_slice;
        m_N_slice_read = std::ceil((float)N_img/(float)m_N_slice);
        m_slice_read_step = m_N_slice - m_N_slice_read*m_N_slice + N_img;
        m_N_slice_proc = std::ceil((float)m_N_pix/(float)m_N_slice);
        m_slice_proc_step = m_N_slice - m_N_slice_proc*m_N_slice + m_N_pix;
        m_N_slice_procMB = std::ceil((float)m_N_pix*N_bands/(float)m_N_slice);
        m_slice_proc_stepMB = m_N_slice - m_N_slice_proc*m_N_slice + m_N_pix*N_bands;
    }


    template <class ReadMatrix>
    void readInputFilesMB(std::string fileUrl,
                        int* bandList,
                        size_t N_bands,
                        ReadMatrix& bandsData,
                        GDALDataType readType) {
        GDALDataset *readDataset = (GDALDataset *)GDALOpen(fileUrl.c_str(), GA_ReadOnly);
        if (readDataset == nullptr) {
            // @TODO: Handle error
            std::cerr << "Error 1111111 \n";
            GDALClose(readDataset);
        }
        // Read the data for one band
        CPLErr outRead = readDataset->RasterIO(GF_Read, 0, 0, m_N_row, m_N_col, bandsData.data() + m_id * m_N_row * m_N_col * N_bands,
                           m_N_row, m_N_col, readType, N_bands, bandList, 0, 0, 0);
        if (outRead != CE_None) {
            // @TODO: Reading was not successful for this file, handle it
            std::cerr << "Error 2222222222 \n";
            std::cerr << "outRead " << outRead << "\n";
            GDALClose(readDataset);
        GDALClose(readDataset);
        }
    }

    template <class ReadMatrix>
    void processMaskMB(ReadMatrix& bandsData,
                       size_t offset) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = offset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = offset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        MatrixBool clearSkyMask =
             ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 1) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 2))
          || ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 5) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 6))
          || ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 11) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 17));

        bandsData.block(band_offset, 0, N_pix_slice, m_N_img).setZero();
        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = clearSkyMask.select((unsigned short) 1, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void computeClearSkyFractionMB(TypesEigen<float>::VectorReal& clearSkyFraction,
                                   size_t offset,
                                   MatrixUI16& bandsData) {
        clearSkyFraction(m_id) = (float) bandsData.block(offset, m_id, m_N_pix, 1).cast<int>().sum() / (float) m_N_pix;
    }

    void maskDataMB(size_t bandOffset,
                    size_t maskOffset,
                    MatrixUI16& bandsData) {
        size_t band_offset;
        size_t mask_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            mask_offset = maskOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            mask_offset = maskOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = (bandsData.block(mask_offset, 0, N_pix_slice, m_N_img).array() == 0)
        .select((unsigned short) 0, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void aggregationSummationMB(MatrixUI16& bandsData,
                                size_t N_bands,
                                TypesEigen<float>::VectorReal& clearSkyFraction,
                                MatrixFloat& aggrData) {

        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_stepMB) {
            band_offset = m_id * m_N_slice_procMB;
            N_pix_slice = m_N_slice_procMB;
        } else {
            band_offset = m_N_slice_procMB * m_slice_proc_stepMB + (m_N_slice_procMB - 1) * (m_id - m_slice_proc_stepMB);
            N_pix_slice = m_N_slice_procMB - 1;
        }

        std::cout << " band_offset " << band_offset << " N_pix_slice " << N_pix_slice << std::endl;

        Eigen::SparseMatrix<float> aggMatrix(m_N_img, m_N_aimg);
        aggMatrix.reserve(m_N_aimg*m_N_aggr);
        // Fill the sparse matrix with ones to perform the aggregation
        for (size_t y = 0; y < m_N_years; ++y) {
            for (size_t i = 0; i < m_N_aipy; ++i) {
                for (size_t j = 0; j < m_N_aggr; ++j) {
                    size_t rowIdx = y*m_N_ipy+i*m_N_aggr+j;
                    size_t colIdx = y*m_N_aipy+i;
                    if (rowIdx < m_N_img) {
                        aggMatrix.insert(rowIdx, colIdx) = clearSkyFraction(rowIdx);
                    }
                }
            }
        }
        aggMatrix.finalize();

        aggrData.block(band_offset, 0, N_pix_slice, m_N_img) = bandsData.block(band_offset, 0, N_pix_slice, m_N_img).cast<float>() * aggMatrix;
    }



    void normalizeAggrDataMB(size_t bandOffset,
                             size_t maskOffset,
                             MatrixFloat& aggrData) {
        size_t band_offset;
        size_t mask_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            mask_offset = maskOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            mask_offset = maskOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        aggrData.block(band_offset, 0, N_pix_slice, m_N_img).array() /= aggrData.block(mask_offset, 0, N_pix_slice, m_N_img).array();
    }

    void creatingValidityMaskMB(size_t offset, MatrixFloat& aggrData) {
        aggrData.block(offset, m_id, m_N_pix, 1) = (aggrData.block(offset, m_id, m_N_pix, 1).array() > 0.).select(1., aggrData.block(offset, m_id, m_N_pix, 1));
    }
};