#include "misc.cpp"
#include "seasconv.cpp"

class LandsatPipeline {
private:    
    size_t m_id;
    size_t m_N_row;
    size_t m_N_col;
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

public:
    LandsatPipeline(size_t id, 
                    size_t N_row, 
                    size_t N_col, 
                    size_t N_img, 
                    size_t N_aimg, 
                    size_t N_ipy, 
                    size_t N_aipy, 
                    size_t N_aggr, 
                    size_t N_years, 
                    size_t N_slice) {
        m_id = id; 
        m_N_row = N_row; 
        m_N_col = N_col; 
        m_N_img = N_img; 
        m_N_aimg = N_aimg; 
        m_N_ipy = N_ipy; 
        m_N_aipy = N_aipy; 
        m_N_aggr = N_aggr; 
        m_N_years = N_years;
        m_N_slice = N_slice;
        m_N_slice_read = std::ceil((float)N_img/(float)m_N_slice);
        m_slice_read_step = m_N_slice - m_N_slice_read*m_N_slice + N_img;
        size_t N_pix = N_row*N_col;
        m_N_slice_proc = std::ceil((float)N_pix/(float)m_N_slice);
        m_slice_proc_step = m_N_slice - m_N_slice_proc*m_N_slice + N_pix;
    }

    template <class ReadMatrix>
    void readInputFiles(std::vector<std::string> fileUrlsRead,
                        size_t band_id,
                        ReadMatrix& readMatrix,
                        GDALDataType readType) {
        std::vector<std::string> fileUrlsReadSlice;
        size_t img_offset;
        if (m_id < m_slice_read_step) {
            img_offset = m_id * m_N_slice_read;
            fileUrlsReadSlice.assign(fileUrlsRead.begin() + img_offset, fileUrlsRead.begin() + img_offset + m_N_slice_read);
        } else {
            img_offset = m_N_slice_read * m_slice_read_step + (m_N_slice_read - 1) * (m_id - m_slice_read_step);
            fileUrlsReadSlice.assign(fileUrlsRead.begin() + img_offset, fileUrlsRead.begin() + img_offset + m_N_slice_read - 1);
        }
        for (size_t i = 0; i < fileUrlsReadSlice.size(); ++i) {
            GDALDataset *readDataset = (GDALDataset *)GDALOpen(fileUrlsReadSlice[i].c_str(), GA_ReadOnly);
            if (readDataset == nullptr) {
                // @TODO: Handle error
                std::cerr << "Error 1111111 \n";
                GDALClose(readDataset);
                continue;
            }
            // Read the data for one band
            GDALRasterBand *band = readDataset->GetRasterBand(band_id);
            CPLErr pixelsRead = band->RasterIO(GF_Read, 0, 0, m_N_row, m_N_col,
                           readMatrix.data() + (img_offset + i) * m_N_row * m_N_col, m_N_row, m_N_col, readType, 0, 0);
            if (pixelsRead != CE_None) {
                // @TODO: Reading was not successful for this file, handle it
                std::cerr << "Error 2222222222 \n";
                std::cerr << "pixelsRead " << pixelsRead << "\n";
                GDALClose(readDataset);
                continue;
            }
            GDALClose(readDataset);
        }
    }

    void processData(MatrixUI16& readMatrix,
                     MatrixBool& clearSkyMask,
                     float attSeas,
                     float attEnv,
                     float scaling,
                     MatrixUI8& writeMatrix,
                     MatrixUI8& qaWriteMatrix) {
        using VectorReal = typename TypesEigen<float>::VectorReal;
        using VectorComplex = typename TypesEigen<float>::VectorComplex;
        // ------------------------------
        // Slicing the pixels
        // ------------------------------
        size_t pix_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            pix_offset = m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            pix_offset = m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        // Setting to 0 the non-clear sky entries
        readMatrix.block(pix_offset, 0, N_pix_slice, m_N_img) = 
            (!clearSkyMask.block(pix_offset, 0, N_pix_slice, m_N_img).array()).select(0, readMatrix.block(pix_offset, 0, N_pix_slice, m_N_img));

        // ------------------------------
        // Aggregating the images
        // ------------------------------
        Eigen::SparseMatrix<unsigned short> aggMatrix(m_N_img, m_N_aimg);
        aggMatrix.reserve(m_N_aimg*m_N_aggr);
        // Fill the sparse matrix with ones to perform the aggregation
        for (size_t y = 0; y < m_N_years; ++y) {
            for (size_t i = 0; i < m_N_aipy; ++i) {
                for (size_t j = 0; j < m_N_aggr; ++j) {
                    size_t rowIdx = y*m_N_ipy+i*m_N_aggr+j;
                    size_t colIdx = y*m_N_aipy+i;
                    if (rowIdx < m_N_img) {
                        aggMatrix.insert(rowIdx, colIdx) = 1;
                    }
                }
            }
        }
        aggMatrix.finalize();
        size_t N_ext = m_N_aimg * 2;
        size_t N_fft = m_N_aimg + 1;
        MatrixFloat aggrMatrix = MatrixFloat::Zero(N_pix_slice, N_ext);
        MatrixFloat validMatrix = MatrixFloat::Zero(N_pix_slice, N_ext);
        { // Use blocks to automatically delate the QA band after extracting the mask
            MatrixUI16 normMatrix = clearSkyMask.block(pix_offset, 0, N_pix_slice, m_N_img).cast<unsigned short>() * aggMatrix;
            aggrMatrix.block(0, 0, N_pix_slice, m_N_aimg) = (readMatrix.block(pix_offset, 0, N_pix_slice, m_N_img) * aggMatrix).cast<float>().array()
                / normMatrix.cast<float>().array() * scaling;
            MatrixBool aggrMask = normMatrix.array() == 0;
            aggrMatrix.block(0, 0, N_pix_slice, m_N_aimg) = aggrMask.select(0, aggrMatrix.block(0, 0, N_pix_slice, m_N_aimg));
            validMatrix.block(0, 0, N_pix_slice, m_N_aimg) = (!aggrMask.array()).select(1, validMatrix.block(0, 0, N_pix_slice, m_N_aimg));
        }

        // ------------------------------
        // Gap-filling with SeasConv
        // ------------------------------
        MatrixComplexFloat aggrMatrixFFT = MatrixComplexFloat::Zero(N_pix_slice, N_fft);
        MatrixComplexFloat validMatrixFFT = MatrixComplexFloat::Zero(N_pix_slice, N_fft);
        // Plan the FFT computation
        WrapFFT<float> wrapFFT = WrapFFT<float>(N_fft, N_ext, N_pix_slice,
            aggrMatrix.data(), validMatrix.data(), aggrMatrixFFT.data(), validMatrixFFT.data());
        // WrapFFT<float> wrapFFT = WrapFFT<float>(N_fft, N_ext, N_pix_slice,
        //     aggrMatrix.data(), aggrMatrix.data(), aggrMatrixFFT.data(), aggrMatrixFFT.data());


        VectorReal conv(m_N_aimg);
        VectorReal convExt = VectorReal::Ones(N_ext);
        VectorComplex convExtFFT(N_fft);
        VectorReal normQa = VectorReal::Ones(N_ext);
        VectorComplex normQaFFT(N_fft);
        compute_conv_vec<float>(conv, m_N_years, m_N_aipy, attSeas, attEnv);
        convExt.segment(0,m_N_aimg) = conv;
        convExt.segment(m_N_aimg+1, m_N_aimg-1) = conv.reverse().segment(0,m_N_aimg-1);
        normQa.segment(m_N_aimg, m_N_aimg) = VectorReal::Zero(m_N_aimg);

        wrapFFT.computeTimeserisFFT();
        wrapFFT.computeMaskFFT();
        wrapFFT.computeFFT(convExt, convExtFFT);
        wrapFFT.computeFFT(normQa, normQaFFT);
        // Convolve the vectors
        aggrMatrixFFT.array().rowwise() *= convExtFFT.array().transpose();
        validMatrixFFT.array().rowwise() *= convExtFFT.array().transpose();
        normQaFFT.array() *= convExtFFT.array();

        // Compute forward transformations    
        wrapFFT.computeTimeserisIFFT();
        wrapFFT.computeMaskIFFT();
        wrapFFT.computeIFFT(normQaFFT, normQa);
        wrapFFT.clean();

        // Renormalize the result
        aggrMatrix.block(0, 0, N_pix_slice, m_N_aimg).array() /= 
            validMatrix.block(0, 0, N_pix_slice, m_N_aimg).array();
        validMatrix.block(0, 0, N_pix_slice, m_N_aimg) = 
            (validMatrix.block(0, 0, N_pix_slice, m_N_aimg).array().rowwise() / 
            normQa.segment(0,m_N_aimg).array().transpose()) / N_ext * 100.;


        writeMatrix.block(pix_offset, 0, N_pix_slice, m_N_aimg) = aggrMatrix.block(0, 0, N_pix_slice, m_N_aimg).cast<unsigned char>();
        qaWriteMatrix.block(pix_offset, 0, N_pix_slice, m_N_aimg) = validMatrix.block(0, 0, N_pix_slice, m_N_aimg).cast<unsigned char>();
    }

    void processMask(MatrixUI8& qualityAssessmentBand,
                     MatrixBool& clearSkyMask) {
        size_t pix_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            pix_offset = m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            pix_offset = m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }
        clearSkyMask.block(pix_offset, 0, N_pix_slice, m_N_img) = 
                     ((qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() >= 1) && (qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() <= 2))
                  || ((qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() >= 5) && (qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() <= 6))
                  || ((qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() >= 11) && (qualityAssessmentBand.block(pix_offset, 0, N_pix_slice, m_N_img).array() <= 17));

    }

    
};
