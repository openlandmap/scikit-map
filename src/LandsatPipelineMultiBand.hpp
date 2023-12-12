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
        m_N_slice_procMB = std::ceil((float)(m_N_pix*N_bands)/(float)m_N_slice);
        m_slice_proc_stepMB = m_N_slice - m_N_slice_procMB*m_N_slice + m_N_pix*N_bands;
    }


    template <class ReadMatrix>
    void readInputFilesMB(std::string fileUrl,
                        int* bandList,
                        size_t N_bands,
                        ReadMatrix& bandsData,
                        GDALDataType readType) {
        GDALDataset *readDataset = (GDALDataset *)GDALOpen(fileUrl.c_str(), GA_ReadOnly);
        if (readDataset == nullptr) {
            bandsData.block(0, m_id, m_N_pix * N_bands, 1).setZero();
            std::cerr << "scikit-map ERROR 1: issues in opening the file with URL " << fileUrl << std::endl;
            std::cout << "Issues in opening the file with URL " << fileUrl << ", considering as gap." << std::endl;
            GDALClose(readDataset);
        } else
        {
            // Read the data for one band
            CPLErr outRead = readDataset->RasterIO(GF_Read, 0, 0, m_N_row, m_N_col, bandsData.data() + m_id * m_N_row * m_N_col * N_bands,
                               m_N_row, m_N_col, readType, N_bands, bandList, 0, 0, 0);
           
            if (outRead != CE_None) {
                bandsData.block(0, m_id, m_N_pix * N_bands, 1).setZero();
                std::cerr << "Error 2: issues in reading the file with URL " << fileUrl << std::endl;
                std::cout << "Issues in reading the file with URL " << fileUrl << ", considering as gap." << std::endl;
                GDALClose(readDataset);
            }
            GDALClose(readDataset);
        }
    }

    
    template <class ReadMatrix>
    unsigned char readAggregatedInputFilesMB(std::string fileDate,
                        std::string in_folder,
                        std::vector<std::string> bandNames,
                        int* bandList,
                        size_t N_bands,
                        ReadMatrix& bandsData,
                        GDALDataType readType) {
        unsigned char ret = 0;
        for (size_t bandId = 0; bandId < N_bands; ++bandId) {
            std::string fileUrl = in_folder + "/" + bandNames[bandId] + ".ard2_m_30m_s_" + fileDate + "_go_epsg.4326_v20230908.tif";
            GDALDataset *readDataset = (GDALDataset *)GDALOpen(fileUrl.c_str(), GA_ReadOnly);
            if (readDataset == nullptr) {
                bandsData.block(m_N_pix * bandId, m_id, m_N_pix, 1).setZero();
                std::cerr << "scikit-map ERROR 1: issues in opening the file with URL " << fileUrl << std::endl;
                std::cout << "Issues in opening the file with URL " << fileUrl << ", considering as gap." << std::endl;
                GDALClose(readDataset);
                ret = 1;
            } else
            {
                // Read the data for one band
                CPLErr outRead = readDataset->RasterIO(GF_Read, 0, 0, m_N_row, m_N_col, bandsData.data() + m_id * m_N_pix * N_bands + bandId * m_N_pix,
                                   m_N_row, m_N_col, readType, 1, bandList, 0, 0, 0);               
                if (outRead != CE_None) {
                    bandsData.block(m_N_pix * bandId, m_id, m_N_pix, 1).setZero();
                    std::cerr << "Error 2: issues in reading the file with URL " << fileUrl << std::endl;
                    std::cout << "Issues in reading the file with URL " << fileUrl << ", considering as gap." << std::endl;
                    GDALClose(readDataset);
                }
                GDALClose(readDataset);
            }
        }
        return ret;
    }

    
    template <class ReadMatrix>
    void processMaskMB(ReadMatrix& bandsData,
                       MatrixBool& clearSkyMask,
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

        clearSkyMask.block(band_offset - offset, 0, N_pix_slice, m_N_img) =
             ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 1) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 2))
          || ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 5) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 6))
          || ((bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() >= 11) && (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() <= 17));

        bandsData.block(band_offset, 0, N_pix_slice, m_N_img).setZero();
        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = clearSkyMask.block(band_offset - offset, 0, N_pix_slice, m_N_img).select(
            (unsigned short) 1, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    
    template <class ReadMatrix>
    void processAggregatedMaskMB(ReadMatrix& bandsData,
                       MatrixBool& clearSkyMask,
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

        clearSkyMask.block(band_offset - offset, 0, N_pix_slice, m_N_img) =
             (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() == 1);
        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = clearSkyMask.block(band_offset - offset, 0, N_pix_slice, m_N_img).select(
            (unsigned short) 1, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void computeClearSkyFractionMB(TypesEigen<double>::VectorReal& clearSkyFraction,
                                   size_t offset,
                                   MatrixUI16& bandsData) {
        clearSkyFraction(m_id) = (float) bandsData.block(offset, m_id, m_N_pix, 1).cast<int>().sum() / (float) m_N_pix;
    }

    template <class ReadMatrix>
    void maskDataMB(size_t bandOffset,
                    MatrixBool& clearSkyMask,
                    ReadMatrix& bandsData) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = (clearSkyMask.block(band_offset - bandOffset, 0, N_pix_slice, m_N_img).array() == false).select(
            (unsigned short) 0, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void aggregationSummationMB(MatrixUI16& bandsData,
                                TypesEigen<double>::VectorReal& clearSkyFraction,
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

        Eigen::SparseMatrix<double> aggMatrix(m_N_img, m_N_aimg);
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

        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg) = bandsData.block(band_offset, 0, N_pix_slice, m_N_img).cast<double>() * aggMatrix;
    }


    void creatingGapMaskMB(size_t offset, MatrixFloat& aggrData, MatrixBool& gapMask) {
        gapMask.block(0, m_id, m_N_pix, 1) = aggrData.block(offset, m_id, m_N_pix, 1).array() == 0;
    }

    void normalizeAggrDataMB(size_t bandOffset,
                             size_t maskOffset,
                             float scaling,
                             MatrixFloat& aggrData,
                             MatrixBool& gapMask) {
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

        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).array() /= (aggrData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array() / scaling);
        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg) = gapMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).select(
            0., aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg));
    }

    
    void normalizeAndConvertAggrDataMB(size_t bandOffset,
                             size_t maskOffset,
                             float scaling,
                             MatrixFloat& aggrData,
                             MatrixUI8& writeData,
                             MatrixBool& gapMask) {
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

        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).array() /= (aggrData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array() / scaling);
        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg) = gapMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).select(
            255., aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg));
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) = aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).cast<unsigned char>();

    }


    void computeConvolutionVectorMB(TypesEigen<double>::VectorComplex& convExtFFT,
                                    bool useFuture,
                                    float attSeas,
                                    float attEnv,
                                    double &minConvValue,
                                    double &maxNormQa) {

        size_t N_ext = m_N_aimg * 2 - 1;
        size_t N_fft = m_N_aimg;
        // Plan the FFT computation

        TypesEigen<double>::VectorReal conv_past(m_N_aimg);
        TypesEigen<double>::VectorReal conv_future(m_N_aimg - 1);
        TypesEigen<double>::VectorReal convExt = TypesEigen<double>::VectorReal::Zero(N_ext);
        TypesEigen<double>::VectorReal normQa = TypesEigen<double>::VectorReal::Ones(N_ext);
        TypesEigen<double>::VectorComplex normQaFFT(N_fft);

        fftw_plan plan_forward = 
            fftw_plan_dft_r2c_1d(N_ext, convExt.data(), reinterpret_cast<fftw_complex*>(convExtFFT.data()), FFTW_MEASURE);
        fftw_plan plan_backward = 
            fftw_plan_dft_c2r_1d(N_ext, reinterpret_cast<fftw_complex*>(convExtFFT.data()), convExt.data(), FFTW_MEASURE);

        compute_conv_vec<double>(conv_past, m_N_years, m_N_aipy, attSeas, attEnv);
        if (useFuture) {
            TypesEigen<double>::VectorReal conv_future = conv_past.reverse().segment(0, m_N_aimg-1);
            convExt.segment(m_N_aimg+1, m_N_aimg-1) = conv_future;
        }
        convExt.segment(0,m_N_aimg) = conv_past;
        minConvValue = convExt.maxCoeff();
        for (size_t i = 0; i < N_ext; i++) {
            if ((convExt(i) > 0.) && (convExt(i) < minConvValue)) {
                minConvValue = convExt(i);
            }
        }
        normQa.segment(m_N_aimg, m_N_aimg-1).setZero();
        normQa(0) = 0.;

        fftw_execute_dft_r2c(plan_forward, convExt.data(), reinterpret_cast<fftw_complex*>(convExtFFT.data()));
        fftw_execute_dft_r2c(plan_forward, normQa.data(), reinterpret_cast<fftw_complex*>(normQaFFT.data()));

        normQaFFT.array() *= convExtFFT.array()/N_ext;

        fftw_execute_dft_c2r(plan_backward, reinterpret_cast<fftw_complex*>(normQaFFT.data()), normQa.data());
        maxNormQa = normQa.segment(0, m_N_aimg).maxCoeff();

        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
    }



    template <class AggrMatrix>
    void convolutionSeasConvMB(size_t offset,
                               MatrixUI8& aggrData,
                               MatrixFloat& convData,
                               TypesEigen<double>::VectorComplex convExtFFT,
                               fftw_plan plan_forward,
                               fftw_plan plan_backward) {

        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = offset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = offset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        size_t N_ext = m_N_aimg * 2 - 1;
        size_t N_fft = m_N_aimg;

        MatrixFloat aggrDataExt = MatrixFloat::Zero(N_pix_slice, N_ext);
        MatrixComplexFloat aggrDataFFT = MatrixComplexFloat::Zero(N_pix_slice, N_fft);
        
        aggrDataExt.block(0, 0, N_pix_slice, m_N_aimg) = aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).cast<double>();

        // Compute the forward transforms 
        for (std::size_t i = 0; i < N_pix_slice; ++i) {
            fftw_execute_dft_r2c(plan_forward, aggrDataExt.data() + i * N_ext, reinterpret_cast<fftw_complex*>(aggrDataFFT.data()) + i * N_fft);
        }
        
        // Convolve the vectors
        aggrDataFFT.array().rowwise() *= convExtFFT.array().transpose();
        
        // Compute backward transformations
        for (std::size_t i = 0; i < N_pix_slice; ++i) {
            fftw_execute_dft_c2r(plan_backward, reinterpret_cast<fftw_complex*>(aggrDataFFT.data()) + i * N_fft, aggrDataExt.data() + i * N_ext);
        }
        
        convData.block(band_offset, 0, N_pix_slice, m_N_aimg) = aggrDataExt.block(0, 0, N_pix_slice, m_N_aimg);
    }

    template <class ReadMatrix>
    void maskNonFilledDataMB(ReadMatrix& bandsData,
                       MatrixBool& nonFilledMask,
                       double minConvValue,
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



        nonFilledMask.block(band_offset - offset, 0, N_pix_slice, m_N_img) = (bandsData.block(band_offset, 0, N_pix_slice, m_N_img).array() < minConvValue);
    }



    void renormalizeSeasConvMB(size_t bandOffset,
                                size_t maskOffset,
                                MatrixFloat& convData,
                                MatrixUI8& aggrData,
                                MatrixBool& clearSkyMask,
                                MatrixBool& nonFilledMask,
                                MatrixUI8& writeData) {
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

        // Renormalize the result
        convData.block(band_offset, 0, N_pix_slice, m_N_aimg).array() /= 
            convData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array();

        // Store in byte the gap-filled values
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) = 
            convData.block(band_offset, 0, N_pix_slice, m_N_aimg).cast<unsigned char>();

        // Set to 255 the values that are still not gap-filled
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) = 
            nonFilledMask.block(band_offset-bandOffset, 0, N_pix_slice, m_N_aimg).select(
                (unsigned char)255, writeData.block(band_offset, 0, N_pix_slice, m_N_aimg));

        // Restore the values that were already not gap
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) =
            clearSkyMask.block(band_offset-bandOffset, 0, N_pix_slice, m_N_aimg).select(
                aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg), 
                writeData.block(band_offset, 0, N_pix_slice, m_N_aimg));

    }

    void convertingToUI8MB(size_t bandOffset,
                                MatrixFloat& aggrData,
                                MatrixUI8& writeData) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        

        // Restore the values that were already not gap
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) = aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).cast<unsigned char>();

    }


    void extractQaMB(double maxNormQa,
                    size_t maskOffset,
                    MatrixFloat& convData,
                    MatrixBool& clearSkyMask,
                    MatrixBool& nonFilledMask,
                    MatrixUI8& writeData) {
        size_t mask_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            mask_offset = maskOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            mask_offset = maskOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        // Renormalize the result
        // Divide by the "best possible filling" QA value: / maxNormQa
        // Scaling from backward transformation in FFTW (N_ext): / (2. * (float)m_N_aimg -1.)
        // Multiply by 249: the gap filled walues will span a QA between 0 to 249
        convData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            convData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array()
            / maxNormQa / (2. * (float)m_N_aimg -1.) * 249.;

        // Saturate to 249 higher values to avoid numerical issues in the conversion
        convData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            (convData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array() > 249.).select(
            249., convData.block(mask_offset, 0, N_pix_slice, m_N_aimg));

        // Store in byte the QA values rounding to the nearest integer
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            (convData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array() + 0.5).floor().cast<unsigned char>();

        // Set to 255 the values that are still not gap-filled
        // In the following the absolute value in the selections of not filled pixels (NaN) takes care of the case -0.0
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            nonFilledMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).select(
            (unsigned char)255, writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg));

        // Set to 250 the values that were already not gap
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) =
            clearSkyMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).select(
            (unsigned char)250, writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg));
    }


  
    template <class T, class U>
    void writeOutputFilesMB(std::string fileName,
                          std::string seaweedPath,
                          T projection,
                          U spatialRef,
                          double *geotransform,
                          MatrixUI8& writeData,
                          size_t dateOffset) {
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        GDALDataset *writeDataset = driver->Create(
            (fileName + "tmp").c_str(),
            m_N_row, m_N_col, 1, GDT_Byte, nullptr
        );        
        writeDataset->SetGeoTransform(geotransform);
        writeDataset->SetSpatialRef(spatialRef);
        writeDataset->SetProjection(projection);
        GDALRasterBand *writeBand = writeDataset->GetRasterBand(1);
        writeBand->SetNoDataValue(255);
        // Write the converted data to the new dataset
        auto outWrite = writeBand->RasterIO(
            GF_Write, 0, 0, m_N_row, m_N_col,
            writeData.data() + m_id * m_N_row * m_N_col + dateOffset,
            m_N_row, m_N_col, GDT_Byte, 0, 0);
        if (outWrite != CE_None) {
            std::cerr << "scikit-map ERROR 3: issues in writing the file " << fileName << std::endl;
            std::cout << "scikit-map ERROR 3: issues in writing the file " << fileName << ", blocking the execution." << std::endl;
            assert(false);
        }
        GDALClose(writeDataset);
        std::string command = "gdal_translate -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 " + fileName + "tmp " + fileName;
        command += " > /dev/null 2>&1";
        int result = system(command.c_str());
        if (result != 0) {
            std::cerr << "scikit-map ERROR 4: issues in comressing the file " << fileName << std::endl;
        }
        command = "rm " + fileName + "tmp";
        command += " > /dev/null 2>&1";
        result = system(command.c_str());
        if (result != 0) {
            std::cerr << "scikit-map ERROR 5: issues in delete the tmp file " << fileName << std::endl;
        }
        command = "mc cp " + fileName + " " + seaweedPath;
        command += " > /dev/null 2>&1";
        result = system(command.c_str());
        if (result != 0) {
            std::cerr << "scikit-map ERROR 6: issues in sending to seaweed the file " << fileName << std::endl;
        }
        command = "rm " + fileName;
        command += " > /dev/null 2>&1";
        result = system(command.c_str());
        if (result != 0) {
            std::cerr << "scikit-map ERROR 7: issues in delete the file " << fileName << std::endl;
        }

    }

};