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
        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = clearSkyMask.select((unsigned short) 1, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void computeClearSkyFractionMB(TypesEigen<float>::VectorReal& clearSkyFraction,
                                   size_t offset,
                                   MatrixUI16& bandsData) {
        clearSkyFraction(m_id) = (float) bandsData.block(offset, m_id, m_N_pix, 1).cast<int>().sum() / (float) m_N_pix;
    }

    void maskDataMB(size_t bandOffset,
                    MatrixBool& clearSkyMask,
                    MatrixUI16& bandsData) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        bandsData.block(band_offset, 0, N_pix_slice, m_N_img) = (!clearSkyMask.block(band_offset - bandOffset, 0, N_pix_slice, m_N_img).array()).select(
            (unsigned short) 0, bandsData.block(band_offset, 0, N_pix_slice, m_N_img));
    }

    void aggregationSummationMB(MatrixUI16& bandsData,
                                size_t bandOffset,
                                TypesEigen<float>::VectorReal& clearSkyFraction,
                                MatrixFloat& aggrData,
                                float scaling) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = bandOffset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = bandOffset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

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

        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg) = bandsData.block(band_offset, 0, N_pix_slice, m_N_img).cast<float>() * (aggMatrix * scaling);
    }


    void creatingGapMaskMB(size_t offset, MatrixFloat& aggrData, MatrixBool& gapMask) {
        gapMask.block(0, m_id, m_N_pix, 1) = aggrData.block(offset, m_id, m_N_pix, 1).array() == 0;
    }

    void normalizeAggrDataMB(size_t bandOffset,
                             size_t maskOffset,
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

        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).array() /= aggrData.block(mask_offset, 0, N_pix_slice, m_N_aimg).array();
        aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg) = gapMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).select(
            0., aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg));
    }


    void computeConvolutionVectorMB(TypesEigen<float>::VectorReal& normQa,
                                    TypesEigen<float>::VectorComplex& convExtFFT,
                                    TypesFFTW<float>::PlanType fftPlan_fw_ts,
                                    TypesFFTW<float>::PlanType fftPlan_fw_mask,
                                    TypesFFTW<float>::PlanType fftPlan_bw_conv_ts,
                                    TypesFFTW<float>::PlanType fftPlan_bw_conv_mask,
                                    TypesFFTW<float>::PlanType plan_forward,
                                    TypesFFTW<float>::PlanType plan_backward,
                                    float attSeas,
                                    float attEnv) {

        using VectorReal = typename TypesEigen<float>::VectorReal;
        using VectorComplex = typename TypesEigen<float>::VectorComplex;
        size_t N_ext = m_N_aimg * 2;
        size_t N_fft = m_N_aimg + 1;
        MatrixFloat tmp1 = MatrixFloat::Zero(1, N_ext);
        MatrixFloat tmp2 = MatrixFloat::Zero(1, N_ext);
        MatrixComplexFloat tmp3 = MatrixComplexFloat::Zero(1, N_fft);
        MatrixComplexFloat tmp4 = MatrixComplexFloat::Zero(1, N_fft);
        // Plan the FFT computation
        WrapFFT<float> wrapFFT = WrapFFT<float>(N_fft, N_ext, 1,
                                                fftPlan_fw_ts,
                                                fftPlan_fw_mask,
                                                fftPlan_bw_conv_ts,
                                                fftPlan_bw_conv_mask,
                                                plan_forward,
                                                plan_backward,
                                                tmp1.data(), 
                                                tmp1.data(),
                                                tmp3.data(),
                                                tmp4.data());

        TypesEigen<float>::VectorReal conv(m_N_aimg);
        TypesEigen<float>::VectorReal convExt = VectorReal::Ones(N_ext);
        TypesEigen<float>::VectorComplex normQaFFT(N_fft);
        compute_conv_vec<float>(conv, m_N_years, m_N_aipy, attSeas, attEnv);
        convExt.segment(0,m_N_aimg) = conv;
        convExt.segment(m_N_aimg+1, m_N_aimg-1) = conv.reverse().segment(0,m_N_aimg-1);
        normQa.setOnes();
        normQa.segment(m_N_aimg, m_N_aimg) = VectorReal::Zero(m_N_aimg);
        normQa(0) = 0.;

        wrapFFT.computeFFT(convExt, convExtFFT);
        wrapFFT.computeFFT(normQa, normQaFFT);
        normQaFFT.array() *= convExtFFT.array();

        // Compute forward transformations
        wrapFFT.computeIFFT(normQaFFT, normQa);
        wrapFFT.clean();
    }


    void convolutionSeasConvMB(size_t offset,
                               MatrixFloat& aggrData,
                               MatrixFloat& convData,
                               TypesEigen<float>::VectorComplex convExtFFT,
                               TypesFFTW<float>::PlanType fftPlan_fw_ts,
                               TypesFFTW<float>::PlanType fftPlan_fw_mask,
                               TypesFFTW<float>::PlanType fftPlan_bw_conv_ts,
                               TypesFFTW<float>::PlanType fftPlan_bw_conv_mask,
                               TypesFFTW<float>::PlanType plan_forward,
                               TypesFFTW<float>::PlanType plan_backward) {
        size_t band_offset;
        size_t N_pix_slice;
        if (m_id < m_slice_proc_step) {
            band_offset = offset + m_id * m_N_slice_proc;
            N_pix_slice = m_N_slice_proc;
        } else {
            band_offset = offset + m_N_slice_proc * m_slice_proc_step + (m_N_slice_proc - 1) * (m_id - m_slice_proc_step);
            N_pix_slice = m_N_slice_proc - 1;
        }

        size_t N_ext = m_N_aimg * 2;
        size_t N_fft = m_N_aimg + 1;

        MatrixFloat aggrDataExt = MatrixFloat::Zero(N_pix_slice, N_ext);
        MatrixFloat tmp1(N_pix_slice, N_ext);
        MatrixComplexFloat aggrMatrixFFT = MatrixComplexFloat::Zero(N_pix_slice, N_fft);
        MatrixComplexFloat tmp2(N_pix_slice, N_fft);
        // Plan the FFT computation
        WrapFFT<float> wrapFFT = WrapFFT<float>(N_fft, N_ext, N_pix_slice,
                                                fftPlan_fw_ts,
                                                fftPlan_fw_mask,
                                                fftPlan_bw_conv_ts,
                                                fftPlan_bw_conv_mask,
                                                plan_forward,
                                                plan_backward,
                                                aggrDataExt.data(), 
                                                tmp1.data(),
                                                aggrMatrixFFT.data(),
                                                tmp2.data());
     
        aggrDataExt.block(0, 0, N_pix_slice, m_N_aimg) = aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg);

        // Compute forward transformations    
        wrapFFT.computeTimeserisFFT();
        // Convolve the vectors
        aggrMatrixFFT.array().rowwise() *= convExtFFT.array().transpose();

        // Compute backward transformations
        wrapFFT.computeTimeserisIFFT();
        wrapFFT.clean();

        convData.block(band_offset, 0, N_pix_slice, m_N_aimg) = aggrDataExt.block(0, 0, N_pix_slice, m_N_aimg);
    }


    void renormalizeSeasConvMB(size_t bandOffset,
                                size_t maskOffset,
                                MatrixFloat& convData,
                                MatrixFloat& aggrData,
                                MatrixBool& gapMask,
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
        // In the following the absolute value in the selections of not filled pixels (NaN) takes care of the case -0.0
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) = 
            (convData.block(mask_offset, 0, N_pix_slice, m_N_aimg).cwiseAbs().array() == (float)0.).select(
                (unsigned char)255, writeData.block(band_offset, 0, N_pix_slice, m_N_aimg));

        // Restore the values that were already not gap
        writeData.block(band_offset, 0, N_pix_slice, m_N_aimg) =
                (!gapMask.block(band_offset-bandOffset, 0, N_pix_slice, m_N_aimg).array()).select(
                aggrData.block(band_offset, 0, N_pix_slice, m_N_aimg).cast<unsigned char>(), 
                writeData.block(band_offset, 0, N_pix_slice, m_N_aimg));

    }


    void extractQaMB(TypesEigen<float>::VectorReal& normQa,
                    size_t maskOffset,
                    MatrixFloat& aggrDataExt,
                    MatrixBool& gapMask,
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
        aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            (aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg).array().rowwise() / 
            normQa.segment(0,m_N_aimg).array().transpose()) / (2. * (float)m_N_aimg) * 249.;

        // Saturate to 249 higher values to avoid numerical issues in the conversion
        aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            (aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg).array() > 249.).select(
                249., aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg));

        // Store in byte the QA values rounding to the higher integer
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg).array().ceil().cast<unsigned char>();

        // Set to 255 the values that are still not gap-filled
        // In the following the absolute value in the selections of not filled pixels (NaN) takes care of the case -0.0
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) = 
            (aggrDataExt.block(mask_offset, 0, N_pix_slice, m_N_aimg).cwiseAbs().array() == (float)0.).select(
            (unsigned char)255, writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg));

        // Set to 250 the values that were already not gap
        writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg) =
                (!gapMask.block(mask_offset-maskOffset, 0, N_pix_slice, m_N_aimg).array()).select(
                (unsigned char)250, writeData.block(mask_offset, 0, N_pix_slice, m_N_aimg));
    }


    template <class T, class U>
    void writeOutputFilesMB(std::string fileName,
                          T projectionRef,
                          U noDataValue,
                          double *geotransform,
                          MatrixUI8& writeData,
                          size_t bandOffset) {
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        GDALDataset *writeDataset = driver->Create(
            (fileName).c_str(),
            m_N_row, m_N_col, 1, GDT_Byte, nullptr
        );        
        writeDataset->SetProjection(projectionRef);
        writeDataset->SetGeoTransform(geotransform);
        GDALRasterBand *writeBand = writeDataset->GetRasterBand(1);
        writeBand->SetNoDataValue(noDataValue);
        char **papszOptions = NULL;
        papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
        writeDataset->SetMetadata(papszOptions, "IMAGE_STRUCTURE");
        CSLDestroy(papszOptions);
        // Write the converted data to the new dataset
        auto outWrite = writeBand->RasterIO(
            GF_Write, 0, 0, m_N_row, m_N_col,
            writeData.data() + m_id * m_N_row * m_N_col + bandOffset,
            m_N_row, m_N_col, GDT_Byte, 0, 0);
        if (outWrite != CE_None) {
            // @TODO: Reading was not successful for this file, handle it
            std::cerr << "Error 33333333 \n";
            std::cerr << "outWrite " << outWrite << "\n";
        }
        GDALClose(writeDataset);

    }


    
};