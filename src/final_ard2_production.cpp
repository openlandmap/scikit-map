#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <fftw3.h>
#include "LandsatPipelineMultiBand.hpp"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // ################################################
    // ############ Input parameters ##################
    // ################################################
    std::string tile = "034E_10N";
    size_t start_year = 1997;
    size_t end_year = 2022;
    size_t N_threads = 96;
    size_t N_bands = 8;
    size_t N_aggr = 4; // Aggregation
    float attSeas = 50.;
    float attEnvPerYear = 5.;
    float savingScaling1 = 250./40000.;
    float savingScaling2 = 250./100.;
    std::vector<float> scalings = {savingScaling1,
                                   savingScaling1,
                                   savingScaling1,
                                   savingScaling1,
                                   savingScaling1,
                                   savingScaling1,
                                   savingScaling2,
                                   1.};
    std::string out_folder = "/mnt/inca/ard2_production/scikit-map/src/" + tile;

    // ################################################
    // ################### Setup ######################
    // ################################################
    size_t N_row = 4004;
    size_t N_col = 4004;
    size_t N_pix = N_row * N_col;
    size_t N_ipy = 23; // Images per year 
    size_t N_aipy = std::ceil((float)N_ipy/(float)N_aggr); // Aggregated images per year
    size_t N_years = end_year-start_year+1;
    float attEnv = attEnvPerYear*(float)N_years;
    size_t N_img = N_years*N_ipy; // Images
    size_t N_aimg = N_years*N_aipy; // Aggregated images
    size_t N_slice;
    size_t offsetQA = (N_bands-1)*N_pix;
    size_t band_id_read;
    size_t band_id_proc;
    size_t band_id_write;
    omp_set_num_threads(N_threads);
    std::vector<std::string> fileUrlsRead;
    std::vector<std::vector<std::string>> filePathsWrite;
    std::vector<std::string> qaPathsWrite;
    CPLSetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif");
    CPLSetConfigOption("GDAL_HTTP_MULTIRANGE", "SINGLE_GET");
    CPLSetConfigOption("GDAL_HTTP_CONNECTTIMEOUT", "320");
    CPLSetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO");
    CPLSetConfigOption("GDAL_HTTP_VERSION", "1.0");
    CPLSetConfigOption("GDAL_HTTP_TIMEOUT", "320");
    CPLSetConfigOption("CPL_CURL_GZIP", "NO");
    GDALAllRegister();  // Initialize GDAL
    Eigen::initParallel();
    Eigen::setNbThreads(N_threads);

    // ################################################
    // ######### Input/output files ###################
    // ################################################

    std::cout << "Creating the URLs" << std::endl;
    // Creating reading and writng urls/paths
    for (size_t y = start_year; y <= end_year; ++y) {
        for (size_t i = 0; i < N_ipy; ++i) {
            size_t img_offset = (y - 1997)*N_ipy + i;
            std::string str = "http://192.168.49."+ std::to_string(30 +  img_offset%9) +":8333/landsat-ard2/" + std::to_string(y) + "/" + tile + "/" + std::to_string(392+img_offset) + ".tif";
            fileUrlsRead.push_back(str);
        }
    }
    std::cout << "Creating the output files paths" << std::endl;
    for (size_t band_id = 1; band_id <= N_bands; ++band_id) {
        std::vector<std::string> tmpVector;
        for (size_t y = start_year; y <= end_year; ++y) {
            for (size_t i = 0; i < 6; ++i) {
                std::string str = out_folder + "/ogh_b" + std::to_string(band_id) + "_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
                tmpVector.push_back(str);
            }
        }
        filePathsWrite.push_back(tmpVector);
    }

    for (size_t y = start_year; y <= end_year; ++y) {
        for (size_t i = 0; i < 6; ++i) {
            std::string str = out_folder + "/ogh_qa_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
            qaPathsWrite.push_back(str);
        }
    }

    // Getting format info from the original files to write the output files consistently
    GDALDataset *inputDataset = (GDALDataset *)GDALOpen(fileUrlsRead[0].c_str(), GA_ReadOnly);
    double geotransform[6];
    inputDataset->GetGeoTransform(geotransform);
    auto projectionRef = inputDataset->GetProjectionRef();
    auto noDataValue = inputDataset->GetRasterBand(1)->GetNoDataValue();
    GDALClose(inputDataset);

    // ###################################################################
    // ######### Creating data structures and saving folder ##############
    // ###################################################################

    std::cout << "Creating data structures and saving folder" << std::endl;
    auto t_tmp = tic();
    // Creating matricies to storage the necessary data
    // Storage in ColMajor to match the RasterIO reading order
    // Compared to the 3D tensor in Python we have: C++ (p, t) -> Python (floor(p/N_col), p%N_col, t)    

    TypesEigen<float>::VectorReal clearSkyFraction(N_img);

    // Create the output folder
    if (!fs::exists(out_folder)) {
        if (!fs::create_directory(out_folder)) {
            std::cerr << "Failed to create the output folder." << std::endl;
            return 1;
        }
    }

    auto fftw_flags = FFTW_MEASURE;
    size_t N_fft = N_aimg + 1;
    size_t N_ext = N_aimg * 2;
    TypesEigen<float>::VectorReal tmpRealVector(N_ext);
    TypesEigen<float>::VectorComplex tmpComplexVector(N_fft);
    // Create plans for forward and backward DFT (Discrete Fourier Transform)
    TypesFFTW<float>::PlanType fftPlan_fw_ts = 
        TypesFFTW<float>::PlanDFT_R2C(N_ext, tmpRealVector.data(), reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), fftw_flags);
    TypesFFTW<float>::PlanType fftPlan_fw_mask = 
        TypesFFTW<float>::PlanDFT_R2C(N_ext, tmpRealVector.data(), reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), fftw_flags);
    TypesFFTW<float>::PlanType fftPlan_bw_conv_ts = 
        TypesFFTW<float>::PlanDFT_C2R(N_ext, reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), tmpRealVector.data(), fftw_flags);
    TypesFFTW<float>::PlanType fftPlan_bw_conv_mask = 
        TypesFFTW<float>::PlanDFT_C2R(N_ext, reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), tmpRealVector.data(), fftw_flags);
    TypesFFTW<float>::PlanType plan_forward = 
        TypesFFTW<float>::PlanDFT_R2C(N_ext, tmpRealVector.data(), reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), fftw_flags);
    TypesFFTW<float>::PlanType plan_backward = 
        TypesFFTW<float>::PlanDFT_C2R(N_ext, reinterpret_cast<TypesFFTW<float>::ComplexType*>(tmpComplexVector.data()), tmpRealVector.data(), fftw_flags);

    std::cout << "Done in " << toc(t_tmp) << " s \n";
    

    // #######################################################################
    // ################## Reading, processing and saving data ################
    // #######################################################################

    MatrixUI8 writeData(N_pix*N_bands, N_aimg);
    { // Use blocks to automatically delate not used Eigen matricies band after extracting the mask
        MatrixFloat aggrData(N_pix*N_bands, N_aimg);
        {
            std::cout << "Reading all the bands" << "\n";
            int bandList[N_bands];
            for (int i = 0; i < N_bands; ++i) {
                bandList[i] = i + 1;
            }
            t_tmp = tic();
            MatrixUI16 bandsData(N_pix*N_bands, N_img);
            N_slice = N_img;
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.readInputFilesMB<MatrixUI16>(fileUrlsRead[i], bandList, N_bands, bandsData, GDT_UInt16);
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";

            std::cout << "Creating the clear-sky mask" << "\n";
            t_tmp = tic();
            MatrixBool clearSkyMask(N_pix, N_img);
            N_slice = N_threads;
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.processMaskMB<MatrixUI16>(bandsData, clearSkyMask, offsetQA);
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";

            std::cout << "Computing clear-sky fraction per time frame" << "\n";
            t_tmp = tic();
            N_slice = N_img;
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.computeClearSkyFractionMB(clearSkyFraction, offsetQA, bandsData);
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";

            std::cout << "Setting to zero the non clear-sky pixels" << "\n";
            std::cout << "####### TODOOOOOO precompute the boolean mask for all the bands" << "\n";
            t_tmp = tic();
            N_slice = N_threads;
            for (size_t b = 0; b < N_bands-1; ++b) {
                size_t bandOffset = b*N_pix;
                // std::cout << "---- Band " << b+1 << "\n";
                #pragma omp parallel for
                for (size_t i = 0; i < N_slice; ++i) {        
                    LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                    lsp.maskDataMB(bandOffset, clearSkyMask, bandsData);
                }
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";            

            std::cout << "Computing wighted summation for the aggregation" << "\n";            
            std::cout << "####### TODOOOOOO do not split by band but only by threads" << "\n";
            t_tmp = tic();
            N_slice = N_threads;
            for (size_t b = 0; b < N_bands; ++b) {
                size_t bandOffset = b*N_pix;
                // std::cout << "---- Band " << b << "\n";
                #pragma omp parallel for
                for (size_t i = 0; i < N_slice; ++i) {        
                    LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                    lsp.aggregationSummationMB(bandsData, bandOffset, clearSkyFraction, aggrData, scalings[b]);
                }
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";    
        }
        
        std::cout << "Creating gap mask" << "\n";
        MatrixBool gapMask(N_pix, N_aimg);
        t_tmp = tic();
        N_slice = N_aimg;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
            lsp.creatingGapMaskMB(offsetQA, aggrData, gapMask);
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Normalizing the aggregated values" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        for (size_t b = 0; b < N_bands; ++b) {
            size_t bandOffset = b*N_pix;
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.normalizeAggrDataMB(bandOffset, offsetQA, aggrData, gapMask);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Computing convolution vector" << "\n";
        t_tmp = tic();
        TypesEigen<float>::VectorReal normQa(N_ext);
        TypesEigen<float>::VectorComplex convExtFFT(N_fft);
        LandsatPipelineMultiBand lsp(0, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
        lsp.computeConvolutionVectorMB(normQa, convExtFFT, 
            fftPlan_fw_ts, fftPlan_fw_mask, fftPlan_bw_conv_ts, fftPlan_bw_conv_mask,
            plan_forward, plan_backward, attSeas, attEnv);
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Performing the convolution for SeasConv" << "\n";
        MatrixFloat convData(N_pix*N_bands, N_aimg);
        t_tmp = tic();
        N_slice = N_threads;
        for (size_t b = 0; b < N_bands; ++b) {
            size_t bandOffset = b*N_pix;
            // std::cout << "---- Band " << b << "\n";
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.convolutionSeasConvMB(bandOffset, aggrData, convData, convExtFFT,
                    fftPlan_fw_ts, fftPlan_fw_mask, fftPlan_bw_conv_ts, fftPlan_bw_conv_mask, plan_forward, plan_backward);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Renormalizing the gap-filled data" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        for (size_t b = 0; b < N_bands-1; ++b) {
            size_t bandOffset = b*N_pix;
            // std::cout << "---- Band " << b << "\n";
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {        
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                lsp.renormalizeSeasConvMB(bandOffset, offsetQA, convData, aggrData, gapMask, writeData);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Extracting the QA band" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
            lsp.extractQaMB(normQa, offsetQA, convData, gapMask, writeData);
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";


        std::cout << "#################################### \n";
        std::cout << "Aggregated band 0 \n";
        std::cout << aggrData.block(0+1,0,3,12) << std::endl;
        std::cout << "Aggregated QA \n";
        std::cout << aggrData.block(offsetQA+1,0,3,12) << std::endl;
        std::cout << "Gap-filled band 0  \n";
        std::cout << convData.block(0+1,0,3,12) << std::endl;
        std::cout << "Gap-filled QA  \n";
        std::cout << convData.block(offsetQA+1,0,3,12) << std::endl;
        std::cout << "Final band 0 \n";
        std::cout << writeData.block(0+1,0,3,12).cast<int>() << std::endl;
        std::cout << "Final QA \n";
        std::cout << writeData.block(offsetQA+1,0,3,12).cast<int>() << std::endl;
    }

    std::cout << "Saving files" << "\n";
    t_tmp = tic();
    N_slice = N_aimg;
    for (size_t b = 0; b < N_bands; ++b) {
        size_t bandOffset = b*N_pix;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
            lsp.writeOutputFilesMB(filePathsWrite[b][i], projectionRef, noDataValue, geotransform, writeData, bandOffset);
        }
    }
    std::cout << "Done " << toc(t_tmp) << " s \n";

    std::cout << "Total time " << toc(t_start) << " s \n";

    // #######################################################################
    // ################## Cleaning up ################
    // #######################################################################

    TypesFFTW<float>::DestroyPlan(fftPlan_fw_ts);
    TypesFFTW<float>::DestroyPlan(fftPlan_fw_mask);
    TypesFFTW<float>::DestroyPlan(fftPlan_bw_conv_ts);
    TypesFFTW<float>::DestroyPlan(fftPlan_bw_conv_mask);
    TypesFFTW<float>::DestroyPlan(plan_forward);
    TypesFFTW<float>::DestroyPlan(plan_backward);

    return 0;
}
