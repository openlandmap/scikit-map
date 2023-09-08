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

int main(int argc, char* argv[]) {

    
    // ################################################
    // ############ Input parameters ##################
    // ################################################
    size_t start_year = 1997;
    size_t end_year = 1998;
    size_t N_threads = omp_get_max_threads();
    size_t N_bands = 8;
    size_t N_aggr = 4; // Aggregation
    float savingScaling = 250./40000.;
    std::vector<float> scalings = {savingScaling,
                                   savingScaling,
                                   savingScaling,
                                   savingScaling,
                                   savingScaling,
                                   savingScaling,
                                   savingScaling,
                                   1.};

    // Check if at least one tile name was provided
    if (argc < 2) {
        std::cerr << "Input at least one tile name " << std::endl;
        return 1;
    }

    std::cout << std::setprecision(1);
    std::cout << std::fixed << std::endl;
    std::cout << "#TODOOOOOOOOOO LZW compression is not working, check chat with Leandro" << std::endl;

    for (size_t t = 0; t < argc - 1; ++t)
    {
        auto t_start = tic();        
        std::string tile = argv[t + 1]; // This tile is good for debug: 034E_10N

        std::cout << "####################################################" << std::endl;
        std::cout << "############# Processing tile " << tile << " #############" << std::endl;
        std::cout << "####################################################" << std::endl;

        // ################################################
        // ################### Setup ######################
        // ################################################
        std::string out_folder = "/mnt/inca/ard2_production/scikit-map/src/" + tile + "_aggr";
        std::string seaweed_folder = "seaweed/landsat-ard2-prod/" + tile + "/agg";
        size_t N_row = 4004;
        size_t N_col = 4004;
        size_t N_pix = N_row * N_col;
        size_t N_ipy = 23; // Images per year
        size_t N_aipy = std::ceil((float)N_ipy/(float)N_aggr); // Aggregated images per year
        size_t N_years = end_year-start_year+1;
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
        std::vector<std::vector<std::string>> seaweedPathsWrite;
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
            std::vector<std::string> tmpVector2;
            for (size_t y = start_year; y <= end_year; ++y) {
                for (size_t i = 0; i < 6; ++i) {
                    std::string str = out_folder + "/ogh_b" + std::to_string(band_id) + "_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
                    std::string str2 = seaweed_folder + "/ogh_b" + std::to_string(band_id) + "_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
                    tmpVector.push_back(str);
                    tmpVector2.push_back(str2);
                }
            }
            filePathsWrite.push_back(tmpVector);
            seaweedPathsWrite.push_back(tmpVector2);
        }

        for (size_t y = start_year; y <= end_year; ++y) {
            for (size_t i = 0; i < 6; ++i) {
                std::string str = out_folder + "/ogh_qa_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
                qaPathsWrite.push_back(str);
            }
        }

        

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

        // The planning can not be constructed in parallel, so it is shared between the threads
        auto fftw_flags = FFTW_MEASURE;
        size_t N_fft = N_aimg + 1;
        size_t N_ext = N_aimg * 2;
        TypesEigen<float>::VectorReal tmpRealVector(N_ext);
        TypesEigen<float>::VectorComplex tmpComplexVector(N_fft);
        // Create plans for forward and backward DFT (Discrete Fourier Transform)
        fftwf_plan plan_forward = 
                fftwf_plan_dft_r2c_1d(N_ext, tmpRealVector.data(), reinterpret_cast<fftwf_complex*>(tmpComplexVector.data()), FFTW_MEASURE);
            fftwf_plan plan_backward = 
                fftwf_plan_dft_c2r_1d(N_ext, reinterpret_cast<fftwf_complex*>(tmpComplexVector.data()), tmpRealVector.data(), FFTW_MEASURE);
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

                std::cout << "Computing weighted summation for the aggregation" << "\n";
                t_tmp = tic();
                N_slice = N_threads;
                #pragma omp parallel for
                for (size_t i = 0; i < N_slice; ++i) {        
                    LandsatPipelineMultiBand lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice, N_bands);
                    lsp.aggregationSummationMB(bandsData, clearSkyFraction, aggrData);
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
                    lsp.normalizeAndConvertAggrDataMB(bandOffset, offsetQA, scalings[b], aggrData, writeData, gapMask);
                }
            }
            std::cout << "Done " << toc(t_tmp) << " s \n";


            std::cout << "#################################### \n";
            std::cout << "Aggregated band 0 \n";
            std::cout << aggrData.block(0+1,0,3,12) << std::endl;
            std::cout << "Aggregated QA \n";
            std::cout << aggrData.block(offsetQA+1,0,3,12) << std::endl;
            std::cout << "Final band 0 \n";
            std::cout << writeData.block(0+1,0,3,12).cast<int>() << std::endl;
            std::cout << "Final QA \n";
            std::cout << writeData.block(offsetQA+1,0,3,12).cast<int>() << std::endl;
        }

        std::cout << "Saving files" << "\n";
        t_tmp = tic();
        // Getting format info from the original files to write the output files consistently
        GDALDataset *inputDataset = (GDALDataset *)GDALOpen(fileUrlsRead[0].c_str(), GA_ReadOnly);
        double geotransform[6];
        inputDataset->GetGeoTransform(geotransform);
        auto projection = inputDataset->GetProjectionRef();
        auto spatialRef = inputDataset->GetSpatialRef();
        #pragma omp parallel for
        for (size_t i = 0; i < N_aimg; ++i) {
            size_t dateOffset = i * N_pix * N_bands;    
            for (size_t b = 0; b < N_bands; ++b) {
                LandsatPipelineMultiBand lsp(b, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_bands, N_bands);
                lsp.writeOutputFilesMB(filePathsWrite[b][i], seaweedPathsWrite[b][i], projection, spatialRef, geotransform, writeData, dateOffset);
            }
        }

        GDALClose(inputDataset);
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Total time " << toc(t_start) << " s \n";

        // #######################################################################
        // ################## Cleaning up ################
        // #######################################################################

        TypesFFTW<float>::DestroyPlan(plan_forward);
        TypesFFTW<float>::DestroyPlan(plan_backward);
    }

    return 0;
}
