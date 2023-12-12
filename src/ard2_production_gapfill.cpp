#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <fftw3.h>
#include "LandsatPipelineMultiBand.hpp"
#include <experimental/filesystem>
#include <fstream>

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    
    // ################################################
    // ############ Input parameters ##################
    // ################################################
    size_t start_year = 1997;
    size_t end_year = 2022;
    size_t N_threads = omp_get_max_threads();
    size_t N_bands = 8;
    std::vector<std::string> bandNames = {"blue_glad",
                                           "green_glad",
                                           "red_glad",
                                           "nir_glad",
                                           "swir1_glad",
                                           "swir2_glad",
                                           "thermal_glad",
                                           "clear_sky_mask"};

    // ################################################
    // ################### Setup ######################
    // ################################################
    size_t N_row = 4004;
    size_t N_col = 4004;
    size_t N_pix = N_row * N_col;
    size_t N_aipy = 6; // Aggregated images per year
    size_t N_years = end_year-start_year+1;
    size_t N_aimg = N_years*N_aipy; // Aggregated images
    size_t N_slice;
    float attSeas = 45.;
    float attEnv = 46.;
    size_t offsetQA = (N_bands-1)*N_pix;
    omp_set_num_threads(N_threads);
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

    std::cout << std::setprecision(1);
    std::cout << std::fixed << std::endl;

    
    // Check if at least one tile name was provided
    if (argc < 4) {
        std::cerr << "Input the first and the last tile index to process, and the tiles filename" << std::endl;
        return 1;
    }

    std::cout << std::setprecision(1);
    std::cout << std::fixed << std::endl;

    // Reading tiles to be processed
    std::string tilesFilename = argv[3];
    std::string tmpFilename = "tmp.txt";
    std::ifstream file(tilesFilename);
    if (!file.is_open()) {
        std::cerr << "scikit-map ERROR 15: Failed to open the tile file." << std::endl;
        return 1;
    }
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    for (size_t t = std::stoi(argv[1]); t <= std::stoi(argv[2]); ++t)
    {
        
        // ################################################
        // ######### Check tile files status  #############
        // ################################################
        std::string tile;
        if (t >= 0 && t < lines.size()) {
            tile = lines[t];
        } else {
            std::cerr << "scikit-map ERROR 9: invalid line index." << std::endl;
        }

        std::vector<std::string> filePathsDatesRead;
        std::vector<std::vector<std::string>> filePathsWrite;
        std::vector<std::vector<std::string>> seaweedPathsWrite;
        
        
        std::string out_folder = "./" + tile + "_seasconv";
        std::string seaweed_folder = "/prod-landsat-ard2/" + tile + "/seasconv/" ;
        
        int N_already_computed_files;
        std::string command = "mc ls seaweed/prod-landsat-ard2/" + tile + "/seasconv/ | wc -l > " + tmpFilename;
        int result = system(command.c_str());
        if (result != 0) {
            std::cerr << "scikit-map ERROR 10: issues in checking the number of already processed files" << std::endl;
            return 1;
        } else {            
            std::ifstream fileTmp(tmpFilename);
            if (!fileTmp.is_open()) {
                std::cerr << "scikit-map ERROR 11: failed to open the file for tile processing check." << std::endl;
                return 1;
            }
            std::string lineTmp;
            std::getline(fileTmp, lineTmp);
            N_already_computed_files = std::stoi(lineTmp);
            fileTmp.close();
        }
        if (N_already_computed_files == N_years*N_aipy*N_bands) {
            std::cout << "Skipping tile " << tile << ", number of already processed files for this tile = " << N_already_computed_files << std::endl;
            continue;
        }
        
        
        
        
        std::cout << "####################################################" << std::endl;
        std::cout << "############# Processing tile " << tile << " #############" << std::endl;
        std::cout << "####################################################" << std::endl;

        // ################################################
        // ######### Input/output files ###################
        // ################################################


        std::cout << "Creating the input files paths" << std::endl;
        for (size_t y = start_year; y <= end_year; ++y) {
            for (size_t i = 0; i < 6; ++i) {
                std::string str =  std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i];
                filePathsDatesRead.push_back(str);            
            }
        }

        std::cout << "Creating the output files paths" << std::endl;
        for (size_t band_id = 1; band_id <= N_bands; ++band_id) {
            std::vector<std::string> tmpVector;
            std::vector<std::string> tmpVector2;
            for (size_t y = start_year; y <= end_year; ++y) {
                for (size_t i = 0; i < 6; ++i) {
                    std::string str = out_folder + "/" + bandNames[band_id-1] + ".SeasConv.ard2_m_30m_s_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + "_go_epsg.4326_v20230908.tif";
                    std::string str2 = "g" + std::to_string(1 +  y%13) + seaweed_folder + bandNames[band_id-1] + ".SeasConv.ard2_m_30m_s_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + "_go_epsg.4326_v20230908.tif";
                    tmpVector.push_back(str);
                    tmpVector2.push_back(str2);
                }
            }
            filePathsWrite.push_back(tmpVector);
            seaweedPathsWrite.push_back(tmpVector2);
        }

        // ###################################################################
        // ######### Creating data structures and saving folder ##############
        // ###################################################################

        std::cout << "Creating data structures and saving folder" << std::endl;
        auto t_tmp = tic();
        // Creating matricies to storage the necessary data
        // Storage in ColMajor to match the RasterIO reading order
        // Compared to the 3D tensor in Python we have: C++ (p, t) -> Python (floor(p/N

        // Create the output folder
        if (!fs::exists(out_folder)) {
            if (!fs::create_directory(out_folder)) {
                std::cerr << "scikit-map ERROR 17: Failed to create the output folder." << std::endl;
                return 1;
            }
        }

        // The planning can not be constructed in parallel, so it is shared between the threads
        auto fftw_flags = FFTW_MEASURE;
        size_t N_fft = N_aimg;
        size_t N_ext = N_aimg * 2 - 1;
        TypesEigen<double>::VectorReal tmpRealVector(N_ext);
        TypesEigen<double>::VectorComplex tmpComplexVector(N_fft);
        // Create plans for forward and backward DFT (Discrete Fourier Transform)
        fftw_plan plan_forward = 
                fftw_plan_dft_r2c_1d(N_ext, tmpRealVector.data(), reinterpret_cast<fftw_complex*>(tmpComplexVector.data()), FFTW_MEASURE);
            fftw_plan plan_backward = 
                fftw_plan_dft_c2r_1d(N_ext, reinterpret_cast<fftw_complex*>(tmpComplexVector.data()), tmpRealVector.data(), FFTW_MEASURE);
        std::cout << "Done in " << toc(t_tmp) << " s \n";


        // #######################################################################
        // ################## Reading, processing and saving data ################
        // #######################################################################

        MatrixUI8 aggrData(N_pix*N_bands, N_aimg);

        std::cout << "Reading all the bands" << "\n";
        int bandList[1];
        bandList[0] = 1;
        t_tmp = tic();
        MatrixUI8 flag_read(N_aimg, 1);
        N_slice = N_aimg;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
            std::string in_folder = "http://192.168.49." + std::to_string(30 +  i%13) + ":8333/prod-landsat-ard2/" + tile + "/agg";
            unsigned char read_res = lsp.readAggregatedInputFilesMB<MatrixUI8>(filePathsDatesRead[i], in_folder, bandNames, bandList, N_bands, aggrData, GDT_Byte);            
            flag_read(i, 0) = read_res; 
        }
        if (flag_read.sum() > 0) {                    
            std::cerr << "scikit-map ERROR 18: missing files for tile " << tile << ", skipping it." << std::endl;
            std::cout << "scikit-map ERROR 18: missing files for tile " << tile << ", skipping it." << std::endl;
            continue;
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        auto t_start = tic();

        MatrixUI8 writeData(N_pix*N_bands, N_aimg);
        MatrixBool clearSkyMask(N_pix, N_aimg);
        MatrixBool nonFilledMask(N_pix, N_aimg);


        std::cout << "#################################### \n";
        std::cout << "Aggregated band 0 \n";
        std::cout << aggrData.block(0+1,0,3,12).cast<int>() << std::endl;
        std::cout << "Aggregated band 1 \n";
        std::cout << aggrData.block(0+1+N_pix,0,3,12).cast<int>() << std::endl;
        std::cout << "Aggregated QA \n";
        std::cout << aggrData.block(offsetQA+1,0,3,12).cast<int>() << std::endl;

        std::cout << "Creating the clear-sky mask" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
            lsp.processAggregatedMaskMB<MatrixUI8>(aggrData, clearSkyMask, offsetQA);
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";


        std::cout << "Setting to zero the non clear-sky pixels" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        for (size_t b = 0; b < N_bands; ++b) {
            size_t bandOffset = b*N_pix;
            // std::cout << "---- Band " << b+1 << "\n";
            #pragma omp parallel for
            for (size_t i = 0; i < N_slice; ++i) {
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
                lsp.maskDataMB(bandOffset, clearSkyMask, aggrData);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Computing convolution vector" << "\n";
        t_tmp = tic();
        TypesEigen<double>::VectorReal normQa(N_ext);
        TypesEigen<double>::VectorComplex convExtFFT(N_fft);
        double minConvValue;
        double maxNormQa;
        LandsatPipelineMultiBand lsp(0, 1, 1, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, 1, 1);
        lsp.computeConvolutionVectorMB(convExtFFT, false, attSeas, attEnv, minConvValue, maxNormQa);
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
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
                lsp.convolutionSeasConvMB<MatrixUI8>(bandOffset, aggrData, convData, convExtFFT, plan_forward, plan_backward);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";



        std::cout << "Mask the pixels that are not numerically filled after the convolution" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
            lsp.maskNonFilledDataMB<MatrixFloat>(convData, nonFilledMask, minConvValue, offsetQA);
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
                LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
                lsp.renormalizeSeasConvMB(bandOffset, offsetQA, convData, aggrData, clearSkyMask, nonFilledMask, writeData);
            }
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";

        std::cout << "Extracting the QA band" << "\n";
        t_tmp = tic();
        N_slice = N_threads;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipelineMultiBand lsp(i, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_slice, N_bands);
            lsp.extractQaMB(maxNormQa, offsetQA, convData, clearSkyMask, nonFilledMask, writeData);
        }
        std::cout << "Done " << toc(t_tmp) << " s \n";
        

        std::cout << "Total time excluding IO " << toc(t_start) << " s \n";

        
        std::cout << "Saving files" << "\n";
        t_tmp = tic();
        // Getting format info from the original files to write the output files consistently
        std::string fileUrl = "http://192.168.49.30:8333/prod-landsat-ard2/" + tile + "/agg"  + "/" + bandNames[0] + ".ard2_m_30m_s_" + filePathsDatesRead[0] + "_go_epsg.4326_v20230908.tif";
        GDALDataset *inputDataset = (GDALDataset *)GDALOpen(fileUrl.c_str(), GA_ReadOnly);
        double geotransform[6];
        inputDataset->GetGeoTransform(geotransform);
        auto projection = inputDataset->GetProjectionRef();
        auto spatialRef = inputDataset->GetSpatialRef();
        #pragma omp parallel for
        for (size_t i = 0; i < N_aimg; ++i) {
            size_t dateOffset = i * N_pix * N_bands;    
            for (size_t b = 0; b < N_bands; ++b) {
                LandsatPipelineMultiBand lsp(b, N_row, N_col, N_aimg, N_aimg, N_aipy, N_aipy, 1, N_years, N_bands, N_bands);
                lsp.writeOutputFilesMB(filePathsWrite[b][i], seaweedPathsWrite[b][i], projection, spatialRef, geotransform, writeData, dateOffset);
            }
        }

        GDALClose(inputDataset);
        std::cout << "Done " << toc(t_tmp) << " s \n";
        // #######################################################################
        // ################## Cleaning up ################
        // #######################################################################

        TypesFFTW<double>::DestroyPlan(plan_forward);
        TypesFFTW<double>::DestroyPlan(plan_backward);
        
    }


    return 0;
}
