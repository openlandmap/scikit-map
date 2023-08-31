#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <chrono>
#include <fftw3.h>
#include <experimental/filesystem>
#include "LandsatPipeline.hpp"
#include <unistd.h>

namespace fs = std::experimental::filesystem;

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // ################################################
    // #############Input parameters ##################
    // ################################################
    std::string tile = "013W_14N";
    size_t start_year = 1997;
    size_t end_year = 2022;
    size_t N_threads = 96;
    size_t N_bands = 8; // Must be >= 3
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
                                   savingScaling2};
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


    // #######################################################################
    // ################## Reading, processing and saving data ################
    // #######################################################################


    // ################## Stage 1 ################

    {

        int bandList[N_bands];
        for (int i = 0; i < N_bands; ++i) {
            bandList[i] = i + 1;
        }
        auto t_start_tmp = std::chrono::high_resolution_clock::now();
        MatrixUI16 bandsData(N_pix*N_bands, N_img);
        std::cout << "Reading multiple bands" << "\n";
        std::cout << fileUrlsRead[0] << "\n";
        N_slice = N_img;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipeline lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice);
            lsp.readInputFilesMB<MatrixUI16>(fileUrlsRead[i], bandList, N_bands, bandsData, GDT_UInt16);
        }
        std::cout << "Done " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start_tmp).count()/1000. << " s \n";
    }


    // std::cout << "Band 1 im 1" << std::endl;
    // std::cout << bandsData.block(0,0,5,1) << std::endl;
    // std::cout << "Band 1 im 2" << std::endl;
    // std::cout << bandsData.block(0,1,5,1) << std::endl;
    // std::cout << "Band 2 im 1" << std::endl;
    // std::cout << bandsData.block(N_pix,0,5,1) << std::endl;
    // std::cout << "Band 2 im 2" << std::endl;
    // std::cout << bandsData.block(N_pix,1,5,1) << std::endl;
    std::cout << "Total time " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count()/1000. << " s \n";


    return 0;
}
