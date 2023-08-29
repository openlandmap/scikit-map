#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <chrono>
#include <experimental/filesystem>
#include "LandsatPipeline.hpp"

namespace fs = std::experimental::filesystem;

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    #pragma pack(push, 1) // @TODO: understuand well what this does https://stackoverflow.com/questions/49628615/understanding-corrupted-size-vs-prev-size-glibc-error

    // ################################################
    // #############Input parameters ##################
    // ################################################
    std::string tile = "034E_10N";
    size_t start_year = 1997;
    size_t end_year = 2022;
    size_t N_threads = 96;
    size_t N_bands = 7;
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
    size_t N_aggr = 4; // Aggregation
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
    CPLSetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif");
    CPLSetConfigOption("GDAL_HTTP_MULTIRANGE", "SINGLE_GET");
    CPLSetConfigOption("GDAL_HTTP_CONNECTTIMEOUT", "320");
    CPLSetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO");
    CPLSetConfigOption("GDAL_HTTP_VERSION", "1.0");
    CPLSetConfigOption("GDAL_HTTP_TIMEOUT", "320");
    CPLSetConfigOption("CPL_CURL_GZIP", "NO");
    GDALAllRegister();  // Initialize GDAL

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

    // Getting format info from the original files to write the output files consistently
    GDALDataset *inputDataset = (GDALDataset *)GDALOpen(fileUrlsRead[0].c_str(), GA_ReadOnly);
    double geotransform[6];
    inputDataset->GetGeoTransform(geotransform);
    auto projectionRef = inputDataset->GetProjectionRef();
    auto noDataValue = inputDataset->GetRasterBand(1)->GetNoDataValue();
    GDALClose(inputDataset);


    // @TODO checl that fileUrlsRead.size() == N_img;

    // ###################################################################
    // ######### Creating data structures and saving folder ##############
    // ###################################################################

    std::cout << "Creating data structures" << std::endl;
    // Creating matricies to storage the necessary data
    // Storage in ColMajor to match the RasterIO reading order
    // Compared to the 3D tensor in Python we have: C++ (p, t) -> Python (floor(p/N_col), p%N_col, t)    
    MatrixBool clearSkyMask(N_pix, N_img);
    MatrixUI16 evenReadMatrix(N_pix, N_img);
    MatrixUI16 oddReadMatrix(N_pix, N_img);
    MatrixUI8 qaWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg);
    MatrixUI8 evenWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg); // @TODO check why I filled it with ones
    MatrixUI8 oddWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg);

    // Create the output folder
    if (!fs::exists(out_folder)) {
        if (!fs::create_directory(out_folder)) {
            std::cerr << "Failed to create the output folder." << std::endl;
            return 1;
        }
    }
    

    // #######################################################################
    // ################## Reading, processing and saving data ################
    // #######################################################################


    // ################## Stage 1 ################

    auto t_start_tmp = std::chrono::high_resolution_clock::now();
    { // Use blocks to automatically delate the QA band after extracting the mask
        MatrixUI8 qualityAssessmentBand(N_pix, N_img);
        std::cout << "Reading band 8" << "\n";
        band_id_read = 8;
        N_slice = N_img;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipeline lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice);
            lsp.readInputFiles<MatrixUI8>(fileUrlsRead, band_id_read, qualityAssessmentBand, GDT_Byte);
        }
        std::cout << "Done in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start_tmp).count()/1000. << " s \n";
        std::cout << "Reading band 1 and processing the QA band" << "\n";
        t_start_tmp = std::chrono::high_resolution_clock::now();
        band_id_read = 1;
        N_slice = N_img;
        #pragma omp parallel for
        for (size_t i = 0; i < N_slice; ++i) {        
            LandsatPipeline lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice);
            lsp.readInputFiles<MatrixUI16>(fileUrlsRead, band_id_read, evenReadMatrix, GDT_UInt16);
            lsp.processMask(qualityAssessmentBand, clearSkyMask);
        }
        std::cout << "Done in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start_tmp).count()/1000. << " s \n";
    }

    // ################## Stage 2 ################

    std::cout << "Reading band 2, processing band 1" << "\n";
    t_start_tmp = std::chrono::high_resolution_clock::now();
    band_id_read = 2;
    band_id_proc = 1;
    N_slice = N_img;
    #pragma omp parallel for
    for (size_t i = 0; i < N_slice; ++i) {
        LandsatPipeline lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice);        
        lsp.readInputFiles<MatrixUI16>(fileUrlsRead, band_id_read, oddReadMatrix, GDT_UInt16);        
        lsp.processData(evenReadMatrix, clearSkyMask, attSeas, attEnv, scalings[band_id_proc], evenWriteMatrix, qaWriteMatrix);        
    }
    std::cout << "Done in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start_tmp).count()/1000. << " s \n";


    // ################## Stage 2 ################

    std::cout << "Reading band 3, processing band 3, saving band 1" << "\n";
    t_start_tmp = std::chrono::high_resolution_clock::now();
    band_id_read = 3;
    band_id_proc = 2;
    band_id_write = 1;
    N_slice = N_aimg;
    #pragma omp parallel for
    for (size_t i = 0; i < N_slice; ++i) {
        LandsatPipeline lsp(i, N_row, N_col, N_img, N_aimg, N_ipy, N_aipy, N_aggr, N_years, N_slice);        
        lsp.readInputFiles<MatrixUI16>(fileUrlsRead, band_id_read, evenReadMatrix, GDT_UInt16);        
        lsp.processData(oddReadMatrix, clearSkyMask, attSeas, attEnv, scalings[band_id_proc], oddWriteMatrix, qaWriteMatrix);
        lsp.writeOutputFiles(filePathsWrite[band_id_write-1][i], projectionRef, noDataValue, geotransform, evenWriteMatrix);    
    }
    std::cout << "Done in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start_tmp).count()/1000. << " s \n";


    // @TODO remember to save the QA at the end


    #pragma pack(pop)

    std::cout << "Total time " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count()/1000. << " s \n";
    



    
    return 0;
}
