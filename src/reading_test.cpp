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

    // ################################################
    // #############Input parameters ##################
    // ################################################

    std::string tile = "034E_10N";
    unsigned int start_year = 1997;
    unsigned int end_year = 1998;
    unsigned int band_id_read;
    unsigned int band_id_proc;
    unsigned int band_id_write;
    std::string out_folder = "/mnt/inca/ard2_production/scikit-map/src/" + tile;

    // ################################################
    // ################### Setup ######################
    // ################################################
    unsigned int N_row = 4004;
    unsigned int N_col = 4004;
    unsigned int nWriteFiles = (end_year-start_year+1)*6;
    unsigned int nThreads = nWriteFiles;
    omp_set_num_threads(nThreads);
    std::vector<std::string> fileUrlsRead;

    // ################################################
    // ######### Input/output files ###################
    // ################################################

    // Creating reading and writng urls/paths
    for (unsigned int y = start_year; y <= end_year; ++y) {
        for (unsigned int i = 0; i < 23; ++i) {
            unsigned int img_offset = (y - 1997)*23 + i;
            std::string str = "http://192.168.49."+ std::to_string(30 +  img_offset%9) +":8333/landsat-ard2/" + std::to_string(y) + "/" + tile + "/" + std::to_string(392+img_offset) + ".tif";
            fileUrlsRead.push_back(str);
        }
    }
    // for (unsigned int y = start_year; y <= end_year; ++y) {
    //     for (unsigned int i = 0; i < 6; ++i) {
    //         std::string str = out_folder + "/ogh_b" + std::to_string(band_id) + "_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
    //         fileUrlsWrite.push_back(str);
    //     }
    // }
    int nReadFiles = fileUrlsRead.size();

    // ################################################
    // ######### Creating data structures #############
    // ################################################
    // Creating matricies to storage the necessary data
    // Storage in ColMajor to match the RasterIO reading order
    ReadMatrix evenReadMatrix(N_row * N_col, nReadFiles);
    ReadMatrix oddReadMatrix(N_row * N_col, nReadFiles);
    WriteMatrix evenWriteMatrix = WriteMatrix::Ones(N_row * N_col, nWriteFiles);
    WriteMatrix oddWriteMatrix = WriteMatrix::Ones(N_row * N_col, nWriteFiles);
    

    // ################################################
    // ################## Reading data ################
    // ################################################    
    int nSlice = std::ceil((float)nReadFiles/(float)nThreads);
    int sliceStep = nThreads - nSlice*nThreads + nReadFiles;

    std::cout << "Reading band 1" << "\n";
    // Reading B1
    band_id_read = 1;
    #pragma omp parallel for
    for (unsigned int i = 0; i < nThreads; ++i) {        
        LandsatPipeline lsp(i);
        lsp.readInputFiles(fileUrlsRead, nSlice, sliceStep, band_id_read, evenReadMatrix);
    }

    std::cout << "Reading band 2" << "\n";
    // Reading B2
    band_id_read = 2;
    #pragma omp parallel for
    for (unsigned int i = 0; i < nThreads; ++i) {        
        LandsatPipeline lsp(i);
        lsp.readInputFiles(fileUrlsRead, nSlice, sliceStep, band_id_read, oddReadMatrix);
    }

    std::cout << "Done" << "\n";

    // ################################################
    // ################## Writing data ################
    // ################################################

    // // Getting format info from the original files to write the output files consistently
    // GDALDataset *inputDataset = (GDALDataset *)GDALOpen(fileUrlsRead[0].c_str(), GA_ReadOnly);
    // double geotransform[6];
    // inputDataset->GetGeoTransform(geotransform);
    // char **papszOptions = NULL;
    //     papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

    // std::cout << "Writing the files" << "\n";
    // // Create the output folder
    // if (!fs::exists(out_folder)) {
    //     if (!fs::create_directory(out_folder)) {
    //         std::cerr << "Failed to create the output folder." << std::endl;
    //         return 1;
    //     }
    // }

    // // #pragma omp parallel for
    // for (unsigned int i = 0; i < 1; ++i) {
    //     GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //     GDALDataset *writeDataset = driver->Create(
    //         (fileUrlsWrite[i]).c_str(),
    //         N_row, N_col, 1, GDT_Byte, nullptr
    //     );
    //     writeDataset->SetProjection(inputDataset->GetProjectionRef());
    //     writeDataset->SetGeoTransform(geotransform);
    //     GDALRasterBand *writeBand = writeDataset->GetRasterBand(1);
    //     writeBand->SetNoDataValue(inputDataset->GetRasterBand(band_id)->GetNoDataValue());
    //     // Write the converted data to the new dataset
    //     auto res = writeBand->RasterIO(
    //         GF_Write, 0, 0, N_row, N_col,
    //         evenWriteMatrix.data() + i * N_row * N_col,
    //         N_row, N_col, GDT_Byte, 0, 0
    //     );
    //     (void) res;
    //     GDALClose(writeDataset);
    // }
    // GDALClose(inputDataset);
    // std::cout << "Done" << "\n";


    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "Elapsed ms: " << elapsed_time_ms << "\n";
    std::cout << "Matrix even" << "\n";
    std::cout << "Matrix value (0,0) : " << evenReadMatrix(0,0) << "\n";
    std::cout << "Matrix value (0,1) : " << evenReadMatrix(0,1) << "\n";
    std::cout << "Matrix value (0,22): " << evenReadMatrix(0,22) << "\n";
    std::cout << "Matrix value (0,24): " << evenReadMatrix(0,24) << "\n";
    std::cout << "Matrix odd" << "\n";
    std::cout << "Matrix value (0,0) : " << oddReadMatrix(0,0) << "\n";
    std::cout << "Matrix value (0,1) : " << oddReadMatrix(0,1) << "\n";
    std::cout << "Matrix value (0,22): " << oddReadMatrix(0,22) << "\n";
    std::cout << "Matrix value (0,24): " << oddReadMatrix(0,24) << "\n";



    return 0;
}
