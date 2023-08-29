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

    // #pragma pack(push, 1) // @TODO: understuand well what this does https://stackoverflow.com/questions/49628615/understanding-corrupted-size-vs-prev-size-glibc-error

    // ################################################
    // #############Input parameters ##################
    // ################################################
    std::string tile = "034E_10N";
    size_t start_year = 1997;
    size_t end_year = 1997;
    size_t N_threads = 96;
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

    // Creating reading and writng urls/paths
    for (size_t y = start_year; y <= end_year; ++y) {
        for (size_t i = 0; i < N_ipy; ++i) {
            size_t img_offset = (y - 1997)*N_ipy + i;
            std::string str = "http://192.168.49."+ std::to_string(30 +  img_offset%9) +":8333/landsat-ard2/" + std::to_string(y) + "/" + tile + "/" + std::to_string(392+img_offset) + ".tif";
            fileUrlsRead.push_back(str);
        }
    }
    // for (size_t y = start_year; y <= end_year; ++y) {
    //     for (size_t i = 0; i < 6; ++i) {
    //         std::string str = out_folder + "/ogh_b" + std::to_string(band_id) + "_" + std::to_string(y) + outDatesStart[i] + "_" + std::to_string(y) + outDatesEnd[i] + ".tif";
    //         fileUrlsWrite.push_back(str);
    //     }
    // }
    
    // @TODO checl that fileUrlsRead.size() == N_img;

    // ################################################
    // ######### Creating data structures #############
    // ################################################
    // Creating matricies to storage the necessary data
    // Storage in ColMajor to match the RasterIO reading order
    // Compared to the 3D tensor in Python we have: C++ (p, t) -> Python (floor(p/N_col), p%N_col, t)    
    MatrixBool clearSkyMask(N_pix, N_img);
    MatrixUI16 evenReadMatrix(N_pix, N_img);
    MatrixUI16 oddReadMatrix(N_pix, N_img);
    MatrixUI8 qaWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg);
    MatrixUI8 evenWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg); // @TODO check why I filled it with ones
    MatrixUI8 oddWriteMatrix = MatrixUI8::Ones(N_pix, N_aimg);
    

    // ################################################
    // ################## Reading data ################
    // ################################################    
    
    auto t_start_tmp = std::chrono::high_resolution_clock::now();
    { // Use blocks to automatically delate the QA band after extracting the mask
        MatrixUI8 qualityAssessmentBand(N_pix, N_img);
        std::cout << "Reading band 8" << "\n";        
        // Reading B8
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
        // Reading B1
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

    std::cout << "Reading band 2, processing band 1" << "\n";
    t_start_tmp = std::chrono::high_resolution_clock::now();
    // Reading B1
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

    std::cout << "-------------- evenWriteMatrix ---------------------\n ";
    std::cout << evenWriteMatrix.block(0, 0, 5, N_aimg).cast<int>() << std::endl;
    std::cout << "-------------- qaWriteMatrix ---------------------\n ";
    std::cout << qaWriteMatrix.block(0, 0, 5, N_aimg).cast<int>() << std::endl;

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
    // for (size_t i = 0; i < 1; ++i) {
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
    //         evenWriteMatrix.data() + i * N_pix,
    //         N_row, N_col, GDT_Byte, 0, 0
    //     );
    //     (void) res;
    //     GDALClose(writeDataset);
    // }
    // GDALClose(inputDataset);
    // std::cout << "Done" << "\n";


    // @TODO remember to save the QA at the end


    // #pragma pack(pop)

    std::cout << "Total time " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count()/1000. << " s \n";
    



    return 0;
}
