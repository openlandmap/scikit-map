#ifndef IOARRAY_H
#define IOARRAY_H

#include "ParArray.h"


namespace skmap
{

class IoArray: public ParArray 
{
    public :

        IoArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);

        /**
        * @ingroup io
        * @brief Warp a single-band mosaic to match a reference tile.
        *
        * @deprecated This function is deprecated. Use GDAL VRTs instead, which
        * provide on-the-fly mosaicking and warping, support multiple bands, and
        * avoid loading full rasters into memory.
        *
        * @param ref_tile_path  File path to reference raster tile.
        * @param mosaic_path    File path to mosaic raster.
        * @param resample       Resampling method ("nearest", "bilinear", etc.)
        *
        * @details
        * This function opens the reference tile and mosaic, computes the target
        * dimensions, sets up GDALWarpOptions, and performs an in-memory warp
        * to match the reference. The resulting raster is read into `m_data`.
        *
        * @note Only works for single-band float32 rasters.
        */
        [[deprecated("Use GDAL VRTs instead of warpTile")]]
        void warpTile(std::string ref_tile_path,
                      std::string mosaic_path,
                      std::string resample);


        /**
        * @brief Reads a portion of a raster dataset into a row buffer.
        *
        * This function opens a raster dataset using GDAL, reads a rectangular
        * window defined by offsets and sizes into an Eigen row expression,
        * and optionally replaces a specific masked value with another value.
        *
        * @param row          Reference to an Eigen row expression where the data will be stored.
        * @param file_loc     File path or URL of the raster dataset to read.
        * @param x_off        Horizontal offset of the window to read.
        * @param y_off        Vertical offset of the window to read.
        * @param x_size       Width of the window to read.
        * @param y_size       Height of the window to read.
        * @param read_type    GDAL data type to read (e.g., GDT_Float32).
        * @param bands_list   List of band indices to read from the dataset.
        * @param value_to_mask Optional value in the dataset to treat as "masked" (e.g., nodata).
        * @param value_to_set  Optional value to replace the masked values with.
        *
        * @note If only `value_to_set` is provided, the function automatically
        *       determines the dataset's nodata value to use as `value_to_mask`.
        *
        * @throws skmapAssertIfTrue if the dataset cannot be opened or read.
        *
        * @details
        * Opens the GDAL dataset in read-only mode, reads the specified rectangular
        * portion into the provided Eigen row buffer, and closes the dataset.
        * If both `value_to_mask` and `value_to_set` are provided and are different,
        * the function replaces all occurrences of `value_to_mask` in the buffer
        * with `value_to_set`.
        */
        void readDataCore(Eigen::Ref<MatFloat::RowXpr> row,
                           std::string file_loc,
                           uint_t x_off,
                           uint_t y_off,
                           uint_t x_size,
                           uint_t y_size,
                           GDALDataType read_type,
                           std::vector<int> bands_list,
                           std::optional<float_t> value_to_mask,
                           std::optional<float_t> value_to_set);

        /**
        * @ingroup io
        * @brief Reads multiple raster datasets into the internal matrix in parallel.
        *
        * This function reads specified regions from a set of raster files
        * (`file_locs`) into the internal `m_data` matrix, using a permutation
        * vector to control the row order. Reading is performed in parallel
        * via the `parRowPerm` helper, which calls `readDataCore` for each file.
        *
        * @param file_locs     Vector of file paths or URLs of raster datasets to read.
        * @param perm_vec      Vector of row indices specifying the permutation order
        *                      for writing into `m_data`.
        * @param x_off         Horizontal offset of the reading window in the rasters.
        * @param y_off         Vertical offset of the reading window in the rasters.
        * @param x_size        Width of the window to read.
        * @param y_size        Height of the window to read.
        * @param read_type     GDALDataType specifying the type to read (e.g., GDT_Float32).
        * @param bands_list    List of band indices to read from each dataset.
        * @param value_to_mask Optional value to treat as masked (e.g., nodata). If not
        *                      provided, no masking is performed.
        * @param value_to_set  Optional value to replace masked values with.
        *
        * @throws skmapAssertIfTrue if the number of columns in `m_data` is smaller
        *         than the requested reading window (`x_size * y_size`).
        *
        * @details
        * The function defines a lambda `readTiff` that wraps `readDataCore`, which
        * reads a single raster into a given row buffer. `parRowPerm` executes this
        * lambda in parallel over the permutation vector, efficiently filling the
        * internal matrix `m_data`.
        *
        * @note Both `value_to_mask` and `value_to_set` are forwarded to `readDataCore`
        *       for per-pixel masking and replacement.
        */
        void readData(std::vector<std::string> file_locs,
                       std::vector<uint_t> perm_vec,
                       uint_t x_off,
                       uint_t y_off,
                       uint_t x_size,
                       uint_t y_size,
                       GDALDataType read_type,
                       std::vector<int> bands_list,
                       std::optional<float_t> value_to_mask,
                       std::optional<float_t> value_to_set);

        void readDataBlocks(std::vector<std::string> file_locs,
                           std::vector<uint_t> perm_vec,
                           std::vector<uint_t> x_off_vec,
                           std::vector<uint_t> y_off_vec,
                           std::vector<uint_t> x_size_vec,
                           std::vector<uint_t> y_size_vec,
                           GDALDataType read_type,
                           std::vector<int> bands_list,
                           std::optional<std::vector<float_t>> value_to_mask_vec,
                           std::optional<float_t> value_to_set);

        /**
        * @brief Initialize GDAL with custom configuration options and error handling.
        *
        * Sets GDAL runtime options from the input dictionary, registers all GDAL drivers, and
        * configures error logging to suppress console output if possible.
        *
        * @param dict Dictionary of GDAL configuration options (key-value pairs) where keys and
        *             values are strings recognized by GDAL.
        *
        * @details
        * Each key-value pair in `dict` is applied via CPLSetConfigOption(). GDALAllRegister() is
        * called to ensure all drivers are available. Error logging is redirected to "/dev/null" if
        * possible; otherwise, a quiet error handler is used to suppress warnings and errors.
        *
        * @note This function modifies global GDAL state and should be called before performing any
        *       GDAL I/O operations. All Python-exposed functions call it already.
        */
        void setupGdal(dict_t dict);

        void getLatLonArray(std::string file_loc,
                               uint_t x_off,
                               uint_t y_off,
                               uint_t x_size,
                               uint_t y_size);

        /**
        * @ingroup io
        * @brief Extract overlay values from raster blocks for a set of points.
        *
        * For each pixel (given by `pix_block_ids` and `pix_inblock_idxs`),
        * this function finds the matching block and copies the corresponding
        * raster values from `m_data` into `data_overlay`.
        *
        * The operation is parallelized across pixels.
        *
        * @param pix_block_ids Vector of block IDs corresponding to each pixel.
        * @param pix_inblock_idxs Vector of linear indices of each pixel within its block.
        * @param unique_blocks_ids_comb Vector of all unique block IDs for the current chunk.
        * @param key_layer_ids_comb Vector of layer indices corresponding to each block-layer combination.
        * @param data_overlay Output array (layers x pixels) where extracted values will be stored.
        */
        void extractOverlay(std::vector<uint_t> pix_block_ids,
                                 std::vector<uint_t> pix_inblock_idxs,
                                 std::vector<uint_t> unique_blocks_ids_comb,
                                 std::vector<uint_t> key_layer_ids_comb,
                                 Eigen::Ref<MatFloat> data_overlay);


        /**
        * @ingroup io
        * @brief Write portions of the internal matrix `m_data` to multiple GeoTIFF files.
        *
        * This function writes selected rows of the `m_data` matrix into GeoTIFF files
        * based on a set of base files and offsets. It supports in-place row casting,
        * NoData masking, optional bash compression commands, and optional uploading
        * to a remote storage (e.g., SeaweedFS).
        *
        * @tparam T The data type to write to disk (e.g., float, double).
        *
        * @param base_files Vector of file paths to base raster files to copy metadata from.
        * @param base_folder Folder where output files will be written.
        * @param file_names Names of the output files (without folder path).
        * @param data_indices Indices of rows in `m_data` corresponding to each output file.
        * @param x_off X-offset within the raster to start writing data.
        * @param y_off Y-offset within the raster to start writing data.
        * @param x_size Width of the region to write.
        * @param y_size Height of the region to write.
        * @param write_type GDAL data type to write (e.g., GDT_Float32).
        * @param no_data_value Value to use for missing or NaN cells.
        * @param bash_compression_command Optional: shell command to compress the output files (e.g., gdal_translate or gzip).
        * @param seaweed_path Optional: vector of remote storage paths to upload each output file.
        *
        * @details
        * For each row specified in `data_indices`, the function:
        * 1. Opens the corresponding base file and copies geotransform, projection, and spatial reference.
        * 2. Masks NaN values with `no_data_value`.
        * 3. Writes the row to a temporary raster if compression is requested, otherwise directly to the final file.
        * 4. Applies optional bash compression.
        * 5. Uploads the file to `seaweed_path` if provided and removes local copies as needed.
        */
        template<typename T>
        void writeData(std::vector<std::string> base_files,
                        std::string base_folder,
                        std::vector<std::string> file_names,
                        std::vector<uint_t> data_indices,
                        uint_t x_off,
                        uint_t y_off,
                        uint_t x_size,
                        uint_t y_size,
                        GDALDataType write_type,
                        T no_data_value,
                        std::optional<std::string> bash_compression_command,
                        std::optional<std::vector<std::string>> seaweed_path)
        {
                         

            auto writeTiff = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
            {
                if ((uint_t)m_data.cols() < x_size * y_size) {
                    throw std::runtime_error("scikit-map ERROR 9: reading region size smaller than the number of columns");
                }
                GDALDataset *inputDataset = (GDALDataset *)GDALOpen(base_files[i].c_str(), GA_ReadOnly);
                if (inputDataset == nullptr) {
                    throw std::runtime_error("scikit-map ERROR 10: issues in reading the file " + base_files[i]);
                }
                double geotransform[6];
                if (inputDataset->GetGeoTransform(geotransform) != CE_None) {
                    throw std::runtime_error("scikit-map ERROR 10: Failed to get GeoTransform.");
                }
                auto projection = inputDataset->GetProjectionRef();
                if (projection == nullptr) {
                    throw std::runtime_error("scikit-map ERROR 10: Failed to get ProjectionRef.");
                }
                auto spatial_ref = inputDataset->GetSpatialRef();
                if (spatial_ref == nullptr) {
                    throw std::runtime_error("scikit-map ERROR 10: Failed to get SpatialRef.");
                }
                int x_size_in = inputDataset->GetRasterXSize();
                int y_size_in = inputDataset->GetRasterYSize();
                std::string layer_name = file_names[i];
                const std::string suffix = ".tif";
                if (layer_name.size() >= suffix.size() && layer_name.compare(layer_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    layer_name = layer_name.substr(0, layer_name.size() - suffix.size());
                }
                GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
                row = row.array().isNaN().select(static_cast<float_t>(no_data_value), row);
                Eigen::RowVectorX<T> casted_row = row.cast<T>();
                std::string file_name = base_folder + "/" + layer_name;
                std::string ending = bash_compression_command.has_value() ? "_tmp.tif" : ".tif";
                std::string tmp_file_name = file_name + ending;
                GDALDataset *writeDataset = driver->Create(tmp_file_name.c_str(),
                    inputDataset->GetRasterXSize(), inputDataset->GetRasterYSize(), 1, write_type, nullptr);
                if (writeDataset == nullptr) {
                    throw std::runtime_error("scikit-map ERROR 10: issues in creating the file " + tmp_file_name);
                }
                writeDataset->SetGeoTransform(geotransform);
                writeDataset->SetSpatialRef(spatial_ref);
                writeDataset->SetProjection(projection);
                GDALRasterBand *writeBand = writeDataset->GetRasterBand(1);
                writeBand->SetNoDataValue(static_cast<double>(no_data_value));
                using MatType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                MatType init_raster = MatType::Constant(1, x_size_in * y_size_in, no_data_value);
                auto out_write1 = writeBand->RasterIO(
                    GF_Write, 0, 0, x_size_in, y_size_in, init_raster.data(),
                    x_size_in, y_size_in, write_type, 0, 0);
                skmapAssertIfTrue(out_write1 != CE_None,
                   "scikit-map ERROR 11: issues in writing the file " + layer_name);
                auto out_write2 = writeBand->RasterIO(
                    GF_Write, x_off, y_off, x_size, y_size, casted_row.data(),
                    x_size, y_size, write_type, 0, 0);
                skmapAssertIfTrue(out_write2 != CE_None,
                   "scikit-map ERROR 11: issues in writing the file " + layer_name);
                GDALClose(inputDataset);
                GDALClose(writeDataset);
                if (bash_compression_command.has_value())
                {                    
                    runBashCommand(bash_compression_command.value() + " " + tmp_file_name + " " + file_name + ".tif");
                    runBashCommand("rm " + tmp_file_name);
                }
                if (seaweed_path.has_value())
                {                    
                    runBashCommand("mc cp " + file_name + ".tif " + seaweed_path.value()[i] + "/" + layer_name + ".tif ");
                    runBashCommand("rm " + file_name + ".tif");
                }

            };
            this->parRowPerm(writeTiff, data_indices);
        }

};

}
 
#endif
