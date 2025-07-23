#ifndef IOARRAY_H
#define IOARRAY_H

#include "ParArray.h"


namespace skmap
{

class IoArray: public ParArray 
{
    public :

        IoArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);

        void warpTile(std::string ref_tile_path,
                      std::string mosaic_path,
                      std::string resample);


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

        void setupGdal(dict_t dict);

        void getLatLonArray(std::string file_loc,
                               uint_t x_off,
                               uint_t y_off,
                               uint_t x_size,
                               uint_t y_size);

        void extractOverlay(std::vector<uint_t> pix_blok_ids,
                                 std::vector<uint_t> pix_inblock_idxs,
                                 std::vector<uint_t> unique_blocks_ids_comb,
                                 std::vector<uint_t> key_layer_ids_comb,
                                 Eigen::Ref<MatFloat> data_overlay);


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
