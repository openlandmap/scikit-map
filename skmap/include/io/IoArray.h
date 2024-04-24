#ifndef IOARRAY_H
#define IOARRAY_H

#include "ParArray.h"


namespace skmap
{

class IoArray: public ParArray 
{
    public :

        IoArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);

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

        void setupGdal(dict_t dict);

        void getLatLonArray(std::string file_loc,
                               uint_t x_off,
                               uint_t y_off,
                               uint_t x_size,
                               uint_t y_size);


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
                skmapAssertIfTrue((uint_t) m_data.cols() < x_size * y_size, "scikit-map ERROR 9: reading region size smaller then the number of columns");
                GDALDataset *inputDataset = (GDALDataset *)GDALOpen(base_files[i].c_str(), GA_ReadOnly);
                double geotransform[6];
                inputDataset->GetGeoTransform(geotransform);
                auto projection = inputDataset->GetProjectionRef();
                auto spatial_ref = inputDataset->GetSpatialRef();
                int x_size_in = inputDataset->GetRasterXSize();
                int y_size_in = inputDataset->GetRasterYSize();
                std::string ending;
                if (bash_compression_command.has_value())
                    ending = "_tmp.tif";
                else 
                    ending = ".tif";
                    
                std::string layer_name = file_names[i];
                GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
                Eigen::RowVectorX<T> casted_row = row.cast<T>();
                GDALDataset *writeDataset = driver->Create((base_folder + "/" + layer_name + ending).c_str(),
                    inputDataset->GetRasterXSize(), inputDataset->GetRasterYSize(), 1, write_type, nullptr);
                writeDataset->SetGeoTransform(geotransform);
                writeDataset->SetSpatialRef(spatial_ref);
                writeDataset->SetProjection(projection);
                GDALRasterBand *writeBand = writeDataset->GetRasterBand(1);
                writeBand->SetNoDataValue((double) no_data_value);
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
                GDALClose(writeDataset);
                if (bash_compression_command.has_value())
                {                    
                    runBashCommand(bash_compression_command.value() + " " + base_folder + "/" + layer_name + ending + " "
                                 + base_folder + "/" + layer_name + ".tif");
                    runBashCommand("rm " + base_folder + "/" + layer_name + ending);
                }
                if (seaweed_path.has_value())
                {                    
                    runBashCommand("mc cp " + base_folder + "/" + layer_name + ".tif " +
                                              seaweed_path.value()[i] + "/" + layer_name + ".tif ");
                    runBashCommand("rm " + base_folder + "/" + layer_name + ".tif");
                }

            };
            this->parRowPerm(writeTiff, data_indices);
        }

};

}
 
#endif