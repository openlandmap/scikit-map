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
                       std::vector<int> bands_list);

        void setupGdal(dict_t dict);

};

}
 
#endif