#ifndef IOARRAY_H
#define IOARRAY_H

#include "misc.cpp"
#include "ParArray.h"


namespace skmap
{

class IoArray: public ParArray 
{
     public :

          IoArray(Eigen::Ref<MatFloat> data, const uint_t n_feat, const uint_t n_pix, const uint_t n_threads);

          void readData(uint_t n_row,
                        uint_t n_col,
                        std::vector<std::string> file_urls,
                        std::vector<int> perm_vec,
                        GDALDataType read_type);

};

}
 
#endif