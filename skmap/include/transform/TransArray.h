#ifndef TRANSARRAY_H
#define TRANSARRAY_H

#include "ParArray.h"

namespace skmap
{

class TransArray: public ParArray 
{
    public :

        TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);


        void computeNormalizedDifference(std::vector<uint_t> positive_indices,
                                         std::vector<uint_t> negative_indices,
                                         std::vector<uint_t> result_indices,
                                         float_t positive_scaling,
                                         float_t negative_scaling,
                                         float_t result_scaling,
                                         float_t result_offset);


};

}
 
 
#endif