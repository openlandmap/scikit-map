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
                                         float_t result_offset,
                                         std::vector<float_t> clip_value);

        void computeBsi(std::vector<uint_t> swir1_indices,
                            std::vector<uint_t> red_indices,
                            std::vector<uint_t> nir_indices,
                            std::vector<uint_t> blue_indices,
                            std::vector<uint_t> result_indices,
                            float_t swir1_scaling,
                            float_t red_scaling,
                            float_t nir_scaling,
                            float_t blue_scaling,
                            float_t result_scaling,
                            float_t result_offset,
                            std::vector<float_t> clip_value);

        void computeEvi(std::vector<uint_t> red_indices,
                            std::vector<uint_t> nir_indices,
                            std::vector<uint_t> blue_indices,
                            std::vector<uint_t> result_indices,
                            float_t red_scaling,
                            float_t nir_scaling,
                            float_t blue_scaling,
                            float_t result_scaling,
                            float_t result_offset,
                            std::vector<float_t> clip_value);

        void computeNirv(std::vector<uint_t> red_indices,
                            std::vector<uint_t> nir_indices,
                            std::vector<uint_t> result_indices,
                            float_t red_scaling,
                            float_t nir_scaling,
                            float_t result_scaling,
                            float_t result_offset,
                            std::vector<float_t> clip_value);

        void computeFapar(std::vector<uint_t> red_indices,
                            std::vector<uint_t> nir_indices,
                            std::vector<uint_t> result_indices,
                            float_t red_scaling,
                            float_t nir_scaling,
                            float_t result_scaling,
                            float_t result_offset,
                            std::vector<float_t> clip_value);


};

}
 
 
#endif
