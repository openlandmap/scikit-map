#ifndef TRANSARRAY_H
#define TRANSARRAY_H

#include "ParArray.h"

namespace skmap
{

class TransArray: public ParArray 
{
    public :

        TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);
        
        void linearRegression(Eigen::Ref<VecFloat> x,
                                      Eigen::Ref<VecFloat> beta_0,
                                      Eigen::Ref<VecFloat> beta_1);

        void copyVecInMatrixRow(Eigen::Ref<VecFloat> in_vec,
                                uint_t row_idx);

        void offsetAndScale(float_t offset,
                            float_t scaling);

        void scaleAndOffset(float_t offset,
                            float_t scaling);

        void averageAggregate(Eigen::Ref<MatFloat> out_data,
                              uint_t agg_factor);

        void maskDifference(float_t diff_th,
                            uint_t count_th,
                            Eigen::Ref<MatFloat> ref_data,
                                    Eigen::Ref<MatFloat> mask_out);

        void reorderArray(Eigen::Ref<MatFloat> out_data,
                          std::vector<std::vector<uint_t>> indices_matrix);

        void selArrayRows(Eigen::Ref<MatFloat> out_data,
                              std::vector<uint_t> row_select);

        void nanMean(Eigen::Ref<VecFloat> out_data);
        
        void computeMannKendallPValues(Eigen::Ref<VecFloat> out_data);
        
        void expandArrayRows(Eigen::Ref<MatFloat> out_data,
                              std::vector<uint_t> row_select);


        void extractArrayRows(Eigen::Ref<MatFloat> out_data,
                          std::vector<uint_t> row_select);
        
        void fillArray(float_t val);

        void fitPercentage(Eigen::Ref<MatFloat> in1,
                           Eigen::Ref<MatFloat> in2);

        void averageAi4sh(Eigen::Ref<MatFloat> in1,
                          Eigen::Ref<MatFloat> in2,
                          uint_t n_pix,
                          uint_t y);

        void hadamardProduct(Eigen::Ref<MatFloat> in1,
                             Eigen::Ref<MatFloat> in2);

        void inverseReorderArray(Eigen::Ref<MatFloat> out_data,
                                   std::vector<std::vector<uint_t>> indices_matrix);

        void transposeArray(Eigen::Ref<MatFloat> out_data);

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

        void computeNirv(std::vector<uint_t> nir_indices,
                     std::vector<uint_t> red_indices,
                     std::vector<uint_t> result_indices,
                     float_t nir_scaling,
                     float_t red_scaling,
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
        

        void computeGeometricTemperature(Eigen::Ref<MatFloat> latitude,
                                         Eigen::Ref<MatFloat> elevation,
                                         float_t elevation_scaling,
                                         float_t a,
                                         float_t b,
                                         float_t result_scaling,
                                         std::vector<uint_t> result_indices,
                                         std::vector<float_t> days_of_year);

        void swapRowsValues(std::vector<uint_t> row_select,
                                 float_t value_to_mask,
                                 float_t new_value);

        void computePercentiles(Eigen::Ref<MatFloat> out_data,
                                uint_t out_index_offset,
                                std::vector<float_t> percentiles);

        void maskNan(std::vector<uint_t> row_select,
                     float_t new_value_in_data);

        void maskData(std::vector<uint_t> row_select,
                      Eigen::Ref<MatFloat> mask,
                      float_t value_of_mask_to_mask,
                      float_t new_value_in_data);


        void applySircle(Eigen::Ref<MatFloat> out_data,
                         uint_t out_index_offset,
                         float_t w_0,
                         Eigen::Ref<VecFloat> w_p,
                         Eigen::Ref<VecFloat> w_f,
                         bool keep_original_values,
                         const std::string& version,
                         const std::string& backend);

        void transposeReorderArray(Eigen::Ref<MatFloat> out_data,
                                   std::vector<std::vector<uint_t>> permutation_matrix);




};

}
 
 
#endif
