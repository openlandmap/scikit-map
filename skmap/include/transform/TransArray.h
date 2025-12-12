#ifndef TRANSARRAY_H
#define TRANSARRAY_H

#include "ParArray.h"

namespace skmap {

class TransArray : public ParArray {
public:
  TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);

  /**
   * @ingroup mangling
   * @brief Copy a vector into the matrix
   *
   * If you created a vector in Python, this is the way to add it to the matrix
   *
   * @param in_vec Vector to copy
   * @param row_idx Matrix row to copy into
   */
  void copyVecInMatrixRow(Eigen::Ref<VecFloat> in_vec, uint_t row_idx);

  /**
   * @ingroup mangling
   * @brief Select `row_select` rows to put in `out_data` in-order
   *
   * \image HTML sel_expand_rows.svg
   *
   * `out_data.rows()` MUST be `>=row_select.len()` and `out_data.cols()` MUST
   * match the width.
   *
   * @param out_data output matrix
   * @param row_select rows to select from initial matrix
   */
  void selArrayRows(Eigen::Ref<MatFloat> out_data,
                    std::vector<uint_t> row_select);

  /**
   * @ingroup mangling
   * @brief Select `col_select` cols to put in `out_data` in-order.
   *
   * `out_data.cols()` MUST be `>=row_select.len()` and `out_data.rows()` MUST
   * match the height.
   *
   * @param out_data output matrix
   * @param col_select columns to select from initial matrix
   */
  void selArrayCols(Eigen::Ref<MatFloat> out_data,
                    std::vector<uint_t> col_select);
  /**
   * @ingroup mangling
   * @brief Inserts rows of self into `out_data` at `row_select` indices
   *
   * \image HTML sel_expand_rows.svg
   *
   * length of `row_select` MUST equal `m_data.rows()`
   *
   * @param out_data Output array
   * @param row_select indices at which to put `m_data`
   */
  void expandArrayRows(Eigen::Ref<MatFloat> out_data,
                       std::vector<uint_t> row_select);

  /**
   * @ingroup mangling
   * @brief Inserts columns of self into `out_data` at `col_select` indices
   *
   * length of `col_select` MUST equal `m_data.cols()`
   *
   * @param out_data Output array
   * @param col_select indices at which to put `m_data`
   */
  void expandArrayCols(Eigen::Ref<MatFloat> out_data,
                       std::vector<uint_t> col_select);

  /**
   * @ingroup mangling
   * @brief Expands the indices of `indices_matrix` to get a wider matrix.
   *
   * \image HTML reorder_transpose_inverse.svg
   *
   * This function is almost always slower than `transposeReorderArray`
   *
   * @param out_data The wider output matrix
   * @param indices_matrix a _rectangular_ matrix of indices to expand
   */
  void reorderArray(Eigen::Ref<MatFloat> out_data,
                    std::vector<std::vector<uint_t>> indices_matrix);

  /**
   * @ingroup mangling
   * @brief Transposes the array
   *
   * \image HTML reorder_transpose_inverse.svg
   *
   * Transposes the array
   *
   * @param out_data Array to be filled with transposed values
   */
  void transposeArray(Eigen::Ref<MatFloat> out_data);

  /**
   * @ingroup mangling
   * @brief reorders and transposes in one step
   *
   * \image HTML reorder_transpose_inverse.svg
   *
   * @warning Currently not working as intended
   *
   * @param out_data output data matrix
   * @param permutation_matrix permutation matrix
   */
  void
  transposeReorderArray(Eigen::Ref<MatFloat> out_data,
                        std::vector<std::vector<uint_t>> permutation_matrix);

  /**
   * @ingroup mangling
   * @brief Inverse of `TransArray::reorderArray`
   *
   * \image HTML reorder_transpose_inverse.svg
   *
   * Selects blocks from the array based on `out_data` width and
   * `indices_matrix`
   *
   * @param out_data Narrower output matrix
   * @param indices_matrix matrix of indices to be appended
   */
  void inverseReorderArray(Eigen::Ref<MatFloat> out_data,
                           std::vector<std::vector<uint_t>> indices_matrix);

  /**
   * @ingroup manipulation
   * @brief Fill array with `val`
   *
   * @param val The fill value
   */
  void fillArray(float_t val);

  /**
   * @ingroup manipulation
   * @brief Fill `NAN`s in `row_select` with `new_value_in_data`
   *
   * If the fill value should be different per row, see
   * `TransArray::maskNanRows`
   *
   * @param row_select vec of row indices to mask
   * @param new_value_in_data mask value
   */
  void maskNan(std::vector<uint_t> row_select, float_t new_value_in_data);

  /**
   * @ingroup manipulation
   * @brief Fill `NAN`s in `row_select` with corresponding value in
   * `new_value_vec`
   *
   * The fill value will correspond to the index in `row_select`
   *
   * @param row_select vec of row indices to mask
   * @param new_value_vec mask values per index
   */
  void maskNanRows(std::vector<uint_t> row_select,
                   Eigen::Ref<VecFloat> new_value_vec);

  /**
   * @ingroup manipulation
   * @brief fills data with `new_value_in_data` where
   * `mask==value_of_mask_to_mask`
   *
   * The mask row corresponds to the index of `row_select`. For using the same
   * mask for all rows, see `TransArray::maskDataRows`.
   *
   * @warning `maskDataRows` and `maskData` may swap names
   *
   * @param row_select vec of rows to apply masks to
   * @param masks masks, one for each row
   * @param value_of_mask_to_mask if the mask cell equals this value, overwrite
   * data
   * @param new_value_in_data the value to overwrite data with
   */
  void maskData(std::vector<uint_t> row_select, Eigen::Ref<MatFloat> masks,
                float_t value_of_mask_to_mask, float_t new_value_in_data);

  /**
   * @ingroup manipulation
   * @brief fills data with `new_value_in_data` where
   * `mask==value_of_mask_to_mask`
   *
   * This function applies the same mask to all rows. To apply a different mask
   * per row, see `TransArray::maskData`
   *
   * @warning `maskDataRows` and `maskData` may swap names
   *
   * @param row_select vec of rows to apply masks to
   * @param mask mask for all rows
   * @param value_of_mask_to_mask if the mask cell equals this value, overwrite
   * data
   * @param new_value_in_data the value to overwrite data with
   */
  void maskDataRows(std::vector<uint_t> row_select, Eigen::Ref<MatFloat> mask,
                    float_t value_of_mask_to_mask, float_t new_value_in_data);

  /**
   * @ingroup manipulation
   * @brief swap `value_to_mask` with `new_value` in `row_select` rows
   *
   * @warning This may be renamed to `swapValues`
   *
   * @param row_select vec of rows to swap values on
   * @param value_to_mask value to filter out (noData value?)
   * @param new_value fill value
   */
  void swapRowsValues(std::vector<uint_t> row_select, float_t value_to_mask,
                      float_t new_value);

  /**
   * @ingroup manipulation
   * @brief element-wise product of two matrices
   *
   * will fill this array with the result. Therefore, `in1`, `in2` and this
   * array's dimensions MUST match exactly.
   *
   * @param in1, in2 The two matrices to multiply element-wise
   */
  void hadamardProduct(Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2);

  /**
   * @ingroup manipulation
   * @brief Applies offset and scale to all rows
   *
   * v_new = (v_old+offset)*scaling
   *
   * @param offset Offset to apply
   * @param scaling Scale factor
   */
  void offsetAndScale(float_t offset, float_t scaling);

  /**
   * @ingroup manipulation
   * @brief Applies scale and offset to all rows
   *
   * v_new = v_old*scaling+offset
   *
   * @param offset Offset to add
   * @param scaling Scale factor
   */
  void scaleAndOffset(float_t offset, float_t scaling);

  /**
   * @ingroup manipulation
   * @brief Applies multiple offsets and scales
   *
   * @warning UNSTABLE Currently index of offsets and scalings is the same as in
   * `m_data`. This may be updated so that it is the same as row_select.
   *
   * v_new = (v_old+offset)*scaling
   *
   * @param row_select permutation matrix with rows to select
   * @param offsets Offsets to apply.
   * @param scalings Scale factors
   */
  void offsetsAndScales(std::vector<uint_t> row_select,
                        Eigen::Ref<VecFloat> offsets,
                        Eigen::Ref<VecFloat> scalings);

  void linearRegression(Eigen::Ref<VecFloat> x, Eigen::Ref<VecFloat> beta_0,
                        Eigen::Ref<VecFloat> beta_1);

  void averageAggregate(Eigen::Ref<MatFloat> out_data, uint_t agg_factor);

  void maskDifference(float_t diff_th, uint_t count_th,
                      Eigen::Ref<MatFloat> ref_data,
                      Eigen::Ref<MatFloat> mask_out);

  void fitPercentage(Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2);

  /**
   * @ingroup processing
   * @brief Computes the per-row mean of non-Nan values
   *
   * Processes chunks in parallel
   *
   * @param out_data Vec of rows length
   */
  void nanMean(Eigen::Ref<VecFloat> out_data);

  void computeMannKendallPValues(Eigen::Ref<VecFloat> out_data);

  void blocksAverage(Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2,
                     uint_t n_pix, uint_t y);

  void blocksAverageVecs(Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2,
                         uint_t n_pix, uint_t y, uint_t row_offset);

  void elementwiseAverage(Eigen::Ref<MatFloat> in1, Eigen::Ref<MatFloat> in2);

  void texturesBwTransform(Eigen::Ref<MatFloat> texture_2, float_t k, float_t a,
                           Eigen::Ref<MatFloat> sand, Eigen::Ref<MatFloat> silt,
                           Eigen::Ref<MatFloat> clay);

  void slidingWindowClassMode(Eigen::Ref<MatFloat> out_data,
                              size_t window_size);

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
                  std::vector<uint_t> result_indices, float_t swir1_scaling,
                  float_t red_scaling, float_t nir_scaling,
                  float_t blue_scaling, float_t result_scaling,
                  float_t result_offset, std::vector<float_t> clip_value);

  void computeEvi(std::vector<uint_t> red_indices,
                  std::vector<uint_t> nir_indices,
                  std::vector<uint_t> blue_indices,
                  std::vector<uint_t> result_indices, float_t red_scaling,
                  float_t nir_scaling, float_t blue_scaling,
                  float_t result_scaling, float_t result_offset,
                  std::vector<float_t> clip_value);

  void computeNirv(std::vector<uint_t> nir_indices,
                   std::vector<uint_t> red_indices,
                   std::vector<uint_t> result_indices, float_t nir_scaling,
                   float_t red_scaling, float_t result_scaling,
                   float_t result_offset, std::vector<float_t> clip_value);

  void computeFapar(std::vector<uint_t> red_indices,
                    std::vector<uint_t> nir_indices,
                    std::vector<uint_t> result_indices, float_t red_scaling,
                    float_t nir_scaling, float_t result_scaling,
                    float_t result_offset, std::vector<float_t> clip_value);

  void computeSavi(std::vector<uint_t> red_indices,
                   std::vector<uint_t> nir_indices,
                   std::vector<uint_t> result_indices, float_t red_scaling,
                   float_t nir_scaling, float_t result_scaling,
                   float_t result_offset, std::vector<float_t> clip_value);

  void computeGeometricTemperature(Eigen::Ref<MatFloat> latitude,
                                   Eigen::Ref<MatFloat> elevation,
                                   float_t elevation_scaling, float_t a,
                                   float_t b, float_t result_scaling,
                                   std::vector<uint_t> result_indices,
                                   std::vector<float_t> days_of_year);

  void computePercentiles(std::vector<uint_t> col_in_select,
                          Eigen::Ref<MatFloat> out_data,
                          std::vector<uint_t> col_out_select,
                          std::vector<float_t> percentiles);

  void fitProbabilities(Eigen::Ref<MatFloat> out_data, float_t input_scaling,
                        uint_t target_scaling,
                        Eigen::Ref<MatFloat> best_classes_data,
                        uint_t n_best_classes);

  void applyTsirf(Eigen::Ref<MatFloat> out_data, uint_t out_index_offset,
                  float_t w_0, Eigen::Ref<VecFloat> w_p,
                  Eigen::Ref<VecFloat> w_f, bool keep_original_values,
                  const std::string &version, const std::string &backend);

  void convolveRows(Eigen::Ref<MatFloat> out_data, float_t w_0,
                    Eigen::Ref<VecFloat> w_p, Eigen::Ref<VecFloat> w_f);

  void extractIndicators(Eigen::Ref<MatFloat> data_out, uint_t col_in_select,
                         std::vector<uint_t> col_out_select,
                         std::vector<uint_t> classes);

  void nanMeanAggregatePattern(Eigen::Ref<MatFloat> out_data,
                               std::vector<std::vector<uint_t>> &agg_pattern);
};

} // namespace skmap

#endif
