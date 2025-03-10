#include "transform/TransArray.h"

namespace skmap {

    void argsortMatrixAscending(Eigen::Ref<MatFloat> in_matrix, Eigen::Ref<MatUint> perm_matrix, bool sort_each_row_true_sort_each_col_false) 
    {
        if (sort_each_row_true_sort_each_col_false) 
        {
            #pragma omp parallel for
            for (int i = 0; i < in_matrix.rows(); ++i) 
            {
                std::vector<uint_t> indices(in_matrix.cols());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&](uint_t a, uint_t b)
                {return in_matrix(i, a) < in_matrix(i, b);});

                MatFloat sorted_row(1, in_matrix.cols());
                for (int j = 0; j < in_matrix.cols(); ++j) 
                {
                    sorted_row(0, j) = in_matrix(i, indices[j]);
                    perm_matrix(i, j) = indices[j];
                }
                in_matrix.row(i) = sorted_row;
            }
        } else {
            #pragma omp parallel for
            for (int j = 0; j < in_matrix.cols(); ++j) 
            {
                std::vector<uint_t> indices(in_matrix.rows());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&](uint_t a, uint_t b)
                {return in_matrix(a, j) < in_matrix(b, j);});

                MatFloat sorted_col(in_matrix.rows(), 1);
                for (int i = 0; i < in_matrix.rows(); ++i) 
                {
                    sorted_col(i, 0) = in_matrix(indices[i], j);
                    perm_matrix(i, j) = indices[i];
                }
                in_matrix.col(j) = sorted_col;
            }
        }
    }

    void applyPermutation(Eigen::Ref<MatFloat> sorted_matrix, Eigen::Ref<MatUint> perm_matrix, bool sort_each_row_true_sort_each_col_false) 
    {
        MatFloat original_matrix = sorted_matrix;
        if (sort_each_row_true_sort_each_col_false) {
            #pragma omp parallel for
            for (int i = 0; i < sorted_matrix.rows(); ++i) {
                for (int j = 0; j < sorted_matrix.cols(); ++j) {
                    original_matrix(i, perm_matrix(i, j)) = sorted_matrix(i, j);
                }
            }
        } else {
            #pragma omp parallel for
            for (int j = 0; j < sorted_matrix.cols(); ++j) {
                for (int i = 0; i < sorted_matrix.rows(); ++i) {
                    original_matrix(perm_matrix(i, j), j) = sorted_matrix(i, j);
                }
            }
        }
        sorted_matrix = original_matrix;
    }
    


    TransArray::TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
            : ParArray(data, n_threads)
    {
    }

    void TransArray::copyVecInMatrixRow(Eigen::Ref<VecFloat> in_vec,
                                        uint_t row_idx)
    {
        skmapAssertIfTrue((uint_t) m_data.cols() != (uint_t) in_vec.size(),
                          "scikit-map ERROR 33: number of input and output columns differ");

        uint_t chuck_size = std::ceil((float_t) m_data.cols() / (float_t) m_n_threads);
        auto copyRowsChunk = [&] (uint_t i)
        {
            uint_t col_start = i * chuck_size;
            uint_t col_end = std::min((i+1) * chuck_size, (uint_t) m_data.cols());
            m_data.row(row_idx).segment(col_start,col_end-col_start) = in_vec.segment(col_start,col_end-col_start);
        };
        this->parForRange(copyRowsChunk, m_n_threads);
    }

    void TransArray::scaleAndOffset(float_t offset,
                                    float_t scaling)
    {
        auto scaleAndOffsetChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            chunk.array() = chunk.array() * scaling + offset;
        };
        this->parChunk(scaleAndOffsetChunk);
    }


    void TransArray::reorderArray(Eigen::Ref<MatFloat> out_data,
                                  std::vector<std::vector<uint_t>> indices_matrix)
    {
        skmapAssertIfTrue((indices_matrix.size() != (uint_t) out_data.rows()) ||
                          (indices_matrix[0].size() * (uint_t) m_data.cols() != (uint_t) out_data.cols()),
                          "scikit-map ERROR 5: size of the new array does not match the input array and the requred reordering");
        auto reorderArrayRow = [&] (uint_t i)
        {
            for (uint_t j = 0; j < indices_matrix[i].size(); j++)
            {
                out_data.block(i, j*m_data.cols(), 1, m_data.cols()) = m_data.row(indices_matrix[i][j]);
            }
        };
        this->parForRange(reorderArrayRow, out_data.rows());
    }


    void TransArray::inverseReorderArray(Eigen::Ref<MatFloat> out_data,
                                         std::vector<std::vector<uint_t>> indices_matrix)
    {
        skmapAssertIfTrue((indices_matrix.size() != (uint_t) out_data.rows()),
                          "scikit-map ERROR 6: size of the new array does not match the size of the reordering matrix");
        auto inverseReorderArrayCol = [&] (uint_t i)
        {
            out_data.row(i) = m_data.block(indices_matrix[i][0], indices_matrix[i][1]*out_data.cols(), 1, out_data.cols());
        };
        this->parForRange(inverseReorderArrayCol, out_data.rows());
    }


    void TransArray::transposeReorderArray(Eigen::Ref<MatFloat> out_data,
                                           std::vector<std::vector<uint_t>> permutation_matrix)
    {
        auto transposeReorderArrayRow = [&] (uint_t i)
        {
            out_data.row(permutation_matrix[i][0]) = m_data.block(permutation_matrix[i][1]*out_data.cols(), permutation_matrix[i][2], 1, out_data.cols()).transpose();
        };
        this->parForRange(transposeReorderArrayRow, permutation_matrix.size());
    }



    void TransArray::selArrayRows(Eigen::Ref<MatFloat> out_data,
                                  std::vector<uint_t> row_select)
    {
        skmapAssertIfTrue((row_select.size() != (uint_t) out_data.rows()),
                          "scikit-map ERROR 14: size of the new array does not match the size of selected");
        auto selArrayRow = [&] (uint_t i)
        {
            out_data.row(i) = m_data.row(row_select[i]);
        };
        this->parForRange(selArrayRow, out_data.rows());
    }



    void TransArray::slidingWindowClassMode(Eigen::Ref<MatFloat> out_data,
                                  size_t window_size)
    {
        auto slidingWindowClassModeCol = [&] (uint_t j)
        {
            VecUint col_data = m_data.col(j).cast<uint_t>();
            uint_t half_window = std::floor(window_size / 2);
            for (uint_t i = 0; i < (uint_t) m_data.rows(); ++i) 
            {
                uint_t start = (uint_t) std::max(0, (int) i - (int) half_window);
                uint_t end = (uint_t) std::min((int)  m_data.rows(), (int)  i +  (int) half_window + 1);
                std::vector<uint_t> window(col_data.data() + start, col_data.data() + end);
                std::unordered_map<uint_t, uint_t> class_counts;
                for (uint_t val : window) 
                    class_counts[val]++;
                
                uint_t max_count = 0;
                for (const auto& pair : class_counts)
                    max_count = std::max(max_count, pair.second);
                
                std::vector<uint_t> dominant_classes;
                for (const auto& pair : class_counts) {
                    if (pair.second == max_count) 
                        dominant_classes.push_back(pair.first);
                }
                if (dominant_classes.size() == 1) {  // Single dominant class
                    out_data(i,j) = float_t (dominant_classes[0]);
                } else {  // Multiple dominant classes
                    out_data(i,j) = float_t (*std::min_element(dominant_classes.begin(), dominant_classes.end()));
                }
            }

        };
        this->parForRange(slidingWindowClassModeCol, m_data.cols());
    }



    void TransArray::expandArrayRows(Eigen::Ref<MatFloat> out_data,
                                     std::vector<uint_t> row_select)
    {
        skmapAssertIfTrue((row_select.size() > (uint_t) m_data.rows()),
                          "scikit-map ERROR 15: size of the old array does not match the size of selected");
        auto expandArrayRow = [&] (uint_t i)
        {
            out_data.row(row_select[i]) = m_data.row(i);
        };
        this->parForRange(expandArrayRow, row_select.size());
    }
    

    void TransArray::selArrayCols(Eigen::Ref<MatFloat> out_data,
                                  std::vector<uint_t> col_select)
    {
        skmapAssertIfTrue((col_select.size() != (uint_t) out_data.cols()),
                          "scikit-map ERROR 14: size of the new array does not match the size of selected");
        auto selArrayCol = [&] (uint_t i)
        {
            out_data.col(i) = m_data.col(col_select[i]);
        };
        this->parForRange(selArrayCol, out_data.cols());
    }


    void TransArray::expandArrayCols(Eigen::Ref<MatFloat> out_data,
                                     std::vector<uint_t> col_select)
    {
        skmapAssertIfTrue((col_select.size() > (uint_t) m_data.cols()),
                          "scikit-map ERROR 15: size of the old array does not match the size of selected");
        auto expandArrayCol = [&] (uint_t i)
        {
            out_data.col(col_select[i]) = m_data.col(i);
        };
        this->parForRange(expandArrayCol, col_select.size());
    }
    

    void TransArray::blocksAverage(Eigen::Ref<MatFloat> in1,
                                  Eigen::Ref<MatFloat> in2,
                                  uint_t n_pix,
                                  uint_t y)
    {
        auto blocksAverageChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            chunk.array() = 0.25 * (in1.block(row_start, y*n_pix, row_end - row_start, (y+1)*n_pix).array() + 
                                    in1.block(row_start, (y+1)*n_pix, row_end - row_start, (y+2)*n_pix).array() + 
                                    in2.block(row_start, y*n_pix, row_end - row_start, (y+1)*n_pix).array() + 
                                    in2.block(row_start, (y+1)*n_pix, row_end - row_start, (y+2)*n_pix).array());
        };
        this->parChunk(blocksAverageChunk);
    }

  

    void TransArray::blocksAverageVecs(Eigen::Ref<MatFloat> in1,
                                  Eigen::Ref<MatFloat> in2,
                                  uint_t n_pix,
                                  uint_t y,
                                  uint_t row_offset)
    {
        auto blocksAverageVecsElem = [&] (uint_t i)
        {
            m_data(row_offset,i) = 0.25 * (in1(0, i + y*n_pix) + 
                                           in1(0, i + (y+1)*n_pix) + 
                                           in2(0, i + y*n_pix) + 
                                           in2(0, i + (y+1)*n_pix));
        };
        this->parForRange(blocksAverageVecsElem, n_pix);
    }


    void TransArray::elementwiseAverage(Eigen::Ref<MatFloat> in1,
                                  Eigen::Ref<MatFloat> in2)
    {
        skmapAssertIfTrue(((uint_t) in1.cols() != (uint_t) in2.cols()),
                          "scikit-map ERROR 52: the two input arrays must have the same number of columns");
        auto elementwiseAverageChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            chunk.array() = 0.5 * (in1.block(row_start, 0, row_end - row_start, in1.cols()).array() + 
                                   in2.block(row_start, 0, row_end - row_start, in2.cols()).array());
        };
        this->parChunk(elementwiseAverageChunk);
    }


    void TransArray::extractArrayRows(Eigen::Ref<MatFloat> out_data,
                                      std::vector<uint_t> row_select)
    {
        skmapAssertIfTrue((row_select.size() > (uint_t) out_data.rows()),
                          "scikit-map ERROR 35: size of the old array does not match the size of selected");
        auto extractArrayRow = [&] (uint_t i)
        {
            out_data.row(i) = m_data.row(row_select[i]);
        };
        this->parForRange(extractArrayRow, row_select.size());
    }


    void TransArray::extractArrayCols(Eigen::Ref<MatFloat> out_data,
                                      std::vector<uint_t> col_select)
    {
        skmapAssertIfTrue((col_select.size() > (uint_t) out_data.cols()),
                          "scikit-map ERROR 35: size of the old array does not match the size of selected");
        auto extractArrayCol = [&] (uint_t i)
        {
            out_data.col(i) = m_data.col(col_select[i]);
        };
        this->parForRange(extractArrayCol, col_select.size());
    }




    void TransArray::swapRowsValues(std::vector<uint_t> row_select,
                                    float_t value_to_mask,
                                    float_t new_value)
    {
        auto swapRowValues = [&] (uint_t i)
        {
            m_data.row(row_select[i]) = (m_data.row(row_select[i]).array() == value_to_mask)
                    .select(new_value, m_data.row(row_select[i]));
        };
        this->parForRange(swapRowValues, row_select.size());
    }


    void TransArray::maskNan(std::vector<uint_t> row_select,
                             float_t new_value)
    {
        auto swapRowValues = [&] (uint_t i)
        {
            auto tmp_row = m_data.row(row_select[i]);
            tmp_row = tmp_row.array().isNaN().select(new_value, tmp_row);
        };
        this->parForRange(swapRowValues, row_select.size());
    }

    void TransArray::maskNanRows(std::vector<uint_t> row_select,
                             Eigen::Ref<VecFloat> new_value_vec)
    {
        skmapAssertIfTrue((row_select.size() != (uint_t) new_value_vec.size()),
                          "scikit-map ERROR 48: row_select and new_value_vec must be of the same size");
        auto swapRowValues = [&] (uint_t i)
        {
            auto tmp_row = m_data.row(row_select[i]);
            tmp_row = tmp_row.array().isNaN().select(new_value_vec(i), tmp_row);
        };
        this->parForRange(swapRowValues, row_select.size());
    }

    void TransArray::maskData(std::vector<uint_t> row_select,
                              Eigen::Ref<MatFloat> mask,
                              float_t value_of_mask_to_mask,
                              float_t new_value_in_data)
    {
        auto maskDataRow = [&] (uint_t i)
        {
            m_data.row(row_select[i]) = (mask.row(i).array() == value_of_mask_to_mask)
                    .select(new_value_in_data, m_data.row(row_select[i]));
        };
        this->parForRange(maskDataRow, row_select.size());
    }





    void TransArray::maskDataRows(std::vector<uint_t> row_select,
                              Eigen::Ref<MatFloat> mask,
                              float_t value_of_mask_to_mask,
                              float_t new_value_in_data)
    {
        auto maskDataRow = [&] (uint_t i)
        {
            m_data.row(row_select[i]) = (mask.row(0).array() == value_of_mask_to_mask)
                    .select(new_value_in_data, m_data.row(row_select[i]));
        };
        this->parForRange(maskDataRow, row_select.size());
    }




    void TransArray::fillArray(float_t val)
    {
        auto fillArrayRow = [&] (uint_t i)
        {
            m_data.row(i).array() =  val;
        };
        this->parForRange(fillArrayRow, m_data.rows());
    }

    void TransArray::texturesBwTransform(Eigen::Ref<MatFloat> texture_2,
                                        float_t k,
                                        float_t a,
                                        Eigen::Ref<MatFloat> sand,
                                        Eigen::Ref<MatFloat> silt,
                                        Eigen::Ref<MatFloat> clay)
    {
        auto texturesBwTransformChunk = [&](Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end) 
        {
            // Extract the relevant block from texture_1 and texture_2
            MatFloat x1 = Eigen::pow(2., chunk.array());
            MatFloat x2 = Eigen::pow(2., texture_2.block(row_start, 0, row_end - row_start, texture_2.cols()).array());

            // Compute the inverse transformation
            MatFloat C = ((1. - (x1.array() + x2.array() - 2.) * k).array() / (x1.array() + x2.array() + 1.).array()).cwiseMax(0.);
            MatFloat S = (x1.array() * C.array() + x1.array() * k - k).cwiseMax(0);
            MatFloat L = (x2.array() * C.array() + x2.array() * k - k).cwiseMax(0);
            
            // Compute total sum for normalization
            MatFloat total = S.array() + L.array() + C.array();

            // Avoid division by zero: Compute scale factor
            MatFloat scale_factor = (total.array() > 0).select(a / total.array(), 0.);  // If total > 0, scale; else, set to 0

            // Store results in the respective blocks
            sand.block(row_start, 0, row_end - row_start, sand.cols()) = S.array() * scale_factor.array();
            silt.block(row_start, 0, row_end - row_start, silt.cols()) = L.array() * scale_factor.array();
            clay.block(row_start, 0, row_end - row_start, clay.cols()) = C.array() * scale_factor.array();

            
            // Apply NaN mask where necessary (assuming NaN handling is required)
            MatBool mask = (chunk.array().isNaN() || texture_2.block(row_start, 0, row_end - row_start, texture_2.cols()).array().isNaN());

            sand.block(row_start, 0, row_end - row_start, sand.cols()) = mask.select(nan_v, sand.block(row_start, 0, row_end - row_start, sand.cols()).array());
            clay.block(row_start, 0, row_end - row_start, clay.cols()) = mask.select(nan_v, clay.block(row_start, 0, row_end - row_start, clay.cols()).array());
            silt.block(row_start, 0, row_end - row_start, silt.cols()) = mask.select(nan_v, silt.block(row_start, 0, row_end - row_start, silt.cols()).array());
        };

        // Apply parallel execution
        this->parChunk(texturesBwTransformChunk);
    }

    
    void TransArray::fitPercentage(Eigen::Ref<MatFloat> in1,
                                   Eigen::Ref<MatFloat> in2)
    {
    auto fitPercentageChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
    {
    chunk.array() = 100. / (in1.block(row_start, 0, row_end - row_start, in1.cols()).array() + 
            in2.block(row_start, 0, row_end - row_start, in2.cols()).array() +
            1.);
    };
    this->parChunk(fitPercentageChunk);
    }


    void TransArray::hadamardProduct(Eigen::Ref<MatFloat> in1,
                                     Eigen::Ref<MatFloat> in2)
    {
        auto hadamardProductChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            chunk.array() = in1.block(row_start, 0, row_end - row_start, in1.cols()).array() *
                            in2.block(row_start, 0, row_end - row_start, in2.cols()).array();
        };
        this->parChunk(hadamardProductChunk);
    }


    void TransArray::transposeArray(Eigen::Ref<MatFloat> out_data)
    {
        skmapAssertIfTrue((m_data.rows() != out_data.cols()) ||
                          (m_data.cols() != out_data.rows()),
                          "scikit-map ERROR 13: size of the new array does not match the transpose input array");
        auto transposeArrayCol = [&] (uint_t i)
        {
            out_data.row(i) = m_data.col(i).transpose();
        };
        this->parForRange(transposeArrayCol, out_data.rows());
    }


    void TransArray::computeNormalizedDifference(std::vector<uint_t> positive_indices,
                                                 std::vector<uint_t> negative_indices,
                                                 std::vector<uint_t> result_indices,
                                                 float_t positive_scaling,
                                                 float_t negative_scaling,
                                                 float_t result_scaling,
                                                 float_t result_offset,
                                                 std::vector<float_t> clip_value)
    {
        skmapAssertIfTrue((positive_indices.size() != negative_indices.size()) || (positive_indices.size() != result_indices.size()),
                          "scikit-map ERROR 3: positive_i, negative_i, result_i must be of the same size");
        auto computeNormalizedDifferenceRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t positive_i = positive_indices[i];
            uint_t negative_i = negative_indices[i];
            row = (m_data.row(positive_i) * positive_scaling - m_data.row(negative_i) * negative_scaling).array() /
                  (m_data.row(positive_i) * positive_scaling + m_data.row(negative_i) * negative_scaling).array() *
                  result_scaling + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeNormalizedDifferenceRow, result_indices);
    }


    void TransArray::computeNirv(std::vector<uint_t> nir_indices,
                                 std::vector<uint_t> red_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t nir_scaling,
                                 float_t red_scaling,
                                 float_t result_scaling,
                                 float_t result_offset,
                                 std::vector<float_t> clip_value)
    {
        skmapAssertIfTrue((nir_indices.size() != red_indices.size()) || (nir_indices.size() != result_indices.size()),
                          "scikit-map ERROR 3: nir_i, red_i, result_i must be of the same size");
        auto computeNirvRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t nir_i = nir_indices[i];
            uint_t red_i = red_indices[i];
            row = ((((m_data.row(nir_i) * nir_scaling - m_data.row(red_i) * red_scaling).array() /
                     (m_data.row(nir_i) * nir_scaling + m_data.row(red_i) * red_scaling).array()) - 0.08).array() * (m_data.row(nir_i) * nir_scaling).array()) *
                  result_scaling + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeNirvRow, result_indices);
    }


    void TransArray::computeBsi(std::vector<uint_t> swir1_indices,
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
                                std::vector<float_t> clip_value)
    {
        auto computeBsiRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t swir1_i = swir1_indices[i];
            uint_t red_i = red_indices[i];
            uint_t nir_i = nir_indices[i];
            uint_t blue_i = blue_indices[i];
            row = (m_data.row(swir1_i) * swir1_scaling
                   + m_data.row(red_i) * red_scaling
                   - m_data.row(nir_i) * nir_scaling
                   - m_data.row(blue_i) * blue_scaling).array() /
                  (m_data.row(swir1_i) * swir1_scaling
                   + m_data.row(red_i) * red_scaling
                   + m_data.row(nir_i) * nir_scaling
                   + m_data.row(blue_i) * blue_scaling).array() *
                  result_scaling + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeBsiRow, result_indices);
    }


    void TransArray::computeEvi(std::vector<uint_t> red_indices,
                                std::vector<uint_t> nir_indices,
                                std::vector<uint_t> blue_indices,
                                std::vector<uint_t> result_indices,
                                float_t red_scaling,
                                float_t nir_scaling,
                                float_t blue_scaling,
                                float_t result_scaling,
                                float_t result_offset,
                                std::vector<float_t> clip_value)
    {
        auto computeEviRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t red_i = red_indices[i];
            uint_t nir_i = nir_indices[i];
            uint_t blue_i = blue_indices[i];
            row = ((m_data.row(nir_i) * nir_scaling - m_data.row(red_i) * red_scaling).array() /
                   ((m_data.row(nir_i) * nir_scaling + m_data.row(red_i) * red_scaling * 6.0 - m_data.row(blue_i) * blue_scaling * 7.5).array() + 1.0).array()).array() *
                  result_scaling * 2.5 + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeEviRow, result_indices);
    }



    void TransArray::computeFapar(std::vector<uint_t> red_indices,
                                  std::vector<uint_t> nir_indices,
                                  std::vector<uint_t> result_indices,
                                  float_t red_scaling,
                                  float_t nir_scaling,
                                  float_t result_scaling,
                                  float_t result_offset,
                                  std::vector<float_t> clip_value)
    {
        auto computeFaparRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t red_i = red_indices[i];
            uint_t nir_i = nir_indices[i];
            row = ((((((m_data.row(nir_i) * nir_scaling
                        - m_data.row(red_i) * red_scaling).array() /
                       (m_data.row(nir_i) * nir_scaling
                        + m_data.row(red_i) * red_scaling).array())
                      - 0.03).array() * (0.95 - 0.001)).array() / (0.96 - 0.03)) + 0.001).array() *
                  result_scaling + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeFaparRow, result_indices);
    }


    void TransArray::computeSavi(std::vector<uint_t> red_indices,
                                  std::vector<uint_t> nir_indices,
                                  std::vector<uint_t> result_indices,
                                  float_t red_scaling,
                                  float_t nir_scaling,
                                  float_t result_scaling,
                                  float_t result_offset,
                                  std::vector<float_t> clip_value)
    {
        auto computeSaviRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            uint_t red_i = red_indices[i];
            uint_t nir_i = nir_indices[i];
            row = (((m_data.row(nir_i) * nir_scaling
                        - m_data.row(red_i) * red_scaling).array() * 1.5).array() /
                       ((m_data.row(nir_i) * nir_scaling
                        + m_data.row(red_i) * red_scaling).array() + 0.5).array()).array() *
                  result_scaling + result_offset;
            row = (row.array()).round();
            row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
            row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
            row = (row.array() < clip_value[0]).select(clip_value[0], row);
            row = (row.array() > clip_value[1]).select(clip_value[1], row);
        };
        this->parRowPerm(computeSaviRow, result_indices);
    }



    void TransArray::computeGeometricTemperature(Eigen::Ref<MatFloat> latitude,
                                                 Eigen::Ref<MatFloat> elevation,
                                                 float_t elevation_scaling,
                                                 float_t a,
                                                 float_t b,
                                                 float_t result_scaling,
                                                 std::vector<uint_t> result_indices,
                                                 std::vector<float_t> days_of_year)
    {
        skmapAssertIfTrue(days_of_year.size() != result_indices.size(),
                          "scikit-map ERROR 8: result_indices must be same size of days_of_year");

        auto computeGeometricTemperatureRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {
            float_t day_of_year = days_of_year[i];
            row = latitude.unaryExpr([&day_of_year, &a, &b] (float_t lat)
                                     {
                                         float_t cos_teta = std::cos(((day_of_year - 18.) / 182.5 + std::pow(4., (float_t) (lat < 0.))) * M_PI);
                                         float_t cos_fi = std::cos(lat * M_PI / 180.);
                                         float_t sin_abs_fi = std::abs(std::sin(lat * M_PI / 180.));
                                         float_t res = a * cos_fi + b * (1. - cos_teta) * sin_abs_fi;
                                         return res;
                                     }).array() - 0.006 * elevation_scaling * elevation.array();
            row = (row.array() * result_scaling).round();
        };
        this->parRowPerm(computeGeometricTemperatureRow, result_indices);

    }

    void TransArray::extractIndicators(Eigen::Ref<MatFloat> data_out,
                               uint_t col_in_select,
                               std::vector<uint_t> col_out_select,
                               std::vector<uint_t> classes)
    {
        skmapAssertIfTrue(col_out_select.size() != classes.size(),
                          "scikit-map ERROR 38: size of col_out_select differs from size of col_in_select times size of classes");

        auto extractIndicatorsChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            for (uint_t k = 0; k < classes.size(); ++k) {
                data_out.block(row_start, col_out_select[k], row_end - row_start, 1) = (chunk.col(col_in_select).array() == classes[k]).cast<float_t>();
            }
        };
        this->parChunk(extractIndicatorsChunk);
    }



    void TransArray::computePercentiles(std::vector<uint_t> col_in_select,
                                        Eigen::Ref<MatFloat> out_data,
                                        std::vector<uint_t> col_out_select,
                                        std::vector<float_t> percentiles)
    {

        skmapAssertIfTrue(col_out_select.size() != percentiles.size(),
                          "scikit-map ERROR 19: out_data too small");
        // @FIXME: the clean way would be that also for out_data a ParArray oject is created and that only the corresponding cunk is sent
        auto computePercentilesChunk = [&] (Eigen::Ref<MatFloat> rows_block, uint_t row_start, uint_t row_end)
        {
            MatFloat chunk(rows_block.rows(), col_in_select.size());
            for (size_t i = 0; i < col_in_select.size(); ++i) {
                chunk.col(i) = rows_block.col(col_in_select[i]);
            }

            MatFloat sorted_chunk(chunk);
            float_t max_float = std::numeric_limits<float_t>::max();
            sorted_chunk = sorted_chunk.array().isNaN().select(max_float, sorted_chunk);
            Eigen::VectorXi not_nan_count(chunk.rows());
            for (uint_t i = 0; i < (uint_t) sorted_chunk.rows(); ++i) 
            {
                std::vector<float_t> sorted_row(sorted_chunk.row(i).data(), sorted_chunk.row(i).data() + sorted_chunk.row(i).size());
                std::sort(sorted_row.begin(), sorted_row.end());
                for (uint_t j = 0; j < (uint_t) sorted_chunk.cols(); ++j) {
                    sorted_chunk(i, j) = sorted_row[j];
                }
                not_nan_count(i) = chunk.cols() - (chunk.row(i).array().isNaN()).count();
            }

            for (uint_t k = 0; k < percentiles.size(); ++k) {
                for (uint_t i = 0; i < (uint_t) sorted_chunk.rows(); ++i) {
                    if (not_nan_count(i) == 0)
                    {
                        out_data(row_start+i, col_out_select[k]) = nan_v;
                    } else
                    {
                        float_t percentile_pos = (float_t) (not_nan_count(i) - 1) * (percentiles[k] / 100.0);
                        uint_t percentile_pos_floor = floor(percentile_pos);
                        uint_t percentile_pos_ceil = ceil(percentile_pos);
                        if (percentile_pos_floor == percentile_pos_ceil)
                        {
                            out_data(row_start+i, col_out_select[k]) = sorted_chunk(i, percentile_pos_floor);
                        } else
                        {
                            float_t percentile_val_floor = sorted_chunk(i, percentile_pos_floor) * ((float_t) percentile_pos_ceil - percentile_pos);
                            float_t percentile_val_ceil = sorted_chunk(i, percentile_pos_ceil) * (percentile_pos - (float_t) percentile_pos_floor);
                            out_data(row_start+i, col_out_select[k]) = percentile_val_floor + percentile_val_ceil;
                        }
                    }
                }

            }
        };
        this->parChunk(computePercentilesChunk);
    }


    void TransArray::fitProibabilites(Eigen::Ref<MatFloat> out_data,
                                      float_t input_scaling,
                                      uint_t target_scaling,
                                      Eigen::Ref<MatFloat> best_classes_data,
                                      uint_t n_best_classes)
    {
        skmapAssertIfTrue(out_data.cols() != m_data.cols(),
                          "scikit-map ERROR 56: in and out data should have the same number of columns");
        skmapAssertIfTrue((uint_t) best_classes_data.cols() != n_best_classes,
                                  "scikit-map ERROR 57: best_classes_data must have n_best_classes columns");

        uint_t n_classes = m_data.cols();
        int_t power_of_two_start = 1;
        while (std::pow(2, power_of_two_start + 1) < n_classes)
            power_of_two_start++;
        
        auto fitProibabilitesChunk = [&] (Eigen::Ref<MatFloat> in_chunk, uint_t row_start, uint_t row_end)
        {
            uint_t n_pix = row_end - row_start;
            auto out_chunk = out_data.block(row_start, 0, n_pix, n_classes);
            out_chunk = in_chunk;
            MatUint perm_matrix(n_pix, n_classes);
            out_chunk = out_chunk.array() / input_scaling * (float_t) target_scaling;

            argsortMatrixAscending(out_chunk, perm_matrix, true);
            
            best_classes_data.block(row_start, 0, n_pix, n_best_classes) = perm_matrix.block(0, n_classes-n_best_classes, n_pix, n_best_classes).cast<float_t>();
            
            MatFloat out_chunk_floor = out_chunk.array().floor();
            MatFloat out_chunk_ceil = out_chunk.array().ceil();
            VecFloat sum_floor = out_chunk_floor.rowwise().sum();
            VecFloat sum_ceil = out_chunk_ceil.rowwise().sum();
    
            #pragma omp parallel for
            for (uint_t i = 0; i < n_pix; ++i)
            {
                if (std::isnan(sum_floor(i)) || std::isnan(sum_ceil(i)))
                {
                    out_chunk.row(i).array() = nan_v;
                    best_classes_data.row(row_start+i).array() = nan_v;
                    continue;
                }

                // Adjust floor sum
                uint_t last_modified_floor = 0;
                while (sum_floor(i) > target_scaling)
                {
                    while (out_chunk_floor(i, last_modified_floor) == 0)
                        last_modified_floor = (last_modified_floor + 1) % n_classes;
                    out_chunk_floor(i, last_modified_floor) -= 1;
                    sum_floor(i) -= 1;
                }
    
                // Adjust ceil sum
                uint_t last_modified_ceil = n_classes - 1;
                while (sum_ceil(i) < target_scaling)
                {
                    out_chunk_ceil(i, last_modified_ceil) += 1;
                    sum_ceil(i) += 1;
                    last_modified_ceil = (last_modified_ceil - 1) % n_classes;
                }
    
                // Apply binary search-based for mid selection
                uint_t tmp_sum = target_scaling + 1;
                int_t power_of_two = power_of_two_start;
                uint_t mid = 1 << power_of_two; // mid = 2^power_of_two
                VecFloat mix_vec(n_classes);
                while (tmp_sum != target_scaling)
                {
                    power_of_two--;
                    mix_vec.head(mid) = out_chunk_floor.row(i).head(mid);
                    mix_vec.tail(n_classes - mid) = out_chunk_ceil.row(i).tail(n_classes - mid);
                    tmp_sum = (uint_t) mix_vec.sum();
                    if (tmp_sum == target_scaling || power_of_two < 0)
                        break;
                    else if (tmp_sum < target_scaling)
                    {
                        mid = mid - (1 << power_of_two);
                        skmapAssertIfTrue(mid < 0, "Mid should not get inferior to 0, something went wrong");
                    }
                    else
                    {
                        mid = std::min(mid + (1 << power_of_two), n_classes);
                    }
                }
                out_chunk.row(i) = mix_vec.transpose();
            }

            applyPermutation(out_chunk, perm_matrix, true);

        };
        this->parChunk(fitProibabilitesChunk);

    }


    void TransArray::offsetAndScale(float_t offset,
                                    float_t scaling)
    {
        auto offsetAndScaleChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            chunk.array() = (chunk.array() + offset) * scaling;
        };
        this->parChunk(offsetAndScaleChunk);
    }



    void TransArray::offsetsAndScales(std::vector<uint_t> row_select,
                                    Eigen::Ref<VecFloat> offsets,
                                    Eigen::Ref<VecFloat> scalings)
    {
        auto offsetsAndScalesRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
        {            
            row = (row.array() + offsets(row_select[i])) * scalings(row_select[i]);
        };
        this->parRowPerm(offsetsAndScalesRow, row_select);
    }




    void TransArray::maskDifference(float_t diff_th,
                                    uint_t count_th,
                                    Eigen::Ref<MatFloat> ref_data,
                                    Eigen::Ref<MatFloat> mask_out)
    {
        auto maskDifferenceChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            auto ref_data_chunk = ref_data.block(row_start, 0, row_end - row_start, ref_data.cols());
            auto mask_out_chunk = mask_out.block(row_start, 0, row_end - row_start, ref_data.cols());
            MatFloat diff = (chunk.array() - ref_data_chunk.array()).abs();
            MatBool mask = (diff.array() > diff_th);
            VecUint count = mask.cast<uint_t>().rowwise().sum();
            // If the masking is recurring in several columns it is not considered outlier and so not masked
            VecBool mask_threshold = (count.array() > count_th);
            for (uint_t i = 0; i < (uint_t) mask.rows(); ++i) 
            {
                if (mask_threshold(i)) 
                    mask.row(i).setConstant(false);
            }
            mask_out_chunk = mask.cast<uint_t>().cast<float_t>();
        };
        this->parChunk(maskDifferenceChunk);
    }


    void TransArray::applyTsirf(Eigen::Ref<MatFloat> out_data,
                                 uint_t out_index_offset,
                                 float_t w_0,
                                 Eigen::Ref<VecFloat> w_p,
                                 Eigen::Ref<VecFloat> w_f,
                                 bool keep_original_values,
                                 const std::string& version,
                                 const std::string& backend)
    {
        auto applyTsirfChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            uint_t n_s = chunk.cols();
            uint_t n_e = n_s + std::max(w_p.size(), w_f.size());
            VecFloat w_e = VecFloat::Zero(n_e);
            w_e(0) = w_0;
            w_e.segment(1, w_f.size()) = w_f;
            w_e.segment(n_e-w_p.size(), w_p.size()) = w_p;
            auto out_block = out_data.block(out_index_offset + row_start, 0, row_end - row_start, n_s);
            MatFloat W(n_s, n_s);
            for (uint_t i = 0; i < n_s; ++i)
            {
                for (uint_t j = 0; j < n_s; ++j)
                {
                    W(j, i) = w_e((-i + j + n_e) % n_e);
                }
            }
            MatBool gaps_mask = chunk.array().isNaN();
            MatBool valid_mask = chunk.array().isFinite();
            MatFloat chunk_masked = chunk;
            chunk_masked = gaps_mask.select(0.0, chunk_masked);
            MatFloat den = valid_mask.cast<float_t>() * W;
            out_block = (chunk_masked * W).array() / den.array();
            MatFloat NaNs = MatFloat::Constant(out_block.rows(), out_block.cols(), nan_v);
            out_block = (den.array() < w_e.minCoeff()).select(NaNs, out_block);
            out_block = valid_mask.select(chunk_masked, out_block);
        };
        this->parChunk(applyTsirfChunk);
    }

    void TransArray::convolveRows(Eigen::Ref<MatFloat> out_data,
                                 float_t w_0,
                                 Eigen::Ref<VecFloat> w_p,
                                 Eigen::Ref<VecFloat> w_f)
    {

        auto convolveRowsChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            uint_t n_s = chunk.cols();
            uint_t n_e = n_s + std::max(w_p.size(), w_f.size());
            VecFloat w_e = VecFloat::Zero(n_e);
            w_e(0) = w_0;
            w_e.segment(1, w_f.size()) = w_f;
            w_e.segment(n_e-w_p.size(), w_p.size()) = w_p;
            auto out_block = out_data.block(row_start, 0, row_end - row_start, n_s);
            MatFloat W(n_s, n_s);
            for (uint_t i = 0; i < n_s; ++i)
            {
                for (uint_t j = 0; j < n_s; ++j)
                {
                    W(j, i) = w_e((-i + j + n_e) % n_e);
                }
            }
            out_block = chunk * W;
        };
        this->parChunk(convolveRowsChunk);

    }

   
    
    void TransArray::averageAggregate(Eigen::Ref<MatFloat> out_data,
                                      uint_t agg_factor)
    {
        auto averageAggregateChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            uint_t n_chucks = std::ceil((float_t) chunk.cols() / (float_t) agg_factor);
            skmapAssertIfTrue((uint_t) out_data.cols() != n_chucks,
                              "scikit-map ERROR 34: wrong number of columns of the aggregated matrix");
            for (uint_t i = 0; i < n_chucks; ++i)
            {
                uint_t col_start = i * agg_factor;
                uint_t col_end = std::min((i+1) * agg_factor, (uint_t) chunk.cols());
                out_data.col(i).segment(row_start, row_end - row_start) = chunk.middleCols(col_start, col_end-col_start).rowwise().mean();

            }
        };
        this->parChunk(averageAggregateChunk);
    }

    
    void TransArray::nanMeanAggregatePattern(Eigen::Ref<MatFloat> out_data,
                                             std::vector<std::vector<uint_t>>& agg_pattern)
    {
        auto nanMeanAggregatePatternChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            skmapAssertIfTrue((uint_t) out_data.cols() != agg_pattern.size(),
                              "scikit-map ERROR 43: wrong number of columns of the output matrix");
            for (uint_t i = 0; i < agg_pattern.size(); ++i)
            {
                MatFloat chunk_col_sel(chunk.rows(), agg_pattern[i].size());
                for (uint_t j = 0; j < agg_pattern[i].size(); ++j)
                    chunk_col_sel.col(j) = chunk.col(agg_pattern[i][j]);
                auto res = out_data.block(row_start, i, row_end - row_start, 1);
                VecFloat count = (float_t) chunk_col_sel.cols() - chunk_col_sel.array().isNaN().cast<float_t>().rowwise().sum().array();
                MatFloat data_no_nan = chunk_col_sel;
                data_no_nan = data_no_nan.array().isNaN().select(0., data_no_nan);
                VecFloat sum = data_no_nan.rowwise().sum();
                res = sum.array() / count.array();
                res = (count.array() == 0.).select(nan_v, res);
            }
        };
        this->parChunk(nanMeanAggregatePatternChunk);
    }


    void TransArray::nanMean(Eigen::Ref<VecFloat> out_data)
    {
        auto nanMeanChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            auto res = out_data.segment(row_start, row_end - row_start);
            VecFloat count = (float_t) chunk.cols() - chunk.array().isNaN().cast<float_t>().rowwise().sum().array();
            MatFloat data_no_nan = chunk;
            data_no_nan = data_no_nan.array().isNaN().select(0., data_no_nan);
            VecFloat sum = data_no_nan.rowwise().sum();
            res = sum.array() / count.array();
            res = (count.array() == 0.).select(nan_v, res);
        };
        this->parChunk(nanMeanChunk);

    }

    void TransArray::computeMannKendallPValues(Eigen::Ref<VecFloat> out_data)
    {
        auto computeMannKendallPValuesChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
        {
            float_t n_cols = chunk.cols();
            float_t n = (float_t) n_cols;
            float_t var_s = (n * (n - 1.) * (2. * n + 5.))/18.;
            for (uint_t i = 0; i < (uint_t) chunk.rows(); i ++)
            {
                // Vectorized computation of differences
                MatFloat mat = MatFloat::Zero(n_cols, n_cols);
                mat.triangularView<Eigen::Upper>() = chunk.row(i).transpose().replicate(n_cols, 1) - chunk.row(i).replicate(1, n_cols);
                // Mann-Kendal's score
                float_t s = mat.unaryExpr(std::ptr_fun(signFunc)).array().sum();
                float_t z_score = 0.;
                if (s > 0.)
                    z_score = (s - 1.) / std::sqrt(var_s);
                else if (s < 0.)
                    z_score = (s + 1.) / std::sqrt(var_s);
                // p-value
                out_data(row_start+i) = 2. * (1. - standardNormalCdf(std::abs(z_score)));
            }
        };
        this->parChunk(computeMannKendallPValuesChunk);
    }

void TransArray::linearRegression(Eigen::Ref<VecFloat> x,
                                  Eigen::Ref<VecFloat> beta_0,
                                  Eigen::Ref<VecFloat> beta_1)
{
    skmapAssertIfTrue((uint_t) m_data.cols() != (uint_t) x.size(),
                      "scikit-map ERROR 37: regression variable is assumed to match size of data columns");
    auto linearRegressionChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
    {
        float_t n_samp = chunk.cols();
        VecFloat Y_mean = chunk.rowwise().mean();
        float_t x_mean = x.mean();
        MatFloat covariance_p1 = chunk.colwise() - Y_mean;
        VecFloat covariance_p2 = x.array() - x_mean;
        VecFloat covariance = covariance_p1 * covariance_p2;
        covariance /= n_samp;
        float_t x_variance = (x.array() - x_mean).square().sum() / n_samp;
        beta_1.segment(row_start, row_end - row_start) = covariance.array() / x_variance;
        beta_0.segment(row_start, row_end - row_start) = Y_mean - beta_1 * x_mean;

    };
    this->parChunk(linearRegressionChunk);
}


}

