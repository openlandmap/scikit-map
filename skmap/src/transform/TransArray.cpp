#include "transform/TransArray.h"

namespace skmap {

TransArray::TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
: ParArray(data, n_threads) 
{
}


void TransArray::computeNormalizedDifference(std::vector<uint_t> positive_indices,
                                             std::vector<uint_t> negative_indices,
                                             std::vector<uint_t> result_indices,
                                             float_t positive_scaling,
                                             float_t negative_scaling,
                                             float_t result_scaling,
                                             float_t result_offset)
{

    skmapAssertIfTrue((positive_indices.size() != negative_indices.size()) || (positive_indices.size() != result_indices.size()),
                       "scikit-map ERROR 3: positive_index, negative_index, result_index must be of the same size");
    auto computeNormalizedDifferenceRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
    {
        uint_t positive_index = positive_indices[i];
        uint_t negative_index = negative_indices[i];
        row = (m_data.row(positive_index) * positive_scaling - m_data.row(negative_index) * negative_scaling).array() / 
              (m_data.row(positive_index) * positive_scaling + m_data.row(negative_index) * negative_scaling).array() *
              result_scaling + result_offset;
        row = (row.array()).round();
        row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
        row = (row.array() == -inf_v).select(result_scaling + result_offset, row);

    };
    this->parRowPerm(computeNormalizedDifferenceRow, result_indices);

}



}
