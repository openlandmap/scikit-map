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
        row = (m_data.row(nir_i) * nir_scaling
             - m_data.row(red_i) * red_scaling).array() / 
             ((m_data.row(nir_i) * nir_scaling
             + m_data.row(red_i) * red_scaling * 6.0
             - m_data.row(blue_i) * blue_scaling * 7.5).array()
             + 1.0).array() *
              result_scaling * 2.5 + result_offset;
        row = (row.array()).round();
        row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
        row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
        row = (row.array() < clip_value[0]).select(clip_value[0], row);
        row = (row.array() > clip_value[1]).select(clip_value[1], row);
    };
    this->parRowPerm(computeEviRow, result_indices);
}


void TransArray::computeNirv(std::vector<uint_t> red_indices,
                            std::vector<uint_t> nir_indices,
                            std::vector<uint_t> result_indices,
                            float_t red_scaling,
                            float_t nir_scaling,
                            float_t result_scaling,
                            float_t result_offset,
                            std::vector<float_t> clip_value)
{
    auto computeNirvRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
    {
        uint_t red_i = red_indices[i];
        uint_t nir_i = nir_indices[i];
        row = (((m_data.row(nir_i) * nir_scaling
               - m_data.row(red_i) * red_scaling).array() / 
                (m_data.row(nir_i) * nir_scaling
               + m_data.row(red_i) * red_scaling).array())
               - 0.08).array() *
                 (m_data.row(nir_i) * nir_scaling).array() *
                 result_scaling + result_offset;
        row = (row.array()).round();
        row = (row.array() == -inf_v).select(-result_scaling + result_offset, row);
        row = (row.array() == +inf_v).select(result_scaling + result_offset, row);
        row = (row.array() < clip_value[0]).select(clip_value[0], row);
        row = (row.array() > clip_value[1]).select(clip_value[1], row);
    };
    this->parRowPerm(computeNirvRow, result_indices);
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

}
