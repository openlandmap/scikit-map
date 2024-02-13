#include "transform/TransArray.h"

namespace skmap {

TransArray::TransArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
: ParArray(data, n_threads) 
{
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


void TransArray::reorderTransposeArray(Eigen::Ref<MatFloat> out_data,
                              std::vector<std::vector<uint_t>> indices_matrix)
{
    skmapAssertIfTrue((indices_matrix.size() != (uint_t) out_data.cols()) ||
                      (indices_matrix[0].size() * (uint_t) m_data.cols() != (uint_t) out_data.rows()),
                       "scikit-map ERROR 6: size of the new array does not match the input array and the requred reordering");
    auto reorderTransposeArrayCol = [&] (uint_t i)
    {
        for (uint_t j = 0; j < indices_matrix[i].size(); j++)
        {
            out_data.block(j*m_data.cols(), i, m_data.cols(), 1) = m_data.row(indices_matrix[i][j]).transpose();
        }
    };
    this->parForRange(reorderTransposeArrayCol, out_data.cols());
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

void TransArray::computeBsf(std::vector<std::vector<uint_t>> indices_matrix,
                std::vector<uint_t> result_indices,
                float_t threshold,
                float_t result_scaling)
{
    skmapAssertIfTrue(indices_matrix.size() != result_indices.size(),
                       "scikit-map ERROR 4: rows of indices_matrix must be same size of result_indices");
    auto computeBsfRow = [&] (uint_t i, Eigen::Ref<MatFloat::RowXpr> row)
    {

        std::vector<uint_t> rows_indices = indices_matrix[i];
        row.setZero();
        for (uint_t row_index : rows_indices) {
            auto res = (m_data.row(row_index).array() <= threshold).cast<float_t>();
            row = row.array() + res.array();
        }
        row = (row.array() * result_scaling / rows_indices.size()).round();
    };
    this->parRowPerm(computeBsfRow, result_indices);
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

}
