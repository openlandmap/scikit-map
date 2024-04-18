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




void TransArray::expandArrayRows(Eigen::Ref<MatFloat> out_data,
                              std::vector<uint_t> row_select)
{
    skmapAssertIfTrue((row_select.size() != (uint_t) m_data.rows()),
                       "scikit-map ERROR 15: size of the old array does not match the size of selected");
    auto expandArrayRow = [&] (uint_t i)
    {
        out_data.row(row_select[i]) = m_data.row(i);
    };
    this->parForRange(expandArrayRow, m_data.rows());
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




void TransArray::fillArray(float_t val)
{
    auto fillArrayRow = [&] (uint_t i)
    {
        m_data.row(i).array() =  val;
    };
    this->parForRange(fillArrayRow, m_data.rows());
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

void TransArray::computePercentiles(Eigen::Ref<MatFloat> out_data,
                                    std::vector<float_t> percentiles)
{

    skmapAssertIfTrue((uint_t) out_data.cols() != percentiles.size(),
                      "scikit-map ERROR 19: out_data second dimension must be same size of percentiles");
    // @FIXME: the clean way would be that also for out_data a ParArray oject is created and that only the corresponding cunk is sent
    auto computePercentilesChunk = [&] (Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
    {
        MatFloat sorted_chunk(chunk);
        float_t max_float = std::numeric_limits<float_t>::max();
        sorted_chunk = sorted_chunk.array().isNaN().select(max_float, sorted_chunk);
        Eigen::VectorXi not_nan_count(chunk.rows());
        for (uint_t i = 0; i < (uint_t) sorted_chunk.rows(); ++i) {
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
                    out_data(row_start+i, k) = nan_v;
                } else
                {
                    float_t percentile_pos = (float_t) (not_nan_count(i) - 1) * (percentiles[k] / 100.0);
                    uint_t percentile_pos_floor = floor(percentile_pos);
                    uint_t percentile_pos_ceil = ceil(percentile_pos);
                    if (percentile_pos_floor == percentile_pos_ceil)
                    {
                        out_data(row_start+i, k) = sorted_chunk(i, percentile_pos_floor);
                    } else
                    {
                        float_t percentile_val_floor = sorted_chunk(i, percentile_pos_floor) * ((float_t) percentile_pos_ceil - percentile_pos);
                        float_t percentile_val_ceil = sorted_chunk(i, percentile_pos_ceil) * (percentile_pos - (float_t) percentile_pos_floor);
                        out_data(row_start+i, k) = percentile_val_floor + percentile_val_ceil;
                    }
                }
            }

        }
    };
    this->parChunk(computePercentilesChunk);

}

}
