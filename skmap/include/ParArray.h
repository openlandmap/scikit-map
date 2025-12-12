#ifndef PARARRAY_H
#define PARARRAY_H

#include "misc.cpp"

namespace skmap {

class ParArray {
protected:
  uint_t m_n_threads;
  uint_t m_n_pix;
  uint_t m_n_feat;
  Eigen::Ref<MatFloat> m_data;

public:
  ParArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);

  void printData();

  // A simple threaded parallel execution of the input function(i) for from 0 to
  // n_max-1
  template <typename F> void parForRange(F f, uint_t n_max) {
    omp_set_num_threads(m_n_threads);
    Eigen::initParallel();
    Eigen::setNbThreads(m_n_threads);
#pragma omp parallel for
    for (uint_t i = 0; i < n_max; ++i) {
      f(i);
    }
  }

  /**
   * @brief process `f_in` parallelly on rows, based on the order in `perm_vec`
   *
   * perm_vec is allowed to be any length less than `rows()`.
   *
   * @warning If an index appears twice, this will cause a data race
   *
   * @param f_in The function to apply to each row
   * @param perm_vec vector with row indices
   */
  template <typename F> void parRowPerm(F f_in, std::vector<uint_t> perm_vec) {
    auto f_out = [&](uint_t i) { f_in(i, m_data.row(perm_vec[i])); };
    this->parForRange(f_out, perm_vec.size());
  }

  /**
   * @brief Process row chunks in parallel by `f_in`
   *
   * Will divide the matrix into approximately equal blocks, where the remainder is shared over the
   * first number of chunks
   *
   * @param f_in The function to apply to each chunk
   */
  template <typename F> void parChunk(F f_in) {
    // rows per chunk: rows/threads
    uint_t a = std::floor((float_t)m_data.rows() / (float_t)m_n_threads);
    // "rows" from exact chunks
    uint_t b = (uint_t)((float_t)a * (float_t)m_n_threads);
    // remainder
    uint_t c = (uint_t)((float_t)m_data.rows() - (float_t)b);

    // divide the matrix into row chunks
    auto f_out = [&](uint_t i) {
      uint_t row_start = i * (a + 1);
      uint_t chunk_size = 0;
      if (i >= c) {
        // last rows-c chunks will be `a` long
        chunk_size = a;
        row_start = (uint_t)((float_t)row_start - (float_t)i + (float_t)c);
      } else // first c chunks will be 1 larger to fill the remainder
        chunk_size = a + 1;
      uint_t row_end = row_start + chunk_size;
      if (chunk_size > 0) // not sure if this ever happens?
        f_in(m_data.block(row_start, 0, row_end - row_start, m_data.cols()),
             row_start, row_end);
      else
        std::cout << "zero-length chunk?" << std::endl;
    };
    this->parForRange(f_out, m_n_threads);
  }
};

} // namespace skmap

#endif