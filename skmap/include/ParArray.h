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
    // Pass a raw pointer + element count instead of an Eigen row expression.
    // m_data is an Eigen::Ref<MatFloat>, so m_data.row(i) is a
    // Block<Ref<MatFloat>,1,-1> — a different type from MatFloat::RowXpr.
    // Binding that to Eigen::Ref<MatFloat::RowXpr> makes Eigen silently
    // evaluate a temporary copy, so writes through `row` would be discarded
    // (the segfault class fixed in readDataCore). Raw pointers guarantee the
    // callback always operates on m_data's backing buffer.
    auto f_out = [&](uint_t i) {
      auto row = m_data.row(perm_vec[i]);
      f_in(i, row.data(), static_cast<uint_t>(row.size()));
    };
    this->parForRange(f_out, perm_vec.size());
  }

  /**
   * @brief Process row chunks in parallel by `f_in`
   *
   * Will divide the matrix into approximately equal blocks, where the remainder
   * is shared over the first number of chunks
   *
   * @param f_in The function to apply to each chunk
   */
  template <typename F> void parChunk(F f_in) {
    // Bug fix: use pure integer arithmetic so float32 rounding errors on large
    // matrices (>~16M rows) never produce wrong chunk boundaries, which could
    // make two threads cover the same row (data race) or access past the end.
    const uint_t n_rows = static_cast<uint_t>(m_data.rows());
    const uint_t a = n_rows / m_n_threads; // base rows per chunk
    const uint_t c = n_rows % m_n_threads; // first `c` threads get one extra row

    auto f_out = [&](uint_t i) {
      const uint_t chunk_size = (i < c) ? (a + 1) : a;
      const uint_t row_start  = (i < c) ? i * (a + 1)
                                         : c * (a + 1) + (i - c) * a;
      const uint_t row_end    = row_start + chunk_size;
      if (chunk_size > 0)
        f_in(m_data.block(row_start, 0, chunk_size, m_data.cols()),
             row_start, row_end);
    };
    this->parForRange(f_out, m_n_threads);
  }
};

} // namespace skmap

#endif