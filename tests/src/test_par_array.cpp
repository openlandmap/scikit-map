#include "ParArray.h"
#include "common.h"
#include "misc.cpp"
#include <gtest/gtest.h>
#include <omp.h>

// child class so we can add dummy lambda functions for parForRange
class ParArrayHelper : public ParArray {
public:
  using ParArray::m_data;
  ParArrayHelper(Eigen::Ref<MatFloat> data, const uint_t n_threads)
      : ParArray(data, n_threads) {}

  /// Most boringest function ever; fills rows with index
  void testParForRangeRows() {
    auto f = [&](uint_t i) { m_data.row(i).fill(i); };
    this->parForRange(f, m_data.rows());
  }

  /// fill rows with the thread number processing them
  void testParForRangeRowsThreadNum() {
    auto f = [&](uint_t i) {
      int n = omp_get_thread_num();
      m_data.row(i).fill(n);
    };
    this->parForRange(f, m_data.rows());
  }

  /// Most boringest function ever; fills cols with index
  void testParForRangeCols() {
    auto f = [&](uint_t i) { m_data.col(i).fill(i); };
    this->parForRange(f, m_data.cols());
  }

  /// fill cols with the thread number processing them
  void testParForRangeColsThreadNum() {
    auto f = [&](uint_t i) {
      int n = omp_get_thread_num();
      m_data.col(i).fill(n);
    };
    this->parForRange(f, m_data.cols());
  }

  void testParChunk() {
    auto f = [&](Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end) {
      int n = omp_get_thread_num();
      chunk.fill(n);
    };
    this->parChunk(f);
  }
};

TEST_F(TransArrayTest, parForRange) {
  // clang-format off
  MatFloat expect_rows(3,4);
  expect_rows <<
    0.0,0.0,0.0,0.0,
    1.0,1.0,1.0,1.0,
    2.0,2.0,2.0,2.0;
  MatFloat expect_cols(3,4);
  expect_cols <<
    0.0,1.0,2.0,3.0,
    0.0,1.0,2.0,3.0,
    0.0,1.0,2.0,3.0;
  // clang-format on

  ParArrayHelper pa(input, THREADS);

  pa.testParForRangeRows();
  EXPECT_EQ(input, expect_rows);
  pa.testParForRangeRowsThreadNum();
  EXPECT_EQ(input, expect_rows);

  pa.testParForRangeCols();
  EXPECT_EQ(input, expect_cols);
  pa.testParForRangeColsThreadNum();
  // clang-format off
  // for some reason the first two cols share the same thread?
  MatFloat expect_colthreads(3,4);
  expect_colthreads <<
    0,0,1,2,
    0,0,1,2,
    0,0,1,2;
  // clang-format on
  EXPECT_EQ(input, expect_colthreads);
}

TEST_F(TransArrayTest, parRowPerm) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    2.0,4.0,6.0,8.0, // 2 in perm_vec => *2
    0.0,0.0,0.0,0.0, // 0 in perm_vec => *0
    9.0,10.,11.,12.; // 1 in perm_vec => *1
  //clang-format on

  auto f = [&](uint_t i, float_t *row_ptr, uint_t row_n_elems) {
    Eigen::Map<Eigen::Matrix<float_t, 1, Eigen::Dynamic, Eigen::RowMajor>> row(
        row_ptr, row_n_elems);
    row = row * i; // multiply by perm_vec index
  };

  TransArray ta(input, THREADS);
  //                                0   1   2
  ta.parRowPerm(f, {1,2,0});
  EXPECT_EQ(input, expected);
}

// perm_vec does not need to apply to all rows
// this test only works on rows 1 and 2
TEST_F(TransArrayTest, parRowPermIncomplete) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,3.0,4.0, // skipped
    0.0,0.0,0.0,0.0, // 0 in perm_vec => *0
    9.0,10.,11.,12.; // 1 in perm_vec => *1
  //clang-format on

  auto f = [&](uint_t i, float_t *row_ptr, uint_t row_n_elems) {
    Eigen::Map<Eigen::Matrix<float_t, 1, Eigen::Dynamic, Eigen::RowMajor>> row(
        row_ptr, row_n_elems);
    row = row * i; // multiply by perm_vec index
  };

  ParArray pa(input, THREADS);
  //                                0   1
  pa.parRowPerm(f, {1,2});
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, DISABLED_parRowPermDataRace) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,3.0,4.0,
    8.0,9.0,10.,11., // 0,1,2 in perm_vec => DATA RACE
    9.0,10.,11.,12.;
  //clang-format on

  auto f = [&](uint_t i, float_t *row_ptr, uint_t row_n_elems) {
    Eigen::Map<Eigen::Matrix<float_t, 1, Eigen::Dynamic, Eigen::RowMajor>> row(
        row_ptr, row_n_elems);
    row.array() += i; // add perm_vec index
  };

  TransArray ta(input, THREADS);
  //                                0   1   2
  ta.parRowPerm(f, {1,1,1});
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, parChunk) {
  // we use 2 threads, so make 2 vectors that can be written to
  std::vector<uint_t> seen_start(3), seen_end(3);

  ParArray pa(input, 3);
  auto f_chunk = [&](Eigen::Ref<MatFloat> chunk, uint_t row_start, uint_t row_end)
  {
    // write to the index of calling thread
    int tid = omp_get_thread_num();
    seen_start[tid] = row_start;
    seen_end[tid] = row_end;
  };
  pa.parChunk(f_chunk);

  EXPECT_EQ(seen_start[0], 0);
  EXPECT_EQ(seen_end[0],   1);
  EXPECT_EQ(seen_start[1], 1);
  EXPECT_EQ(seen_end[1],   2);
  EXPECT_EQ(seen_start[2], 2);
  EXPECT_EQ(seen_end[2], 3);
}

TEST_F(TransArrayTest, parChunkFill) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    0,0,0,0,
    1,1,1,1,
    2,2,2,2;
  //clang-format on
  ParArrayHelper pah(input, 3);
  pah.testParChunk();
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, parChunkFillRemainder) {
  // clang-format off
  MatFloat input(7,1);
  MatFloat expected(7,1);
  expected <<
    0,
    0,
    1,
    1,
    2,
    3,
    4;
  //clang-format on
  ParArrayHelper pah(input, 5);
  pah.testParChunk();
  EXPECT_EQ(input, expected);
}
