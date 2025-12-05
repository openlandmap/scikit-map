#include "ParArray.h"
#include "common.h"
#include "misc.cpp"
#include <gtest/gtest.h>

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

  /// Most boringest function ever; fills cols with index
  void testParForRangeCols() {
    auto f = [&](uint_t i) { m_data.col(i).fill(i); };
    this->parForRange(f, m_data.cols());
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

  pa.testParForRangeCols();
  EXPECT_EQ(input, expect_cols);
}

TEST_F(TransArrayTest, parRowPerm) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    2.0,4.0,6.0,8.0, // 2 in perm_vec => *2
    0.0,0.0,0.0,0.0, // 0 in perm_vec => *0
    9.0,10.,11.,12.; // 1 in perm_vec => *1
  //clang-format on

  auto f = [&](uint_t i, Eigen::Ref<MatFloat::RowXpr> row) {
      row = row*i; // multiply by perm_vec index
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

  auto f = [&](uint_t i, Eigen::Ref<MatFloat::RowXpr> row) {
      row = row*i; // multiply by perm_vec index
  };

  ParArray pa(input, THREADS);
  //                                0   1
  pa.parRowPerm(f, {1,2});
  EXPECT_EQ(input, expected);
}
