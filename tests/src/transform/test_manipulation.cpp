#include "../common.h"
#include "transform/TransArray.h"
#include <gtest/gtest.h>

TEST_F(TransArrayTest, fillArray) {
  // clang-format off
  MatFloat expected(3,3);
  expected <<
    5.0,5.0,5.0,
    5.0,5.0,5.0,
    5.0,5.0,5.0;
  //clang-format on
  TransArray ta(nanny, THREADS);
  ta.fillArray(5.0);
  EXPECT_EQ(nanny, expected);
}

TEST_F(TransArrayTest, maskNan) {
  // clang-format off
  MatFloat expected(3,3);
  expected <<
    2.0,1.0,2.0, // 0->2.0
    1.0,3.0,1.0, // 1->3.0
    2.0,1.0,2.0; // 2->2.0
  //clang-format on
  TransArray ta(nanny, THREADS);
  ta.maskNan({0,2},2.0);
  ta.maskNan({1}, 3.0);
  EXPECT_EQ(nanny, expected);
}

TEST_F(TransArrayTest, maskNanRows) {
  // clang-format off
  MatFloat expected(3,3);
  expected <<
    2.0,1.0,2.0,
    1.0,3.0,1.0,
    4.0,1.0,4.0;
  //clang-format on
  VecFloat new_value_vec{{2.0,3.0,4.0}};
  TransArray ta(nanny, THREADS);
  ta.maskNanRows({0,1,2},new_value_vec);
  EXPECT_EQ(nanny, expected);
}

TEST_F(TransArrayTest, maskData) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,14.,4.0,
    5.0,6.0,7.0,8.0,
    14.,10.,11.,14.;
  MatFloat mask(2,4);
  mask <<
    0.0,0.0,1.0,0.0,
    // 0.0,1.0,0.0,0.0,
    1.0,0.0,0.0,1.0;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.maskData({0,2}, mask, 1.0, 14.);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, maskDataRows) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,14.,4.0,
    5.0,6.0,7.0,8.0,
    9.0,10.,14.,12.;
  MatFloat mask(1,4);
  mask <<
    0.0,0.0,1.0,0.0;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.maskDataRows({0,2}, mask, 1.0, 14.);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, maskDataBoundsCheck) {
  // Regression: maskData used to access masks.row(i) for i up to
  // row_select.size() without checking masks.rows(), causing an out-of-bounds
  // Eigen access / SIGSEGV when masks has fewer rows than row_select.
  MatFloat mask(1, 4); // only 1 row, but row_select has 2 entries
  mask << 0.0, 0.0, 1.0, 0.0;
  TransArray ta(input, THREADS);
  EXPECT_THROW(ta.maskData({0, 2}, mask, 1.0, 14.), std::runtime_error);
}

TEST_F(TransArrayTest, maskDataRowSelectBoundsCheck) {
  // Regression: row_select indices must be validated against m_data.rows().
  MatFloat mask(2, 4);
  mask.setZero();
  TransArray ta(input, THREADS);
  EXPECT_THROW(ta.maskData({0, 99}, mask, 1.0, 14.), std::runtime_error);
}

TEST_F(TransArrayTest, swapRowsValues) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    5.0,2.0,3.0,4.0,
    5.0,6.0,7.0,8.0,
    9.0,10.,11.,12.;
  // clang-format on
  MatFloat same = input;
  TransArray ta(input, THREADS);
  ta.swapRowsValues({1, 2}, 1.0, 5.0);
  EXPECT_EQ(input, same);
  ta.swapRowsValues({0}, 1.0, 5.0);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, hadamardProduct) {
  // clang-format off
  MatFloat in1(3,3);
  in1 <<
    1.0,2.0,3.0,
    4.0,5.0,6.0,
    7.0,8.0,9.0;
  MatFloat in2(3,3);
  in2 <<
    0.0,1.0,2.0,
    1.0,2.0,3.0,
    2.0,3.0,4.0;
  MatFloat expected(3,3);
  expected <<
    0.0,2.0,6.0,
    4.0,10.,18.,
    14.,24.,36.;
  //clang-format on
  TransArray ta(nanny, THREADS);
  ta.hadamardProduct(in1, in2);
  EXPECT_EQ(nanny, expected);
}

TEST_F(TransArrayTest, offsetAndScale) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    0.0,2.0,4.0,6.0,
    8.0,10.,12.,14.,
    16.,18.,20.,22.;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.offsetAndScale(-1.0, 2.0);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, scaleAndOffset) {
  // clang-format off
  MatFloat expected(3,4);
  expected <<
    1.0,3.0,5.0,7.0,
    9.0,11.,13.,15.,
    17.,19.,21.,23.;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.scaleAndOffset(-1.0, 2.0);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, offsetsAndScalesUnstable) {
  // clang-format off
  VecFloat offsets(3);
  offsets <<
    -1.,-2.,-3.;
  VecFloat scalings(3);
  scalings <<
    2.0,3.0,4.0;
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,3.0,4.0, // skipped
    9.0,12.,15.,18., // (v-2)*3
    24.,28.,32.,36.; // (v-3)*4
  //clang-format on
  TransArray ta(input, THREADS);

  ta.offsetsAndScales({2,1},offsets,scalings);
  EXPECT_EQ(input, expected);
}

TEST_F(TransArrayTest, DISABLED_offsetsAndScalesNew) {
  // clang-format off
  VecFloat offsets(3);
  offsets <<
    -1.,-2.;
  VecFloat scalings(3);
  scalings <<
    2.0,3.0;
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,3.0,4.0, // skipped
    9.0,12.,15.,18., // (v-2)*3
    21.,24.,27.,30.; // (v-1)*2
  //clang-format on
  TransArray ta(input, THREADS);

  ta.offsetsAndScales({2,1},offsets,scalings);
  EXPECT_EQ(input, expected);
}
