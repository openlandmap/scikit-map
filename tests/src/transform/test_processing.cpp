#include "../common.h"
#include "misc.cpp"
#include "transform/TransArray.h"

#include <cmath>
#include <gtest/gtest.h>

TEST_F(TransArrayTest, nanMean) {
  // clang-format off
  // MatFloat nan(3,2);
  // nan <<
  //   NAN, 1.0, 2.0,
  //   2.0, 3.0, NAN;
  VecFloat out(3);
  VecFloat expected(3);
  expected <<
    1,1,1;
  //clang-format on
  TransArray ta(nanny, THREADS);
  ta.nanMean(out);
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, elementwiseAverage) {
  // clang-format off
  MatFloat in1(3,4);
  in1 <<
    1, 2, 3, 4,
    5, 6, 7, 8,
    9,10,11,12;
  MatFloat in2 (3,4);
  in2 <<
   12,11,10, 9,
    8, 7, 6, 5,
    4, 3, 2, 1;
  MatFloat expected(3,4);
  expected <<
    6.5,6.5,6.5,6.5,
    6.5,6.5,6.5,6.5,
    6.5,6.5,6.5,6.5;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.elementwiseAverage(input, input);
  // EXPECT_EQ(input, expected);
}

// TEST_F(TransArrayTest, blocksAverage) {
//   // clang-format off
//   MatFloat in1(3,3);
//   in1 <<
//     0,1,2,
//     1,2,3,
//     2,3,4;
//   MatFloat in2(3,3);
//   in2 <<
//     0,1,2,
//     3,4,5,
//     6,7,8;
//   MatFloat expected(3,3);
//   expected <<
//     1,2,3,4,
//     5,6,7,8,
//     9;
//   //clang-format on
//   TransArray ta(nanny, THREADS);
//   ta.blocksAverage(in1, in2, 3, 0);
//   EXPECT_EQ(nanny, expected);
// }

TEST_F(TransArrayTest, DISABLED_blocksAverage) {
  // clang-format off
  MatFloat out(3,1);
  MatFloat in1(3,2); // 2 columns only
  in1 <<
    0,1,
    1,2,
    2,3;
  MatFloat in2(3,2);
  in2 <<
    0,1,
    3,4,
    6,7;
  MatFloat expected(3,1); // output has 1 column = n_pix
  expected <<
    0.25*(0+1+0+1),  // row 0
    0.25*(1+2+3+4),  // row 1
    0.25*(2+3+6+7);  // row 2
  // clang-format on

  TransArray ta(out, THREADS);
  ta.blocksAverage(in1, in2, 1, 0); // n_pix=1, y=0
  // EXPECT_EQ(nanny, expected);
}

TEST_F(TransArrayTest, DISABLED_blocksAverageVecs) {
  // clang-format off
  MatFloat in1(2,4); // 1 row, 4 columns
  in1 <<
    0, 1, 2, 3,
    4, 5, 6, 7;

  MatFloat in2(2,4); // 1 row, 4 columns
  in2 <<
    4, 5, 6, 7,
    8, 9,10,11;

  // Output row in m_data will hold n_pix = 2 averaged values
  MatFloat expected(2,2);
  expected <<
    3, 4,
    1, 8;
  // clang-format on

  // Prepare TransArray with a single row in m_data
  MatFloat out(2, 2);
  TransArray ta(out, THREADS);      // 1 thread
  ta.blocksAverage(in1, in2, 2, 1); // n_pix=2, y=0, row_offset=0

  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, computePercentiles) {
  // this works on a transposed array, so we have 3 pixels in input
  // clang-format off
  MatFloat out(3,1);
  MatFloat expected(3,1);
  expected <<
    2.5,
    6.5,
    10.5;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.computePercentiles(
    {0,1,2,3},
    out,
    {0},
    {50}
  );
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, convolveRows) {
  // clang-format off
  MatFloat in(2, 5);
  in <<
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10;
  // clang-format on
  MatFloat out(2, 5);
  float_t w_0 = 0.5;
  VecFloat w_p(2);
  w_p << 0.2, 0.1; // past weights (look forward)
  VecFloat w_f(1);
  w_f << 0.3; // future weights (look back)

  TransArray ta(in, 1);
  ta.convolveRows(out, w_0, w_p, w_f);

  // Reference: the old dense circulant matrix W.
  uint_t n_s = 5;
  uint_t n_e = n_s + std::max(w_p.size(), w_f.size());
  VecFloat w_e = VecFloat::Zero(n_e);
  w_e(0) = w_0;
  w_e.segment(1, w_f.size()) = w_f;
  w_e.segment(n_e - w_p.size(), w_p.size()) = w_p;
  MatFloat W(n_s, n_s);
  for (uint_t i = 0; i < n_s; ++i)
    for (uint_t j = 0; j < n_s; ++j)
      W(j, i) = w_e((-i + j + n_e) % n_e);
  MatFloat expected = in * W;

  EXPECT_TRUE(out.isApprox(expected, 1e-5));
}

TEST_F(TransArrayTest, computeMannKendallPValues) {
  // clang-format off
  MatFloat in(2, 4);
  in <<
    1, 2, 3, 4, // increasing -> S = -6
    4, 3, 2, 1; // decreasing -> S = +6
  // clang-format on
  VecFloat out(2);
  TransArray ta(in, 1);
  ta.computeMannKendallPValues(out);

  // n=4 -> var_s = 4*3*13/18 = 8.6667, sqrt = 2.9439
  // z = (|S|-1)/sqrt(var_s) = 5/2.9439 = 1.6984
  // p = 2*(1 - Phi(1.6984)) = 0.0897
  EXPECT_NEAR(out(0), 0.0897, 1e-3);
  EXPECT_NEAR(out(1), 0.0897, 1e-3);
}
