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
