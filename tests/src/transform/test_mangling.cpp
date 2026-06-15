#include "../common.h"
#include "misc.cpp"
#include "transform/TransArray.h"
#include <iostream>
#include <stdexcept>

using namespace skmap;

TEST_F(TransArrayTest, ReorderArray) {
  // clang-format off
  MatFloat out(2,8);
  MatFloat expected(2,8);
  expected <<
    1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0, // 0,1
    9.0,10.,11.,12.,9.0,10.,11.,12.; // 2,2
  // clang-format on

  TransArray ta(input, THREADS);
  ta.reorderArray(out, {{0, 1}, {2, 2}});

  // std::cout << "input\n" << input << std::endl;
  // std::cout << "expected\n" << expected << std::endl;
  // std::cout << "out\n" << out << std::endl;

  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, ReorderArrayThrowsOnInvalidSize) {
  // clang-format off
  // wrong-size matrix
  MatFloat out(2,6);
  // clang-format on
  TransArray ta(input, THREADS);
  EXPECT_THROW(ta.reorderArray(out, {{0, 1}, {2, 2}}), std::runtime_error);
}

TEST_F(TransArrayTest, TransposeArray) {
  // clang-format off
  MatFloat out (4,3);
  MatFloat expected(4,3);
  expected <<
    1.0,5.0,9.0,
    2.0,6.0,10.,
    3.0,7.0,11.,
    4.0,8.0,12.;
  // clang-format on
  TransArray ta(input, THREADS);
  ta.transposeArray(out);
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, TransposeArrayThrowsOnInvalidSize) {
  MatFloat out(5, 3);
  TransArray ta(input, THREADS);
  EXPECT_THROW(ta.transposeArray(out), std::runtime_error);
}

// This seems impossible to do in one step???
TEST_F(TransArrayTest, DISABLED_TransposeReorderArray) {
  // clang-format off
  MatFloat out(8,2);
  MatFloat expected(8,2);
  expected <<
    1.0,9.0,
    2.0,10.,
    3.0,11.,
    4.0,12.,
    5.0,9.0,
    6.0,10.,
    7.0,11.,
    8.0,12.;
  // clang-format on

  // block-testing debug
  // for (int i=0;i<8;i++)
  //   for (int j=0;j<2;j++)
  //     for (int p=8-i;p>0;p--)
  //       for (int q=2-j;q>0;q--)
  //         std::cout<<"block
  //         "<<i<<j<<p<<q<<std::endl<<expected.block(i,j,p,q)<<std::endl;
  TransArray ta(input, THREADS);
  ta.transposeReorderArray(out, {
                                    {0, 0, 0},
                                    {1, 0, 0},
                                    {2, 0, 0},
                                    {3, 0, 0},
                                    {4, 0, 0},
                                    {5, 0, 0},
                                    {6, 0, 0},
                                    {7, 0, 0},
                                });
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, InverseReorderArray) {
  // clang-format off
  MatFloat out(3,4);
  MatFloat trans(2,8);
  MatFloat expected(8,2);
  expected <<
    1.0,9.0,
    2.0,10.,
    3.0,11.,
    4.0,12.,
    5.0,9.0,
    6.0,10.,
    7.0,11.,
    8.0,12.;
  // clang-format on
  // this is an inverse test, so we create input from expected
  // so it doesn't do everything in one step: transpose
  TransArray ta(expected, THREADS);
  ta.transposeArray(trans);
  // reorder
  TransArray ra(trans, THREADS);
  ra.inverseReorderArray(out, {{0, 0}, {0, 1}, {1, 0}});

  EXPECT_EQ(out, input);
}

TEST_F(TransArrayTest, SelArrayRows) {
  // clang-format off
  MatFloat out(2,4);
  MatFloat expected(2,4);
  expected <<
    1.0,2.0,3.0,4.0,
    9.0,10.,11.,12.;
  // clang-format on
  TransArray ta(input, THREADS);
  ta.selArrayRows(out, {0, 2});
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, SelArrayCols) {
  // clang-format off
  MatFloat out(3,2);
  MatFloat expected(3,2);
  expected <<
    2.0,4.0,
    6.0,8.0,
    10.,12.;
  // clang-format on
  TransArray ta(input, THREADS);
  ta.selArrayCols(out, {1, 3});
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, ExpandArrayRows) {
  // clang-format off
  MatFloat out = MatFloat::Zero(5,4);
  MatFloat expected(5,4);
  expected <<
    1.0,2.0,3.0,4.0, // 0
    0.0,0.0,0.0,0.0,
    9.0,10.,11.,12., // 2
    0.0,0.0,0.0,0.0,
    5.0,6.0,7.0,8.0; // 4
  // clang-format on
  TransArray ta(input, THREADS);
  ta.expandArrayRows(out, {0, 4, 2});
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, ExpandArrayCols) {
  // clang-format off
  MatFloat out = MatFloat::Zero(3,7);
  MatFloat expected(3,7);
  expected <<
  // 0       2       1       3
  // 0   1   2   3   4   5   6
    1.0,0.0,3.0,0.0,2.0,0.0,4.0,
    5.0,0.0,7.0,0.0,6.0,0.0,8.0,
    9.0,0.0,11.,0.0,10.,0.0,12.;
  // clang-format on
  TransArray ta(input, THREADS);
  ta.expandArrayCols(out, {0, 4, 2, 6});
  EXPECT_EQ(out, expected);
}

TEST_F(TransArrayTest, copyVecInMatrixRow) {
  // clang-format off
  VecFloat vec(4);
  vec <<
    13.,14.,15.,16.;
  MatFloat expected(3,4);
  expected <<
    1.0,2.0,3.0,4.0,
    13.,14.,15.,16.,
    9.0,10.,11.,12.;
  //clang-format on
  TransArray ta(input, THREADS);
  ta.copyVecInMatrixRow(vec, 1);
  EXPECT_EQ(input, expected);
}
