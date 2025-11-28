#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "transform/TransArray.h"

TEST(TransArrayTest, ReorderArraySimple)
{
    using MatFloat = Eigen::MatrixXf;

    // Original data: 4 rows, 3 cols
    MatFloat input(4, 3);
    input <<
        1,2,3,
        4,5,6,
        7,8,9,
        10,11,12;

    // Create test object and assign data
    TransArray ta;
    ta.m_data = input;    // If m_data is private: add a setter or constructor for testing

    // Indices_matrix: each row tells which rows to copy
    // Example:
    // New row 0 contains original rows [2, 0]
    // New row 1 contains original rows [3, 1]
    std::vector<std::vector<uint_t>> indices = {
        {2, 0},
        {3, 1}
    };

    // Output should have:
    // rows = indices_matrix.size() = 2
    // cols = (#indices per row) * original_cols = 2 * 3 = 6
    MatFloat out(2, 6);

    // Call function
    ta.reorderArray(out, indices);

    // Expected:
    // Row 0 = [7 8 9 | 1 2 3]
    // Row 1 = [10 11 12 | 4 5 6]
    MatFloat expected(2, 6);
    expected <<
        7,8,9, 1,2,3,
        10,11,12, 4,5,6;

    EXPECT_TRUE(out.isApprox(expected));
}

TEST(TransArrayTest, ThrowsOnInvalidSize)
{
    using MatFloat = Eigen::MatrixXf;

    MatFloat input(3, 2);
    input <<
        1,2,
        3,4,
        5,6;

    TransArray ta;
    ta.m_data = input;

    // indices specify 1*2 rows = OK, but out_data has wrong size
    std::vector<std::vector<uint_t>> indices = {
        {0, 1}
    };

    MatFloat out(1, 3); // Wrong: required cols = 2 * 2 = 4 → should throw/assert

    // Expect your skmapAssertIfTrue to throw std::runtime_error (adjust if needed)
    EXPECT_ANY_THROW(ta.reorderArray(out, indices));
}
