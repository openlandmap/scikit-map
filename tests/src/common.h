
#include "misc.cpp"
#include "transform/TransArray.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace skmap;

const uint_t THREADS = 42;
constexpr float_t NAN_FLOAT = std::numeric_limits<float_t>::quiet_NaN();
// constexpr float NAN_FLOAT = std::numeric_limits<float>::quiet_NaN();

class TransArrayTest : public ::testing::Test {
protected:
  MatFloat input;
  MatFloat nanny;
  // We use the same 3x4 matrix for all unit tests
  void SetUp() override {
    // clang-format off
    input.resize(3, 4);
    input <<
        1.0,2.0,3.0,4.0,
        5.0,6.0,7.0,8.0,
        9.0,10.,11.,12.;
    nanny.resize(3,3);
    nanny <<
      NAN,1.0,NAN,
      1.0,NAN,1.0,
      NAN,1.0,NAN;
    // clang-format on
  }
};

// We may want to access protected members at some point, this is the way to do
// that
class TransArrayExposed : public TransArray {
public:
  using TransArray::m_data;
  TransArrayExposed(Eigen::Ref<MatFloat> data, const uint_t n_threads)
      : TransArray(data, n_threads) {}
};
