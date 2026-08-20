#include "ParArray.h"

namespace skmap {

ParArray::ParArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
    : m_n_threads(std::min(n_threads, (uint_t)data.rows())), m_data(data) {
  // Set OpenMP/Eigen thread counts once per binding call instead of on every
  // parForRange/parChunk invocation (the old code re-initialised Eigen's
  // thread pool on every call, which is measurable overhead for the many
  // small calls the Python layer chains together).
  omp_set_num_threads(m_n_threads);
  Eigen::initParallel();
  Eigen::setNbThreads(m_n_threads);
}

void ParArray::printData() { std::cout << m_data << std::endl; }

} // namespace skmap