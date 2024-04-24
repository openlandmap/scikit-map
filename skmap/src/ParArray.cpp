#include "ParArray.h"

namespace skmap 
{

ParArray::ParArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
	: m_n_threads(std::min(n_threads, (uint_t) data.rows()))
    , m_data(data)
{
}

void ParArray::printData()
{
    std::cout << m_data << std::endl;
}



}