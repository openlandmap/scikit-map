#ifndef PARARRAY_H
#define PARARRAY_H

#include "misc.cpp"

namespace skmap {

class ParArray {
     protected :
          uint_t m_n_threads;
          uint_t m_n_pix;
          uint_t m_n_feat;
          Eigen::Ref<MatFloat> m_data;

     public :

          ParArray(Eigen::Ref<MatFloat> data, const uint_t n_feat, const uint_t n_pix, const uint_t n_threads);
    
          void printData();

          template<typename F>
          void parFeat(F f)
          {
              omp_set_num_threads(m_n_threads);
              Eigen::initParallel();
              Eigen::setNbThreads(m_n_threads);
              #pragma omp parallel for
              for (size_t i = 0; i < m_n_feat; ++i)
              {
                  f(i);
              }
          }

          
};

}
 
#endif