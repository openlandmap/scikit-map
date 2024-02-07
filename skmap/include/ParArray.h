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

        ParArray(Eigen::Ref<MatFloat> data, const uint_t n_threads);
    
        void printData();

        // A simple threaded parallel execution of the input function(i) for from 0 to n_max-1
        template<typename F>
        void parForRange(F f, uint_t n_max)
        {
            omp_set_num_threads(m_n_threads);
            Eigen::initParallel();
            Eigen::setNbThreads(m_n_threads);
            #pragma omp parallel for
            for (size_t i = 0; i < n_max; ++i)
            {
                f(i);
            }
        }

        
        template<typename F>
        void parRowPerm(F f_in, std::vector<uint_t> perm_vec)
        {
            auto f_out = [&] (uint_t i)
            {
                f_in(i, m_data.row(perm_vec[i]));
            };
            this->parForRange(f_out, perm_vec.size());
        }        
};

}
 
#endif