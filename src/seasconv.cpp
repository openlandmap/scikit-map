#include <iostream>
#include <Eigen/Dense>
#include "WrapFFT.hpp"

template <class T>
struct TypesEigen {
    using NumpyMatReal = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using NumpyMatComplex = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorReal = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VectorComplex = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>;
};


template <class T>
void compute_conv_vec(typename TypesEigen<T>::VectorReal& conv,
    const unsigned int N_years,
    const unsigned int N_ipy,
    const float att_seas,
    const float att_env) {

    const unsigned int N_samp = N_years * N_ipy;
    typename TypesEigen<T>::VectorReal base_func = TypesEigen<T>::VectorReal::Zero(N_ipy);
    typename TypesEigen<T>::VectorReal env_func = TypesEigen<T>::VectorReal::Zero(N_samp);
    const float period_y = (float) N_ipy / 2.;
    const float slope_y = att_seas/10./period_y;

    for (unsigned int i = 0; i < N_ipy; ++i) {
        if (i <= period_y) 
            base_func(i) = -slope_y*(float)i;
        else
            base_func(i) = slope_y*((float)i-period_y) - att_seas/10.;
    }

    const float delta_e = N_samp;
    const float slope_e = att_env/10./delta_e;
    for (unsigned int i = 0; i < N_samp; ++i) {
        env_func(i) = -slope_e*(float)i;
    }   

    for (unsigned int i = 0; i < N_samp; ++i) {
        conv(i) = std::pow(10., base_func[i%N_ipy] + env_func[i]);
    }
}
    
    
template <class T>
int run(const unsigned int N_years,
    const unsigned int N_ipy,
    const unsigned int N_pix,
    const float att_seas,
    const float att_env,
    T* ts_in,
    T* qa_out) {    
    
    using NumpyMatReal = typename TypesEigen<T>::NumpyMatReal;
    using NumpyMatComplex = typename TypesEigen<T>::NumpyMatComplex;
    using VectorReal = typename TypesEigen<T>::VectorReal;
    using VectorComplex = typename TypesEigen<T>::VectorComplex;

    const unsigned int N_samp = N_years * N_ipy;
    const unsigned int N_ext = N_samp * 2;
    const unsigned int N_fft = N_samp + 1;
 
    // Link the input/output data to Eigen matrices
    Eigen::Map<NumpyMatReal> ts_ext(ts_in, N_pix, N_ext);
    Eigen::Map<NumpyMatReal> mask_ext(qa_out, N_pix, N_ext);

    // Create needed variables
    NumpyMatComplex ts_ext_fft(N_pix, N_fft);
    NumpyMatComplex mask_ext_fft(N_pix, N_fft);
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mask = ts_ext.array().isNaN(); // Validity mask
    VectorReal conv(N_samp);
    VectorReal conv_ext = VectorReal::Ones(N_ext);
    VectorComplex conv_ext_fft(N_fft);
    VectorReal norm_qa = VectorReal::Ones(N_ext);
    VectorComplex norm_qa_fft(N_fft);

    // Plan the FFT computation
    WrapFFT<T> wrapFFT = WrapFFT<T>(N_fft, N_ext, N_pix, ts_ext.data(), mask_ext.data(), ts_ext_fft.data(), mask_ext_fft.data());

    // Preparing the inputs for the comnputation
    ts_ext = mask.select(0.0, ts_ext);
    mask = !mask.array(); // Switch from a NaN mask to a validity mask for the next steps
    mask_ext.block(0, 0, N_pix, N_samp) = mask.block(0, 0, N_pix, N_samp).cast<T>();
    compute_conv_vec<T>(conv, N_years, N_ipy, att_seas, att_env);
    conv_ext.segment(0,N_samp) = conv;
    conv_ext.segment(N_samp+1,N_samp-1) = conv.reverse().segment(0,N_samp-1);
    norm_qa.segment(N_samp,N_samp) = VectorReal::Zero(N_samp);

    // Compute forward transformations
    wrapFFT.computeTimeserisFFT();
    wrapFFT.computeMaskFFT();
    wrapFFT.computeFFT(conv_ext, conv_ext_fft);
    wrapFFT.computeFFT(norm_qa, norm_qa_fft);

    // Convolve the vectors
    ts_ext_fft.array().rowwise() *= conv_ext_fft.array().transpose();
    mask_ext_fft.array().rowwise() *= conv_ext_fft.array().transpose();
    norm_qa_fft.array() *= conv_ext_fft.array();

    // Compute forward transformations    
    wrapFFT.computeTimeserisIFFT();
    wrapFFT.computeMaskIFFT();
    wrapFFT.computeIFFT(norm_qa_fft, norm_qa);
    wrapFFT.clean();

    // Renormalize the result
    ts_ext.block(0, 0, N_pix, N_samp).array() /= mask_ext.block(0, 0, N_pix, N_samp).array();
    mask_ext.block(0, 0, N_pix, N_samp) = (mask_ext.block(0, 0, N_pix, N_samp).array().rowwise() / norm_qa.segment(0,N_samp).array().transpose()) / N_ext * 100.;

    return 0;
}

extern "C"
int runDouble(const unsigned int N_years,
    const unsigned int N_ipy,
    const unsigned int N_pix,
    const float att_seas,
    const float att_env,
    double* ts_in,
    double* qa_out) {
    return run<double>(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, qa_out);
}
 

extern "C"
int runFloat(const unsigned int N_years,
    const unsigned int N_ipy,
    const unsigned int N_pix,
    const float att_seas,
    const float att_env,
    float* ts_in,
    float* qa_out) {
    return run<float>(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, qa_out);
}
 