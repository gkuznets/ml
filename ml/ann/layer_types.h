#pragma once

#include <ml/ann/activations.h>

namespace ml {
namespace ann {

template <unsigned size>
struct Input {};

template <unsigned width, unsigned height>
struct ImageInput {};

template <unsigned windowSize>
struct Convolution {};

template <unsigned windowSize>
struct Subsampling {};

template <unsigned size
         ,typename ActivationFn = Tanh>
struct FullyConnected {
    typedef ActivationFn Activation;
};

template <unsigned size
         ,typename ActivationFn = Tanh>
struct SymmetricFullyConnected {
    typedef ActivationFn Activation;
};

} // namespace ann
} // namespace ml

