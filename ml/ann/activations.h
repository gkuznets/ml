#pragma once

#include <cmath>

namespace ml {
namespace ann {

/*static const double TANH_COEFF = 1.0;

// TODO: add vectorized implementation
static inline double sigmoid(double x) {
    //return 1.0 / (1.0 + std::exp(-x));
    //return TANH_COEFF * std::tanh(0.666 * x);
    return TANH_COEFF * std::tanh(x);
}*/


struct Tanh {
    double operator()(double x) const {
        return std::tanh(x);
    }
};

} // namespace ann
} // namespace ml

