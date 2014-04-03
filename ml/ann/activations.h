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

struct Linear {
    template <typename Input>
    auto&& operator() (Input&& input) const {
        return std::forward<Input>(input);
    }
};


struct Tanh {
    double operator()(double x) const {
        return std::tanh(x);
    }

    template <typename Input>
    auto operator() (Input input) const {
        for (unsigned row = 0; row < input.rows(); ++row) {
            for (unsigned col = 0; col < input.cols(); ++col) {
                input(row, col) = tanh(input(row, col));
            }
        }
        return input;
    }
};

} // namespace ann
} // namespace ml

