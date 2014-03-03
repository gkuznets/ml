#pragma once

#include <ml/bit_vec.h>
#include <ml/dot.h>

#include <cmath>

namespace ml {

struct RBFKernel {
    RBFKernel() : sigma2_(1.0) {}
    explicit RBFKernel(double sigma2) : sigma2_(2.0 * sigma2) {}

    template <typename RowVectorX, typename RowVectorY>
    double operator() (const RowVectorX& x, const RowVectorY& y) const {
        auto delta = x - y;
        return std::exp(-dot(delta, delta) / sigma2_);
    }

    template <unsigned size>
    double operator() (const BitVec<size>& x, const BitVec<size>& y) const {
        return std::exp(-static_cast<double>(distance(x, y)) / sigma2_);
    }

private:
    const double sigma2_;
};

namespace {

template <typename T>
inline constexpr T pow(T base, unsigned exp) {
    return (exp == 0) ? 1 :
           (exp % 2 == 0) ? pow(base, exp / 2) * pow(base, exp / 2) :
                            base * pow(base, exp - 1);
}

} // namespace

template <unsigned N>
struct PolynomialKernel {
    template <typename RowVectorX, typename RowVectorY>
    double operator() (const RowVectorX& x, const RowVectorY& y) const {
        double base = 1.0 + static_cast<double>(dot(x, y));
        return pow(base, N);
    }
};

} // namespace ml

