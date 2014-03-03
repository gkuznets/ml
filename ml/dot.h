#pragma once

#include <ml/bit_vec.h>

#include <numeric>
#include <vector>

namespace ml {

template <typename RowVectorX, typename RowVectorY>
double dot(const RowVectorX& x, const RowVectorY& y) {
    return x * y.transpose();
}

template <typename T>
T dot(const std::vector<T>& a, const std::vector<T>& b) {
    const T zero = 0;
    return std::inner_product(
            std::begin(a), std::end(a),
            std::begin(b), zero);
}

template <unsigned N>
unsigned dot(const BitVec<N>& a, const BitVec<N>& b) {
    return a * b;
}

} // namespace ml

