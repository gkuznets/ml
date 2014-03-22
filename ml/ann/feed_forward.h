#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <tuple>

#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

namespace {

template <typename Vec>
double& last(Vec& vec) {
    return vec(vec.size() - 1);
}

} // namespace

template <typename ColVector
         ,typename Connections
         ,typename Layers>
double feedForward(
        const ColVector& input,
        const Connections& connections,
        Layers) {
    Eigen::VectorXd result(input.rows() + 1);
    last(result) = 1.0;
    result.head(input.rows()) = input;
    meta::tup_each(
            [&result](const auto& conn, auto layer) {
                result = conn.transform(result);
                result.resize(result.size() + 1);
                last(result) = 1.0;
                for (unsigned i = 0; i < result.size() - 1; ++i) {
                    result[i] = layer.activation(result[i]);
                }
            },
            connections,
            meta::apply<std::tuple, meta::tail<Layers>>());
    return result[0];
}

template <typename ColVector
         ,typename Connections
         ,typename Activations
         ,typename Layers>
void feedForward(
        const ColVector& input,
        const Connections& connections,
        Activations& activations,
        Layers) {
    std::get<0>(activations).head(input.size()) = input;
    meta::tup_each(
            [](const auto& conn, const auto& in, auto& out, auto layer) {
                out.head(out.size() - 1) = conn.transform(in);
                for (unsigned j = 0; j < out.size() - 1; ++j) {
                    out(j) = layer.activation(out(j));
                }
            },
            connections,
            activations,
            meta::tup_tail(activations),
            meta::apply<std::tuple, meta::tail<Layers>>());
}

} // namespace detail
} // namespace ann
} // namespace ml

