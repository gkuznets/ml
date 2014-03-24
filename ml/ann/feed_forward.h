#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <tuple>

#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

template <typename ColVector
         ,typename Connections
         ,typename Layers>
Eigen::VectorXd feedForward(
        const ColVector& input,
        const Connections& connections,
        Layers) {
    Eigen::VectorXd result = input;
    meta::tup_each(
            [&result](const auto& conn, auto layer) {
                result = conn.transform(result);
                for (unsigned i = 0; i < result.size(); ++i) {
                    result[i] = layer.activation(result[i]);
                }
            },
            connections,
            meta::apply<std::tuple, meta::tail<Layers>>());
    return result;
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
    std::get<0>(activations) = input;
    meta::tup_each(
            [](const auto& conn, const auto& in, auto& out, auto layer) {
                out = conn.transform(in);
                for (unsigned j = 0; j < out.size(); ++j) {
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

