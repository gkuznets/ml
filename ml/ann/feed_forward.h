#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <tuple>

#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

template <typename Input
         ,typename Connections
         ,typename Layers>
Eigen::MatrixXd feedForward(
        const Input& input,
        const Connections& connections,
        Layers) {
    Eigen::MatrixXd result = input;
    meta::tup_each(
            [&result](const auto& conn, auto layer) {
                typename decltype(layer)::Activation act;
                result = act(conn.transform(result));
            },
            connections,
            meta::apply<std::tuple, meta::tail<Layers>>());
    return result;
}

template <typename Input
         ,typename Connections
         ,typename Activations
         ,typename Layers>
void feedForward(
        const Input& input,
        const Connections& connections,
        Activations& activations,
        Layers) {
    std::get<0>(activations) = input;
    meta::tup_each(
            [](const auto& conn, const auto& in, auto& out, auto layer) {
                typename decltype(layer)::Activation act;
                out = act(conn.transform(in));
            },
            connections,
            activations,
            meta::tup_tail(activations),
            meta::apply<std::tuple, meta::tail<Layers>>());
}

} // namespace detail
} // namespace ann
} // namespace ml

