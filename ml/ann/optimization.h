#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <cmath> // see todo
#include <cstdint>
#include <limits>
#include <tuple>

namespace ml {
namespace ann {
namespace detail {

template <typename Dataset, typename Theta, typename NetConf>
void batchGradDescend(
        const Dataset& dataset,
        const double regParam,
        Theta& theta,
        NetConf) {
    const uint64_t m = size(dataset);
    const double learningRate = 0.05 / static_cast<double>(m);

    Theta accumulatedDeltas;
    meta::apply<std::tuple, typename NetConf::Activations> activations;
    meta::apply<std::tuple, typename NetConf::Deltas> delta;

    double oldCost = std::numeric_limits<double>::max();
    double newCost = 0.0;
    while (std::abs(oldCost - newCost) > 0.000001 * m) {
        oldCost = newCost;
        newCost = 0.0;

        meta::tup_each([] (auto& a) { last(a) = 1.0; }, activations);
        meta::tup_each([] (auto& x) { x.zero(); }, accumulatedDeltas);

        for (uint64_t i = 0; i < m; ++i) {
            feedForward(
                    dataset.examples.col(i), theta, activations,
                    typename NetConf::Layers{});

            // Backpropagation
            const auto& yi = dataset.labels(i);
            meta::tup_last(delta)(0) = meta::tup_last(activations)(0) - yi;
            meta::tup_each(
                    [](auto& delta, const auto& prevDelta,
                       const auto& theta, const auto& activations) {

                        const size_t size = delta.size();
                        delta = theta.backTransform(prevDelta).head(size);
                        delta.array() *=
                            (1 - activations.head(size).array().square());
                    },
                    meta::tup_tail(meta::tup_reverse(delta)),
                    meta::tup_reverse(delta),
                    meta::tup_reverse(theta),
                    meta::tup_tail(meta::tup_reverse(activations)));
            meta::tup_each(
                    [](auto& acc, const auto& delta, const auto& activations) {
                        acc.add(delta * activations.transpose());
                    },
                    accumulatedDeltas, delta, activations);
            double hxi = meta::tup_last(activations)(0);
            // TODO: add LossFunction to NetworkConf
            newCost -= ((1.0 + yi) * log(1.0 + hxi) + (1.0 - yi) * log(1.0 - hxi));
        }
        meta::tup_each(
                [regParam, learningRate] (auto& theta, const auto& acc) {
                    theta.update(regParam, learningRate, acc);
                },
                theta, accumulatedDeltas);
        // Regularization term
        meta::tup_each(
                [&newCost, regParam](const auto& theta) {
                    newCost += regParam * theta.squaredNorm();
                },
                theta);
    }
}

} // namespace detail
} // namespace ann
} // namespace ml

