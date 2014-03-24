#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <cmath>
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
    while (std::abs(oldCost - newCost) > 0.00001 * m) {
        oldCost = newCost;
        newCost = 0.0;

        meta::tup_each([] (auto& x) { x.zero(); }, accumulatedDeltas);

        for (uint64_t i = 0; i < m; ++i) {
            feedForward(
                    dataset.examples.col(i), theta, activations,
                    typename NetConf::Layers{});

            // Backpropagation
            const auto& label = dataset.labels.col(i);
            meta::tup_last(delta) = meta::tup_last(activations) - label;
            meta::tup_each(
                    [](auto& delta, const auto& prevDelta,
                       const auto& theta, const auto& activations) {

                        delta = theta.backTransform(prevDelta);
                        delta.array() *= (1 - activations.array().square());
                    },
                    meta::tup_tail(meta::tup_reverse(delta)),
                    meta::tup_reverse(delta),
                    meta::tup_reverse(theta),
                    meta::tup_tail(meta::tup_reverse(activations)));
            meta::tup_each(
                    [](auto& acc, const auto& delta, const auto& activations) {
                        acc.propagate(delta, activations);
                    },
                    accumulatedDeltas, delta, activations);
            const auto& prediction = meta::tup_last(activations);
            newCost += NetConf::lossFn(prediction, label);
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

