#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace ml {
namespace ann {
namespace detail {

template <typename Labels
         ,typename Activations
         ,typename Delta
         ,typename Connections>
void backProp(
        const Labels& labels,
        const Activations& activations,
        const Connections& connections,
        Delta& delta,
        Connections& accumulator) {
    meta::tup_last(delta) = meta::tup_last(activations) - labels;
    meta::tup_each(
            [](auto& delta, const auto& prevDelta,
               const auto& connections, const auto& activations) {

                delta = connections.backTransform(prevDelta);
                delta.array() *= (1 - activations.array().square());
            },
            meta::tup_tail(meta::tup_reverse(delta)),
            meta::tup_reverse(delta),
            meta::tup_reverse(connections),
            meta::tup_tail(meta::tup_reverse(activations)));
    meta::tup_each(
            [](auto& acc, const auto& delta, const auto& activations) {
                acc.propagate(delta, activations);
            },
            accumulator, delta, activations);
}

template <unsigned batchSize
         ,typename Dataset
         ,typename StopCriterion
         ,typename NetConf>
auto gradDescent(const Dataset& dataset,
                 double regParam, // weight decay parameter
                 StopCriterion stopCiterion) {
    const uint64_t dsSize = size(dataset);
    const double learningRate = 0.05;

    typename NetConf::Connections connections;
    meta::tup_each([] (auto& c) { c.init(); }, connections);
    decltype(connections) accumulatedDeltas;
    typename NetConf::template Activations<batchSize> activations;
    typename NetConf::template Delta<batchSize> delta;
    unsigned epoch = 0;
    while (!stopCiterion.fulfilled(epoch, connections)) {
        // TODO: sweep through the whole dataset starting at random position
        for (unsigned step = 0; step < dsSize / batchSize; ++step) {
            meta::tup_each([] (auto& x) { x.zero(); }, accumulatedDeltas);
            unsigned batchStart = step * batchSize;
            auto examples = dataset.examples.template middleCols<batchSize>(batchStart);
            auto labels = dataset.labels.template middleCols<batchSize>(batchStart);

            feedForward(
                examples, connections, activations, typename NetConf::Layers{});
            backProp(labels, activations, connections, delta, accumulatedDeltas);
            meta::tup_each(
                [regParam, learningRate] (auto& theta, const auto& acc) {
                    theta.update(
                        regParam, learningRate / static_cast<double>(batchSize), acc);
                },
                connections, accumulatedDeltas);
        }
        epoch++;
    }
    return connections;
}


template <typename Dataset, typename Theta, typename NetConf>
void batchGradDescent(
        const Dataset& dataset,
        const double regParam,
        Theta& theta,
        NetConf) {
    const uint64_t m = size(dataset);
    const double learningRate = 0.05 / static_cast<double>(m);

    Theta accumulatedDeltas;
    typename NetConf::template Activations<1> activations;
    typename NetConf::template Delta<1> delta;

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

