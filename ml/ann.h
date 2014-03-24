#pragma once

#include <ml/ann/feed_forward.h>
#include <ml/ann/guts.h>
#include <ml/ann/layer_types.h>
#include <ml/ann/optimization.h>

#include <meta/meta.h>
#include <meta/tuple.h>

#include <cmath>
#include <tuple>

namespace ml {
namespace ann {

// TODO: move me somewhere
struct CrossEntropy {
    template <typename Prediction, typename Label>
    double operator()(Prediction&& prediction, Label&& label) const {
        // TODO: figure out dependency on last activation fn
        return -((1.0 + label.array()) * (1.0 + prediction.array()).log() +
                 (1.0 - label.array()) * (1.0 - prediction.array()).log()).sum();
    }
};

struct QuadLoss {
   template <typename Prediction, typename Label>
    double operator()(Prediction&& prediction, Label&& label) const {
        return 0.5 * (prediction - label).squaredNorm();
    }
};

template <typename... LrTypes>
struct NetworkConf {
    using Layers = typename detail::Layers<meta::none, LrTypes...>::type;
    using Connections = meta::zipWith<
                            detail::Connections,
                            Layers,
                            meta::tail<Layers>>;
    using Activations = meta::map<detail::Activations, Layers>;
    using Deltas = meta::map<detail::Deltas, meta::tail<Layers>>;
    static const QuadLoss lossFn;
};

namespace {

template <typename Connections
         ,typename Layers>
class ANNClassifier {
public:
    explicit ANNClassifier(Connections connections)
        : connections_(std::move(connections)) {}

    template <typename ColVector>
    auto operator() (const ColVector& input) const {
        return detail::feedForward(input, connections_, Layers{});
    }

private:
    Connections connections_;
};

} // namespace

template <typename NetConf, typename Dataset>
auto train(NetConf, const Dataset& dataset) {
    meta::apply<std::tuple, typename NetConf::Connections> connections;
    meta::tup_each([] (auto& c) { c.init(); }, connections);
    detail::batchGradDescend(dataset, 2.0, connections, NetConf{});
    return ANNClassifier<
        decltype(connections),
        typename NetConf::Layers>(std::move(connections));
}

} // namespace ann
} // namespace ml

