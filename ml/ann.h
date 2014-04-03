#pragma once

#include <ml/ann/feed_forward.h>
#include <ml/ann/guts.h>
#include <ml/ann/layer_types.h>
#include <ml/ann/optimization.h>
#include <ml/ann/params.h>

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

private:
    typedef meta::map<detail::Nodes, Layers> ActivationTypes;
    typedef meta::map<detail::Nodes, meta::tail<Layers>> DeltaTypes;
    typedef meta::zipWith<
                    detail::Connection,
                    Layers,
                    meta::tail<Layers>> ConnectionTypes;
public:
    typedef meta::apply<std::tuple, ConnectionTypes> Connections;
    typedef meta::apply<std::tuple, ActivationTypes> Activations;
    typedef meta::apply<std::tuple, DeltaTypes> Delta;

    static const QuadLoss lossFn;
};

namespace {

template <typename NetConf>
class ANNClassifier {
public:
    explicit ANNClassifier(typename NetConf::Connections connections)
        : connections_(std::move(connections)) {
    }

    template <typename Input>
    auto operator() (const Input& input) const {
        return detail::feedForward(
                input, connections_, typename NetConf::Layers{});
    }
private:
    typename NetConf::Connections connections_;
};

} // namespace


template <typename Dataset
         ,typename NetworkParmas
         ,typename OptzParams>
auto train(const Dataset& trainingSet,
        OptzParams optimizationParams, NetworkParmas) {
    detail::gradDescent(trainingSet, optimizationParams, NetworkParmas{});
    auto monitor = meta::get<optimizationMonitorP>(optimizationParams);
    return ANNClassifier<NetworkParmas>(monitor.connections());
}

} // namespace ann
} // namespace ml

