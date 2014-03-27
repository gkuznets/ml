#pragma once

#include <ml/ann/feed_forward.h>
#include <ml/ann/guts.h>
#include <ml/ann/layer_types.h>
#include <ml/ann/optimization.h>

#include <meta/meta.h>
#include <meta/tuple.h>

#include <cmath>
#include <limits>
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
    template <unsigned batchSize>
    using ActivationTypes = meta::zipWith<
        detail::Nodes,
        Layers,
        meta::list_of<
            detail::BatchSize<batchSize>,
            meta::length<Layers>::value>>;
    template <unsigned batchSize>
    using DeltaTypes = meta::zipWith<
        detail::Nodes,
        meta::tail<Layers>,
        meta::list_of<
            detail::BatchSize<batchSize>,
            meta::length<Layers>::value - 1>>;

    typedef meta::zipWith<
                    detail::Connection,
                    Layers,
                    meta::tail<Layers>> ConnectionTypes;
public:
    typedef meta::apply<std::tuple, ConnectionTypes> Connections;

    template <unsigned batchSize>
    using Activations =
        meta::apply<std::tuple, ActivationTypes<batchSize>>;

    template <unsigned batchSize>
    using Delta =
        meta::apply<std::tuple, DeltaTypes<batchSize>>;

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
         ,typename NetConf>
class EarlyStopping {
public:
    EarlyStopping(const Dataset& validationSet,
                  const unsigned epochsBetweenUpdates,
                  typename NetConf::Connections& out)
        : validationSet_(validationSet)
        , epochsBetweenUpdates_(epochsBetweenUpdates)
        , lastUpdateEpoch_(0)
        , lowestLoss_(std::numeric_limits<double>::max())
        , bestEpoch_(0)
        , out_(out) {}

    bool fulfilled(
            const unsigned epoch,
            const typename NetConf::Connections& connections) {
        if (epoch - lastUpdateEpoch_ > epochsBetweenUpdates_) {
            lastUpdateEpoch_ = epoch;
            double currentLoss = loss(connections);
            if (currentLoss < lowestLoss_) {
                lowestLoss_ = currentLoss;
                bestEpoch_ = epoch;
                out_ = connections;
            } else if (epoch - bestEpoch_ > 200) {
                return true;
            }
        }
        return false;
    }
private:
    double loss(const typename NetConf::Connections& connections) const {
        auto prediction = detail::feedForward(
                                validationSet_.examples,
                                connections,
                                typename NetConf::Layers{});
        return NetConf::lossFn(prediction, validationSet_.labels);
    }

    const Dataset& validationSet_;
    const unsigned epochsBetweenUpdates_;
    unsigned lastUpdateEpoch_;
    double lowestLoss_;
    unsigned bestEpoch_;
    typename NetConf::Connections& out_;
};


template <typename NetConf, typename Dataset>
auto train(NetConf, const Dataset& trainingSet, const Dataset& validationSet) {
    typename NetConf::Connections connections;
    //meta::tup_each([] (auto& c) { c.init(); }, connections);
    //detail::batchGradDescent(trainingSet, 2.0, connections, NetConf{});
    EarlyStopping<Dataset, NetConf> earlyStopping(validationSet, 5, connections);
    detail::gradDescent<32, Dataset, decltype(earlyStopping), NetConf>(
                trainingSet, 1.0, earlyStopping);
    return ANNClassifier<NetConf>(std::move(connections));
}

} // namespace ann
} // namespace ml

