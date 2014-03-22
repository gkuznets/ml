#pragma once

#include <ml/ann/feed_forward.h>
#include <ml/ann/guts.h>
#include <ml/ann/layer_types.h>
#include <ml/ann/optimization.h>

#include <meta/meta.h>
#include <meta/tuple.h>

#include <tuple>

namespace ml {
namespace ann {

template <typename... LrTypes>
struct NetworkConf {
    using Layers = typename detail::Layers<meta::none, LrTypes...>::type;
    using Connections = meta::zipWith<
                            detail::Connections,
                            Layers,
                            meta::tail<Layers>>;
    using Activations = meta::map<detail::Activations, Layers>;
    using Deltas = meta::map<detail::Deltas, meta::tail<Layers>>;
};

namespace {

template <typename Connections
         ,typename Layers>
class ANNClassifier {
public:
    explicit ANNClassifier(Connections connections)
        : connections_(std::move(connections)) {}

    template <typename ColVector>
    double operator() (const ColVector& input) const {
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

