#pragma once

#include <meta/meta.h>
#include <ml/ann/connection.h>
#include <ml/ann/layer_types.h>

#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

template <typename PrevLayer
         ,typename Type>
struct Layer {};

template <unsigned N>
struct Layer<meta::none, Input<N>> {
    static const unsigned numNodes = N;
};

template <unsigned W, unsigned H>
struct Layer<meta::none, ImageInput<W, H>> {
    static const unsigned width = W;
    static const unsigned height = H;
    static const unsigned numNodes = width * height;
};

template <typename PrevLayer
         ,unsigned N
         ,typename ActivationFn>
struct Layer<PrevLayer, FullyConnected<N, ActivationFn>> {
    static const unsigned numNodes = N;
    typedef ActivationFn Activation;
};

template <typename PrevLayer
         ,unsigned N
         ,typename ActivationFn>
struct Layer<PrevLayer, SymmetricFullyConnected<N, ActivationFn>> {
    static const unsigned numNodes = PrevLayer::numNodes;

    template <typename Input>
    auto activation(Input&& x) const {
        return ActivationFn{}(std::forward(x));
    }
};


template <typename LrInst, typename... LrTypes>
struct Layers {};

template <typename LrInst, typename LrType, typename... LrTypes>
struct Layers<LrInst, LrType, LrTypes...> {
    typedef Layer<LrInst, LrType> ThisLayer;
    typedef meta::prepend<
                ThisLayer,
                typename Layers<ThisLayer, LrTypes...>::type> type;
};

template <typename LrInst>
struct Layers<LrInst> {
    typedef meta::list<> type;
};

template <typename InputLayer
         ,typename OutputLayer>
struct ConnectionSelector {};

template <typename InputLayer
         ,unsigned N
         ,typename ActivationFn>
struct ConnectionSelector<
                InputLayer,
                Layer<InputLayer, FullyConnected<N, ActivationFn>>> {
    typedef Layer<InputLayer, FullyConnected<N, ActivationFn>> OutputLayer;
    typedef FullConnection<InputLayer::numNodes, OutputLayer::numNodes> type;
};

template <typename InputLayer
         ,typename OutputLayer>
using Connection = typename ConnectionSelector<InputLayer, OutputLayer>::type;

template <typename Layer>
using Nodes = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace detail
} // namespace ann
} // namespace ml

