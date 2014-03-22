#pragma once

#include <meta/meta.h>
#include <ml/ann/layer_types.h>

#include <random>

#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

template <typename PrevLayer
         ,typename Type>
struct Layer { };

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
    static const ActivationFn activation;
};

template <typename PrevLayer
         ,unsigned N
         ,typename ActivationFn>
struct Layer<PrevLayer, SymmetricFullyConnected<N, ActivationFn>> {
    static const unsigned numNodes = PrevLayer::numNodes;
    static const ActivationFn activation;
};


template <typename LrInst, typename... LrTypes>
struct Layers { };

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

template <typename InputLayer, typename OutputLayer>
class Connections {
public:
    void init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (unsigned row = 0; row < rows_; ++row) {
            for (unsigned col = 0; col < cols_; ++col) {
                weights_(row, col) = dis(gen);
            }
        }
    }

    void zero() {
        weights_.fill(0.0);
    }

    double squaredNorm() const {
        // Don't regularize bias terms
        return weights_.leftCols(cols_ - 1).squaredNorm();
    }

    // TODO: use a better name
    template <typename ColVector>
    auto transform(const ColVector& input) const {
        return weights_ * input;
    }

    template <typename ColVector>
    auto backTransform(const ColVector& vec) const {
        return weights_.transpose() * vec;
    }

    template <typename Matrix>
    void add(Matrix&& delta) {
        weights_ += delta;
    }

    void update(
            double regParam, double learningRate, const Connections& delta) {
        weights_ *= (1.0 - regParam * learningRate);
        weights_ -= learningRate * delta.weights_;
    }

private:
    const unsigned rows_ = OutputLayer::numNodes;
    const unsigned cols_ = InputLayer::numNodes + 1;
    Eigen::Matrix<double
                 ,OutputLayer::numNodes
                 ,InputLayer::numNodes + 1> weights_;
};

template <typename Layer>
using Activations = Eigen::Matrix<double, Layer::numNodes + 1, 1>;

template <typename Layer>
using Deltas = Eigen::Matrix<double, Layer::numNodes, 1>;


} // namespace detail
} // namespace ann
} // namespace ml

