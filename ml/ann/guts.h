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

namespace {

template <typename Matrix>
void fillRandom(Matrix& m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (unsigned row = 0; row < m.rows(); ++row) {
        for (unsigned col = 0; col < m.cols(); ++col) {
            m(row, col) = dis(gen);
        }
    }
}

} // namespace

template <typename InputLayer
         ,typename OutputLayer>
class Connections {};

template <typename InputLayer
         ,unsigned N
         ,typename ActivationFn>
class Connections<
        InputLayer,
        Layer<InputLayer, FullyConnected<N, ActivationFn>>> {
public:
    void init() {
        fillRandom(weights_);
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
    typedef Layer<InputLayer, FullyConnected<N, ActivationFn>> OutputLayer;
    static const unsigned rows_ = OutputLayer::numNodes;
    static const unsigned cols_ = InputLayer::numNodes + 1;
    Eigen::Matrix<double, rows_, cols_> weights_;
};

template <typename InputLayer
         ,unsigned N
         ,typename ActivationFn>
class Connections<
        InputLayer,
        Layer<InputLayer, SymmetricFullyConnected<N, ActivationFn>>> {
public:
    void init() {
        fillRandom(weights_);
    }

    template <typename ColVector>
    auto transform(const ColVector& input) const {
        auto hidden = weights_ * input;
    }

private:
    static const unsigned rows_ = N;
    static const unsigned cols_ = InputLayer::numNodes + 1;
    Eigen::Matrix<double, rows_, cols_> weights_;
};


template <typename Layer>
using Activations = Eigen::Matrix<double, Layer::numNodes + 1, 1>;

template <typename Layer>
using Deltas = Eigen::Matrix<double, Layer::numNodes, 1>;


} // namespace detail
} // namespace ann
} // namespace ml

