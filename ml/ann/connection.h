#pragma once

#include <ml/exception.h>
#include <random>
#include <Eigen/Dense>

namespace ml {
namespace ann {
namespace detail {

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

template <unsigned inSize, unsigned outSize>
class FullConnection {
    Eigen::Matrix<double, outSize, Eigen::Dynamic> weights_;
    Eigen::Matrix<double, outSize, 1> bias_;

public:
    FullConnection() : weights_(outSize, inSize) {}

    void init() {
        fillRandom(weights_);
        fillRandom(bias_);
    }

    template <typename Weights, typename Bias>
    void initWith(const Weights& weights, const Bias& bias) {
        REQUIRE(weights.rows() == outSize && weights.cols() == inSize,
            "Invalid weights size ("
            << weights.rows() << "x" << weights.cols() << ")"
            " != (" << outSize << "x" << inSize << ")");
        REQUIRE(bias.rows() == outSize && bias.cols() == 1,
            "Invalid bias size ("
            << bias.rows() << "x" << bias.cols() << ")"
            " != (" << outSize << "x" << 1 << ")");
        weights_ = weights;
        bias_ = bias;
    }

    void zero() {
        weights_.fill(0.0);
        bias_.fill(0.0);
    }

    double squaredNorm() const {
        // Don't regularize bias terms
        return weights_.squaredNorm();
    }

    const auto& weights() const {
        return weights_;
    }

    const auto& bias() const {
        return bias_;
    }

    // TODO: use a better name
    // TODO: investigate problem with clang
    template <typename Input>
    Eigen::Matrix<double, outSize, Input::ColsAtCompileTime>
    transform(const Input& input) const {
        return (weights_ * input).colwise() + bias_;
    }

    template <typename Input>
    auto backTransform(const Input& delta) const {
        return weights_.transpose() * delta;
    }

    template <typename Delta, typename Activations>
    void propagate(Delta&& delta, Activations&& activations) {
        weights_ += delta * activations.transpose();
        bias_ += delta.rowwise().sum();
    }

    template <typename Regularizer>
    void applyRegularizer(const Regularizer& rglrz, double learningRate) {
        rglrz.apply(weights_, learningRate);
        // Do not regularize bias
    }

    void update(
            const FullConnection& delta,
            double learningRate) {
        weights_ -= learningRate * delta.weights_;
        bias_ -= learningRate * delta.bias_;
    }

    template <typename OStream
             ,unsigned inSize_
             ,unsigned outSize_>
    friend OStream& operator<< (
            OStream& o, const FullConnection<inSize_, outSize_>& fc) {
        return (o << fc.weights_);
    }
};

/*template <typename InputLayer
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
};*/


} // namespace detail
} // namespace ann
} // namespace ml

