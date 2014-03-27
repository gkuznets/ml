#pragma once

#include <ostream>
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

    void zero() {
        weights_.fill(0.0);
        bias_.fill(0.0);
    }

    double squaredNorm() const {
        // Don't regularize bias terms
        return weights_.squaredNorm();
    }

    void dump(std::ostream& out) const {
        out << weights_ << "\n\n";
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

    void update(
            double regParam,
            double learningRate,
            const FullConnection& delta) {
        weights_ *= (1.0 - regParam * learningRate);
        bias_ *= (1.0 - regParam * learningRate);
        weights_ -= learningRate * delta.weights_;
        bias_ -= learningRate * delta.bias_;
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

