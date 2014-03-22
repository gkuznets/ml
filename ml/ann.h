#pragma once

#include <meta/meta.h>
#include <meta/tuple.h>

#include <cmath>
#include <random>
#include <tuple>

#include <Eigen/Dense>

namespace ml {
namespace ann {

static const double TANH_COEFF = 1.0;

// TODO: add vectorized implementation
static inline double sigmoid(double x) {
    //return 1.0 / (1.0 + std::exp(-x));
    //return TANH_COEFF * std::tanh(0.666 * x);
    return TANH_COEFF * std::tanh(x);
}

// Layer types
template <unsigned size>
struct Input {};

template <unsigned width, unsigned height>
struct ImageInput {};

template <unsigned windowSize>
struct Convolution { };

template <unsigned windowSize>
struct Subsampling { };

template <unsigned size>
struct FullyConnected { };


template <typename PrevLayer, typename Type>
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

template <typename PrevLayer, unsigned N>
struct Layer<PrevLayer, FullyConnected<N>> {
    static const unsigned numNodes = N;
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

template <typename... LrTypes>
struct NetworkConf {
    using Layers = typename Layers<meta::none, LrTypes...>::type;
    using Connections = meta::zipWith<
                            Connections,
                            Layers,
                            meta::tail<Layers>>;
    using Activations = meta::map<Activations, Layers>;
    using Deltas = meta::map<Deltas, meta::tail<Layers>>;
};

template <typename Vec>
double& last(Vec& vec) {
    return vec(vec.size() - 1);
}

template <typename ColVector, typename Connections>
double feedForward(
        const ColVector& input,
        const Connections& connections) {
    Eigen::VectorXd result(input.rows() + 1);
    last(result) = 1.0;
    result.head(input.rows()) = input;
    meta::tup_each(
            [&result](const auto& conn) {
                result = conn.transform(result);
                result.resize(result.size() + 1);
                last(result) = 1.0;
                for (unsigned i = 0; i < result.size() - 1; ++i) {
                    result[i] = sigmoid(result[i]);
                }
            },
            connections);
    return result[0];
}

template <typename ColVector
         ,typename Connections
         ,typename Activations>
void feedForward(
        const ColVector& input,
        const Connections& connections,
        Activations& activations) {
    std::get<0>(activations).head(input.size()) = input;
    meta::tup_each(
            [](const auto& conn, const auto& in, auto& out) {
                out.head(out.size() - 1) = conn.transform(in);
                for (unsigned j = 0; j < out.size() - 1; ++j) {
                    out(j) = sigmoid(out(j));
                }
            },
            connections,
            activations,
            meta::tup_tail(activations));
}

template <typename Weights>
class ANNClassifier {
public:
    explicit ANNClassifier(Weights weights)
        : weights_(std::move(weights)) {}

    template <typename ColVector>
    double operator() (const ColVector& input) const {
        return feedForward(input, weights_);
    }

private:
    Weights weights_;

};

template <typename Dataset, typename Theta, typename NetConf>
void batchGradDescend(
        const Dataset& dataset,
        const double regParam,
        Theta& theta,
        NetConf) {
    const uint64_t m = size(dataset);
    const double learningRate = 0.05 / static_cast<double>(m);

    Theta accumulatedDeltas;
    meta::apply<std::tuple, typename NetConf::Activations> activations;
    meta::apply<std::tuple, typename NetConf::Deltas> delta;

    double oldCost = std::numeric_limits<double>::max();
    double newCost = 0.0;
    while (std::abs(oldCost - newCost) > 0.000001 * m) {
        oldCost = newCost;
        newCost = 0.0;

        meta::tup_each([] (auto& a) { last(a) = 1.0; }, activations);
        meta::tup_each([] (auto& x) { x.zero(); }, accumulatedDeltas);

        for (uint64_t i = 0; i < m; ++i) {
            feedForward(dataset.examples.col(i), theta, activations);

            // Backpropagation
            const auto& yi = dataset.labels(i);
            meta::tup_last(delta)(0) = meta::tup_last(activations)(0) - yi;
            meta::tup_each(
                    [](auto& delta, const auto& prevDelta,
                       const auto& theta, const auto& activations) {

                        const size_t size = delta.size();
                        delta = theta.backTransform(prevDelta).head(size);
                        delta.array() *=
                            (1 - activations.head(size).array().square());
                    },
                    meta::tup_tail(meta::tup_reverse(delta)),
                    meta::tup_reverse(delta),
                    meta::tup_reverse(theta),
                    meta::tup_tail(meta::tup_reverse(activations)));
            meta::tup_each(
                    [](auto& acc, const auto& delta, const auto& activations) {
                        acc.add(delta * activations.transpose());
                    },
                    accumulatedDeltas, delta, activations);
            double hxi = meta::tup_last(activations)(0);
            newCost -= ((1.0 + yi) * log(TANH_COEFF + hxi) + (1.0 - yi) * log(TANH_COEFF - hxi));
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


template <typename NetConf, typename Dataset>
auto train(NetConf, const Dataset& dataset) {
    meta::apply<std::tuple, typename NetConf::Connections> connections;
    meta::tup_each([] (auto& c) { c.init(); }, connections);

    batchGradDescend(dataset, 2.0, connections, NetConf{});
    return ANNClassifier<decltype(connections)>(std::move(connections));
}

} // namespace ann
} // namespace ml

