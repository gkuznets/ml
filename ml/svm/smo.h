#pragma once

#include <ml/dataset/dataset.h>
#include <ml/dataset/dataset_traits.h>
#include <ml/dot.h>
#include <ml/sign.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <boost/range/algorithm/count_if.hpp>
#include <boost/range/irange.hpp>

namespace ml {
namespace svm {
namespace smo {

namespace {

uint64_t random(uint64_t n) {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist(0, n - 1);
    return dist(rd);
}

bool nonBound(double x, double bound) {
    return x > 0.0 && x < bound;
}

double clip(double x, double low, double high) {
    return std::max(low, std::min(high, x));
}

class ErrorCache {
public:
    explicit ErrorCache(const unsigned size) : cache_(size) {}

    double operator() (unsigned i) const {
        return cache_[i];
    }

    void set(unsigned i, double value) {
        cache_[i] = value;
    }

    template <typename Kernel>
    void update(
            unsigned i0, double dAlpha0,
            unsigned i1, double dAlpha1,
            double dThreshold,
            const Kernel& kernel) {
        const unsigned size = cache_.size();
        for (unsigned i = 0; i < size; ++i) {
            cache_[i] += dThreshold +
                dAlpha0 * kernel(i, i0) + dAlpha1 * kernel(i, i1);
        }
    }

private:
    std::vector<double> cache_;

};

template <typename Dataset, typename Kernel>
double learnedFunction(
        unsigned i,
        const double threshold,
        const Dataset& dataset,
        const std::vector<double>& alphas,
        const Kernel& K) {
    double sum = threshold;
    for (unsigned j = 0; j < alphas.size(); ++j) {
        if (alphas[j] > 0.0)
            sum += alphas[j] * label(j, dataset) * K(i, j);
    }
    return sum;
}

unsigned choosePair(
        unsigned i0,
        const double error0,
        const double C,
        const std::vector<double>& alphas,
        ErrorCache& errorCache) {

    const unsigned N = alphas.size();
    unsigned result = i0;
    double dErrorMax = 0.0;
    // trying to maximize |error0 - error1|
    for (unsigned i = 0; i < N; ++i) {
        if (!nonBound(alphas[i], C))
            continue;

        const double dError = std::abs(errorCache(i) - error0);
        if (dError > dErrorMax) {
            result = i;
            dErrorMax = dError;
        }
    }
    return result;
}

std::pair<double, double> computeBounds(
        double alpha0, double alpha1,
        int label0, int label1,
        const double C) {
    if (label0 != label1) {
        return {std::max(0.0, alpha1 - alpha0),
                std::min(C, C + alpha1 - alpha0)};
    } else {
        return {std::max(0.0, alpha0 + alpha1 - C),
                std::min(C, alpha0 + alpha1)};
    }
}

template <typename Dataset, typename Kernel>
bool step(
        unsigned i0, unsigned i1,
        const double error0,
        const Dataset& dataset,
        const double C,
        double& threshold,
        std::vector<double>& alphas,
        const Kernel& K,
        ErrorCache& errorCache) {
    static const double EPS = 0.001;
    if (i0 == i1)
        return false;

    const double label0 = label(i0, dataset);
    const double label1 = label(i1, dataset);
    double L = 0.0;
    double H = 0.0;
    std::tie(L, H) = computeBounds(
            alphas[i0], alphas[i1], label0, label1, C);

    if (L == H)
        return false;

    const double error1 = errorCache(i1);

    const double k01 = K(i0, i1);
    const double k00 = K(i0, i0);
    const double k11 = K(i1, i1);
    const double eta = 2.0 * k01 - k00 - k11;
    double alpha1 = 0.0;
    if (eta < 0.0) {
        alpha1 = clip(alphas[i1] + label1 * (error1 - error0) / eta, L, H);
    } else {
        return false;
        double c0 = - eta / 2.0;
        double c1 = label1 * (error0 - error1) - eta * alphas[i1];
        const double Lobj = c0 * L * L + c1 * L;
        const double Hobj = c0 * H * H + c1 * H;
        if (Lobj < Hobj - EPS) {
            alpha1 = L;
        } else if (Lobj > Hobj + EPS) {
            alpha1 = H;
        } else {
            alpha1 = alphas[i1];
        }
    }

    if (std::abs(alpha1 - alphas[i1]) < EPS * (alpha1 + alphas[i1] + EPS)) {
        return false;
    }

    const int s = label0 * label1;
    double alpha0 = alphas[i0] + s * (alphas[i1] - alpha1);
    if (alpha0 < 0.0) {
        alpha0 = 0.0;
        alpha1 = alphas[i1] + s * alphas[i0];
    } else if (alpha0 > C) {
        alpha0 = C;
        alpha1 = alphas[i1] + s * (alphas[i0] - C);
    }

    const double dAlpha0 = label0 * (alpha0 - alphas[i0]);
    const double dAlpha1 = label1 * (alpha1 - alphas[i1]);
    double d0 = error0 + dAlpha0 * k00 + dAlpha1 * k01;
    double d1 = error1 + dAlpha0 * k01 + dAlpha1 * k11;

    double dThreshold = 0.0;
    if (nonBound(alpha0, C)) {
        dThreshold = d0;
    } else if (nonBound(alpha1, C)) {
        dThreshold = d1;
    } else {
        dThreshold = (d1 + d0) / 2.0;
    }
    threshold -= dThreshold;

    alphas[i0] = alpha0;
    alphas[i1] = alpha1;
    errorCache.update(i0, dAlpha0, i1, dAlpha1, -dThreshold, K);

    return true;
}

template <typename Dataset, typename Kernel>
bool examine(
        const unsigned i0,
        const Dataset& dataset,
        const double C,
        double& threshold,
        std::vector<double>& alphas,
        const Kernel& K,
        ErrorCache& errorCache) {

    const unsigned N = alphas.size();
    static const double TOLERANCE = 0.001;
    const double alpha0 = alphas[i0];
    const int label0 = label(i0, dataset);
    const double error0 = errorCache(i0);

    if ((label0 * error0 < -TOLERANCE && alpha0 < C) ||
        (label0 * error0 > TOLERANCE && alpha0 > 0.0)) {

        const unsigned i1 = choosePair(i0, error0, C, alphas, errorCache);
        if (step(i0, i1, error0, dataset, C, threshold, alphas, K, errorCache)) {
            return true;
        }
        // trying all non-bound alphas starting at random position
        unsigned rnd = random(N);
        for (unsigned j = 0; j < N; ++j) {
            unsigned i = (j + rnd) % N;
            if (i != i1 && !nonBound(alphas[i], C) &&
                    step(i0, i, error0, dataset, C, threshold, alphas, K, errorCache)) {
                return true;
            }
        }
        // trying all bound alphas
        for (unsigned j = 0; j < N; ++j) {
            unsigned i = (j + rnd) % N;
            if (i != i1 && nonBound(alphas[i], C) &&
                    step(i0, i, error0, dataset, C, threshold, alphas, K, errorCache)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

//! Implementation of Sequential Minimal Optimization (SMO) algorithm
template <typename Dataset, typename Kernel>
std::pair<std::vector<double>, double>
solve(const Dataset& dataset, const double C, const Kernel& K) {
    const uint64_t N = size(dataset);
    std::vector<double> alphas(N, 0.0);
    double threshold = 0.0;

    ErrorCache errorCache(size(dataset));
    for (unsigned i = 0; i < size(dataset); ++i) {
        errorCache.set(i, -label(i, dataset));
    }

    bool examineAll = true;

    while (true) {
        const unsigned numChanged = boost::count_if(
            boost::irange<unsigned>(0, N),
            [&] (unsigned i) {
                return (examineAll || nonBound(alphas[i], C)) &&
                    examine(i, dataset, C, threshold, alphas, K, errorCache);
            });

        if (examineAll) {
            examineAll = false;
        } else if (numChanged == 0) {
            examineAll = true;
        }
        if (numChanged == 0 && !examineAll)
            break;
    }

    double b_minus = -1e7;
    double b_plus = 1e7;
    for (unsigned i = 0; i < alphas.size(); ++i) {
        const double f = learnedFunction(i, 0.0, dataset, alphas, K);
        if (label(i, dataset) == 1) {
            b_plus = std::min(b_plus, f);
        } else {
            b_minus = std::max(b_minus, f);
        }
    }
    threshold = - (b_minus + b_plus) / 2.0;
    return {alphas, threshold};
}

} // namespace smo
} // namespace svm
} // namespace ml

