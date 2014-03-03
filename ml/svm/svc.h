#pragma once

#include <ml/dataset/dataset.h>
#include <ml/sign.h>
#include <ml/svm/smo.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ml {
namespace svm {
namespace c12n {

namespace detail {

template <typename KernelFn, typename Dataset>
class WrappedKernel {
public:
    WrappedKernel(WrappedKernel&&) = default;

    WrappedKernel(KernelFn kernelFn, const Dataset& dataset)
        : kernelFn_(kernelFn), dataset_(dataset) {}

    double operator()(unsigned i, unsigned j) const {
        return kernelFn_(example(i, dataset_), example(j, dataset_));
    }

private:
    KernelFn kernelFn_;
    const Dataset& dataset_;

};

template <typename KernelFn, typename Dataset>
WrappedKernel<KernelFn, Dataset>
wrapKernel(KernelFn kernelFn, const Dataset& dataset) {
    return WrappedKernel<KernelFn, Dataset>(kernelFn, dataset);
}

//! Binary SVM classifier
template <typename Dataset, typename Kernel>
class SVMClassifier {
public:
    SVMClassifier(SVMClassifier&&) = default;
    SVMClassifier(
            Dataset dataset,
            std::vector<double> alphas,
            double threshold,
            Kernel kernel)
        : dataset_(std::move(dataset))
        , alphas_(std::move(alphas))
        , threshold_(threshold)
        , kernel_(kernel) {}

    SVMClassifier& operator= (SVMClassifier&&) = default;

    template <typename RowVector>
    int operator() (const RowVector& row) const {
        double sum = threshold_;
        for (uint64_t i = 0; i < alphas_.size(); ++i) {
            sum += alphas_[i] * kernel_(row, example(i, dataset_));
        }
        return sign(sum);
    }

private:
    Dataset dataset_;
    std::vector<double> alphas_;
    double threshold_;
    Kernel kernel_;

};

} // namespace detail

//! Train soft margin SVM binary classifier
//!
//! \param trainingSet Labelled training dataset
//! \param C Regularization parameter
//! \param kernel Kernel function
//! \return Binary classifier
template <typename Dataset, typename Kernel>
detail::SVMClassifier<Dataset, Kernel>
train(const Dataset& trainingSet, const double C, Kernel kernel) {
    std::vector<double> alphas;
    double threshold;
    std::tie(alphas, threshold) =
        smo::solve(trainingSet, C, detail::wrapKernel(kernel, trainingSet));

    std::vector<double> nonzeroAlphas;
    std::vector<uint64_t> nonzeroPositions;
    for (uint64_t i = 0; i < alphas.size(); ++i) {
        if (alphas[i] > 0.0) {
            nonzeroPositions.push_back(i);
            nonzeroAlphas.push_back(alphas[i] * label(i, trainingSet));
        }
    }

    Dataset supportSubset(nonzeroPositions.size());
    for (uint64_t i = 0; i < nonzeroPositions.size(); ++i) {
        uint64_t position = nonzeroPositions[i];
        set(i, example(position, trainingSet),
                label(position, trainingSet), supportSubset);
    }
    return detail::SVMClassifier<Dataset, Kernel>(
            std::move(supportSubset),
            std::move(nonzeroAlphas),
            threshold,
            kernel);
}

} // namespace c12n
} // namespace svm

namespace svc = svm::c12n;

} // namespace ml

