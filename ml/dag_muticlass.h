#pragma once

#include <ml/exception.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace ml {
namespace dag {

namespace detail {

//! DAG classifier
template <typename OneVsOneClassifier>
class DAGClassifier {
public:
    typedef std::vector<OneVsOneClassifier> OneVsOneClassifiers;

    DAGClassifier(DAGClassifier&&) = default;
    explicit DAGClassifier(OneVsOneClassifiers classifiers)
        : oneVsOneClassifiers_(std::move(classifiers))
        , numClasses_(numClasses(oneVsOneClassifiers_.size())) {}

    template <typename RowVector>
    int operator() (const RowVector& row) const {
        std::vector<int> candidates(numClasses_);
        std::iota(candidates.begin(), candidates.end(), 0);
        while (candidates.size() > 1) {
            int cls0 = candidates[candidates.size() - 2];
            int cls1 = candidates[candidates.size() - 1];
            candidates.resize(candidates.size() - 1);
            const auto& classifier = pairClassifier(cls0, cls1);

            if (classifier(row) == -1) {
                // popping cls0
                candidates.back() = cls1;
            }
        }
        return candidates[0];
    }

private:
    static int numClasses(int numPairs) {
        return (1 + std::sqrt(1 + 8 * numPairs)) / 2;
    }

    // TODO: add explanation
    const OneVsOneClassifier& pairClassifier(int cls0, int cls1) const {
        auto index = cls0 * (2 * numClasses_ - cls0 - 3) / 2 + cls1 - 1;
        return oneVsOneClassifiers_[index];
    }

    OneVsOneClassifiers oneVsOneClassifiers_;
    const int numClasses_;
};

template <typename OneVsOneClassifier>
DAGClassifier<OneVsOneClassifier>
makeDAGClassifier(std::vector<OneVsOneClassifier> classifiers) {
    return DAGClassifier<OneVsOneClassifier>(std::move(classifiers));
}

} // namespace detail

namespace {

template <typename Dataset>
std::vector<std::vector<unsigned>>
splitIndices(const Dataset& dataset) {
    std::vector<std::pair<int, unsigned>> indices(size(dataset));
    std::unordered_set<int> seenClasses;
    for (unsigned i = 0; i < size(dataset); ++i) {
        int cls = label(i, dataset);
        seenClasses.insert(cls);
        indices[i] = {cls, i};
    }
    std::sort(indices.begin(), indices.end());
    REQUIRE(indices.front().first == 0 &&
            indices.back().first == static_cast<int>(seenClasses.size() - 1),
            "Class labels shouldn't contain gaps");

    std::vector<std::vector<unsigned>> result;
    result.reserve(seenClasses.size());
    auto cmpFirst = [](
            const std::pair<int, unsigned>& a,
            const std::pair<int, unsigned>& b) {
        return a.first < b.first;
    };
    auto clsBegin = indices.begin();
    while (clsBegin != indices.end()) {
        auto clsEnd =
            std::upper_bound(clsBegin, indices.end(), *clsBegin, cmpFirst);
        std::vector<unsigned> clsIndices;
        std::for_each(clsBegin, clsEnd,
                [&clsIndices] (const std::pair<int, unsigned>& clsIndex) {
                    clsIndices.push_back(clsIndex.second);
                });
        result.emplace_back(std::move(clsIndices));
        clsBegin = clsEnd;
    }
    return result;
}

template <typename Dataset>
Dataset makePairDataSet(
        const std::vector<unsigned>& class0Indices,
        const std::vector<unsigned>& class1Indices,
        const Dataset& dataset) {
    Dataset result(class0Indices.size() + class1Indices.size());
    unsigned pos = 0;
    for (unsigned i: class0Indices) {
        set(pos++, example(i, dataset), 1, result);
    }
    for (unsigned i: class1Indices) {
        set(pos++, example(i, dataset), -1, result);
    }
    return result;
}

} // namespace


//! Train multiclass classifier using DAG algorithm
//!
//! \param dataset Labelled training
//! \param model Model of a bianary classification
//!         algorithm
//! \return DAG multiclass classifier
template <typename Dataset, typename Model>
auto train(const Dataset& dataset, Model model)
    -> detail::DAGClassifier<decltype(model(dataset))> {

    auto classes = splitIndices(dataset);
    unsigned numClasses = classes.size();
    typedef decltype(model(dataset)) OneVsOneClassifier;
    std::vector<OneVsOneClassifier> oneVsOneClassifiers;
    oneVsOneClassifiers.reserve(numClasses * (numClasses - 1) / 2);

    for (unsigned cls0 = 0; cls0 < numClasses - 1; ++cls0) {
        for (unsigned cls1 = cls0 + 1; cls1 < numClasses; ++cls1) {
            auto pairDataset = makePairDataSet(
                    classes[cls0], classes[cls1], dataset);
            oneVsOneClassifiers.emplace_back(model(pairDataset));
        }
    }

    return detail::makeDAGClassifier(std::move(oneVsOneClassifiers));
}

} // namespace dag
} // namespace ml

