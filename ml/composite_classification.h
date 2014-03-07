#pragma once

#include <ml/exception.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace ml {
namespace c12n {
namespace composite {

namespace {

template <typename OneVsOneClassifier, typename DecisionFn>
class CompositeClassifier {
public:
    typedef std::vector<OneVsOneClassifier> OneVsOneClassifiers;

    CompositeClassifier(CompositeClassifier&&) = default;
    explicit CompositeClassifier(OneVsOneClassifiers classifiers)
        : oneVsOneClassifiers_(std::move(classifiers))
        , numClasses_(numClasses(oneVsOneClassifiers_.size())) {}

    template <typename RowVector>
    int operator() (const RowVector& row) const {
        return DecisionFn()(numClasses_,
                [&](int cls0, int cls1) {
                    return pairClassifier(cls0, cls1)(row);
                });
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

//! Train composite multiclass classifier
//
//! \param dataset Labelled training set
//! \param model Model of a bianary classification algorithm
//! \return Composite multiclass classifier
// NOTICE: clang 3.3 can't compile this, if we try to add
// DecisionFn argument and make it deduce template params.
template <
    typename Dataset,
    typename Model,
    typename DecisionFn>
auto trainOneVsOneComposite(
        const Dataset& dataset, Model model)
        -> CompositeClassifier<decltype(model(dataset)), DecisionFn> {

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

    return CompositeClassifier<decltype(model(dataset)), DecisionFn>{
        std::move(oneVsOneClassifiers)};
}

} // namespace composite
} // namespace c12n
} // namespace ml

