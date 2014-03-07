#pragma once

#include <ml/composite_classification.h>

#include <vector>

namespace ml {
namespace dag {

namespace {

//! DAG decision function
struct DAGDecide {
    template <typename ClassifyOneVsOne>
    int operator()(const int numClasses, ClassifyOneVsOne classify) const {
        std::vector<int> candidates(numClasses);
        std::iota(candidates.begin(), candidates.end(), 0);
        while (candidates.size() > 1) {
            int cls0 = candidates[candidates.size() - 2];
            int cls1 = candidates[candidates.size() - 1];
            candidates.resize(candidates.size() - 1);

            if (classify(cls0, cls1) == -1) {
                // popping cls0
                candidates.back() = cls1;
            }
        }
        return candidates[0];
    }
};

} // namespace detail

//! Train DAG multiclass classifier
//!
//! \param dataset Labelled training set
//! \param model Model of a bianary classification algorithm
//! \return DAG multiclass classifier
template <typename Dataset, typename Model>
auto train(const Dataset& dataset, Model model)
    -> decltype(c12n::composite::trainOneVsOneComposite<
            Dataset, Model, DAGDecide>(dataset, model)) {
    return c12n::composite::trainOneVsOneComposite<
        Dataset, Model, DAGDecide>(dataset, model);
}

} // namespace dag
} // namespace ml

