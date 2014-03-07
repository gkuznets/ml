#pragma once

#include <ml/composite_classification.h>

#include <algorithm>
#include <vector>

namespace ml {
namespace max_wins {

namespace {

//! 'Max Wins' decision function
struct MaxWinsDecide {
    template <typename ClassifyOneVsOne>
    int operator()(const int numClasses, ClassifyOneVsOne classify) const {
        std::vector<unsigned> wins(numClasses, 0);
        for (int cls0 = 0; cls0 < numClasses - 1; ++cls0) {
            for (int cls1 = cls0 + 1; cls1 < numClasses; ++cls1) {
                if (classify(cls0, cls1) == 1) {
                    wins[cls0]++;
                } else {
                    wins[cls1]++;
                }
            }
        }
        return std::max_element(wins.begin(), wins.end()) - wins.begin();
    }
};

} // namespace

//! Train 'Max Wins' composite classifier
template <typename Dataset, typename Model>
auto train(const Dataset& dataset, Model model)
    -> decltype(c12n::composite::trainOneVsOneComposite<
            Dataset, Model, MaxWinsDecide>(dataset, model)) {
    return c12n::composite::trainOneVsOneComposite<
        Dataset, Model, MaxWinsDecide>(dataset, model);
}

} // namespace max_wins
} // namespace ml

