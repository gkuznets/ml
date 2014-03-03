#pragma once

#include <ml/dataset/dataset_traits.h>
#include <cstdint>
#include <vector>

namespace ml {

template <typename Example, typename Label>
struct VecDataset {
    VecDataset() = default;
    VecDataset(VecDataset&&) = default;
    explicit VecDataset(uint64_t size)
        : examples(size)
        , labels(size) {}

    std::vector<Example> examples;
    std::vector<Label> labels;
};

template <typename X, typename Y>
struct dataset_traits<VecDataset<X, Y>> {
};

template <typename X, typename Y>
uint64_t size(const VecDataset<X, Y>& dataset) {
    return dataset.examples.size();
}

template <typename X, typename Y>
const X& example(uint64_t pos, const VecDataset<X, Y>& dataset) {
    return dataset.examples[pos];
}

template <typename X, typename Y>
const Y& label(uint64_t pos, const VecDataset<X, Y>& dataset) {
    return dataset.labels[pos];
}

template <typename X, typename Y>
void set(
        uint64_t pos,
        const X& example,
        const Y& label,
        VecDataset<X, Y>& dataset) {
    dataset.examples[pos] = example;
    dataset.labels[pos] = label;
}

} // namespace ml

