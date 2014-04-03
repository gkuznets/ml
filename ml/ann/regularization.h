#pragma once

namespace ml {
namespace ann {

class L2Regularization {
public:
    explicit L2Regularization(double lambda) : lambda_(lambda) {}

    template <typename Weights>
    void apply(Weights& wts, double coeff) const {
        wts *= 1.0 - lambda_ * coeff;
    }
private:
    double lambda_;
};

} // namespace ann
} // namespace ml

