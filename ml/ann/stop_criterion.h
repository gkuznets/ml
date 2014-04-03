#pragma once

#include <ml/ann/feed_forward.h>
#include <limits>

namespace ml {
namespace ann {

class EpochsNumber {
    unsigned maxEpoch_;
public:
    explicit EpochsNumber(unsigned maxEpoch)
        : maxEpoch_(maxEpoch) {}

    template <typename Connections>
    bool operator() (unsigned epoch, const Connections&) const {
        return epoch > maxEpoch_;
    }
};

template <typename Dataset
         ,typename NetConf>
class EarlyStopping {
public:
    EarlyStopping(const EarlyStopping&) = delete;
    EarlyStopping(EarlyStopping&&) = default;
    EarlyStopping(const Dataset& validationSet,
                  const unsigned epochsBetweenUpdates)
        : validationSet_(validationSet)
        , epochsBetweenUpdates_(epochsBetweenUpdates)
        , lastUpdateEpoch_(0)
        , lowestLoss_(std::numeric_limits<double>::max())
        , bestEpoch_(0) {}

    EarlyStopping& operator= (const EarlyStopping&) = delete;
    EarlyStopping& operator= (EarlyStopping&&) = default;

    class Ref {
    public:
        explicit Ref(EarlyStopping& parent) : parent_(parent) {}

        bool operator() (
            const unsigned epoch,
            const typename NetConf::Connections& connections) {
            return parent_(epoch, connections);
        }

        const auto& connections() const {
            return parent_.connections();
        }
    private:
        EarlyStopping& parent_;
    };

    Ref ref() { return Ref{*this}; }

    bool operator() (
            const unsigned epoch,
            const typename NetConf::Connections& connections) {
        if (epoch - lastUpdateEpoch_ >= epochsBetweenUpdates_) {
            lastUpdateEpoch_ = epoch;
            double currentLoss = loss(connections);
            if (currentLoss < lowestLoss_) {
                lowestLoss_ = currentLoss;
                bestEpoch_ = epoch;
                out_ = connections;
            } else if (epoch - bestEpoch_ > 10) {
                return true;
            }
        }
        return false;
    }

    const auto& connections() const {
        return out_;
    }

private:
    double loss(const typename NetConf::Connections& connections) const {
        auto prediction = detail::feedForward(
                                validationSet_.examples,
                                connections,
                                typename NetConf::Layers{});
        return NetConf::lossFn(prediction, validationSet_.labels);
    }

    const Dataset& validationSet_;
    const unsigned epochsBetweenUpdates_;
    unsigned lastUpdateEpoch_;
    double lowestLoss_;
    unsigned bestEpoch_;
    typename NetConf::Connections out_;
};

} // namespace ann
} // namespace ml

