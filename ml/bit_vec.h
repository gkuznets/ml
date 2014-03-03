#pragma once

#include <cstdint>
#include <vector>

namespace ml {

template <unsigned size>
class BitVec {
public:
    BitVec(BitVec&&) = default;
    BitVec() : packs_(1 + (size - 1) / 64, 0ul) {}

    BitVec& operator= (BitVec&& othr) = default;
    BitVec& operator= (const BitVec& other) {
        if (this != &other) {
            packs_ = other.packs_;
        }
        return *this;
    }

    void set(unsigned pos) {
        auto& pack = packs_[pos / 64];
        pack |= (1ul << (pos & 0x0000003F));
    }

    unsigned count() const {
        unsigned sum = 0;
        for (const auto& pack: packs_) {
            sum += __builtin_popcountll(pack);
        }
        return sum;
    }

    unsigned operator* (const BitVec<size>& other) const {
        unsigned sum = 0;
        for (unsigned i = 0; i < packs_.size(); ++i) {
            sum += __builtin_popcountll(packs_[i] & other.packs_[i]);
        }
        return sum;
    }

    template <unsigned N>
    friend unsigned distance(const BitVec<N>& a, const BitVec<N>& b);

private:
    std::vector<uint64_t> packs_;

};

template <unsigned size>
unsigned distance(const BitVec<size>& a, const BitVec<size>& b) {
    unsigned sum = 0;
    for (unsigned i = 0; i < a.packs_.size(); ++i) {
        sum += __builtin_popcountll(a.packs_[i] ^ b.packs_[i]);
    }
    return sum;
}

} // namespace ml

