#pragma once

#include <array>
#include <cstdint>

namespace ml {

template <unsigned SIZE>
class BitVec {
public:
    BitVec() {
        packs_.fill(0ul);
    }

    BitVec(BitVec&&) = default;
    BitVec(const BitVec&) = default;

    BitVec& operator= (BitVec&& othr) = default;
    BitVec& operator= (const BitVec& other) {
        if (this != &other) {
            packs_ = other.packs_;
        }
        return *this;
    }

    static unsigned size() { return SIZE; }

    bool operator() (unsigned pos) const {
        auto& pack = packs_[pos / 64];
        return pack & (1ul << (pos & 0x0000003F));
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

    unsigned operator* (const BitVec<SIZE>& other) const {
        unsigned sum = 0;
        for (unsigned i = 0; i < packs_.size(); ++i) {
            sum += __builtin_popcountll(packs_[i] & other.packs_[i]);
        }
        return sum;
    }

    template <unsigned N>
    friend unsigned distance(const BitVec<N>& a, const BitVec<N>& b);

private:
    std::array<uint64_t, 1 + (SIZE - 1) / 64> packs_;
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

