#pragma once

#include <tuple>
#include <type_traits>

namespace meta {

namespace {

template <typename A, typename B>
class Any {
public:
    Any(A a, B b) : a_(a), b_(b) {}

    template <typename... Args>
    bool operator() (Args... args) {
        return a_(std::forward<Args>(args)...) or
            b_(std::forward<Args>(args)...);
    }

private:
    A a_;
    B b_;
};

template <typename A, typename B>
class All {
public:
    All(A a, B b) : a_(a), b_(b) {}

    template <typename... Args>
    bool operator() (Args... args) {
        return a_(std::forward<Args>(args)...) and
            b_(std::forward<Args>(args)...);
    }

private:
    A a_;
    B b_;
};

} // namespace

template <typename A, typename B>
auto any(A a, B b) {
    return Any<A, B>(a, b);
}

template <typename A, typename B>
auto all(A a, B b) {
    return All<A, B>(a, b);
}

} // namespace meta

