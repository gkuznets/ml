#pragma once

namespace meta {

template <template <unsigned> class F
         ,unsigned start
         ,unsigned end
         ,unsigned revPos>
struct for_each_perform {
    void operator() (F f) const {
        f<start + end - revPos>();
        for_each_perform<F, start, end, revPos - 1>()(f);
    }
};

template <template <unsigned> class F
         ,unsigned start
         ,unsigned end>
struct for_each_perform<F, start, end, 0> {
    void operator() (F) const { }
};

template <typename F
         ,unsigned start
         ,unsigned end>
void for_each(F f) {
    for_each_perform<F, start, end, end>()(f);
}

} // namespace meta

