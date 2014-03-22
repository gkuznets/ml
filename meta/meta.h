#pragma once

namespace meta {

struct none {};

template <typename... Ts>
struct list { };


template <typename List>
struct length {};

template <typename... Ts>
struct length<list<Ts...>> {
    static const unsigned value = sizeof...(Ts);
};


template <typename T
         ,typename List>
struct prepend_t {};

template <typename T
         ,typename... Ts>
struct prepend_t<T, list<Ts...>> {
    typedef list<T, Ts...> type;
};

template <typename T
         ,typename List>
using prepend = typename prepend_t<T, List>::type;


template <template <typename... T> class F
         ,typename List>
struct map_t {};

template <template <typename... T> class F
         ,typename... Ts>
struct map_t<F, list<Ts...>> {
    typedef list<F<Ts>...> type;
};

template <template <typename... T> class F
         ,typename List>
using map = typename map_t<F, List>::type;


template <typename List>
struct tail_t {};

template <typename T, typename... Ts>
struct tail_t<list<T, Ts...>> {
    typedef list<Ts...> type;
};

template <typename List>
using tail = typename tail_t<List>::type;


template <template <typename... As> class F
         ,typename List>
struct apply_t {};

template <template <typename... As> class F
         ,typename... Ts>
struct apply_t<F, list<Ts...>> {
    typedef F<Ts...> type;
};

// apply<F, list<A, B, C>> == F<A, B, C>
template <template <typename... As> class F
         ,typename List>
using apply = typename apply_t<F, List>::type;

template <typename ListA, typename ListB>
struct zip_t {};

template <typename ListA>
struct zip_t<ListA, list<>> {
    typedef list<> type;
};

template <typename ListB>
struct zip_t<list<>, ListB> {
    typedef list<> type;
};

template <>
struct zip_t<list<>, list<>> {
    typedef list<> type;
};

template <typename A
         ,typename... As
         ,typename B
         ,typename... Bs>
struct zip_t<list<A, As...>, list<B, Bs...>> {
    typedef prepend<
                list<A, B>,
                typename zip_t<list<As...>, list<Bs...>>::type> type;
};

template <typename ListA
         ,typename ListB>
using zip = typename zip_t<ListA, ListB>::type;


template <template <typename... T> class F
         ,typename ListA
         ,typename ListB>
struct zipWith_t {};

template <template <typename... T> class F
         ,typename ListA>
struct zipWith_t<F, ListA, list<>> {
    typedef list<> type;
};

template <template <typename... T> class F
         ,typename ListB>
struct zipWith_t<F, list<>, ListB> {
    typedef list<> type;
};

template <template <typename... T> class F>
struct zipWith_t<F, list<>, list<>> {
    typedef list<> type;
};

template <template <typename... T> class F
         ,typename A
         ,typename... As
         ,typename B
         ,typename... Bs>
struct zipWith_t<F, list<A, As...>, list<B, Bs...>> {
    typedef prepend<
                F<A, B>,
                typename zipWith_t<F, list<As...>, list<Bs...>>::type> type;
};

template <template <typename... T> class F
         ,typename ListA
         ,typename ListB>
using zipWith = typename zipWith_t<F, ListA, ListB>::type;


/*
template <unsigned>
struct placeholder {};

typedef placeholder<0> _0;

// map<F, list<L1, L2, L2>> == list<F<L1>, F<L2>, F<L3>>
// map<apply<F, _1>, list<L1, L2, L3>> == list<apply<F, L1>, apply<F, L2>, apply<F, L3>>
// map<bind<apply, F, _1>, list<L1, L2, L3>> == list
template <template <typename...> class F, typename... Ts>
struct bind {
    typedef F<Ts...> type;
};

template <template <typename> class F, unsigned N>
struct bind<F, placeholder<N>> {
    template <typename U>
    using type = F<U>;
};

template <template <typename...> class F
         ,typename T
         ,unsigned N>
struct bind<F, T, placeholder<N>> {
    template <typename U>
    using type = F<T, U>;
};
*/
} // namespace meta

