#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace meta {

template <int, int, typename, bool>
class tuple_view;

template <typename Tuple>
struct tup_size {
    static const size_t value =
        tup_size<
            typename std::remove_const<
                typename std::remove_reference<Tuple>::type>::type>::value;
};

template <typename... Ts>
struct tup_size<std::tuple<Ts...>> {
    static const size_t value = std::tuple_size<std::tuple<Ts...>>::value;
};

template <int start
         ,int end
         ,typename Tuple>
struct tup_size<tuple_view<start, end, Tuple, false>> {
    static const size_t value = end - start;
};

template <int start
         ,int end
         ,typename Tuple>
struct tup_size<tuple_view<start, end, Tuple, true>> {
    static const size_t value = start - end;
};


struct tup_reverse_impl {
    template <int start
             ,int end
             ,typename Tuple
             ,bool reverse>
    static auto make(tuple_view<start, end, Tuple, reverse>&& t) {
        static const int delta = reverse ? 1 : -1;
        return tuple_view<end + delta, start + delta, Tuple, !reverse>(
                t.tuple_);
    }

    template <int start
             ,int end
             ,typename Tuple
             ,bool reverse>
    static auto make(const tuple_view<start, end, Tuple, reverse>& t) {
        static const int delta = reverse ? 1 : -1;
        return tuple_view<end + delta, start + delta, Tuple, !reverse>(
                t.tuple_);
    }

    template <int start
             ,int end
             ,typename Tuple
             ,bool reverse>
    static auto make(tuple_view<start, end, Tuple, reverse>& t) {
        static const int delta = reverse ? 1 : -1;
        return tuple_view<end + delta, start + delta, Tuple, !reverse>(
                t.tuple_);
    }

    template <typename... Ts>
    static auto make(std::tuple<Ts...>& t) {
        typedef std::tuple<Ts...> Tuple;
        return tuple_view<
            std::tuple_size<Tuple>::value - 1, -1, Tuple, true>(t);
    }

    template <typename... Ts>
    static auto make(const std::tuple<Ts...>& t) {
        typedef const std::tuple<Ts...> Tuple;
        return tuple_view<
            std::tuple_size<Tuple>::value - 1, -1, Tuple, true>(t);
    }

};

template <int start
         ,int end
         ,typename Tuple
         ,bool reverse = false>
class tuple_view {
public:
    tuple_view(tuple_view&&) = default;
    explicit tuple_view(Tuple& tuple) : tuple_(tuple) {}

    template <size_t pos>
    auto&& get() const {
        return std::get<start + pos>(tuple_);
    }

    auto tail() const {
        static_assert(start < end, "Tail of an empty tuple is undefined");
        return tuple_view<start + 1, end, Tuple, reverse>(tuple_);
    }

private:
    friend struct tup_reverse_impl;

    Tuple& tuple_;
};

template <int start
         ,int end
         ,typename Tuple>
class tuple_view<start, end, Tuple, true> {
public:
    tuple_view(tuple_view&&) = default;
    explicit tuple_view(Tuple& tuple) : tuple_(tuple) {}

    template <size_t pos>
    auto&& get() const {
        return std::get<start - pos>(tuple_);
    }

    auto tail() const {
        static_assert(start > 0, "Tail of an empty tuple is undefined");
        return tuple_view<start - 1, end, Tuple, true>(tuple_);
    }

private:
    friend struct tup_reverse_impl;

    Tuple& tuple_;
};


template <typename Tuple>
auto tup_tail(Tuple&& t) {
    static_assert(tup_size<Tuple>::value > 0,
        "Tail of an empty tuple is undefined");
    return tuple_view<1, tup_size<Tuple>::value, Tuple>(
            std::forward<Tuple>(t));
}

template <int start
         ,int end
         ,typename Tuple
         ,bool reverse>
auto tup_tail(tuple_view<start, end, Tuple, reverse>&& tupView) {
    return tupView.tail();
}

template <typename Tuple>
auto tup_reverse(Tuple&& t) {
    return tup_reverse_impl::make(t);
}

template <size_t pos
         ,int start
         ,int end
         ,typename Tuple
         ,bool reverse>
const auto& get(const tuple_view<start, end, Tuple, reverse>& t) {
    return t.template get<pos>();
}

template <size_t pos
         ,int start
         ,int end
         ,typename Tuple
         ,bool reverse>
auto&& get(tuple_view<start, end, Tuple, reverse>& t) {
    return t.template get<pos>();
}

template <size_t pos
         ,int start
         ,int end
         ,typename Tuple
         ,bool reverse>
auto&& get(tuple_view<start, end, Tuple, reverse>&& t) {
    return t.template get<pos>();
}


template <size_t pos, typename... Ts>
const auto& get(const std::tuple<Ts...>& t) {
    return std::get<pos>(t);
}

template <size_t pos, typename... Ts>
auto& get(std::tuple<Ts...>& t) {
    return std::get<pos>(t);
}


template <typename... Ts>
auto&& tup_last(std::tuple<Ts...>& tuple) {
    return std::get<std::tuple_size<std::tuple<Ts...>>::value - 1>(tuple);
}

namespace {

template <typename F
         ,size_t revPos>
struct tup_each_perform {
    template <typename Tuple
             ,typename... Tuples>
    void operator() (F f, Tuple&& t, Tuples&&... ts) const {
        typedef typename std::remove_reference<Tuple>::type BareTuple;
        f(meta::get<tup_size<BareTuple>::value - revPos>(
                    std::forward<Tuple>(t)),
          meta::get<tup_size<BareTuple>::value - revPos>(
                    std::forward<Tuples>(ts))...);
        tup_each_perform<F, revPos - 1>()(
                f, std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
    }
};

template <typename F>
struct tup_each_perform<F, 0> {
    template <typename... Tuple>
    void operator() (F, Tuple&&...) const {}
};

}

template <typename F
         ,typename Tuple
         ,typename... Tuples>
void tup_each(F f, Tuple&& t, Tuples&&... ts) {
    typedef typename std::remove_reference<Tuple>::type BareTuple;
    tup_each_perform<F, tup_size<BareTuple>::value>()(
            f, std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
}

} // namespace meta

