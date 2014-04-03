#pragma once

#include <meta/tuple.h>

namespace meta {

namespace detail {

template <typename Tag, typename Value>
struct Param {
    Value value;
};

template <typename Tag>
struct find {
    template <typename Value, typename OtherParams>
    Value operator() (
            const Param<Tag, Value>& param, const OtherParams&) const {
        return param.value;
    }

    template <typename Param, typename OtherParams>
    auto operator() (const Param&, const OtherParams& otherParams) const {
        static_assert(meta::tup_size<OtherParams>::value > 0,
                "Tag not found");
        return (*this)(meta::get<0>(otherParams), meta::tup_tail(otherParams));
    }
};

} // namespace detail

template <typename Tag, typename Value>
auto param(Value value) {
    return detail::Param<Tag, Value>{value};
}

template <typename Tag, typename... Params>
auto get(const std::tuple<Params...>& params) {
    return detail::find<Tag>()(meta::get<0>(params), meta::tup_tail(params));
}

} // namespace meta

