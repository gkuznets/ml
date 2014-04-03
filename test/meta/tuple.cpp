#include <tuple>
#include <meta/tuple.h>

#define BOOST_TEST_MODULE meta_tuple
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE ( tup_size ) {
    std::tuple<int, double, std::tuple<int, int>> t;
    size_t meta_size = meta::tup_size<decltype(t)>::value;
    size_t std_size = std::tuple_size<decltype(t)>::value;
    BOOST_CHECK_EQUAL(meta_size, std_size);
}

BOOST_AUTO_TEST_CASE ( tup_each_1 ) {
    std::tuple<int, int, int, int> t{1, 2, 3, 4};
    int sum = 0;
    meta::tup_each([&sum](int x) { sum += x; }, t);
    BOOST_CHECK_EQUAL( sum, 1 + 2 + 3 + 4 );
    meta::tup_each([](auto& x) { x *= 2; }, t);
    sum = 0;
    meta::tup_each([&sum](int x) { sum += x; }, t);
    BOOST_CHECK_EQUAL( sum, 2 * (1 + 2 + 3 + 4) );
}

BOOST_AUTO_TEST_CASE ( tup_each_2 ) {
    std::tuple<int, int, int, int> t1{1, 2, 3, 4};
    std::tuple<int, int, int, int> t2{1, 2, 3, 4};
    int sum = 0;
    meta::tup_each([&sum](int x1, int x2) { sum += x1 + x2; }, t1, t2);
    BOOST_CHECK_EQUAL( sum, 2 * (1 + 2 + 3 + 4) );

    meta::tup_each([](auto& x1, const auto& x2) { x1 = x2 * 2; }, t1, t2);
    meta::tup_each([](int x1, int x2) {
                    BOOST_CHECK_EQUAL( x1, 2 * x2 );
                    }, t1, t2);

    meta::tup_each([](const auto& x1, auto& x2) { x2 = x1 * 2; }, t1, t2);
    meta::tup_each([](int x1, int x2) {
                    BOOST_CHECK_EQUAL( 2 * x1, x2 );
                    }, t1, t2);
}

BOOST_AUTO_TEST_CASE ( tup_reverse ) {
    std::tuple<int, int, int, int> t{1, 2, 3, 4};
    auto rt = meta::tup_reverse(t);
    BOOST_CHECK_EQUAL(meta::get<0>(rt), meta::get<3>(t));
    BOOST_CHECK_EQUAL(meta::get<1>(rt), meta::get<2>(t));
    BOOST_CHECK_EQUAL(meta::get<2>(rt), meta::get<1>(t));
    BOOST_CHECK_EQUAL(meta::get<3>(rt), meta::get<0>(t));

    auto rrt = meta::tup_reverse(rt);
    BOOST_CHECK_EQUAL(meta::get<0>(rrt), meta::get<0>(t));
    BOOST_CHECK_EQUAL(meta::get<1>(rrt), meta::get<1>(t));
    BOOST_CHECK_EQUAL(meta::get<2>(rrt), meta::get<2>(t));
    BOOST_CHECK_EQUAL(meta::get<3>(rrt), meta::get<3>(t));
}

BOOST_AUTO_TEST_CASE ( tup_tail ) {
    auto t = std::make_tuple(1, 2.0, 3, 4);
    auto tl = meta::tup_tail(t);
    BOOST_CHECK_EQUAL(meta::get<0>(tl), meta::get<1>(t));
    BOOST_CHECK_EQUAL(meta::get<1>(tl), meta::get<2>(t));
    auto ttl = meta::tup_tail(meta::tup_tail(t));
    BOOST_CHECK_EQUAL(meta::get<0>(ttl), meta::get<2>(t));
    BOOST_CHECK_EQUAL(meta::get<1>(ttl), meta::get<3>(t));
}

BOOST_AUTO_TEST_CASE ( tup_reverse_each_1 ) {
    std::tuple<int, int, int, int> t1{1, 2, 3, 4}, t2{0, 0, 0, 0};
    tup_each([](auto& x1, const auto& x2) {
                x1 = x2;
            },
            meta::tup_reverse(t2), t1);
    BOOST_CHECK_EQUAL(std::get<0>(t1), std::get<3>(t2));
    BOOST_CHECK_EQUAL(std::get<1>(t1), std::get<2>(t2));
    BOOST_CHECK_EQUAL(std::get<2>(t1), std::get<1>(t2));
    BOOST_CHECK_EQUAL(std::get<3>(t1), std::get<0>(t2));
}

