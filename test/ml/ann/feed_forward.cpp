#include <meta/tuple.h>
#include <ml/ann.h>
#include <ml/ann/activations.h>
#include <ml/ann/connection.h>
#include <ml/ann/feed_forward.h>

#include <Eigen/Dense>

#define BOOST_TEST_MODULE ml_ann_feed_forward
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE ( simple_feed_fwd ) {
    typedef ml::ann::NetworkConf<
        ml::ann::Input<5>,
        ml::ann::FullyConnected<5, ml::ann::Linear>,
        ml::ann::FullyConnected<5, ml::ann::Linear>> NetConf;
    NetConf::Connections connections;
    Eigen::VectorXd input(5);
    input.fill(0.0);
    meta::tup_each([](auto& conn) { conn.zero(); }, connections);
    auto result = ml::ann::detail::feedForward(
            input, connections, NetConf::Layers{});
    BOOST_CHECK_EQUAL(result.size(), 5);
    BOOST_CHECK_EQUAL(result(0), 0.0);
    BOOST_CHECK_EQUAL(result(1), 0.0);
    BOOST_CHECK_EQUAL(result(2), 0.0);
    BOOST_CHECK_EQUAL(result(3), 0.0);
    BOOST_CHECK_EQUAL(result(4), 0.0);

    Eigen::MatrixXd m55(5, 5);
    m55.fill(1.0);
    Eigen::VectorXd v51(5);
    v51.fill(1.0);
    meta::tup_each(
            [&m55, &v51](auto& conn){
                conn.initWith(m55, v51);
            }, connections);

    result = ml::ann::detail::feedForward(
            input, connections, NetConf::Layers{});
    auto shouldBe = m55 * (m55 * input + v51) + v51;
    const double tolerance = 1e-7;
    BOOST_CHECK_CLOSE(result(0), shouldBe(0), tolerance);
    BOOST_CHECK_CLOSE(result(1), shouldBe(1), tolerance);
    BOOST_CHECK_CLOSE(result(2), shouldBe(2), tolerance);
    BOOST_CHECK_CLOSE(result(3), shouldBe(3), tolerance);
    BOOST_CHECK_CLOSE(result(4), shouldBe(4), tolerance);
}


BOOST_AUTO_TEST_CASE ( full_feed_fwd ) {
    typedef ml::ann::NetworkConf<
        ml::ann::Input<5>,
        ml::ann::FullyConnected<5, ml::ann::Linear>,
        ml::ann::FullyConnected<5, ml::ann::Linear>> NetConf;
    NetConf::Connections connections;
    NetConf::Activations activations;
    Eigen::VectorXd input(5);
    input.fill(0.0);
    meta::tup_each([](auto& conn) { conn.zero(); }, connections);
    ml::ann::detail::feedForward(
            input, connections, activations, NetConf::Layers{});
    auto& result = std::get<2>(activations);
    BOOST_CHECK_EQUAL(result.size(), 5);
    BOOST_CHECK_EQUAL(result(0), 0.0);
    BOOST_CHECK_EQUAL(result(1), 0.0);
    BOOST_CHECK_EQUAL(result(2), 0.0);
    BOOST_CHECK_EQUAL(result(3), 0.0);
    BOOST_CHECK_EQUAL(result(4), 0.0);

    input << 1, 2, 3, 4, 5;
    Eigen::MatrixXd m55(5, 5);
    m55.fill(1.0);
    Eigen::VectorXd v51(5);
    v51.fill(1.0);
    meta::tup_each(
            [&m55, &v51](auto& conn){
                conn.initWith(m55, v51);
            }, connections);

    ml::ann::detail::feedForward(
           input, connections, activations, NetConf::Layers{});
    const auto& act1 = std::get<1>(activations);
    const auto& act2 = std::get<2>(activations);
    auto shouldBe1 = m55 * input + v51;
    auto shouldBe2 = m55 * act1 + v51;
    const double tolerance = 1e-7;
    BOOST_CHECK_CLOSE(act1(0), shouldBe1(0), tolerance);
    BOOST_CHECK_CLOSE(act1(1), shouldBe1(1), tolerance);
    BOOST_CHECK_CLOSE(act1(2), shouldBe1(2), tolerance);
    BOOST_CHECK_CLOSE(act1(3), shouldBe1(3), tolerance);
    BOOST_CHECK_CLOSE(act1(4), shouldBe1(4), tolerance);

    BOOST_CHECK_CLOSE(act2(0), shouldBe2(0), tolerance);
    BOOST_CHECK_CLOSE(act2(1), shouldBe2(1), tolerance);
    BOOST_CHECK_CLOSE(act2(2), shouldBe2(2), tolerance);
    BOOST_CHECK_CLOSE(act2(3), shouldBe2(3), tolerance);
    BOOST_CHECK_CLOSE(act2(4), shouldBe2(4), tolerance);
}

