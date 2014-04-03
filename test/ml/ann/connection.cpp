#define BOOST_TEST_MODULE ml_ann_connection
#include <boost/test/included/unit_test.hpp>

#include <ml/ann/connection.h>
#include <Eigen/Dense>

BOOST_AUTO_TEST_CASE ( full_conn_transform ) {
    ml::ann::detail::FullConnection<5, 5> conn;
    Eigen::MatrixXd m55(5, 5);
    m55 << 1,-2,3,4,5,
           4,5,6,-7,8,
           2,-3,4,5,6,
           -4,6,8,3,8,
           1,6,4,7,-9;
    Eigen::VectorXd v51(5);
    v51 << 3,4,5,2,1;
    conn.initWith(m55, v51);
    Eigen::VectorXd input(5);
    input << 1,6,2,-2,-6;
    auto transformed = conn.transform(input);
    auto shouldBe = m55 * input + v51;
    const double tolerance = 1e-7;
    BOOST_CHECK_CLOSE(transformed(0), shouldBe(0), tolerance);
    BOOST_CHECK_CLOSE(transformed(1), shouldBe(1), tolerance);
    BOOST_CHECK_CLOSE(transformed(2), shouldBe(2), tolerance);
    BOOST_CHECK_CLOSE(transformed(3), shouldBe(3), tolerance);
    BOOST_CHECK_CLOSE(transformed(4), shouldBe(4), tolerance);
}

