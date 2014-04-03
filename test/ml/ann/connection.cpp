#define BOOST_TEST_MODULE ml_ann_connection
#include <boost/test/included/unit_test.hpp>

#include <ml/ann/connection.h>
#include <Eigen/Dense>

BOOST_AUTO_TEST_CASE ( full_conn_transform ) {
    ml::ann::detail::FullConnection<5, 5> conn;
}

