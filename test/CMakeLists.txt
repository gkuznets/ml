add_executable (tuple
    meta/tuple.cpp)
add_test (tuple_test tuple)

add_executable (connection
    ml/ann/connection.cpp)
add_test (connection_test connection)

add_executable (feed_forward
    ml/ann/feed_forward.cpp)
add_test (feed_forward_test feed_forward)
