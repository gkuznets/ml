cmake_minimum_required (VERSION 2.6)
project (ml)
set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories (. /usr/local/include)

set (Boost_USE_MULTITHREADED OFF)
find_package (Boost REQUIRED COMPONENTS program_options)
include_directories (${Boost_INCLUDE_DIR})

find_package (Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIR})

find_package (Qt4 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set (CMAKE_CXX_COMPILER clang++)
set (CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wunreachable-code -Werror --std=c++11 --std=c++1y -stdlib=libc++")

add_executable (minst
    minst.cpp
    read_minst.cpp)
target_link_libraries (minst
    ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_executable (autoenc
    autoenc.cpp
    read_minst.cpp)
set_target_properties (autoenc PROPERTIES COMPILE_FLAGS -Wno-deprecated-register)
target_link_libraries (autoenc
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${QT_LIBRARIES})

enable_testing()
add_subdirectory (test)
