cmake_minimum_required (VERSION 2.6)
project (ml)
set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories (. /usr/local/include)

set (Boost_USE_MULTITHREADED OFF)
find_package (Boost REQUIRED COMPONENTS program_options)
include_directories (${Boost_INCLUDE_DIR})

set (CMAKE_CXX_COMPILER clang++)
set (CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wunreachable-code -Werror --std=c++11 --std=c++1y -stdlib=libc++")

add_executable (minst minst.cpp)
target_link_libraries (minst
    ${Boost_PROGRAM_OPTIONS_LIBRARY})

enable_testing()
add_subdirectory (test)
