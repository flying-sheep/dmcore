cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(dmcore)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pedantic -pedantic-errors -Wall -Wextra")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libraries

find_package(Armadillo REQUIRED)
find_package(MLPACK REQUIRED)

# test

enable_testing()

# parts

add_subdirectory(dmcore)
add_subdirectory(dmcore-bin)
