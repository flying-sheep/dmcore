cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dmcore)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
