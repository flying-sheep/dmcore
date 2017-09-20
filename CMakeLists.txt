cmake_minimum_required(VERSION 2.8.12)
project(dmcore)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Armadillo REQUIRED)
find_package(MLPACK REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS} ${MLPACK_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pedantic -pedantic-errors -Wall -Wextra")

#file(GLOB_RECURSE header_files FOLLOW_SYMLINKS "include/*.hpp")
file(GLOB_RECURSE source_files FOLLOW_SYMLINKS "src/*.cpp")

include_directories("include")

add_library(dmcore SHARED ${source_files})
add_executable(dmcore-bin src/bin/main.cpp)
set_target_properties(dmcore-bin PROPERTIES OUTPUT_NAME dmcore)
target_link_libraries(dmcore-bin dmcore ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARIES})

install(TARGETS dmcore LIBRARY DESTINATION lib)

enable_testing()
