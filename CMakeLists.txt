cmake_minimum_required(VERSION 2.8.12)
project(dmcore)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pedantic -pedantic-errors -Wall -Wextra")

#file(GLOB_RECURSE header_files FOLLOW_SYMLINKS "include/*.hpp")
file(GLOB_RECURSE source_files FOLLOW_SYMLINKS "src/*.cpp")

include_directories("include")

add_library(dmcore SHARED ${source_files})
add_executable(dmcore-bin src/bin/main.cpp)
set_target_properties(dmcore-bin PROPERTIES OUTPUT_NAME dmcore)
target_link_libraries(dmcore-bin dmcore)

install(TARGETS dmcore LIBRARY DESTINATION lib)

enable_testing()
