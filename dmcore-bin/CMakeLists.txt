project(dmcore-bin)

add_executable(dmcore_bin main.cpp)
set_target_properties(dmcore_bin PROPERTIES OUTPUT_NAME dmcore)

target_link_libraries(dmcore_bin dmcore armadillo mlpack)
