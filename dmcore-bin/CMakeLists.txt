project(dmcore-bin)

add_executable(dmcore-bin main.cpp)
set_target_properties(dmcore-bin PROPERTIES OUTPUT_NAME dmcore)

target_link_libraries(dmcore-bin dmcore armadillo mlpack)
