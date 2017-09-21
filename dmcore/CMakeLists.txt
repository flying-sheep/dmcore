project(dmcore)

file(GLOB_RECURSE dmcore_header_files FOLLOW_SYMLINKS "include/*.hpp")
file(GLOB_RECURSE dmcore_source_files FOLLOW_SYMLINKS "*.cpp")

add_library(dmcore SHARED ${dmcore_source_files})
target_include_directories(dmcore PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

# install

install(TARGETS dmcore EXPORT dmcore_export
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

# public headers
install(FILES ${dmcore_header_files} DESTINATION include/dmcore)

# CMake export
install(EXPORT dmcore_export
	NAMESPACE dmcore::
	DESTINATION lib/cmake/dmcore
)

