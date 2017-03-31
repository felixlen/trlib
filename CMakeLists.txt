cmake_minimum_required (VERSION 2.8.11)
project (TRLIB)

set (TRLIB_VERSION_MAJOR 0)
set (TRLIB_VERSION_MINOR 1)

option(TRLIB_MEASURE_TIME OFF)
option(TRLIB_MEASURE_SUBTIME OFF)
option(TRLIB_TEST_PLOT OFF)
option(TRLIB_BUILD_PYTHON2 OFF)
option(TRLIB_BUILD_PYTHON3 OFF)
option(TRLIB_BUILD_MEX OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(bindings)
add_subdirectory(examples)
enable_testing()
add_subdirectory(tests)
add_subdirectory(doc)
