cmake_minimum_required (VERSION 2.8.11)
project (TRLIB)

set (TRLIB_VERSION_MAJOR 0)
set (TRLIB_VERSION_MINOR 1)

option(TRLIB_MEASURE_TIME OFF)
option(TRLIB_MEASURE_SUBTIME OFF)
option(TRLIB_TEST_OUTPUT OFF)
option(TRLIB_TEST_PLOT OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

add_subdirectory(include)
add_subdirectory(src)
enable_testing()
add_subdirectory(tests)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file("${CMAKE_SOURCE_DIR}/Doxyfile.in" "${CMAKE_BINARY_DIR}/Doxyfile")
    add_custom_target(doc ${DOXYGEN_EXECUTABLE} DEPENDS "${CMAKE_BINARY_DIR}/Doxyfile" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(DOXYGEN_FOUND)
