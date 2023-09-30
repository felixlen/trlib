cmake_minimum_required (VERSION 3.1)
project (TRLIB VERSION 0.4 LANGUAGES C)

set(PROJECT_DESCRIPTION "Trust Region Subproblem Solver Library")

option(TRLIB_MEASURE_TIME OFF)
option(TRLIB_MEASURE_SUBTIME OFF)
option(TRLIB_TEST_PLOT OFF)
option(TRLIB_BUILD_PYTHON2 OFF)
option(TRLIB_BUILD_PYTHON3 OFF)
option(TRLIB_BUILD_MEX OFF)

option(TRLIB_BLAS_LAPACK_UNDERSCORE_SUFFIX OFF)

if(TRLIB_BLAS_LAPACK_UNDERSCORE_SUFFIX)
    add_compile_definitions(TRLIB_BLAS_LAPACK_UNDERSCORE_SUFFIX)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(bindings)
add_subdirectory(examples)
enable_testing()
add_subdirectory(tests)
add_subdirectory(doc)
