find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} m)
add_executable("simple_qp_callback" "${CMAKE_CURRENT_SOURCE_DIR}/simple_qp_callback.c")
set_property(TARGET "simple_qp_callback" PROPERTY C_STANDARD 11)
target_link_libraries("simple_qp_callback" trlib ${LIBS})
target_include_directories("simple_qp_callback" PRIVATE "${CMAKE_BINARY_DIR}/include")


