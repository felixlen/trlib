enable_testing()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Check)
if(CHECK_FOUND)
    find_package(Protobuf)
    set(MESSAGEFILES)
    if(PROTOBUF_FOUND)
        include_directories(${PROTOBUF_INCLUDE_DIRS})
        add_custom_target(matrix_msg ALL DEPENDS trlib_test_matrix_msg.pb-c.c trlib_test_matrix_msg_pb2.py )
        add_custom_command(OUTPUT trlib_test_matrix_msg.pb-c.c COMMAND protoc-c --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --c_out=${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/trlib_test_matrix_msg.proto")
        add_custom_command(OUTPUT trlib_test_matrix_msg_pb2.py COMMAND protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --python_out=${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/trlib_test_matrix_msg.proto")
        set(MESSAGEFILES ${MESSAGEFILES} trlib_test_matrix_msg.pb-c.c)
        set(LIBS ${LIBS} protobuf-c zmq)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/socketserver.py" ${CMAKE_CURRENT_BINARY_DIR})
    endif(PROTOBUF_FOUND) 

    set(LIBS ${LIBS} ${CHECK_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} pthread)
    set(TESTNAMES 
        quadratic_zero 
        eigen_inverse
        leftmost
        leftmost_irreducible
        factor_tri_simple_posdef
        factor_tri_simple_indef
        factor_tri_reducible
        factor_tri_expanding
        krylov_2d_simple
        krylov_3d_all_cases 
        krylov_tri_simple
        krylov_tri_diagonal_lanczos_trigger
        krylov_tri_diagonal_with_zeros
        krylov_lanczos_rosenbrock prepare_memory
        )
    foreach(t ${TESTNAMES})
        add_executable("test_${t}" "test_${t}.c" ${MESSAGEFILES})
        set_property(TARGET "test_${t}" PROPERTY C_STANDARD 11)
        target_link_libraries("test_${t}" trlib ${LIBS})
        target_include_directories("test_${t}" PRIVATE "${CMAKE_BINARY_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}" ${CHECK_INCLUDE_DIRS})
        add_test("test_${t}" "${CMAKE_CURRENT_BINARY_DIR}/test_${t}")
    endforeach(t)
endif(CHECK_FOUND)
