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

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_test.h.in" "${CMAKE_CURRENT_BINARY_DIR}/trlib_test.h")

    set(LIBS ${LIBS} ${CHECK_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} pthread)
    set(TESTNAMES 
        quadratic_zero 
        eigen_inverse
        leftmost
        leftmost_irreducible
        factor_tri_simple_posdef
        factor_tri_simple_indef
        factor_tri_simple_indef_reg
        factor_tri_reducible
        factor_tri_expanding
        factor_tri_narrow_lam
        krylov_2d_simple
        krylov_3d_all_cases 
        krylov_tri_simple
        krylov_tri_diagonal_lanczos_trigger
        krylov_tri_diagonal_with_zeros
        krylov_lanczos_rosenbrock prepare_memory
        )
    foreach(t ${TESTNAMES})
        add_executable("test_${t}" "test_${t}.c" "${CMAKE_CURRENT_SOURCE_DIR}/trlib_test.c" ${MESSAGEFILES})
        set_property(TARGET "test_${t}" PROPERTY C_STANDARD 11)
        target_link_libraries("test_${t}" trlib ${LIBS})
        target_include_directories("test_${t}" PRIVATE "${CMAKE_BINARY_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}" ${CHECK_INCLUDE_DIRS})
        add_test("test_${t}" "${CMAKE_CURRENT_BINARY_DIR}/test_${t}")

        add_custom_command(OUTPUT "test_${t}_valgrind.c" COMMAND sed "'/sr = srunner_create(s);/a    srunner_set_fork_status(sr, CK_NOFORK);'" "${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.c" > "test_${t}_valgrind.c" DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.c")
        add_executable("test_${t}_valgrind" "test_${t}_valgrind.c" "${CMAKE_CURRENT_SOURCE_DIR}/trlib_test.c" ${MESSAGEFILES})
        set_property(TARGET "test_${t}_valgrind" PROPERTY C_STANDARD 11)
        target_link_libraries("test_${t}_valgrind" trlib ${LIBS})
        target_include_directories("test_${t}_valgrind" PRIVATE "${CMAKE_BINARY_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}" ${CHECK_INCLUDE_DIRS})
        add_test("test_${t}_valgrind" valgrind  --leak-check=full --show-leak-kinds=all --show-reachable=yes --error-exitcode=1 --errors-for-leak-kinds=all "${CMAKE_CURRENT_BINARY_DIR}/test_${t}_valgrind")
    endforeach(t)

endif(CHECK_FOUND)
