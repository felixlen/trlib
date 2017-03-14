find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_private.h.in" trlib_private.h)

add_library(trlib SHARED 
    ${CMAKE_SOURCE_DIR}/src/trlib_krylov.c 
    ${CMAKE_SOURCE_DIR}/src/trlib_tri_factor.c 
    ${CMAKE_SOURCE_DIR}/src/trlib_leftmost.c 
    ${CMAKE_SOURCE_DIR}/src/trlib_eigen_inverse.c 
    ${CMAKE_SOURCE_DIR}/src/trlib_quadratic_zero.c
    trlib_private.h)
target_include_directories(trlib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(trlib PUBLIC "${CMAKE_BINARY_DIR}/include")
target_link_libraries(trlib ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
set_property(TARGET trlib PROPERTY C_STANDARD 11)
install(TARGETS trlib DESTINATION lib)

find_package(HDF5)
find_package(SuiteSparse OPTIONAL_COMPONENTS CHOLMOD)

#if(HDF5_FOUND AND SuiteSparse_FOUND)
#    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_driver.c.in" "${CMAKE_CURRENT_BINARY_DIR}/trlib_driver.c")
#    if(APPLE)
#	set (ARGP_STATIC_LIBRARY "/opt/local/lib/libargp.a")
#    else(APPLE)
#	set (ARGP_STATIC_LIBRARY "")
#    endif(APPLE)
#    add_executable(driver ${CMAKE_CURRENT_BINARY_DIR}/trlib_driver.c)
#    set_property(TARGET driver PROPERTY C_STANDARD 11)
#    target_include_directories(driver PRIVATE ${HDF5_INCLUDE_DIRS} ${SuiteSparse_INCLUDE_DIRS})
#    #message(${SuiteSparse_LIBRARIES})
#    target_link_libraries(driver ${ARGP_STATIC_LIBRARY} trlib ${HDF5_C_LIBRARIES} ${SuiteSparse_LIBRARIES})
#endif(HDF5_FOUND AND SuiteSparse_FOUND)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib.pyx" ${CMAKE_CURRENT_BINARY_DIR})

#foreach(PY_VER "2" "3")
foreach(PY_VER "2")
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHON_INCLUDE_DIR CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(PYTHON_LIBRARY_DEBUG CACHE)
    unset(CYTHON)
    find_package(PythonInterp ${PY_VER} EXACT)
    if(PYTHONINTERP_FOUND)
        find_package(PythonLibs ${PY_VER} EXACT)
        if(PY_VER EQUAL "2")
            execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_vars()['SO'])" OUTPUT_VARIABLE PY_SO)
            set(CYTHON "cython")
        elseif(PY_VER EQUAL "3")
            execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_vars()['EXT_SUFFIX'])" OUTPUT_VARIABLE PY_SO)
            set(CYTHON "cython3")
        endif(PY_VER EQUAL "2")
        string(STRIP ${PY_SO} PY_SO)
	
	if(APPLE)
        	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys, numpy; sys.stdout.write(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_PATH)
	else(APPLE)
		set (NUMPY_INCLUDE_PATH "")
	endif(APPLE)
        
	add_custom_target("pytrlib${PY_VER}" ALL DEPENDS "_trlib${PY_VER}.c")
        add_custom_command(OUTPUT "_trlib${PY_VER}.c" COMMAND ${CYTHON} trlib.pyx -o "_trlib${PY_VER}.c" DEPENDS trlib.pyx "${CMAKE_CURRENT_BINARY_DIR}/ctrlib.pxd")
        add_library("_trlib${PY_VER}" SHARED "_trlib${PY_VER}.c")
        target_include_directories("_trlib${PY_VER}" PRIVATE ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_PATH})
        target_link_libraries("_trlib${PY_VER}" trlib ${LIBS} ${PYTHON_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lib_trlib${PY_VER}.so" DESTINATION ${CMAKE_BINARY_DIR} RENAME trlib${PY_SO})
    endif(PYTHONINTERP_FOUND)
endforeach(PY_VER)
