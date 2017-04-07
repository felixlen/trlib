configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py"             "${CMAKE_CURRENT_BINARY_DIR}/conf.py")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/index.rst.in"        "${CMAKE_CURRENT_BINARY_DIR}/index.rst")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/installation.rst.in" "${CMAKE_CURRENT_BINARY_DIR}/installation.rst")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/matlab.rst.in"       "${CMAKE_CURRENT_BINARY_DIR}/matlab.rst")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python.rst.in"       "${CMAKE_CURRENT_BINARY_DIR}/python.rst")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cexample.rst.in"     "${CMAKE_CURRENT_BINARY_DIR}/cexample.rst")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/references.rst.in"   "${CMAKE_CURRENT_BINARY_DIR}/references.rst")

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_krylov.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_krylov.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_krylov.rst" "API trlib_krylov"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_krylov.h")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_tri_factor.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_tri_factor.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_tri_factor.rst" "API trlib_tri_factor"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_tri_factor.h")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_leftmost.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_leftmost.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_leftmost.rst" "API trlib_leftmost"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_leftmost.h")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_eigen_inverse.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_eigen_inverse.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_eigen_inverse.rst" "API trlib_eigen_inverse"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_eigen_inverse.h")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_quadratic_zero.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_quadratic_zero.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_quadratic_zero.rst" "API trlib_quadratic_zero"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_quadratic_zero.h")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capi_types.rst"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extractdoc.py" "${CMAKE_BINARY_DIR}/include/trlib/trlib_types.h" "${CMAKE_CURRENT_BINARY_DIR}/capi_types.rst" "API trlib_types"
    DEPENDS "${CMAKE_BINARY_DIR}/include/trlib/trlib_types.h")

add_custom_target(doc 
    COMMAND sphinx-build -b html -d "${CMAKE_CURRENT_BINARY_DIR}/doctrees" ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/html" 
    DEPENDS 
      "${CMAKE_CURRENT_BINARY_DIR}/conf.py" 
      "${CMAKE_CURRENT_BINARY_DIR}/index.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/installation.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/matlab.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/python.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_krylov.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_tri_factor.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_leftmost.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_eigen_inverse.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_quadratic_zero.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/capi_types.rst" 
      "${CMAKE_CURRENT_BINARY_DIR}/references.rst" 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

