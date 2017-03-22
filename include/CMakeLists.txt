configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib.h.in"                "${CMAKE_CURRENT_BINARY_DIR}/trlib.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_types.h.in"          "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_types.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_krylov.h.in"         "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_krylov.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_tri_factor.h.in"     "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_tri_factor.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_leftmost.h.in"       "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_leftmost.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_quadratic_zero.h.in" "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_quadratic_zero.h")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_eigen_inverse.h.in"  "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_eigen_inverse.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/trlib.h" DESTINATION include )
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_types.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_krylov.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_tri_factor.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_leftmost.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_quadratic_zero.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_eigen_inverse.h"
                DESTINATION include/trlib )

# upon request collect all files needed for compilation in folder sources
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/trlib.h" DESTINATION "${CMAKE_BINARY_DIR}/sources" )
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_types.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_krylov.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_tri_factor.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_leftmost.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_quadratic_zero.h"
                "${CMAKE_CURRENT_BINARY_DIR}/trlib/trlib_eigen_inverse.h"
                DESTINATION "${CMAKE_BINARY_DIR}/sources/trlib" )
