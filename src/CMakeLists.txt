include(CMakePackageConfigHelpers)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_private.h.in" trlib_private.h)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib.pc.in"
  "${CMAKE_BINARY_DIR}/trlib.pc"
  @ONLY)

add_library(trlib SHARED
  ${CMAKE_SOURCE_DIR}/src/trlib_krylov.c
  ${CMAKE_SOURCE_DIR}/src/trlib_tri_factor.c
  ${CMAKE_SOURCE_DIR}/src/trlib_leftmost.c
  ${CMAKE_SOURCE_DIR}/src/trlib_eigen_inverse.c
  ${CMAKE_SOURCE_DIR}/src/trlib_quadratic_zero.c
  trlib_private.h)

# trlib_private.h
target_include_directories(trlib
  PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(trlib
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(trlib ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} m)
set_property(TARGET trlib PROPERTY C_STANDARD 11)

if(APPLE)
  set_property(TARGET trlib PROPERTY INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
endif(APPLE)

install(TARGETS trlib
  EXPORT trlib-config
  DESTINATION lib)

install(EXPORT trlib-config
  NAMESPACE "trlib::"
  DESTINATION "lib/cmake/trlib")

install(FILES "${CMAKE_BINARY_DIR}/trlib.pc"
  DESTINATION "share/pkgconfig")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/trlib_private.h"
  DESTINATION "include/trlib")

# upon request collect all files needed for compilation in folder sources
install ( FILES ${CMAKE_SOURCE_DIR}/src/trlib_krylov.c
                ${CMAKE_SOURCE_DIR}/src/trlib_tri_factor.c
                ${CMAKE_SOURCE_DIR}/src/trlib_leftmost.c
                ${CMAKE_SOURCE_DIR}/src/trlib_eigen_inverse.c
                ${CMAKE_SOURCE_DIR}/src/trlib_quadratic_zero.c
                ${CMAKE_BINARY_DIR}/src/trlib_private.h
                DESTINATION "share/trlib/src" )

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/trlib-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
)

install(
  FILES "${CMAKE_BINARY_DIR}/trlib-config-version.cmake"
  DESTINATION "lib/cmake/trlib"
)

#find_package(HDF5)
#find_package(SuiteSparse OPTIONAL_COMPONENTS CHOLMOD)
#if(HDF5_FOUND AND SuiteSparse_FOUND)
#    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_driver.c.in" "${CMAKE_CURRENT_BINARY_DIR}/trlib_driver.c")
#    if(APPLE)
#       set (ARGP_STATIC_LIBRARY "/opt/local/lib/libargp.a")
#    else(APPLE)
#       set (ARGP_STATIC_LIBRARY "")
#    endif(APPLE)
#    add_executable(driver ${CMAKE_CURRENT_BINARY_DIR}/trlib_driver.c)
#    set_property(TARGET driver PROPERTY C_STANDARD 11)
#    target_include_directories(driver PRIVATE ${HDF5_INCLUDE_DIRS} ${SuiteSparse_INCLUDE_DIRS})
#    #message(${SuiteSparse_LIBRARIES})
#    target_link_libraries(driver ${ARGP_STATIC_LIBRARY} trlib ${HDF5_C_LIBRARIES} ${SuiteSparse_LIBRARIES})
#endif(HDF5_FOUND AND SuiteSparse_FOUND)
