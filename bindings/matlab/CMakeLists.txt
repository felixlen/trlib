if(TRLIB_BUILD_MEX)
    find_package(Matlab)
    if(Matlab_FOUND)
        include_directories("${PROJECT_BINARY_DIR}/include")
        matlab_add_mex(NAME mex_trlib SRC mex_trlib.c LINK_TO trlib)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib.m"                "${CMAKE_CURRENT_BINARY_DIR}/trlib.m")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_options.m"        "${CMAKE_CURRENT_BINARY_DIR}/trlib_options.m")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_problem.m"        "${CMAKE_CURRENT_BINARY_DIR}/trlib_problem.m")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_solve.m"          "${CMAKE_CURRENT_BINARY_DIR}/trlib_solve.m")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trlib_set_hotstart.m"   "${CMAKE_CURRENT_BINARY_DIR}/trlib_set_hotstart.m")
    endif(Matlab_FOUND)
endif(TRLIB_BUILD_MEX)

add_custom_target( mex_zip 
    COMMAND rm -rf trlib_mex.zip rm -rf trlib_mex && cp -r ../../sources trlib_mex && rm -r trlib_mex/trlib/trlib_types.h && cp ../../../include/trlib_types_mex.h trlib_mex/trlib/trlib_types.h && cp -r ../../../bindings/matlab/* trlib_mex && rm -r trlib_mex/CMakeLists.txt && sed -i 's,plhs \\[0\\] = \(mxArray \\*\) prhs \\[1\\]\\;,// plhs \\[0\\] = \(mxArray \\*\) prhs \\[1\\]\\;,g' trlib_mex/mex_trlib.c && sed -i 's,// plhs \\[0\\] = mxDuplicateArray\(prhs\\[1\\]\)\;,plhs \\[0\\] = mxDuplicateArray\(prhs\\[1\\]\)\;,g' trlib_mex/mex_trlib.c && zip -r trlib_mex.zip trlib_mex && rm -rf trlib_mex
   )
