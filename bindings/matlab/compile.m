if strcmp(computer('arch'), 'win32') | strcmp(computer('arch'), 'win64')
    mex CFLAGS='$CFLAGS -std=c99' mex_trlib.c trlib_eigen_inverse.c trlib_leftmost.c trlib_quadratic_zero.c trlib_tri_factor.c trlib_krylov.c -lmwblas -lmwlapack -largeArrayDims
else
    mex CFLAGS='$CFLAGS -std=c99' mex_trlib.c trlib_eigen_inverse.c trlib_leftmost.c trlib_quadratic_zero.c trlib_tri_factor.c trlib_krylov.c -lmwblas -lmwlapack -largeArrayDims
end
