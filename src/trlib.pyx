import numpy as np
cimport ctrlib
cimport libc.stdio
cimport numpy as np

cdef class Callback:
    cdef public object hvfcn
    
    def __cinit__(self, hvfcn):
        self.hvfcn = hvfcn

cdef void hvfcn_cb(void *userdata, int n, double *d, double* hv):
    cdef object self = <object> userdata
    _d = np.asarray(<np.float64_t[:n]> d)
    _hv = np.asarray(<np.float64_t[:n]> hv)
    _hv[:] = self.hvfcn(_d)

def krylov_prepare_memory(int itmax, double[::1] fwork not None):
    return ctrlib.trlib_krylov_prepare_memory(itmax, &fwork[0] if fwork.shape[0] > 0 else NULL)

def krylov_memory_size(int itmax):
    cdef int iwork_size, fwork_size, h_pointer
    ctrlib.trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size, &h_pointer)
    return iwork_size, fwork_size, h_pointer

def krylov_min(int init, double radius, double g_dot_g, double v_dot_g, double p_dot_Hp,
        int[::1] iwork not None, double [::1] fwork not None,
        equality = False, int itmax = 500, int itmax_lanczos = 100,
        int ctl_invariant=0,
        double tol_rel_i = np.finfo(np.float).eps**.5, double tol_abs_i = 0.0,
        double tol_rel_b = np.finfo(np.float).eps**.3, double tol_abs_b = 0.0,
        double zero = np.finfo(np.float).eps, int verbose=0, refine = True,
        long[::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([20], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    cdef int ret, action, iter, ityp
    cdef double flt1, flt2, flt3
    eprefix = prefix.encode('UTF-8')
    cdef char* cprefix = eprefix
    ret = ctrlib.trlib_krylov_min(init, radius, 1 if equality else 0, itmax, itmax_lanczos,
            tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b, zero, ctl_invariant, g_dot_g, v_dot_g, p_dot_Hp,
            &iwork[0] if iwork.shape[0] > 0 else NULL, &fwork[0] if fwork.shape[0] > 0 else NULL,
            1 if refine else 0, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout, &timing_b[0], &action, &iter, &ityp, &flt1, &flt2, &flt3)
    if timing is None:
        return ret, action, iter, ityp, flt1, flt2, flt3, ttiming
    else:
        return ret, action, iter, ityp, flt1, flt2, flt3
