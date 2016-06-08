import numpy as np
cimport ctrlib
cimport libc.stdio
cimport numpy as np

cdef class Callback:
    cdef public object hvfcn
    
    def __cinit__(self, hvfcn):
        self.hvfcn = hvfcn

cdef void hvfcn_cb(void *userdata, long n, double *d, double* hv):
    cdef object self = <object> userdata
    _d = np.asarray(<np.float64_t[:n]> d)
    _hv = np.asarray(<np.float64_t[:n]> hv)
    _hv[:] = self.hvfcn(_d)

def krylov_prepare_memory(long itmax, double[::1] fwork not None):
    return ctrlib.trlib_krylov_prepare_memory(itmax, &fwork[0] if fwork.shape[0] > 0 else NULL)

def krylov_memory_size(long itmax):
    cdef long iwork_size, fwork_size, h_pointer
    ctrlib.trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size, &h_pointer)
    return iwork_size, fwork_size, h_pointer

def krylov_min(long init, double radius, double g_dot_g, double v_dot_g, double p_dot_Hp,
        long[::1] iwork not None, double [::1] fwork not None,
        equality = False, long itmax = 500, long itmax_lanczos = 100,
        long ctl_invariant=0,
        double tol_rel_i = np.finfo(np.float).eps**.5, double tol_abs_i = 0.0,
        double tol_rel_b = np.finfo(np.float).eps**.3, double tol_abs_b = 0.0,
        double zero = np.finfo(np.float).eps, long verbose=0, refine = True,
        long[::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([20], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    cdef long ret, action, iter, ityp
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

def leftmost_irreducible(double[::1] diag, double [::1] offdiag,
        long warm, double leftmost_minor, long itmax = 500, double tol_abs = np.finfo(np.float).eps**.75,
        long verbose=0, long [::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([20], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    eprefix = prefix.encode('UTF-8')
    cdef double leftmost
    cdef long iter_pr, ret
    ret = ctrlib.trlib_leftmost_irreducible(diag.shape[0], &diag[0] if diag.shape[0] > 0 else NULL, &offdiag[0] if offdiag.shape[0] > 0 else NULL, warm, leftmost_minor, itmax, tol_abs, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout, &timing_b[0], &leftmost, &iter_pr)
    return ret, leftmost, iter_pr


def krylov_timing_size():
    return ctrlib.trlib_krylov_timing_size()
