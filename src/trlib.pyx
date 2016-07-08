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
        ttiming = np.zeros([ctrlib.trlib_krylov_timing_size()], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    cdef long ret, action, iter, ityp
    cdef double flt1, flt2, flt3
    eprefix = prefix.encode('UTF-8')
    ret = ctrlib.trlib_krylov_min(init, radius, 1 if equality else 0, itmax, itmax_lanczos,
            tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b, zero, ctl_invariant, g_dot_g, v_dot_g, p_dot_Hp,
            &iwork[0] if iwork.shape[0] > 0 else NULL, &fwork[0] if fwork.shape[0] > 0 else NULL,
            1 if refine else 0, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout,
            &timing_b[0] if timing_b.shape[0] > 0 else NULL, &action, &iter, &ityp, &flt1, &flt2, &flt3)
    if timing is None:
        return ret, action, iter, ityp, flt1, flt2, flt3
    else:
        return ret, action, iter, ityp, flt1, flt2, flt3, timing

def krylov_timing_size():
    return ctrlib.trlib_krylov_timing_size()

def leftmost_irreducible(double[::1] diag, double [::1] offdiag,
        long warm, double leftmost_minor, long itmax = 500, double tol_abs = np.finfo(np.float).eps**.75,
        long verbose=0, long [::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([ctrlib.trlib_krylov_timing_size()], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    eprefix = prefix.encode('UTF-8')
    cdef double leftmost
    cdef long iter_pr, ret
    ret = ctrlib.trlib_leftmost_irreducible(diag.shape[0], &diag[0] if diag.shape[0] > 0 else NULL, &offdiag[0] if offdiag.shape[0] > 0 else NULL, warm, leftmost_minor, itmax, tol_abs, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout, &timing_b[0], &leftmost, &iter_pr)
    return ret, leftmost, iter_pr

def tri_min(long [::1] irblk, double [::1] diag, double [::1] offdiag,
        double [::1] neglin, double radius, long itmax = 500,
        double tol_rel = np.finfo(np.float).eps, long pos_def = False,
        equality = False, warm0 = False, double lam0 = 0.0, warm = False, double lam = 0.0,
        warm_leftmost = False,long ileftmost = 0, double [::1] leftmost = None,
        warm_fac0 = False, double [::1] diag_fac0 = None, double [::1] offdiag_fac0 = None,
        warm_fac = False, double [::1] diag_fac = None, double [::1] offdiag_fac = None,
        double [::1] sol0 = None, double [::1] sol = None, double [::1] fwork = None,
        refine = True, long verbose = 0, long [::1] timing = None, prefix = ""):

    cdef double [:] leftmost_b, diag_fac0_b, offdiag_fac0_b, diag_fac_b, offdiag_fac_b, sol0_b, sol_b, fwork_b, ones_b
    cdef long [:] timing_b
    tones = np.ones(diag.shape[0])
    ones_b = tones
    if leftmost is None:
        tleftmost = np.zeros(irblk.shape[0])
        leftmost_b = tleftmost
    else:
        leftmost_b = leftmost
    if diag_fac0 is None:
        tdiag_fac0 = np.zeros(irblk[1])
        diag_fac0_b = tdiag_fac0
    else:
        diag_fac0_b = diag_fac0
    if offdiag_fac0 is None:
        toffdiag_fac0 = np.zeros(irblk[1]-1)
        offdiag_fac0_b = toffdiag_fac0
    else:
        offdiag_fac0_b = offdiag_fac0
    if diag_fac is None:
        tdiag_fac = np.zeros(diag.shape[0])
        diag_fac_b = tdiag_fac
    else:
        diag_fac_b = diag_fac
    if offdiag_fac is None:
        toffdiag_fac = np.zeros(offdiag.shape[0])
        offdiag_fac_b = toffdiag_fac
    else:
        offdiag_fac_b = offdiag_fac
    if sol0 is None:
        tsol0 = np.zeros(irblk[1])
        sol0_b = tsol0
    else:
        sol0_b = sol0
    if sol is None:
        tsol = np.zeros(diag.shape[0])
        sol_b = tsol
    else:
        sol_b = sol
    if fwork is None:
        tfwork = np.zeros([ctrlib.trlib_tri_factor_memory_size(diag.shape[0])])
        fwork_b = tfwork
    else:
        fwork_b = fwork
    if timing is None:
        ttiming = np.zeros([ctrlib.trlib_tri_timing_size()], dtype=np.int)
        timing_b = ttiming
    else:
        timing_b = timing
    eprefix = prefix.encode('UTF-8')
    cdef long ret, iwarm0, iwarm, iwarm_fac0, iwarm_fac, iwarm_leftmost, iter_newton, sub_fail
    cdef double obj
    iwarm0 = 1 if warm0 else 0
    iwarm = 1 if warm else 0
    iwarm_fac0 = 1 if warm_fac0 else 0
    iwarm_fac = 1 if warm_fac else 0
    iwarm_leftmost = 1 if warm_leftmost else 0

    ret = ctrlib.trlib_tri_factor_min(irblk.shape[0]-1, &irblk[0],
            &diag[0] if diag.shape[0] > 0 else NULL, &offdiag[0] if offdiag.shape[0] > 0 else NULL,
            &neglin[0] if neglin.shape[0] > 0 else NULL, radius, itmax, tol_rel, 1 if pos_def else 0,
            1 if equality else 0, &iwarm0, &lam0, &iwarm, &lam, &iwarm_leftmost, &ileftmost,
            &leftmost_b[0] if leftmost_b.shape[0] > 0 else NULL, &iwarm_fac0,
            &diag_fac0_b[0] if diag_fac0_b.shape[0] > 0 else NULL, &offdiag_fac0_b[0] if offdiag_fac0_b.shape[0] > 0 else NULL,
            &iwarm_fac, &diag_fac_b[0] if diag_fac_b.shape[0] > 0 else NULL, &offdiag_fac_b[0] if offdiag_fac_b.shape[0] > 0 else NULL,
            &sol0_b[0] if sol0_b.shape[0] > 0 else NULL, &sol_b[0] if sol_b.shape[0] > 0 else NULL,
            &ones_b[0] if ones_b.shape[0] > 0 else NULL, &fwork_b[0] if fwork_b.shape[0] > 0 else NULL,
            1 if refine else 0, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout, &timing_b[0] if timing_b.shape[0] else NULL,
            &obj, &iter_newton, &sub_fail)

    return ret, obj, sol, True if iwarm0==1 else False, lam0, True if iwarm==1 else False, lam, \
        True if iwarm_leftmost==1 else False, ileftmost, leftmost
