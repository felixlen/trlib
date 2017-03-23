import numpy as np
import scipy.sparse.linalg
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
        double tol_rel = np.finfo(np.float).eps, double tol_newton_tiny = 1e-2*np.finfo(np.float).eps**.5,
        long pos_def = False, equality = False,
        warm0 = False, double lam0 = 0.0, warm = False, double lam = 0.0,
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
            &neglin[0] if neglin.shape[0] > 0 else NULL, radius, itmax, tol_rel, tol_newton_tiny,
            1 if pos_def else 0, 1 if equality else 0,
            &iwarm0, &lam0, &iwarm, &lam, &iwarm_leftmost, &ileftmost,
            &leftmost_b[0] if leftmost_b.shape[0] > 0 else NULL, &iwarm_fac0,
            &diag_fac0_b[0] if diag_fac0_b.shape[0] > 0 else NULL, &offdiag_fac0_b[0] if offdiag_fac0_b.shape[0] > 0 else NULL,
            &iwarm_fac, &diag_fac_b[0] if diag_fac_b.shape[0] > 0 else NULL, &offdiag_fac_b[0] if offdiag_fac_b.shape[0] > 0 else NULL,
            &sol0_b[0] if sol0_b.shape[0] > 0 else NULL, &sol_b[0] if sol_b.shape[0] > 0 else NULL,
            &ones_b[0] if ones_b.shape[0] > 0 else NULL, &fwork_b[0] if fwork_b.shape[0] > 0 else NULL,
            1 if refine else 0, verbose, 1, eprefix, <libc.stdio.FILE*> libc.stdio.stdout, &timing_b[0] if timing_b.shape[0] else NULL,
            &obj, &iter_newton, &sub_fail)

    return ret, obj, sol, True if iwarm0==1 else False, lam0, True if iwarm==1 else False, lam, \
        True if iwarm_leftmost==1 else False, ileftmost, leftmost

def trlib_solve(hess, grad, radius, invM = lambda x: x, data=None, reentry=False, verbose=0, ctl_invariant=0):
    if hasattr(hess, 'dot'):
        hmv = lambda x: hess.dot(x)
    else:
        hmv = hess
    itmax = 2*grad.shape[0]
    iwork_size, fwork_size, h_pointer = krylov_memory_size(itmax)
    if data is None:
        data = {}
    if not reentry:
        init = ctrlib._TRLIB_CLS_INIT
        if not 'fwork' in data:
            data['fwork'] = np.empty([fwork_size])
        krylov_prepare_memory(itmax, data['fwork'])
        if not 'iwork' in data:
            data['iwork'] = np.empty([iwork_size], dtype=np.int)
        if not 's' in data:
            data['s'] = np.empty(grad.shape)
        if not 'g' in data:
            data['g'] = np.empty(grad.shape)
        if not 'v' in data:
            data['v'] = np.empty(grad.shape)
        if not 'gm' in data:
            data['gm'] = np.empty(grad.shape)
        if not 'p' in data:
            data['p'] = np.empty(grad.shape)
        if not 'Hp' in data:
            data['Hp'] = np.empty(grad.shape)
        if not 'Q' in data:
            data['Q'] = np.empty([itmax+1, grad.shape[0]])
    else:
        if reentry != 'convex':
            init = ctrlib._TRLIB_CLS_HOTSTART
        else:
            init = ctrlib._TRLIB_CLS_HOTSTART_P

    v_dot_g = 0.0; g_dot_g = 0.0; p_dot_Hp = 0.0
    
    while True:
        ret, action, iter, ityp, flt1, flt2, flt3 = krylov_min(
            init, radius, g_dot_g, v_dot_g, p_dot_Hp, data['iwork'], data['fwork'],
            ctl_invariant=ctl_invariant, itmax=itmax, verbose=verbose)
        init = 0
        if action == ctrlib._TRLIB_CLA_INIT:
            data['s'][:] = 0.0
            data['gm'][:] = 0.0
            data['g'][:] = grad
            data['v'][:] = invM(data['g'])
            g_dot_g = np.dot(data['g'], data['g'])
            v_dot_g = np.dot(data['v'], data['g'])
            data['p'][:] = -data['v']
            data['Hp'][:] = hmv(data['p'])
            p_dot_Hp = np.dot(data['p'], data['Hp'])
            data['Q'][0,:] = data['v']/np.sqrt(v_dot_g)
        if action == ctrlib._TRLIB_CLA_RETRANSF:
            data['s'][:] = np.dot(data['fwork'][h_pointer:h_pointer+iter+1], data['Q'][:iter+1,:])
        if action == ctrlib._TRLIB_CLA_UPDATE_STATIO:
            if ityp == ctrlib._TRLIB_CLT_CG:
                data['s'] += flt1 * data['p']
        if action == 4: # CLA_UPDATE_GRAD
            if ityp == ctrlib._TRLIB_CLT_CG:
                data['Q'][iter,:] = flt2*data['v']
                data['gm'][:] = data['g']
                data['g'] += flt1*data['Hp']
            if ityp == ctrlib._TRLIB_CLT_L:
                data['s'][:] = data['Hp'] + flt1*data['g'] + flt2*data['gm']
                data['gm'][:] = flt3*data['g']
                data['g'][:] = data['s']
            data['v'][:] = invM(data['g'])
            g_dot_g = np.dot(data['g'], data['g'])
            v_dot_g = np.dot(data['v'], data['g'])
        if action == ctrlib._TRLIB_CLA_UPDATE_DIR:
            data['p'][:] = flt1 * data['v'] + flt2 * data['p']
            data['Hp'][:] = hmv(data['p'])
            p_dot_Hp = np.dot(data['p'], data['Hp'])
            if ityp == ctrlib._TRLIB_CLT_L:
                data['Q'][iter,:] = data['p']
        if action == ctrlib._TRLIB_CLA_NEW_KRYLOV:
            # FIXME: adapt for M != I
            QQ, RR = np.linalg.qr(data['Q'][:iter,:].transpose(), 'complete')
            data['g'][:] = QQ[:,iter]
            data['gm'][:] = 0.0
            data['v'][:] = invM(data['g'])
            g_dot_g = np.dot(data['g'], data['g'])
            v_dot_g = np.dot(data['v'], data['g'])
            data['p'][:] = data['v']/np.sqrt(v_dot_g)
            data['Hp'][:] = hmv(data['p'])
            p_dot_Hp = np.dot(data['p'], data['Hp'])
            data['Q'][iter,:] = data['p']
        if action == ctrlib._TRLIB_CLA_CONV_HARD:
            # FIXME: adapt for M != I
            Ms = data['s']
            Hs = hmv(data['s'])
            g_dot_g = np.dot(data['g'], data['g'])
            v_dot_g = np.dot(Hs+grad+flt1*Ms, invM(Hs+grad)+flt1*data['s'])
        if ret < 10:
            break
    if ret < 0:
        print("Warning, status: %d" % ret, data['iwork'][7], data['iwork'][8])
        pass
    data['ret'] = ret
    data['iter'] = iter
    return data['s'], data
