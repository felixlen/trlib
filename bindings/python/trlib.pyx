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
        long ctl_invariant=0, long convexify=1, long earlyterm=1,
        double tol_rel_i = np.finfo(np.float).eps**.5, double tol_abs_i = 0.0,
        double tol_rel_b = np.finfo(np.float).eps**.3, double tol_abs_b = 0.0,
        double zero = np.finfo(np.float).eps, double obj_lo=-1e20, long verbose=0, refine = True,
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
            tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b, zero, obj_lo,
            ctl_invariant, convexify, earlyterm, g_dot_g, v_dot_g, p_dot_Hp,
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

def trlib_solve(hess, grad, radius, invM = lambda x: x, TR=None, reentry=False,
        verbose=0, ctl_invariant=0, convexify=1, earlyterm=1,
        double tol_rel_i = np.finfo(np.float).eps**.5, double tol_abs_i = 0.0,
        double tol_rel_b = np.finfo(np.float).eps**.3, double tol_abs_b = 0.0
        ):
    r"""
    Solves trust-region subproblem
    
    min .5 x^T H x + g^x   s.t.  || x ||_M <= r

    with a projected CG/Lanczos method.

    Parameters:
    -----------
    hess: {sparse matrix, dense matrix, LinearOperator}
        hessian matrix/operator `H` with shape (n,n)
    grad: array
        gradient `g` with shape (n,1)
    radius: float
        trust-region radius `r`
    invM: {sparse matrix, dense matrix, LinearOperator}, optional
        inverse of matrix/operator defining trust-region constraint norm, default: identity, acts as preconditioner in CG/Lanczos
    TR: dict, optional
        TR output of previous call for hotstarting
    reentry: boolean, optional
        set this to `True`, if you want to resolve with all data fixed but changed trust-region radius, provide `TR` of previous call
    verbose: int, optional
        verbosity level
    ctl_invariant: int, optional
        flag that determines how to treat hard-case, see C API of trlib_krylov_min, default `0`
    convexify: int, optional
        flag that determines if resolving with convexified should be tried if solution seems unrealistic
    earlyterm: int, optional
        flag that determines if solver should terminate early prior to convergence if it seems unlikely that progress will be made soon

    Returns:
    --------
    array
        solution vector
    dict
        trust-region instance data `TR`, needed for warmstart.
        
            - TR['ret'] gives return code of trlib_krylov_min,
            - TR['obj'] gives objective funtion value,
            - TR['lam'] lagrange multiplier

    Examples:
    ---------
    Solve a sample large-scale problem with indefinite diagonal hessian matrix:
   
    >>> import scipy.sparse
    >>> import numpy as np
    >>> H = scipy.sparse.diags(np.linspace(-1.0, 100.0, 1000),0)
    >>> g = np.ones(1000)
    >>> x, TR = trlib.trlib_solve(H, g, 1.0)
    >>> np.linalg.norm(x)
    0.99999999999999978
    >>> x, TR = trlib.trlib_solve(H, g, .5, reentry=True, TR=TR)
    0.50000000000005329
    """

    if hasattr(hess, 'dot'):
        hmv = lambda x: hess.dot(x)
    else:
        hmv = hess
    itmax = 2*grad.shape[0]
    iwork_size, fwork_size, h_pointer = krylov_memory_size(itmax)
    if TR is None:
        TR = {}
    if not reentry:
        init = ctrlib._TRLIB_CLS_INIT
        if not 'fwork' in TR:
            TR['fwork'] = np.empty([fwork_size])
        krylov_prepare_memory(itmax, TR['fwork'])
        if not 'iwork' in TR:
            TR['iwork'] = np.empty([iwork_size], dtype=np.int)
        if not 's' in TR:
            TR['s'] = np.empty(grad.shape)
        if not 'g' in TR:
            TR['g'] = np.empty(grad.shape)
        if not 'v' in TR:
            TR['v'] = np.empty(grad.shape)
        if not 'gm' in TR:
            TR['gm'] = np.empty(grad.shape)
        if not 'p' in TR:
            TR['p'] = np.empty(grad.shape)
        if not 'Hp' in TR:
            TR['Hp'] = np.empty(grad.shape)
        if not 'Q' in TR:
            TR['Q'] = np.empty([itmax+1, grad.shape[0]])
    else:
        if reentry != 'convex':
            init = ctrlib._TRLIB_CLS_HOTSTART
        else:
            init = ctrlib._TRLIB_CLS_HOTSTART_P

    v_dot_g = 0.0; g_dot_g = 0.0; p_dot_Hp = 0.0
    
    while True:
        if verbose > 2:
            print(TR['iwork'][:5])
        ret, action, iter, ityp, flt1, flt2, flt3 = krylov_min(
            init, radius, g_dot_g, v_dot_g, p_dot_Hp, TR['iwork'], TR['fwork'],
            ctl_invariant=ctl_invariant, itmax=itmax, verbose=verbose,
            convexify=convexify, earlyterm=earlyterm,
            tol_rel_i=tol_rel_i, tol_abs_i=tol_abs_i, tol_rel_b=tol_rel_b, tol_abs_b=tol_abs_b)
        if verbose > 2:
            print(TR['iwork'][:5], ret, action, iter, ityp, flt1, flt2, flt3)
        init = 0
        if action == ctrlib._TRLIB_CLA_INIT:
            TR['s'][:] = 0.0
            TR['gm'][:] = 0.0
            TR['g'][:] = grad
            TR['v'][:] = invM(TR['g'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(TR['v'], TR['g'])
            TR['p'][:] = -TR['v']
            TR['Hp'][:] = hmv(TR['p'])
            p_dot_Hp = np.dot(TR['p'], TR['Hp'])
            TR['Q'][0,:] = TR['v']/np.sqrt(v_dot_g)
        if action == ctrlib._TRLIB_CLA_RETRANSF:
            TR['s'][:] = np.dot(TR['fwork'][h_pointer:h_pointer+iter+1], TR['Q'][:iter+1,:])
        if action == ctrlib._TRLIB_CLA_UPDATE_STATIO:
            if ityp == ctrlib._TRLIB_CLT_CG:
                TR['s'] += flt1 * TR['p']
        if action == ctrlib._TRLIB_CLA_UPDATE_GRAD:
            if ityp == ctrlib._TRLIB_CLT_CG:
                TR['Q'][iter,:] = flt2*TR['v']
                TR['gm'][:] = TR['g']
                TR['g'] += flt1*TR['Hp']
            if ityp == ctrlib._TRLIB_CLT_L:
                TR['s'][:] = TR['Hp'] + flt1*TR['g'] + flt2*TR['gm']
                TR['gm'][:] = flt3*TR['g']
                TR['g'][:] = TR['s']
            TR['v'][:] = invM(TR['g'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(TR['v'], TR['g'])
        if action == ctrlib._TRLIB_CLA_UPDATE_DIR:
            TR['p'][:] = flt1 * TR['v'] + flt2 * TR['p']
            TR['Hp'][:] = hmv(TR['p'])
            p_dot_Hp = np.dot(TR['p'], TR['Hp'])
            if ityp == ctrlib._TRLIB_CLT_L:
                TR['Q'][iter,:] = TR['p']
        if action == ctrlib._TRLIB_CLA_NEW_KRYLOV:
            # FIXME: adapt for M != I
            QQ, RR = np.linalg.qr(TR['Q'][:iter,:].transpose(), 'complete')
            TR['g'][:] = QQ[:,iter]
            TR['gm'][:] = 0.0
            TR['v'][:] = invM(TR['g'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(TR['v'], TR['g'])
            TR['p'][:] = TR['v']/np.sqrt(v_dot_g)
            TR['Hp'][:] = hmv(TR['p'])
            p_dot_Hp = np.dot(TR['p'], TR['Hp'])
            TR['Q'][iter,:] = TR['p']
        if action == ctrlib._TRLIB_CLA_CONV_HARD:
            # FIXME: adapt for M != I
            Ms = TR['s']
            Hs = hmv(TR['s'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(Hs+grad+flt1*Ms, invM(Hs+grad)+flt1*TR['s'])
        if action == ctrlib._TRLIB_CLA_OBJVAL:
            obj = .5*np.dot(TR['s'], hmv(TR['s'])) + np.dot(TR['s'], grad)
        if ret < 10:
            break
    if ret < 0:
        print("Warning, status: %d" % ret, TR['iwork'][7], TR['iwork'][8])
        pass
    TR['ret'] = ret
    TR['iter'] = iter
    TR['obj'] = TR['fwork'][8]
    TR['lam'] = TR['fwork'][7]
    return TR['s'], TR

def umin(obj, grad, hessvec, x, tol=1e-5, eta1=1e-2, eta2=.95, gamma1=.5, gamma2=2., itmax=-1, verbose=1):
    """
    Standard Trust Region Algorithm for Unconstrained Optimization Problem:

        min f(x)

    This implements Algorithm 6.1 of [Gould1999]_

    with slight modification:
        - check for descent
        - aggresive trust region reduction upon failed step if next iteration will have the same subproblem solution
        
        
    Parameters
    ----------
    obj : function
        callback that computes x |-> f(x)
    grad : function
        callback that computes x |-> nabla f(x)
    hessvec : function
        callback that computes (x,d) |-> hv = nabla^2 f(x) * d
    x : float, n
        starting point
    tol, eta1, eta2, gamma1, gamma2 : float, optional
        algorithm parameters
    itmax : int, optional
        maximum number of iterations
    verbose : int, optional
        verbosity level, default `1`


    Returns
    -------
    x : array
        last point, solution in case of convergence

    Examples
    --------
    To compute the minimizer of the extended Rosenbrock function in R^10:

    >>> import trlib
    >>> import numpy as np
    >>> import scipy.optimize
    >>> trlib.umin(scipy.optimize.rosen, scipy.optimize.rosen_der, scipy.optimize.rosen_hess_prod, np.zeros(10))
    it   obj         ‖g‖        radius     step       rho          ?  nhv
       0 +9.0000e+00 6.0000e+00 3.1623e-01 3.1623e-01 -2.9733e-01  -    2
       1 +9.0000e+00 6.0000e+00 1.5811e-01 1.5811e-01 +9.6669e-01  +    0
       2 +8.6458e+00 5.0685e+00 3.1623e-01 3.1623e-01 +2.4105e-01  +    5
       3 +8.5464e+00 2.0904e+01 3.1623e-01 1.8711e-01 +1.1361e+00  +    7
       4 +7.5523e+00 5.1860e+00 6.3246e-01 6.2354e-01 -5.0077e+00  -    8
       5 +7.5523e+00 5.1860e+00 3.1623e-01 3.1623e-01 +7.2941e-01  +    0
       6 +7.1073e+00 1.3532e+01 3.1623e-01 2.4811e-01 +1.2836e+00  +    9
       7 +6.3028e+00 6.1031e+00 6.3246e-01 4.0848e-01 +5.8886e-01  +    9
       8 +5.9622e+00 1.6234e+01 6.3246e-01 2.1087e-01 +1.1893e+00  +   10
       9 +4.8955e+00 4.6504e+00 1.2649e+00 5.7036e-01 -1.7558e+00  -   10
      10 +4.8955e+00 4.6504e+00 5.1332e-01 5.1332e-01 -6.7456e-01  -    0
      11 +4.8955e+00 4.6504e+00 2.5666e-01 2.5666e-01 +1.0507e+00  +    0
      12 +4.3447e+00 6.1833e+00 5.1332e-01 4.1590e-01 +5.9133e-01  +   10
      13 +3.9902e+00 1.6533e+01 5.1332e-01 2.1157e-01 +1.1863e+00  +   10
      14 +2.9133e+00 4.7129e+00 1.0266e+00 5.6456e-01 -1.6059e+00  -   10
      15 +2.9133e+00 4.7129e+00 5.0810e-01 5.0810e-01 -5.7730e-01  -    0
      16 +2.9133e+00 4.7129e+00 2.5405e-01 2.5405e-01 +1.0568e+00  +    0
      17 +2.3636e+00 6.0237e+00 5.0810e-01 4.1557e-01 +5.4052e-01  +   10
      18 +2.0447e+00 1.6488e+01 5.0810e-01 1.8313e-01 +1.1450e+00  +   10
      19 +1.0851e+00 3.7054e+00 1.0162e+00 3.8037e-01 +2.0492e-01  +   10
    it   obj         ‖g‖        radius     step       rho          ?  nhv
      20 +1.0047e+00 1.4515e+01 1.0162e+00 1.3411e-01 +1.0904e+00  +   10
      21 +3.7776e-01 1.2123e+00 2.0324e+00 4.2342e-01 -1.1558e+00  -   10
      22 +3.7776e-01 1.2123e+00 3.8108e-01 3.8108e-01 -3.7686e-01  -    0
      23 +3.7776e-01 1.2123e+00 1.9054e-01 1.9054e-01 +9.4089e-01  +    0
      24 +2.4119e-01 3.5489e+00 1.9054e-01 1.9054e-01 +1.2584e+00  +   10
      25 +1.2688e-01 2.9419e+00 3.8108e-01 1.9886e-01 +1.2520e+00  +   10
      26 +5.7708e-02 3.2321e+00 7.6216e-01 1.4598e-01 +1.2840e+00  +   10
      27 +2.0136e-02 1.5044e+00 1.5243e+00 1.4197e-01 +1.1901e+00  +   10
      28 +5.2154e-03 1.5659e+00 3.0487e+00 6.9041e-02 +1.1823e+00  +   10
      29 +6.1955e-04 3.0998e-01 6.0973e+00 4.0825e-02 +1.0863e+00  +   10
      30 +1.8524e-05 1.2279e-01 1.2195e+01 5.7325e-03 +1.0204e+00  +   10
      31 +2.2110e-08 2.0846e-03 2.4389e+01 2.7853e-04 +1.0008e+00  +   10
      32 +3.6798e-14 5.5874e-06
    """
    ii = 0
    if itmax == -1:
        itmax = 10*x.shape[0]
    g = np.empty_like(x)
    radius = 1.0/np.sqrt(x.shape[0])
    data = None
    reentry = False
    accept = False
    while ii < itmax:
        if verbose > 0 and ii % 20 == 0:
            print(u"{:4s} {:11s} {:10s} {:10s} {:10s} {:11s} {:s} {:4s}".format("it", "obj", u"\u2016g\u2016", "radius", "step", "rho", " ?", " nhv"))
        g[:] = grad(x)
        if np.linalg.norm(g) <= tol:
            print("{:4d} {:+2.4e} {:2.4e}".format(ii, obj(x), np.linalg.norm(g)))
            break
        niter = np.array([0])
        def hp(vec):
            niter[0] += 1
            return hessvec(x, vec)
        sol, data = trlib_solve(hp, g, radius, TR=data, reentry=reentry)
        accept = False;
        actual = obj(x+sol) - obj(x)
        pred = np.dot(sol, .5*hessvec(x,sol)+g)
        rho = actual/pred
        if rho >= eta1 and actual < 0.0:
            accept = True
        print("{:4d} {:+2.4e} {:2.4e} {:2.4e} {:2.4e} {:+2.4e}  {:s} {:4d}".format(ii, obj(x), np.linalg.norm(g), radius, np.linalg.norm(sol), rho, '+' if accept else '-', niter[0]))
        if rho >= eta1 and actual < 0.0:
            x = x+sol
            if rho >= eta2:
                radius = gamma2*radius
        if rho < eta1 or actual >= 0.0:
            radius = min(max(1e-8*radius, .9*np.linalg.norm(sol)), gamma1*radius)
        reentry = not accept
        ii += 1
    return x
