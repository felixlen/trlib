# MIT License
#
# Copyright (c) 2016--2017 Felix Lenders
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import scipy.sparse.linalg
import warnings
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
        double tol_rel_i = -2.0, double tol_abs_i = 0.0,
        double tol_rel_b = -3.0, double tol_abs_b = 0.0,
        double zero = np.finfo(float).eps, double obj_lo=-1e20, long verbose=0, refine = True,
        long[::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([ctrlib.trlib_krylov_timing_size()], dtype=int)
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
        long warm, double leftmost_minor, long itmax = 500, double tol_abs = np.finfo(float).eps**.75,
        long verbose=0, long [::1] timing = None, prefix=""):
    cdef long [:] timing_b
    if timing is None:
        ttiming = np.zeros([ctrlib.trlib_krylov_timing_size()], dtype=int)
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
        double tol_rel = np.finfo(float).eps, double tol_newton_tiny = 1e-2*np.finfo(float).eps**.5,
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
        ttiming = np.zeros([ctrlib.trlib_tri_timing_size()], dtype=int)
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
        double tol_rel_i = -2.0, double tol_abs_i = 0.0,
        double tol_rel_b = -3.0, double tol_abs_b = 0.0
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
    1.0000000000000002
    >>> x, TR = trlib.trlib_solve(H, g, .5, reentry=True, TR=TR)
    0.50000000000000011
    """

    if hasattr(hess, 'dot'):
        hmv = lambda x: hess.dot(x)
    else:
        hmv = hess

    cdef long equality = 0
    cdef long itmax_lanczos = 100
    
    cdef double tol_r_i = tol_rel_i
    cdef double tol_a_i =  tol_abs_i
    cdef double tol_r_b = tol_rel_b
    cdef double tol_a_b =  tol_abs_b
    cdef double zero = 2e-16
    cdef double obj_lb = -1e20
    
    cdef long ctli = 0
    cdef long cvx = convexify
    cdef long eterm = earlyterm
    
    cdef double g_dot_g = 0.0
    cdef double v_dot_g = 0.0
    cdef double p_dot_Hp = 0.0
    
    cdef long refine = 1
    cdef long verb = verbose
    cdef long unicode = 1
    
    cdef long ret = 0
    cdef long action = 0
    cdef long it = 0
    cdef long ityp = 0
    cdef long itmax = 2*grad.shape[0]
    cdef long init
    cdef long iwork_size
    cdef long fwork_size
    cdef long h_pointer
    
    cdef double flt1 = 0.0
    cdef double flt2 = 0.0
    cdef double flt3 = 0.0
    
    prefix = "".encode('UTF-8')
    
    ctrlib.trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size, &h_pointer)
    
    if TR is None:
        TR = {}
    if not reentry:
        init = ctrlib._TRLIB_CLS_INIT
        if not 'fwork' in TR:
            TR['fwork'] = np.empty([fwork_size])
        krylov_prepare_memory(itmax, TR['fwork'])
        if not 'iwork' in TR:
            TR['iwork'] = np.empty([iwork_size], dtype=int)
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
        if not 'timing' in TR:
            TR['timing'] = np.zeros([ctrlib.trlib_krylov_timing_size()], dtype=int)
    else:
        if reentry != 'convex':
            init = ctrlib._TRLIB_CLS_HOTSTART
        else:
            init = ctrlib._TRLIB_CLS_HOTSTART_P

    cdef long   [:] iwork_view  = TR['iwork']
    cdef double [:] fwork_view  = TR['fwork']
    cdef long   [:] timing_view = TR['timing']

    
    while True:
        ret = ctrlib.trlib_krylov_min(init, radius, equality, itmax, itmax_lanczos,
            tol_r_i, tol_a_i, tol_r_b, tol_a_b, zero, obj_lb,
            ctli, cvx, eterm, g_dot_g, v_dot_g, p_dot_Hp,
            &iwork_view[0] if TR['iwork'].shape[0] > 0 else NULL,
            &fwork_view[0] if TR['fwork'].shape[0] > 0 else NULL,
            refine, verb, unicode, prefix, <libc.stdio.FILE*> libc.stdio.stdout,
            &timing_view[0] if TR['timing'].shape[0] > 0 else NULL,
            &action, &it, &ityp, &flt1, &flt2, &flt3)

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
            TR['s'][:] = np.dot(TR['fwork'][h_pointer:h_pointer+it+1], TR['Q'][:it+1,:])
        if action == ctrlib._TRLIB_CLA_UPDATE_STATIO:
            if ityp == ctrlib._TRLIB_CLT_CG:
                TR['s'] += flt1 * TR['p']
        if action == ctrlib._TRLIB_CLA_UPDATE_GRAD:
            if ityp == ctrlib._TRLIB_CLT_CG:
                TR['Q'][it,:] = flt2*TR['v']
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
                TR['Q'][it,:] = TR['p']
        if action == ctrlib._TRLIB_CLA_NEW_KRYLOV:
            # FIXME: adapt for M != I
            QQ, RR = np.linalg.qr(TR['Q'][:iter,:].transpose(), 'complete')
            TR['g'][:] = QQ[:,it]
            TR['gm'][:] = 0.0
            TR['v'][:] = invM(TR['g'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(TR['v'], TR['g'])
            TR['p'][:] = TR['v']/np.sqrt(v_dot_g)
            TR['Hp'][:] = hmv(TR['p'])
            p_dot_Hp = np.dot(TR['p'], TR['Hp'])
            TR['Q'][it,:] = TR['p']
        if action == ctrlib._TRLIB_CLA_CONV_HARD:
            # FIXME: adapt for M != I
            Ms = TR['s']
            Hs = hmv(TR['s'])
            g_dot_g = np.dot(TR['g'], TR['g'])
            v_dot_g = np.dot(Hs+grad+flt1*Ms, invM(Hs+grad)+flt1*TR['s'])
        if action == ctrlib._TRLIB_CLA_OBJVAL:
            g_dot_g = .5*np.dot(TR['s'], hmv(TR['s'])) + np.dot(TR['s'], grad)
        if ret < 10:
            break
    if ret < 0:
        warnings.warn('trlib status {:d}'.format(ret), RuntimeWarning)
        pass
    TR['ret'] = ret
    TR['iter'] = it
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
    >>> trlib.umin(scipy.optimize.rosen, scipy.optimize.rosen_der, scipy.optimize.rosen_hess_prod, np.zeros(5))
    it   obj         ‖g‖        radius     step       rho          ?  nhv
       0 +4.0000e+00 4.0000e+00 4.4721e-01 4.4721e-01 -3.9705e+00  -    2
       1 +4.0000e+00 4.0000e+00 2.2361e-01 2.2361e-01 +6.4377e-01  +    0
       2 +3.7258e+00 1.0252e+01 2.2361e-01 4.5106e-02 +1.0011e+00  +    1
       3 +3.4943e+00 2.3702e+00 4.4721e-01 4.4721e-01 -1.6635e+00  -    3
       4 +3.4943e+00 2.3702e+00 2.2361e-01 2.2361e-01 +6.3041e-01  +    0
       5 +3.1553e+00 1.0513e+01 2.2361e-01 3.2255e-02 +1.0081e+00  +    1
       6 +2.9844e+00 3.0912e+00 4.4721e-01 4.4721e-01 -8.6302e-01  -    3
       7 +2.9844e+00 3.0912e+00 2.2361e-01 2.2361e-01 +8.2990e-01  +    0
       8 +2.5633e+00 8.4640e+00 2.2361e-01 4.1432e-02 +1.0074e+00  +    2
       9 +2.4141e+00 3.2440e+00 4.4721e-01 4.4721e-01 -3.3029e-01  -    4
      10 +2.4141e+00 3.2440e+00 2.2361e-01 2.2361e-01 +7.7786e-01  +    0
      11 +1.9648e+00 6.5362e+00 2.2361e-01 2.2361e-01 +1.1319e+00  +    4
      12 +1.4470e+00 5.1921e+00 4.4721e-01 4.0882e-01 +1.7910e-01  +    5
      13 +1.3564e+00 1.6576e+01 4.4721e-01 7.9850e-02 +1.0404e+00  +    2
      14 +6.8302e-01 3.7423e+00 8.9443e-01 4.0298e-01 -4.9142e-01  -    5
      15 +6.8302e-01 3.7423e+00 3.6268e-01 3.6268e-01 +8.1866e-02  +    0
      16 +6.5818e-01 1.3817e+01 3.6268e-01 4.8671e-02 +1.0172e+00  +    1
      17 +3.1614e-01 5.9764e+00 7.2536e-01 1.3202e-02 +1.0081e+00  +    2
      18 +2.8033e-01 1.4543e+00 1.4507e+00 3.5557e-01 -8.6703e-02  -    5
      19 +2.8033e-01 1.4543e+00 3.2001e-01 3.2001e-01 +3.4072e-01  +    0
    it   obj         ‖g‖        radius     step       rho          ?  nhv
      20 +2.3220e-01 1.0752e+01 3.2001e-01 2.0244e-02 +1.0101e+00  +    1
      21 +1.2227e-01 5.0526e+00 6.4002e-01 1.1646e-02 +1.0073e+00  +    2
      22 +9.6271e-02 1.6755e+00 1.2800e+00 1.4701e-03 +9.9992e-01  +    1
      23 +9.5040e-02 6.1617e-01 2.5601e+00 3.3982e-01 -3.6365e-01  -    5
      24 +9.5040e-02 6.1617e-01 3.0584e-01 3.0584e-01 +1.3003e-01  +    0
      25 +8.6495e-02 9.2953e+00 3.0584e-01 1.1913e-02 +1.0071e+00  +    1
      26 +3.0734e-02 4.0422e+00 6.1167e-01 7.7340e-03 +1.0056e+00  +    2
      27 +1.7104e-02 1.1570e+00 1.2233e+00 9.7535e-04 +1.0004e+00  +    1
      28 +1.6540e-02 3.6078e-01 2.4467e+00 2.0035e-01 +5.5171e-01  +    5
      29 +8.6950e-03 3.5281e+00 2.4467e+00 3.7360e-03 +1.0023e+00  +    1
      30 +2.0894e-03 1.4262e+00 4.8934e+00 2.4287e-03 +1.0018e+00  +    2
      31 +5.8038e-04 3.5326e-01 9.7868e+00 4.3248e-04 +1.0004e+00  +    2
      32 +5.1051e-04 7.3221e-02 1.9574e+01 4.3231e-02 +9.9862e-01  +    5
      33 +1.4135e-05 1.4757e-01 3.9147e+01 2.0304e-04 +1.0001e+00  +    3
      34 +7.1804e-07 1.3804e-02 7.8294e+01 1.5512e-03 +1.0008e+00  +    5
      35 +2.2268e-11 1.8517e-04 1.5659e+02 2.0956e-06 +1.0000e+00  +    5
      36 +3.7550e-21 1.2923e-10
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
