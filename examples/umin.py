from __future__ import print_function
import numpy as np
import trlib
import ctypes

def standard_tr_unconstrained(obj, grad, hessvec, x, qpsolver,
        tol=1e-5, eta1=1e-2, eta2=.95, gamma1=.5, gamma2=2., itmax=-1):
    """
    Standard Trust Region Algorithm for Unconstrained Optimization
    This implements Algorithm 6.1 of
        Gould, Lucidi, Roma, Toint; SIAM J. Optim. 1999, Vol. 9, No. 2, pp. 504-525
        Solving The Trust-Region Subproblem Using The Lanczos Method
        
    Not a particular efficient implementation
        
    Parameters
    ----------
    obj : function
        callback that computes x |-> f(x)
    grad : function
        callback that computes x |-> nabla f(x)
    hessvec : function
        callback that computes (x,d) |-> hv = nabla^2 f(x) * d
    qpsolver : function
        callback that computes (g, hessvec, radius, tol_abs, tol_rel, data, reentry) |-> (sol, data)
    x : float, n
    tol, eta1, eta2, gamma1, gamma2 : float, optional
    itmax : int, optional
    
    Returns
    -------
    history : array
        contains iteration history
    """
    ii = 0
    if itmax == -1:
        itmax = 10*x.shape[0]
    g = np.empty_like(x)
    radius = 1.0/np.sqrt(x.shape[0])
    data = None
    reentry = False
    accept = False
    totalit = 0
    while ii < itmax:
        if ii % 20 == 0:
            print("{:4s}{:14s}{:14s}{:14s}{:14s}{:3s}{:4s}".format(" it", "   obj", "   norm g",
                                                                   "    rho", " radius", " ? ", " nit"))
        g[:] = grad(x)
        if np.linalg.norm(g) <= tol:
            print("{:4d}{:14e}{:14e}".format(ii, obj(x), np.linalg.norm(g)))
            break
        niter = np.array([0])
        def hp(vec):
            niter[0] += 1
            return hessvec(x, vec)
        sol, data = qpsolver(g, hp, radius, data, reentry)
        totalit += niter[0]
        accept = False
        rho = (obj(x+sol)-obj(x))/np.dot(sol, .5*hessvec(x, sol)+g)
        if rho >= eta1:
            accept = True
        print("{:4d}{:14e}{:14e}{:14e}{:14e}{:3s}{:4d}".format(ii, obj(x), np.linalg.norm(g), rho, radius, '+' if accept else '-', niter[0]))
        if rho >= eta1:
            x = x+sol
            if rho >= eta2:
                radius = gamma2*radius
        if rho < eta1:
            radius = gamma1*radius
        reentry = not accept
        ii += 1
    return x, totalit

def qpsolver_low_level(lin, hmv, radius, data=None, reentry=False, verbose=0):
    itmax = 10*lin.shape[0]
    if data is None:
        data = {}
    if not reentry:
        init = 1
        if not 'fwork' in data:
            data['fwork'] = np.empty([22+17*itmax*lin.shape[0]])
        trlib.prepare_memory(itmax, data['fwork'])
        if not 'iwork' in data:
            data['iwork'] = np.empty([15+itmax], dtype=ctypes.c_int)
        if not 's' in data:
            data['s'] = np.empty(lin.shape)
        if not 'g' in data:
            data['g'] = np.empty(lin.shape)
        if not 'gm' in data:
            data['gm'] = np.empty(lin.shape)
        if not 'p' in data:
            data['p'] = np.empty(lin.shape)
        if not 'Hp' in data:
            data['Hp'] = np.empty(lin.shape)
        if not 'Q' in data:
            data['Q'] = np.empty([itmax+1, lin.shape[0]])
        if not 'timing' in data:
            data['timing'] = np.empty([20], dtype=np.int)
        data['timing'][:] = 0
    else:
        init = 2

    v_dot_g = 0.0; p_dot_Hp = 0.0
    
    while True:
        ret, action, iter, ityp, flt1, flt2, flt3 = trlib.krylov_min(
            init, radius, v_dot_g, v_dot_g, p_dot_Hp, data['iwork'], data['fwork'],
            itmax=itmax, timing=data['timing'], refine=True, verbose=verbose)
        init = 0
        if action == 1: # CLA_INIT
            data['s'][:] = 0.0
            data['gm'][:] = 0.0
            data['g'][:] = lin
            v_dot_g = np.dot(data['g'], data['g'])
            data['p'][:] = -data['g']
            data['Hp'][:] = hmv(data['p'])
            p_dot_Hp = np.dot(data['p'], data['Hp'])
            data['Q'][0,:] = data['g']/np.sqrt(v_dot_g)
        if action == 2: # CLA_RETRANSF
            data['s'][:] = np.dot(data['fwork'][16+3*itmax:16+3*itmax+iter+1], data['Q'][:iter+1,:])
        if action == 3: # CLA_UPDATE_STATIO
            if ityp == 1: # CLT_CG
                data['s'] += flt1 * data['p']
        if action == 4: # CLA_UPDATE_GRAD
            if ityp == 1: # CLT_CG
                data['Q'][iter,:] = flt2*data['g']
                data['gm'][:] = data['g']
                data['g'] += flt1*data['Hp']
            if ityp == 2: # CLT_L
                data['s'][:] = data['Hp'] + flt1*data['g'] + flt2*data['gm']
                data['gm'][:] = flt3*data['g']
                data['g'][:] = data['s']
            v_dot_g = np.dot(data['g'], data['g'])
        if action == 5: # CLA_UPDATE_DIR
            data['p'][:] = flt1 * data['g'] + flt2 * data['p']
            data['Hp'][:] = hmv(data['p'])
            p_dot_Hp = np.dot(data['p'], data['Hp'])
            if ityp == 2: # CLT_L
                data['Q'][iter,:] = data['p']
        if action == 6: # CLA_NEW_KRYLOV
            break # FIXME: actually implement this
        if ret < 10:
            break
    if ret < 0:
        print("Warning, status: %d" % ret, data['iwork'][7], data['iwork'][8])
    return data['s'], data

def qpsolver_driver(lin, hmv, radius, data=None, reentry=False, verbose=0):
    sol = trlib.solve_qp_cb(radius, lin, hmv, verbose=verbose) 
    return sol, None
