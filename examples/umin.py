from __future__ import print_function
import numpy as np
import trlib
import ctypes

def standard_tr_unconstrained(obj, grad, hessvec, x, qpsolver=trlib.trlib_solve,
        tol=1e-5, eta1=1e-2, eta2=.95, gamma1=.5, gamma2=2., itmax=-1):
    """
    Standard Trust Region Algorithm for Unconstrained Optimization
    This implements Algorithm 6.1 of
        Gould, Lucidi, Roma, Toint; SIAM J. Optim. 1999, Vol. 9, No. 2, pp. 504-525
        Solving The Trust-Region Subproblem Using The Lanczos Method

    with slight modification:
        - check for descent
        - resolve with convexified QP if step will be discarded
        - aggresive trust region reduction upon failed step if next iteration will have the same subproblem solution
        
        
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
    x : array
        last point, solution in case of convergence
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
        if ii % 20 == 0:
            print("{:4s} {:11s} {:10s} {:10s} {:10s} {:11s}  {:2s} {:4s}".format("it", "obj", "||g||", "radius", "step", "rho", " ?", " nhv"))
        g[:] = grad(x)
        if np.linalg.norm(g) <= tol:
            print("{:4d} {:+2.4e} {:2.4e}".format(ii, obj(x), np.linalg.norm(g)))
            break
        niter = np.array([0])
        def hp(vec):
            niter[0] += 1
            return hessvec(x, vec)
        sol, data = qpsolver(hp, g, radius, data=data, reentry=reentry)
        accept = False; convex = ' '
        actual = obj(x+sol) - obj(x)
        pred = np.dot(sol, .5*hessvec(x,sol)+g)
        rho = actual/pred
        if pred >= 0.0 or rho < eta1:
            convex = 's'
            sol, data = qpsolver(hp, g, radius, data=data, reentry='convex')
            actual = obj(x+sol) - obj(x)
            pred = np.dot(sol, .5*hessvec(x,sol)+g)
            rho = actual/pred
        if rho >= eta1 and actual < 0.0:
            accept = True
        print("{:4d} {:+2.4e} {:2.4e} {:2.4e} {:2.4e} {:+2.4e}  {:s}{:s} {:4d}".format(ii, obj(x), np.linalg.norm(g), radius, np.linalg.norm(sol), rho, convex, '+' if accept else '-', niter[0]))
        if rho >= eta1 and actual < 0.0:
            x = x+sol
            if rho >= eta2:
                radius = gamma2*radius
        if rho < eta1 or actual >= 0.0:
            radius = min(max(1e-8*radius, .9*np.linalg.norm(sol)), gamma1*radius)
        reentry = not accept
        ii += 1
    return x
