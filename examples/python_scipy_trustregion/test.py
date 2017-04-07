import scipy.optimize
import numpy as np
import _trustregion_trlib
hasexact = True
try:
    import scipy.optimize._trustregion_exact as _trustregion_exact
except:
    try:
        import _trustregion_exact
    except:
        hasexact = False
import scipy.optimize._trustregion_ncg
import time


def hesspgat(x,d):
    global nhp
    nhp += 1
    return scipy.optimize.rosen_hess_prod(x,d)

n = 1

for ii in range(9):
    n = 2*n

    print("n = {:d}".format(n))
    print("subproblem solver      time          #iter  #hpev  #hev   obj            message")

    nhp = 0
    curtime = time.time()
    res = scipy.optimize._trustregion_ncg._minimize_trust_ncg(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat)
    soltime = time.time()-curtime
    print("ncg:                   {:+2.4e}s  {:5d}  {:5d}  {:5d}  {:+2.4e}s   {:s}".format(soltime, res['nit'], nhp, 0, res['fun'], res['message']))

    nhp = 0
    curtime = time.time()
    res = _trustregion_trlib._minimize_trust_trlib(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat)
    soltime = time.time()-curtime
    print("trlib, inexact=True:   {:+2.4e}s  {:5d}  {:5d}  {:5d}  {:+2.4e}s   {:s}".format(soltime, res['nit'], nhp, 0, res['fun'], res['message']))

    nhp = 0
    curtime = time.time()
    res = _trustregion_trlib._minimize_trust_trlib(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat, inexact=False)
    soltime = time.time()-curtime
    print("trlib, inexact=False:  {:+2.4e}s  {:5d}  {:5d}  {:5d}  {:+2.4e}s   {:s}".format(soltime, res['nit'], nhp, 0, res['fun'], res['message']))
    
    if hasexact:
        curtime = time.time()
        res = _trustregion_exact._minimize_trustregion_exact(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hess=scipy.optimize.rosen_hess)
        soltime = time.time()-curtime
        print("exact:                 {:+2.4e}s  {:5d}  {:5d}  {:5d}  {:+2.4e}s   {:s}".format(soltime, res['nit'], 0, res['nhev'], res['fun'], res['message']))

    print("")
