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

    print("trlib n = {:d}".format(n))
    nhp = 0
    curtime = time.time()
    restr = _trustregion_trlib._minimize_trust_trlib(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat)
    trtime = time.time()-curtime
    nhptrlib = nhp
    print("solution time:    {:e} s\n# hessian vector: {:d}\n# NLP iterations: {:d}\nobjective:        {:e}\nmessage:          {:s}".format(trtime, nhp, restr['nit'], restr['fun'], restr['message']))

    print("\ntrlib inexact=False n = {:d}".format(n))
    nhp = 0
    curtime = time.time()
    restr = _trustregion_trlib._minimize_trust_trlib(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat, inexact=False)
    trtime = time.time()-curtime
    nhptrlib = nhp
    print("solution time:    {:e} s\n# hessian vector: {:d}\n# NLP iterations: {:d}\nobjective:        {:e}\nmessage:          {:s}".format(trtime, nhp, restr['nit'], restr['fun'], restr['message']))
    
    print("\nncg n = {:d}".format(n))
    nhp = 0
    curtime = time.time()
    resncg = scipy.optimize._trustregion_ncg._minimize_trust_ncg(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat)
    ncgtime = time.time()-curtime
    nhpncg = nhp
    print("solution time:    {:e} s\n# hessian vector: {:d}\n# NLP iterations: {:d}\nobjective:        {:e}\nmessage:          {:s}".format(ncgtime, nhp, resncg['nit'], resncg['fun'], resncg['message']))

    if hasexact:
        print("\nexact n = {:d}".format(n))
        curtime = time.time()
        resexa = _trustregion_exact._minimize_trustregion_exact(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hess=scipy.optimize.rosen_hess)
        exatime = time.time()-curtime
        print("solution time:    {:e} s\n# hessians:       {:d}\n# NLP iterations: {:d}\nobjective:        {:e}\nmessage:          {:s}".format(exatime, resexa['nhev'], resexa['nit'], resexa['fun'], resexa['message']))

    print("\n\n")
