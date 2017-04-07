import scipy.optimize
import numpy as np
import _trustregion_trlib
import scipy.optimize._trustregion_ncg
import time


def hesspgat(x,d):
    global nhp
    nhp += 1
    return scipy.optimize.rosen_hess_prod(x,d)

n = 1

for ii in range(11):
    n = 2*n
    print("trlib n = {:d}".format(n))
    nhp = 0
    curtime = time.time()
    restr = _trustregion_trlib._minimize_trust_trlib(scipy.optimize.rosen, 0.0*np.ones([n]), jac=scipy.optimize.rosen_der, hessp=hesspgat)
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

    print("\n\n")
