import scipy.optimize
import numpy as np
import umin

n = 100
x, totalit_ll = umin.standard_tr_unconstrained(scipy.optimize.rosen,
                                               scipy.optimize.rosen_der,
                                               scipy.optimize.rosen_hess_prod,
                                               np.zeros(n))
print(totalit_ll)
