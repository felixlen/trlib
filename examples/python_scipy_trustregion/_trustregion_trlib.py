from scipy.optimize._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
import trlib

__all__ = ['_minimize_trustregion_trlib']

def _minimize_trust_trlib(fun, x0, args=(), jac=None, hess=None, hessp=None,
                          inexact = True, **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.
    Options
    -------
    inexact : bool, optional
        if True requires less nonlinear iterations, but more vector products
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region ',
                         'exact minimization.')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Newton-CG trust-region minimization')
    if inexact:
        return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                                  subproblem=TRLIBSubproblem,
                                  **trust_region_options)
    else:
        return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                                  subproblem=TRLIBESubproblem,
                                  **trust_region_options)


class TRLIBSubproblem(BaseQuadraticSubproblem):
    def __init__(self, x, fun, jac, hess, hessp):
        super(TRLIBSubproblem, self).__init__(x, fun, jac, hess, hessp)
        self.TR = None

    def solve(self, trust_radius):
        s, TR = trlib.trlib_solve(self.hessp, self.jac, trust_radius, TR=self.TR, reentry=not self.TR is None)
        self.TR = TR
        return s, TR['lam'] > 0.0

class TRLIBESubproblem(BaseQuadraticSubproblem):
    def __init__(self, x, fun, jac, hess, hessp):
        super(TRLIBESubproblem, self).__init__(x, fun, jac, hess, hessp)
        self.TR = None

    def solve(self, trust_radius):
        s, TR = trlib.trlib_solve(self.hessp, self.jac, trust_radius, TR=self.TR, reentry=not self.TR is None, tol_rel_i=1e-8, tol_rel_b=1e-6)
        self.TR = TR
        return s, TR['lam'] > 0.0
