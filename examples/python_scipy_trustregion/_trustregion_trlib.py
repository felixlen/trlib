from scipy.optimize._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
import trlib

__all__ = ['_minimize_trustregion_trlib']

def _minimize_trust_trlib(fun, x0, args=(), jac=None, hess=None, hessp=None,
                                **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.
    Options
    -------
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region ',
                         'exact minimization.')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Newton-CG trust-region minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                                  subproblem=TRLIBSubproblem,
                                  **trust_region_options)

class TRLIBSubproblem(BaseQuadraticSubproblem):
    def __init__(self, x, fun, jac, hess, hessp):
        super(TRLIBSubproblem, self).__init__(x, fun, jac, hess, hessp)
        self.TR = None

    def solve(self, trust_radius):
        s, TR = trlib.trlib_solve(self.hessp, self.jac, trust_radius, TR=self.TR, reentry=not self.TR is None)
        self.TR = TR
        return s, TR['lam'] > 0.0
