#include "trlib_eigen_inverse.h"

int trlib_eigen_inverse(
        int n, double *diag, double *offdiag, 
        double lam_init, int itmax, double tol_abs,
        double *ones, double *diag_fac, double *offdiag_fac,
        double *eig, int verbose, int unicode, char *prefix, FILE *fout,
        long *timing, double *lam_pert, double *pert, int *iter_inv) {
    // Local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    *iter_inv = 0;                               // iteration counter
    *pert = 0.0;                                 // perturbation factor to update lam until factorization is possible
    int info_fac = 0;                            // status variable for factorization
    double invnorm = 0.0;                        // 1/norm of eig before normalization
    double minuslam = - lam_init;                // negative of current estimation of eigenvalue
    int inc = 1; int nm = n-1;

    // obtain factorization of T - lam*I, perturb until possible
    // iter_inv is misused in this loop as flag if we can find a suitable lambda to start with
    *iter_inv = TRLIB_EIR_FAIL_FACTOR;
    while (*pert <= 1.0/TRLIB_EPS) {
        // set diag_fac to diag - lam
        TRLIB_DCOPY(&n, diag, &inc, diag_fac, &inc) // diag_fac <-- diag
        TRLIB_DAXPY(&n, &minuslam, ones, &inc, diag_fac, &inc) // diag_fac <-- diag_fac - lam
        TRLIB_DCOPY(&nm, offdiag, &inc, offdiag_fac, &inc) // offdiag_fac <-- offdiag
        TRLIB_DPTTRF(&n, diag_fac, offdiag_fac, &info_fac); // compute factorization
        if (info_fac == 0) { *iter_inv = 0; break; }
        if (*pert == 0.0) { 
            *pert = TRLIB_EPS_POW_4 * fmax(1.0, -lam_init);
        }
        else { 
            *pert = 10.0*(*pert);
        }
        minuslam = *pert - lam_init;
    }
    *lam_pert = -minuslam;

    if ( *iter_inv == TRLIB_EIR_FAIL_FACTOR ) { TRLIB_PRINTLN_2("Failure on factorizing in inverse correction!") TRLIB_RETURN(TRLIB_EIR_FAIL_FACTOR) }

    TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
    TRLIB_DSCAL(&n, &invnorm, eig, &inc) // normalize eig
    // perform inverse iteration
    while (1) {
        *iter_inv += 1;

        if ( *iter_inv > itmax ) { TRLIB_RETURN(TRLIB_EIR_ITMAX) }

        // solve (T - lam*I)*eig_new = eig_old
        TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, eig, &n, &info_fac)
        if( info_fac != 0 ) { TRLIB_PRINTLN_2("Failure on solving inverse correction!") TRLIB_RETURN(TRLIB_EIR_FAIL_LINSOLVE) }

        // normalize eig
        TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
        TRLIB_DSCAL(&n, &invnorm, eig, &inc)

        // check for convergence
        if (fabs( invnorm - *pert ) <= tol_abs ) { TRLIB_RETURN(TRLIB_EIR_CONV) }
    }
    
    TRLIB_RETURN(TRLIB_EIR_ITMAX)
};

int trlib_eigen_timing_size() {
#if TRLIB_MEASURE_TIME
    return 1 + TRLIB_SIZE_TIMING_LINALG;
#endif
    return 0;
}

