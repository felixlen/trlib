#include "trlib_quadratic_zero.h"

int trlib_quadratic_zero(double c_abs, double c_lin, double tol,
        int verbose, int unicode, char *prefix, FILE *fout,
        double *t1, double *t2) {
    int n  = 0;   // number of roots
    *t1 = 0.0;    // first root
    *t2 = 0.0;    // second root
    double q = 0.0;
    double dq = 0.0;
    double lin_sq = c_lin*c_lin;

    if (fabs(c_abs) > tol*lin_sq) {
        // well behaved non-degenerate quadratic
        // compute discriminant
        q = lin_sq - 4.0 * c_abs;
        if ( fabs(q) <= (TRLIB_EPS*c_lin)*(TRLIB_EPS*c_lin) ) {
            // two distinct zeros, but discrimant tiny --> numeric double zero
            // initialize on same root obtained by standard formula with zero discrement, let newton refinement do the rest
            n = 2;
            *t1 = -.5*c_lin; *t2 = *t1;
        }
        else if ( q < 0.0 ) {
            n = 2;
            *t1 = 0.0; *t2 = 0.0;
            return n;
        }
        else {
            // discriminant large enough, two distinc zeros
            n = 2;
            // start with root according to plus sign to avoid cancellation
            *t1 = -.5 * ( c_lin + copysign( sqrt(q), c_lin ) );
            *t2 = c_abs/(*t1);
            if (*t2 < *t1) { q = *t2; *t2 = *t1; *t1 = q; }
        }
    }
    else {
        n = 2;
        if (c_lin < 0.0) { *t1 = 0.0; *t2 = - c_lin; }
        else { *t1 = - c_lin; *t2 = 0.0; }
    }

    // newton correction
    q = (*t1+c_lin)*(*t1)+c_abs; dq = 2.0*(*t1)+c_lin;
    if (dq != 0.0) { *t1 = *t1 - q/dq; }
    q = (*t2+c_lin)*(*t2)+c_abs; dq = 2.0*(*t2)+c_lin;
    if (dq != 0.0) { *t2 = *t2 - q/dq; }
    return n;
};
