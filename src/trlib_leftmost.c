#include "trlib_leftmost.h"

int trlib_leftmost(
        int nirblk, int *irblk, double *diag, double *offdiag,
        int warm, double leftmost_minor, int itmax, double tol_abs,
        int verbose, int unicode, char *prefix, FILE *fout,
        long *timing, int *ileftmost, double *leftmost) {
    int ret = 0; int curit;
    if(! warm) {
        ret = 0; int curret;
        for(int ii = 0; ii < nirblk; ++ii) {
            curret = trlib_leftmost_irreducible(irblk[ii+1]-irblk[ii], diag+irblk[ii], offdiag+irblk[ii], 0, 0.0, itmax,
                tol_abs, verbose, unicode, prefix, fout, timing, leftmost+ii, &curit);
            if (curret == 0) { ret = curret; }
        }
        *ileftmost = 0;
        for(int ii = 1; ii < nirblk; ++ii) {
            if (leftmost[ii] < leftmost[*ileftmost]) { *ileftmost = ii; }
        }
    }
    else { 
        ret = trlib_leftmost_irreducible(irblk[nirblk] - irblk[nirblk-1], diag+irblk[nirblk-1], offdiag+irblk[nirblk-1],
                1, leftmost_minor, itmax, tol_abs, verbose, unicode, prefix, fout, timing, leftmost+nirblk-1, &curit);
        if (leftmost[nirblk-1] < leftmost[*ileftmost]) { *ileftmost = nirblk-1; }
    }
    return ret;
}

int trlib_leftmost_irreducible(
        int n, double *diag, double *offdiag,
        int warm, double leftmost_minor, int itmax, double tol_abs,
        int verbose, int unicode, char *prefix, FILE *fout,
        long *timing, double *leftmost, int *iter_pr) {
    // Local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    *iter_pr = 0;                           // iteration counter
    int jj = 0;                             // local counter variable
    double low = 0.0;                       // lower bracket variable: low <= leftmost       for desired value
    double up = 0.0;                        // upper bracket variable:        leftmost <= up for desired value
    *leftmost = 0.0;                        // estimation of desired leftmost eigenvalue
    double leftmost_attempt = 0.0;          // trial step for leftmost eigenvalue
    double dleftmost = 0.0;                 // increment
    double prlp = 0.0;                      // value of Parlett-Reid-Last-Pivot function
    double dprlp = 0.0;                     // derivative of Parlett-Reid-Last-Pivot function wrt to leftmost
    int n_neg_piv = 0;                      // number of negative pivots in factorization
    double quad_abs = 0.0;                  // absolute coefficient in quadratic model
    double quad_lin = 0.0;                  // linear   coefficient in quadratic model
    double zerodum = 0.0;                   // dummy return variables from quadratic equation
    double oabs0 = 0.0; double oabs1 = 0.0; // temporaries in Gershgorin limit computation

    int continue_outer_loop = 0;            // local spaghetti code control variable

    // trivial case: one-dimensional. return diagonal value
    if (n == 1) { *leftmost = diag[0]; TRLIB_RETURN(TRLIB_LMR_CONV) }

    /* set bracket interval derived from Gershgorin circles
       Gershgorin:
        eigenvalues are contained in the union of balls centered at
        diag_i with radius sum of absolute values in column i, except diagonal element
       this estimation is rough and could be improved by circle component analysis
              determine if worth doing */

    oabs0 = fabs(offdiag[0]); oabs1 = fabs(offdiag[n-2]);
    low = fmin( diag[0] - oabs0, diag[n-1] - oabs1 );
    up  = fmax( diag[0] + oabs0, diag[n-1] - oabs1 );
    for( int ii = 1; ii < n-1; ++ii ) {
        oabs1 = fabs(offdiag[ii]);
        low = fmin( low, diag[ii] - oabs0 - oabs1 );
        up  = fmax( up,  diag[ii] + oabs0 + oabs1 );
        oabs0 = oabs1;
    }

    /* set leftmost to sensible initialization
       on warmstart, provided leftmost is eigenvalue of principal (n-1) * (n-1) submatrix
          by eigenvalue interlacing theorem desired value <= provided leftmost
       on coldstart, start with lower bound as hopefully this is a good estimation */
    if ( warm ) {
        // provided leftmost is an upper bound and a pole of Parlett-Reid Value, thus pertub a bit
        up = fmin(up, leftmost_minor); *leftmost = leftmost_minor - TRLIB_EPS_POW_4;
    }  
    else { leftmost_minor = 0.0; *leftmost = low; }; // ensure sanity on leftmost_minor and start with lower bound
    // Parlett-Reid Iteration, note we can assume n > 1
    itmax = itmax*n;
//    def print_iter_counter_exit(low, leftmost, up, n_neg_piv, jj, prlp, action):
//        condprint("{:4s}{:14e}{:14e}{:14e}{:14e}{:4d}{:4d} {:8s}".format("", low, leftmost, up, prlp, n_neg_piv, jj, action))

    while (1) {
        /* iterate to obtain Parlett-Reid last pivot value of -leftmost == 0.0
           this iteration uses a safeguard bracket [low, up] such that alway low <= leftmost <= up
           note that T - t*I is positive definite for t <= desired leftmost
           steps of iteration:
          
           (1) compute Parlett-Reid last pivot value which is D_n in a LDL^T factorization of T
               obtain derivative d D_n / d leftmost as byproduct in the very same recursion
               track if breakdown would occur in factorization, happens if either
               (a) a pivot become zero
               (b) more than one negative pivot present
               if breakdown would occurs this means that Parlett-Reid value is infinite
                 end iteration at premature point and restart with adapted bounds and estimation:
               (a) a pivot became zero:  
                   if last pivot zero   --> goal reached, exit
                   if previous zero     --> T - leftmost I not positive definite, thus desired value <= leftmost
               (b) multiple neg privots --> T - leftmost I            indefinite, thus desired value <= leftmost
           (2) compute a trial update for leftmost. two possibilities
               (a) Newton update
               (b) zero of suitable model of analytic expression,
                   analytic expression is given by prlp(t) = det(T-t*I)/det(U-t*I) with U principal (n-1)*(n-1) submatrix
                   Gould proposes model m(t) = (t-a)(t-b)/(t-leftmost(U))
               do (b) if warmstart where user provided leftmost(U), otherwise go route (a)
          
           (3) take trial step if inside bracket, otherwise midpoint
          
           stop iteration if either bracket is sufficiently small or Parlett-Reid value is close enough to zero

           note the recurrence for \hat D_k(t) := det(T_k+t*I)/det(T_{k-1}+tI) = (t+delta_k)-gamma_k^2/\hat D_{k-1}(t)
           maybe this helps to find a better suited model */

        *iter_pr += 1;
        
        // test if iteration limit exceeded
        if ( *iter_pr > itmax ) { TRLIB_RETURN(TRLIB_LMR_ITMAX) }

        // initialize: no negative pivots so far
        n_neg_piv = 0;

        // print iteration headline every 10 iterations
        if (*iter_pr % 10 == 1) {
            TRLIB_PRINTLN_1("%6s%8s%14s%14s%14s%14s%14s%6s%6s", "  it  ", " action ", "     low      ", "   leftmost   ", "      up      ", "   dleftmost  ", "      prlp    ", " nneg ", "  br  ")
        }
        TRLIB_PRINTLN_1("%6d%8s%14e%14e%14e", *iter_pr, "  entry ", low, *leftmost, up)

        // compute pivot and derivative of LDL^T factorization of T - leftmost I
        continue_outer_loop = 0;
        for( jj = 0; jj < n; ++jj ) {
            /* compute jj-th pivot
               special case for jj == 0 since offdiagonal is missing */
            if (jj == 0) { prlp = diag[0] - *leftmost; dprlp = -1.0; }
            else{
                // update pivot as pivot = d_j - leftmost - o_{j-1}^2/pivot
                // thus dpivot/dleftmost =     - 1.0      - o_{j-1}^2/pivot^2 * dpivot/dleftmost
                dprlp = -1.0 + offdiag[jj-1]*offdiag[jj-1]*dprlp / (prlp*prlp);
                prlp  = diag[jj] - offdiag[jj-1]*offdiag[jj-1]/prlp - *leftmost;
            }

            // check for breakdown
            if (prlp == 0.0) {
                // if last pivot and no negative pivots encountered --> finished
                if (n_neg_piv == 0 && jj+1 == n) { TRLIB_RETURN(TRLIB_LMR_CONV) }
                else{
                    /* if not last pivot or negative pivots encountered:
                       estimation provides a new upper bound; reset estimation */
                    up = *leftmost;
                    *leftmost = 0.5 * (low+up);
                    continue_outer_loop = 1;
                    break; // continue outer loop
                }
            }
            else if ( prlp < 0.0 ) {
                n_neg_piv += 1;
                if (n_neg_piv > 1) {
                    // more than one negative pivot: factorization would fail
                    up = *leftmost;
                    *leftmost = 0.5 * (low+up);
                    continue_outer_loop = 1;
                    break; // continue outer loop
                }
            }
        }

        if (continue_outer_loop) { 
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14s%14e%6d%6d", "", " bisecp ", low, *leftmost, up, "", prlp, n_neg_piv, jj)
            continue; 
        }

        // we have survived computing the Last-Pivot value without finding a zero pivot and at most one negative pivot

        // adapt bracket, no negative pivots encountered: leftmost provides new lower bound, otherwise upper bound
        if (n_neg_piv == 0) { low = *leftmost; }
        else { up = *leftmost; }

        // test if bracket interval is small or last pivot has converged to zero
        if (up-low <= tol_abs * fmax(1.0, fmax(fabs(low), fabs(up))) || fabs(prlp) <= tol_abs) { 
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14s%14e%6d%6d", "", "  conv  ", low, *leftmost, up, "", prlp, n_neg_piv, jj)
            TRLIB_RETURN(TRLIB_LMR_CONV)
        }

        /* compute trial step for new leftmost
           on coldstart do Newton iteration, on warmstart find zero of model of analytic expression */
        if (warm) {
            /* use analytic model m(t) = (t-a)(t-b)/(t-leftmost_minor)
               fit a, b such that m matches function value and derivative
               at current estimation and compute left zero of numerator */
            quad_lin = -(2.0*(*leftmost)+prlp+((*leftmost)-leftmost_minor)*dprlp);
            quad_abs = -(((*leftmost)-leftmost_minor)*prlp+(*leftmost)*(quad_lin+(*leftmost)));
            trlib_quadratic_zero(quad_abs, quad_lin, TRLIB_EPS_POW_75, 0, 0, "", NULL, &leftmost_attempt, &zerodum);
            dleftmost = leftmost_attempt - *leftmost;
        }
        else { dleftmost = -prlp/dprlp; leftmost_attempt = *leftmost + dleftmost; } // Newton step

        // assess if we can use trial step
        if (low <= leftmost_attempt && leftmost_attempt <= up) { 
            if( fabs(dleftmost) <= tol_abs * fmax(1.0, fmax(fabs(low), fabs(up))) ) { TRLIB_RETURN(TRLIB_LMR_NEWTON_BREAK) }
            if ( warm ) { TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%14e%6d%6d", "", " qmodel ", low, *leftmost, up, dleftmost, prlp, n_neg_piv, jj) }
            else { TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%14e%6d%6d", "", " newton ", low, *leftmost, up, dleftmost, prlp, n_neg_piv, jj) }
            *leftmost = leftmost_attempt;
        }
        else { 
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%14e%6d%6d", "", " bisecs ", low, *leftmost, up, .5*(up-low), prlp, n_neg_piv, jj)
            *leftmost = .5*(low+up);
        }
    }
}

int trlib_leftmost_timing_size() {
#if TRLIB_MEASURE_TIME
    return 1;
#endif
    return 0;
}

