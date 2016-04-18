#include "trlib.h"

int trlib_prepare_memory(int itmax, double *fwork) {
    for(int jj = 21+11*itmax; jj<22+12*itmax; ++jj) { *(fwork+jj) = 1.0; } // everything to 1.0 in ones
    memset(fwork+15+2*itmax, 0, itmax*sizeof(double)); // neglin = - gamma_0 e1, thus set neglin[1:] = 0
    return 0;
}

int trlib_krylov_min(
    int init, double radius, int equality, int itmax, int itmax_lanczos,
    double tol_rel_i, double tol_abs_i, double tol_rel_b, double tol_abs_b, double zero,
    double g_dot_g, double v_dot_g, double p_dot_Hp, int *iwork, double *fwork, int refine,
    int verbose, int unicode, char *prefix, FILE *fout, long *timing, int *action,
    int *iter, int *ityp, double *flt1, double *flt2, double *flt3) {
    /* The algorithm runs by solving the trust region subproblem restricted to a Krylov subspace K(ii)
       The Krylov space K(ii) can be either described by the pCG iterates: (notation iM = M^-1)
         K(ii) = span(p_0, ..., p_ii)
       and in an equivalent way by the Lanczos iterates
         K(ii) = span(q_0, ..., q_ii)

       In one iteration the algorithms performs the following steps
       (a) expand K(ii-1) to K(ii):
           if done via pCG:
             alpha = (g, v)/(p, H p); g+ = g + alpha H p; v+ = iM g; beta = (g+, v+)/(g, v); p+ = -v+ + beta p
           if done via Lanczos:
             y = iM t; gamma = sq (t, y); w = t/gamma; q = y/gamma; delta = (q, H q); t+ = Hq - delta w - gamma w-
           we use pCG as long as it does not break down (alpha ~ 0) and continue with Lanczos in that case,
           note the relationship q = v/sq (g, v) * +-1
       (b) compute minimizer s of problem restricted to sample Krylov space K(ii)
           check if this minimizer is expected to be interior:
             do the pCG iterates satisfy the trust region constraint?
             is H positive definite on K(ii), i.e. are all alphas >= 0?
           if the minimizer is interior, set s = p
           if the minimizer is expected on the boundary, set s = Q*h with Q = [q_0, ..., q_ii]
             and let s solve a tridiagonal trust region subprobem with hessian the tridiagonal matrix
             T_ii from the Lanczos process,
             diag(T_ii) = (delta_0, ..., delta_ii) and offdiag(T_ii) = (gamma_1, ..., gamma_ii)
       (c) test for convergence */

    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    // sane names for workspace variables
    int *status = iwork;
    int *ii = iwork+1; *iter = *ii;
    int *pos_def = iwork+2;
    int *interior = iwork+3;
    int *warm_leftmost = iwork+4;
    int *ileftmost = iwork+5;
    int *warm_lam0 = iwork+6;
    int *warm_lam = iwork+7;
    int *lanczos_switch = iwork+8;
    int *exit_tri = iwork+9;
    int *sub_fail_tri = iwork+10;
    int *iter_tri = iwork+11;
    int *iter_last_head = iwork+12;
    int *type_last_head = iwork+13;
    int *irblk = iwork+14;

    double *stop_i = fwork;
    double *stop_b = fwork+1;
    double *v_g = fwork+2;
    double *p_Hp = fwork+3;
    double *cgl = fwork+4;
    double *cglm = fwork+5;
    double *lam0 = fwork+6;
    double *lam = fwork+7;
    double *obj = fwork+8;
    double *s_Mp = fwork+9;
    double *p_Mp = fwork+10;
    double *s_Ms = fwork+11;
    double *sigma = fwork+12;
    double *alpha = fwork+13;
    double *beta = fwork+13+itmax+1;
    double *neglin = fwork+13+2*(itmax+1);
    double *h0 = fwork+13+3*(itmax+1);
    double *h = fwork+13+4*(itmax+1);
    double *delta =  fwork+13+5*(itmax+1);
    double *delta_fac0 = fwork+13+6*(itmax+1);
    double *delta_fac = fwork+13+7*(itmax+1);
    double *gamma = fwork+13+8*(itmax+1); // note that this is shifted by 1, so gamma[0] is gamma_1
    double *gamma_fac0 = fwork+13+8+9*itmax;
    double *gamma_fac = fwork+13+8+10*itmax;
    double *ones = fwork+13+8+11*itmax;
    double *leftmost = fwork+13+9+12*itmax;
    double *fwork_tr = fwork+13+10+13*itmax;

    // local variables
    int returnvalue = TRLIB_CLR_CONTINUE;
    int warm_fac0 = 0; // flag that indicates if you we could successfully update the factorization
    int warm_fac = 0; // flag that indicates if you we could successfully update the factorization
    double sp_Msp = 0.0; // (s+, Ms+)

    if (init == TRLIB_CLS_INIT) { iwork[0] = TRLIB_CLS_INIT; }
    if (init == TRLIB_CLS_HOTSTART) { iwork[0] = TRLIB_CLS_HOTSTART; }

    while(1) {
        switch( *status ) {
            case TRLIB_CLS_INIT:
                // initialization
                *ii = 0; *iter = *ii;  // iteration counter
                *pos_def = 1;  // empty krylov subspace so far, so H is positive definite there for sure
                *interior = !equality;  // we can have interior solution if we are not asked for equality solution
                *warm_leftmost = 0;  // coldstart, so no warmstart information on leftmost available
                irblk[0] = 0; // start pointer to first irreducible block
                *warm_lam0 = 0;  // coldstart, so no warmstart information on multiplier available
                *warm_lam = 0;  // coldstart, so no warmstart information on multiplier available
                *lanczos_switch = -1; // indicate that no lanczos switch occured
                *exit_tri = 0;  // set return code from #trlib_tri_factor_min to 0 just to be on the safe side
                *sub_fail_tri = 0;  // set sub_fail from #trlib_tri_factor_min to 0 just to be on the safe side
                *iter_tri = 0;  // set newton iter from #trlib_tri_factor_min to 0 just to be on the safe side
                *iter_last_head = 0;  // indicate that iteration headline should be printed in first iteration
                *type_last_head = 0;  // just a safety initialization for last iteration headline type
                // ask the user to initialize the vectors he manages, set internal state to resume with vector initialization
                *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_VEC_INIT; *action = TRLIB_CLA_INIT;
                break;
            case TRLIB_CLS_VEC_INIT:
                if (v_dot_g <= 0.0) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_TRIVIAL; returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
                *stop_i = fmax(tol_abs_i, tol_rel_i*sqrt(v_dot_g)); *stop_i = (*stop_i)*(*stop_i); // set interior stopping tolerance, note that this is squared as for efficiency we compare norm^2 <= tol
                *stop_b = fmax(tol_abs_b, tol_rel_b*sqrt(v_dot_g)); // set boundary stopping tolerance, here no square as we directly compare norm <= tol
                *v_g = v_dot_g; // store (v, g)
                *p_Hp = p_dot_Hp; // store (p, Hp)
                neglin[0] = - sqrt(v_dot_g); // set neglin = - gamma_0 e_1 
                *cgl = 1.0; *cglm = 1.0; // ratio between CG and Lanczos vectors is 1 in this and previous iteration
                *sigma = 1.0; // sigma_0 = 1
                *p_Mp = *v_g; // (p0, M p0) = (-v0, -M v0) = (v0, M M^-1 g0); (s, Mp) = (s, Ms) = 0 already properly initialized
                *leftmost = 0.0; *lam = 0.0; // assume interior solution
                *obj = 0.0; *s_Mp = 0.0; *p_Mp = 0.0; *s_Ms = 0.0; // safe initialization for scalar values
                delta[0] = 0.0; // incremental updates in delta, have to initialize it
                *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_NEW_ITER; *action = TRLIB_CLA_TRIVIAL; // continue with CG iteration
                break;
            case TRLIB_CLS_CG_NEW_ITER:
                if (fabs(*p_Hp) <= zero) { *action = TRLIB_CLA_TRIVIAL; *status = TRLIB_CLS_LANCZOS_SWT; break; } // (p, Hp) ~ 0 ---> CG breaks down, continue Lanczos
                alpha[*ii] = (*v_g)/(*p_Hp);
                /* update Lanczos tridiagonal
                   diag(i)    = 1/alpha(i) + beta(i-1)/alpha(i-1)
                   offdiag(i) = sqrt( beta(i-1)/abs(alpha(i-1) )
                     terms with index i-1 have been computed in previous iteration, just add 1/alpha(i) to diag(i) */
                delta[*ii] += (*p_Hp)/(*v_g); // delta(i) += (p,Hp)/(v,g)
                // update if hessian possitive definite in current krylov subspace
                *pos_def = *pos_def && (alpha[*ii] > 0.0);

                // update quantities needed to computed || s_trial ||_M and ratio between Lanczos vector q and pCG vector v
                if (*ii > 0) { *sigma = - copysign( 1.0, alpha[*ii-1] ) * (*sigma); }
                *cglm = *cgl; *cgl = *sigma/sqrt(*v_g);
                if (*interior) {
                    if (*ii>0) {
                        *s_Mp = beta[*ii-1]*(*s_Mp + alpha[*ii-1]*(*p_Mp));
                        *p_Mp = *v_g + beta[*ii-1]*beta[*ii-1]*(*p_Mp);
                    }
                    sp_Msp = *s_Ms + alpha[*ii]*(2.0*(*s_Mp)+alpha[*ii]*(*p_Mp));
                }
                // update if we can expect interior solution
                *interior = *interior && *pos_def && (sp_Msp < radius*radius);

                // update solution candidate
                if (*interior) {
                    // update (s, Ms) and objective
                    *s_Ms = sp_Msp; *obj = *obj - .5*alpha[*ii]*alpha[*ii]*(*p_Hp);
                    // ask user to update stationary point
                    *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_UPDATE_S; *flt1 = alpha[*ii]; *action = TRLIB_CLA_UPDATE_STATIO;
                }
                else {
                    /* solution candidate is on boundary
                       solve tridiagonal reduction
                       first try to update factorization if available to start tridiagonal problem warmstarted */
                    warm_fac0 = 0;
                    if (*warm_lam0) {
                        // check if subminor regular, otherwise warmstart impossible
                        warm_fac0 = delta_fac0[*ii-1] != 0.0;
                        if (warm_fac0) {
                            gamma_fac0[*ii-1] = gamma[*ii-1]/delta_fac0[*ii-1];
                            delta_fac0[*ii] = delta[*ii] + *lam0 - gamma[*ii-1]*gamma[*ii-1]/delta_fac0[*ii-1];
                            // check if regularized tridiagonal is still positive definite for warmstart
                            warm_fac0 = delta_fac0[*ii] > 0.0;
                        }
                    }
                    /* call trlib_tri_factor_min to solve tridiagonal problem, store solution candidate in h
                       the criterion to specify the maximum number of iterations is weird. it should not be dependent on problem size rather than condition of the hessian... */
                    irblk[1] = *ii+1;
                    *exit_tri = trlib_tri_factor_min(
                        1, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                        warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                        &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                        h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                        timing+1, obj, iter_tri, sub_fail_tri);

                    // check for failure, beware: newton break is ok as this means most likely convergence
                    // exit with error and ask the user to get (potentially invalid) solution candidate by backtransformation
                    if (*exit_tri < 0 && *exit_tri != TRLIB_TTR_NEWTON_BREAK) {
                        *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_FAIL_TTR; break;
                    }
                    // also in positive definite case with interior solution
                    if (*exit_tri == TRLIB_TTR_CONV_INTERIOR && *pos_def) {
                        *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_UNEXPECT_INT; break;
                    }

                    // request gradient update from user, skip directly to state TRLIB_CLS_CG_UPDATE_S that does this
                    *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_UPDATE_S; *action = TRLIB_CLA_TRIVIAL;
                }
                break;
            case TRLIB_CLS_CG_UPDATE_S:
                // request gradient update from user
                *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_UPDATE_GV; *flt1 = alpha[*ii]; *flt2= *cgl; *action = TRLIB_CLA_UPDATE_GRAD;
                break;
            case TRLIB_CLS_CG_UPDATE_GV:
                if (v_dot_g <= 0.0) { if (*interior) {*action = TRLIB_CLA_TRIVIAL;} else {*ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF;} returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
                beta[*ii] = v_dot_g/(*v_g);
                /* prepare the next Lanczos tridiagonal matrix as far as possible
                   the diagonal term is given by delta(i+1) = 1/alpha(i+1) + beta(i)/alpha(i)
                   here we can compute already beta(i)/alpha(i) = (v+, g+)/(v, g) / (v, g)/(p, Hp)
                   and the complete offdiagonal term gamma(i+1) = sqrt(beta(i))/abs(alpha(i)) */
                delta[*ii+1] = (v_dot_g*(*p_Hp))/((*v_g)*(*v_g));
                gamma[*ii] = fabs( sqrt(v_dot_g)*(*p_Hp)/(sqrt(*v_g)*(*v_g)) );
                *v_g = v_dot_g; // update (v,g)

                // print iteration details
                // first print headline if necessary
                if (((*ii)-(*iter_last_head)) % 20 == 0 || (*interior && *type_last_head != TRLIB_CLT_CG_INT) || (!(*interior) && *type_last_head != TRLIB_CLT_CG_BOUND)) {
                    if(*interior) {
                        if (unicode) { TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "   \u2016g\u208a\u2016_M\u207b\u00b9   ", "   leftmost   ", "      \u03bb       ", "      \u03b3       ", "      \u03b4       ", "      \u03b1       ", "      \u03b2       ") }
                        else { TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "  ||g+||_M^-1 ", "   leftmost   ", "     lam      ", "    gamma     ", "    delta     ", "    alpha     ", "     beta     ") }
                        *type_last_head = TRLIB_CLT_CG_INT;
                    }
                    else {
                        if (unicode) { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "   \u03b3\u1d62\u208a\u2081|h\u1d62|   ", "   leftmost   ", "      \u03bb       ", "      \u03b3       ", "      \u03b4       ", "      \u03b1       ", "      \u03b2       ") }
                        else { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "gam(i+1)|h(i)|", "   leftmost   ", "     lam      ", "    gamma     ", "    delta     ", "    alpha     ", "     beta     ") }
                        *type_last_head = TRLIB_CLT_CG_BOUND;
                    }
                    *iter_last_head = *ii;
                }
                if (*interior) {
                    TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e%14e%14e%14e%14e", *ii, *iter_tri, "cg_i", *obj, sqrt(*v_g), *leftmost, *lam, *ii == 0 ? -neglin[0] : gamma[*ii-1], delta[*ii], alpha[*ii], beta[*ii])
                }
                else {
                    TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e%14e%14e%14e%14e", *ii, *iter_tri, "cg_b", *obj, gamma[*ii]*fabs(h[*ii]), *leftmost, *lam, *ii == 0 ? -neglin[0] : gamma[*ii-1], delta[*ii], alpha[*ii], beta[*ii]) TRLIB_PRINTLN_2("%s", "")
                }

                // test for convergence
                // interior: ||g^+||_{M^-1} = (g+, M^-1 g+) = (g+, v+) small, boundary gamma(i+1)*|h(i)| small
                if (*interior && (*v_g <= *stop_i)) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_TRIVIAL; returnvalue = TRLIB_CLR_CONV_INTERIOR; break; }
                else if (!(*interior) && (gamma[*ii]*fabs(h[*ii]) <= *stop_b) ) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_CONV_BOUND; break; }
                // otherwise prepare next iteration
                else { *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_UPDATE_P; *flt1 = -1.0; *flt2 = beta[*ii]; *action = TRLIB_CLA_UPDATE_DIR; break; }
                break;
            case TRLIB_CLS_CG_UPDATE_P:
                *p_Hp = p_dot_Hp;
                // prepare next iteration
                *ii += 1;
                // check if we have to boil out due to iteration limit exceeded
                if (*ii >= itmax) { if (*interior) {*action = TRLIB_CLA_TRIVIAL;} else {*action = TRLIB_CLA_RETRANSF;} *ityp = TRLIB_CLT_CG; returnvalue = TRLIB_CLR_ITMAX; break; }
                *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_NEW_ITER; *action = TRLIB_CLA_TRIVIAL;
                break;
            case TRLIB_CLS_HOTSTART:
                /* reentry with smaller trust region radius
                   we implement hotstart by not making use of the CG basis but rather the Lanczos basis
                   as this covers both cases: the interior and the boundary cases
                   the additional cost by doing this is neglible since we most likely will just do one iteration */
                // solve the corresponding tridiagonal problem, check for convergence and otherwise continue to iterate
                irblk[1] = *ii+1;
                *exit_tri = trlib_tri_factor_min(
                    1, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                    warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                    &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                    h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                    timing+1, obj, iter_tri, sub_fail_tri);

                /* check for failure, beware: newton break is ok as this means most likely convergence
                   exit with error and ask the user to get (potentially invalid) solution candidate by backtransformation */
                if (*exit_tri < 0 && *exit_tri != TRLIB_TTR_NEWTON_BREAK) {
                    *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_FAIL_TTR; break;
                }

                // print some information
                if (unicode) { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "   \u03b3\u1d62\u208a\u2081|h\u1d62|   ", "   leftmost   ", "      \u03bb       ", "      \u03b3       ", "      \u03b4       ") }
                else { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "gam(i+1)|h(i)|", "   leftmost   ", "     lam      ", "    gamma     ", "    delta     ") }
                *type_last_head = TRLIB_CLT_HOTSTART;
                *iter_last_head = *ii;

                TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e%14e%14e", *ii, *iter_tri, " hot", *obj, gamma[*ii]*fabs(h[*ii]), *leftmost, *lam, *ii == 0 ? neglin[0] : gamma[*ii-1], delta[*ii]) TRLIB_PRINTLN_2("%s", "")

                // test for convergence
                if ( (*exit_tri != TRLIB_TTR_CONV_INTERIOR) && gamma[*ii]*fabs(h[*ii]) <= *stop_b) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
                else if ( (*exit_tri == TRLIB_TTR_CONV_INTERIOR) && gamma[*ii]*fabs(h[*ii]) <= sqrt(*stop_i)) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
                else {
                    // prepare next iteration
                    if (lanczos_switch < 0) { *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_UPDATE_P; *flt1 = -1.0; *flt2 = beta[*ii]; *action = TRLIB_CLA_UPDATE_DIR; break; }
                }
                break;

            case TRLIB_CLS_LANCZOS_SWT:
                /* switch from CG to Lanczos. perform the first iteration by hand, after the that the coefficients match
                   so far pCG has been used, which means there is some g^{CG}(ii), v^{CG}(ii), p^{CG}(ii) and H p^{CG}(ii)
                   furthermore gamma(ii) is correct

                   what we need now is p^L(ii) ~ v^{CG}(ii), g^{L}(ii) ~ g^{CG}(ii) and H p^L(ii) (new)

                   set p^L := sigma/sqrt( (v^CG, g^CG) ); Hp := Hp^L and compute (p^L, Hp^L) */
                *lanczos_switch = *ii;
                if ( *ii > 0 ) { *sigma = - copysign( 1.0, alpha[*ii-1] ) * (*sigma); }
                *ityp = TRLIB_CLT_L; *status = TRLIB_CLS_L_UPDATE_P; *flt1 = (*sigma)/sqrt(*v_g); *flt2 = 0.0; *action = TRLIB_CLA_UPDATE_DIR;
                break;
            case TRLIB_CLS_L_UPDATE_P:
                delta[*ii] = p_dot_Hp;
                /* solve tridiagonal reduction
                   first try to update factorization if available to start tridiagonal problem warmstarted */
                warm_fac0 = 0;
                if (*warm_lam0) {
                    // check if subminor regular, otherwise warmstart impossible
                    warm_fac0 = delta_fac0[*ii-1] != 0.0;
                    if (warm_fac0) {
                        gamma_fac0[*ii-1] = gamma[*ii-1]/delta_fac0[*ii-1];
                        delta_fac0[*ii] = delta[*ii] + *lam0 - gamma[*ii-1]*gamma[*ii-1]/delta_fac0[*ii-1];
                        // check if regularized tridiagonal is still positive definite for warmstart
                        warm_fac0 = delta_fac0[*ii] > 0.0;
                    }
                }
                /* call factor_min to solve tridiagonal problem, store solution candidate in h
                   the criterion to specify the maximum number of iterations is weird. it should not be dependent on problem size rather than condition of the hessian... */
                irblk[1] = *ii+1;
                *exit_tri = trlib_tri_factor_min(
                    1, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                    warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                    &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                    h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                    timing+1, obj, iter_tri, sub_fail_tri);

                /* check for failure, beware: newton break is ok as this means most likely convergence
                   exit with error and ask the user to get (potentially invalid) solution candidate by backtransformation */
                if (*exit_tri < 0 && *exit_tri != TRLIB_TTR_NEWTON_BREAK) {
                    *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_FAIL_TTR; break;
                }

                /* convergence check is logical at this position, *but* requires gamma(ii+1).
                   wait until gradient has been updated */
                // compute g^L(ii+1)
                *flt1 = -delta[*ii]/gamma[*ii-1]; *flt2 = -gamma[*ii-1]/gamma[*ii-2]; *flt3 = 1.0;
                // in the case that we just switched to Lanczos, we have to use different coefficients
                if (*ii == *lanczos_switch) {
                    *flt1 = -delta[*ii]/sqrt(*v_g)*(*sigma); *flt2 = -gamma[*ii-1]*(*cgl); *flt3 = gamma[*ii-1]/sqrt(*v_g);
                    *cgl = 1.0; *cglm = 1.0;
                }
                *ityp = TRLIB_CLT_L;  *action = TRLIB_CLA_UPDATE_GRAD; *status = TRLIB_CLS_L_CHK_CONV;
                break;
            case TRLIB_CLS_L_CHK_CONV:
                // convergence check after new gradient has been computed
                if (v_dot_g <= 0.0) { if (*interior) {*action = TRLIB_CLA_TRIVIAL;} else {*ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF;} returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
                gamma[*ii] = sqrt(v_dot_g);

                // print some information
                // first print headline if necessary
                if (((*ii)-(*iter_last_head)) % 20 == 0 || *type_last_head != TRLIB_CLT_LANCZOS) {
                    if (unicode) { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "   \u03b3\u1d62\u208a\u2081|h\u1d62|   ", "   leftmost   ", "      \u03bb       ", "      \u03b3       ", "      \u03b4       ") }
                    else { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "gam(i+1)|h(i)|", "   leftmost   ", "     lam      ", "    gamma     ", "    delta     ") }
                    *type_last_head = TRLIB_CLT_LANCZOS;
                    *iter_last_head = *ii;
                }
                TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e%14e%14e", *ii, *iter_tri, " lcz", *obj, gamma[*ii]*fabs(h[*ii]), *leftmost, *lam, *ii == 0 ? neglin[0] : gamma[*ii-1], delta[*ii]) TRLIB_PRINTLN_2("%s", "")

                // test for convergence
                if ( (*exit_tri != TRLIB_TTR_CONV_INTERIOR) && gamma[*ii]*fabs(h[*ii]) <= *stop_b) { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
                else if ( (*exit_tri == TRLIB_TTR_CONV_INTERIOR) && gamma[*ii]*fabs(h[*ii]) <= sqrt(*stop_i)) { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
                else {
                    // prepare next iteration
                    *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_TRIVIAL; *status = TRLIB_CLS_L_NEW_ITER; break;
                }
                break;
            case TRLIB_CLS_L_NEW_ITER:
                // prepare next iteration
                *ii += 1;
                // check if we have to boil out due to iteration limit exceeded
                if (*ii >= itmax || (*ii - *lanczos_switch >= itmax_lanczos)) { *action = TRLIB_CLA_RETRANSF; *ityp = TRLIB_CLT_L; returnvalue = TRLIB_CLR_ITMAX; break; }
                *iter = *ii; *ityp = TRLIB_CLT_L; *flt1 = 1.0/gamma[*ii-1]; *flt2= 0.0; *action = TRLIB_CLA_UPDATE_DIR; *status = TRLIB_CLS_L_UPDATE_P;
                break;
            default: *action = TRLIB_CLA_TRIVIAL;
        }
        if (action != TRLIB_CLA_TRIVIAL || returnvalue <= 0) { break; }
    }
    TRLIB_RETURN(returnvalue)
}

int trlib_tri_factor_min(
    int nirblk, int *irblk, double *diag, double *offdiag,
    double *neglin, double radius, 
    int itmax, double tol_rel, int pos_def, int equality,
    int *warm0, double *lam0, int *warm, double *lam,
    int *warm_leftmost, int *ileftmost, double *leftmost,
    int *warm_fac0, double *diag_fac0, double *offdiag_fac0,
    int *warm_fac, double *diag_fac, double *offdiag_fac,
    double *sol0, double *sol, double *ones, double *fwork,
    int refine,
    int verbose, int unicode, char *prefix, FILE *fout,
    long *timing, double *obj, int *iter_newton, int *sub_fail) {
    // use notation of Gould paper
    // h = h(lam) denotes solution of (T+lam I) * h = -lin

    // local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    /* this is based on Theorem 5.8 in Gould paper,
     * the data for the first block has a 0 suffix,
     * the data for the \ell block has a l suffix */
    int n0 = irblk[1];                               // dimension of first block
    int nl;                                          // dimension of block corresponding to leftmost
    int nm0 = irblk[1]-1;                            // length of offdiagonal of first block
    int info_fac = 0;                                // factorization information
    int ret = 0;                                     // return code
    int newton = 0;                                  // perform newton iteration
    double lam_pert = 0.0;                           // perturbation of leftmost eigenvalue as starting value for lam
    double norm_sol0 = 0.0;                          // norm of h_0(lam)
    int interior = 0;                                // solution is interior
    *iter_newton = 0;                                // newton iteration counter
    int jj = 0;                                      // local iteration counter
    double dlam     = 0.0;                           // increment in newton iteration
    int inc = 1;                                     // increment in vector storage
    double *w = fwork;                               // auxiliary vector to be used in newton iteration
    double *diag_lam = fwork+(irblk[nirblk]);        // vector that holds diag + lam, could be saved if we would implement iterative refinement ourselve
    double *work = fwork+2*(irblk[nirblk]);          // workspace for iterative refinement
    double ferr = 0.0;                               // forward  error bound from iterative refinement
    double berr = 0.0;                               // backward error bound from iterative refinement
    double dot = 0.0;                                // save dot products
    double invD_norm_w_sq = 0.0;                     // || w ||_{D^-1}^2

    // FIXME: ensure diverse warmstarts work as expected
    
    // initialization:
    *sub_fail = 0;

    // set sol to 0 as a safeguard
    memset(sol, 0, irblk[nirblk]*sizeof(double));

    // first make sure that lam0, h_0 is accurate
    TRLIB_PRINTLN_1("Solving trust region problem, radius %e; starting on first irreducible block", radius)
    // if only blocks changed that differ from the first then there is nothing to do
    if (nirblk > 1 && *warm0) {
        TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
        TRLIB_PRINTLN_1("Solution provided via warmstart, \u03bb\u2080 = %e, \u2016h\u2080\u2016 = %e", *lam0, norm_sol0)
        if (norm_sol0-radius < 0.0) { TRLIB_PRINTLN_1("  violates \u2016h\u2080\u2016 - radius \u2265 0, but is %e, switch to coldstart", norm_sol0-radius) *warm0 = 0; }
        else { newton = 1; }
    }
    if (nirblk == 1 || !*warm0) {
        // seek for lam0, h_0 with (T0+lam0*I) pos def and ||h_0(lam_0)|| = radius

        /* as a first step to initialize the newton iteration,
         *  find such a pair with the losened requierement ||h_0(lam_0)|| >= radius */
        if(*warm0) {
            if(!*warm_fac0) {
                // factorize T + lam0 I
                TRLIB_DCOPY(&n0, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
                TRLIB_DAXPY(&n0, lam0, ones, &inc, diag_lam, &inc) // diag_lam <-- lam0 + diag_lam
                TRLIB_DCOPY(&n0, diag_lam, &inc, diag_fac0, &inc) // diag_fac0 <-- diag_lam
                TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac0, &inc) // offdiag_fac0 <-- offdiag
                TRLIB_DPTTRF(&n0, diag_fac0, offdiag_fac0, &info_fac) // compute factorization
                if (info_fac != 0) { *warm0 = 0; } // factorization failed, switch to coldastart
                else { *warm_fac0 = 1; }
            }
            if(*warm_fac0) {
                // solve for h0(lam0) and compute norm
                TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol0 <-- neglin
                TRLIB_DPTTRS(&n0, &inc, diag_fac0, offdiag_fac0, sol0, &n0, &info_fac) // sol <-- (T+lam0 I)^-1 sol
                if(info_fac!=0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
                TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
                if (norm_sol0 >= radius) { *warm0 = 1; newton = 1; } else { *warm0 = 0; }
            }
        }
        if(!*warm0) {
            *lam0 = 0.0;
            TRLIB_PRINTLN_1("Coldstart. Seeking suitable initial \u03bb\u2080, starting with 0")
            TRLIB_DCOPY(&n0, diag, &inc, diag_fac0, &inc) // diag_fac0 <-- diag0
            TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac0, &inc) // offdiag_fac0 <-- offdiag0
            TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol0 <-- neglin0
            TRLIB_DPTTRF(&n0, diag_fac0, offdiag_fac0, &info_fac) // compute factorization
            if (info_fac == 0) {
                // test if lam0 = 0 is suitable
                TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol0 <-- neglin
                TRLIB_DPTTRS(&n0, &inc, diag_fac0, offdiag_fac0, sol0, &n0, &info_fac) // sol0 <-- T0^-1 sol0
                if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
                TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
                if (norm_sol0<radius && equality) { info_fac = 1; } // in equality case we have to find suitable lam
            }
            if (info_fac != 0) { 
                TRLIB_PRINTLN_1(" \u03bb\u2080 = 0 unsuitable \u2265 get leftmost ev of first block!")
                *sub_fail = trlib_leftmost_irreducible(irblk[1], diag, offdiag, *warm_leftmost, *leftmost, 1000, TRLIB_EPS_POW_75, verbose-2, unicode, " LM ", fout, timing+10, leftmost, &jj); // ferr can safely be overwritten by computed leftmost for the moment as can jj with the number of rp iterations
                // if (*sub_fail != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LM) } failure of leftmost: may lead to inefficiency, since what we are doing may be slow...
                // T - leftmost*I is singular, so perturb it a bit to catch factorization; start with small perturbation and increase
                // first enlarge -leftmost to get into the region of positive definiteness
                lam_pert = (1.0+fabs(*leftmost)) * TRLIB_EPS_POW_75;
                while (lam_pert < (1.0+fabs(*leftmost))/TRLIB_EPS) {
                    *lam0 = -(*leftmost) + lam_pert;
                    TRLIB_PRINTLN_2(" attempting factorization with \u03bb\u2080 = %e, pert %e", *lam0, lam_pert)
                    // factorize T + lam I
                    TRLIB_DCOPY(&n0, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
                    TRLIB_DAXPY(&n0, lam0, ones, &inc, diag_lam, &inc) // diag_lam <-- lam + diag_lam
                    TRLIB_DCOPY(&n0, diag_lam, &inc, diag_fac0, &inc) // diag_fac <-- diag_lam
                    TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac0, &inc) // offdiag_fac <-- offdiag
                    TRLIB_DPTTRF(&n0, diag_fac0, offdiag_fac0, &info_fac) // compute factorization
                    if (info_fac == 0) { break; } // factorization possible, exit loop
                    lam_pert = 2.0*lam_pert;
                }
                if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_FACTOR) } // ensure we finally got a factorization
                // now we have lam0 such that (T0+lam0*I) pos def and is factorized
                // check if ||h0(lam0)|| >= radius (||h0(-leftmost)|| should be infinity)
                TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol0 <-- neglin
                TRLIB_DPTTRS(&n0, &inc, diag_fac0, offdiag_fac0, sol0, &n0, &info_fac) // sol0 <-- T0^-1 sol0
                if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
                //if (refine) { TRLIB_DPTRFS(&n0, &inc, diag, offdiag, diag_fac0, offdiag_fac0, neglin, &n0, sol0, &n0, &ferr, &berr, work, &info_fac) }
                if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
                TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
                // if ||h0(lam0)|| < radius, bisect lam0 until ||h0(lam0)|| >= radius
                jj = 0;
                ferr = -(*leftmost) + 0.5*lam_pert; // lower bound on lam0
                berr = *lam0; // uppper bound on lam0
                while ( (info_fac != 0 || norm_sol0 < radius) && jj < 500 && fabs(berr-ferr) > TRLIB_EPS*fabs(*lam0)) {
                    TRLIB_PRINTLN_2(" \u03bb\u2080 = %e unsuitable, \u2016h\u2080\u2016 = %e", *lam0, norm_sol0);
                    *lam0 = .5*(ferr+berr);// + .3*(berr-ferr); jj++;
                    TRLIB_PRINTLN_2(" ")
                    TRLIB_PRINTLN_2(" testing \u03bb\u2080 = %e, enclosure size %e", *lam0, berr - ferr);
                    // factorize T + lam I
                    TRLIB_DCOPY(&n0, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
                    TRLIB_DAXPY(&n0, lam0, ones, &inc, diag_lam, &inc) // diag_lam <-- lam + diag_lam
                    TRLIB_DCOPY(&n0, diag_lam, &inc, diag_fac0, &inc) // diag_fac <-- diag_lam
                    TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac0, &inc) // offdiag_fac <-- offdiag
                    TRLIB_DPTTRF(&n0, diag_fac0, offdiag_fac0, &info_fac) // compute factorization
                    if ( info_fac!= 0) { TRLIB_PRINTLN_2( " factorization failed, increase lower bound, difference %e", *lam0 - ferr) ferr = *lam0; } // factorization failed, increase lower bound
                    else{ 
                        TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol0 <-- neglin
                        TRLIB_DPTTRS(&n0, &inc, diag_fac0, offdiag_fac0, sol0, &n0, &info_fac) // sol0 <-- T0^-1 sol0
                        if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
                        TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
                        if(norm_sol0 < radius) { TRLIB_PRINTLN_2( " norm too small, decrease upper bound, difference %e", berr - *lam0) berr = *lam0; } // still to small, decrease upper bound
                    }
                }
                *warm_fac0 = 0;
            }
            if (norm_sol0<radius) {
                if(*lam0 == 0.0 && !equality) { ret = TRLIB_TTR_CONV_INTERIOR; }
                else { TRLIB_PRINTLN_1(" \u03bb\u2080 = -leftmost = %e with \u2016h\u2080\u2016 = %e turned out to be unsuitable, should be in theory!", *lam0, norm_sol0) TRLIB_RETURN(TRLIB_TTR_FAIL_HARD) }
            }
            else { newton = 1; }
            TRLIB_PRINTLN_1(" \u03bb\u2080 = %e suitable, \u2016h\u2080\u2016 = %e", *lam0, norm_sol0)
        }
    }
    if (newton) {
        /* now a suitable pair lam0, h0 has been found.
         * perform a newton iteration on 0 = 1/||h0(lam0)|| - 1/radius */
        TRLIB_PRINTLN_1("Starting Newton iteration for \u03bb\u2080")
        while (1) {
            /* compute newton correction to lam, by
                (1) Factoring T0 + lam0 I = LDL^T
                (2) Solving (T0+lam0 I)*h0 = -lin
                (3) L*w = h0/||h0||
                (4) compute increment (||h0||-Delta)/Delta/||w||_{D^-1}^2 */
    
            // steps (1) and (2) have already been performed on initializaton or previous iteration
    
            /* step (3) L*w = h/||h||
               compute ||w||_{D^-1}^2 in same loop */
            ferr = 1.0/norm_sol0; TRLIB_DCOPY(&n0, sol0, &inc, w, &inc) TRLIB_DSCAL(&n0, &ferr, w, &inc) // w <-- sol/||sol||
            invD_norm_w_sq = w[0]*w[0]/diag_fac0[0];
            for( jj = 1; jj < n0; ++jj ) { w[jj] = w[jj] - offdiag_fac0[jj-1]*w[jj-1]; invD_norm_w_sq += w[jj]*w[jj]/diag_fac0[jj]; }
    
            // step (4) compute increment (||h||-Delta)/Delta/||w||_{D^-1}^2
            dlam = (norm_sol0-radius)/(radius*invD_norm_w_sq);
    
            // iteration completed
            *iter_newton += 1;
    
            // test if dlam is not tiny or newton limit exceeded, return eventually
            if (fabs(dlam) <= TRLIB_EPS * fabs(*lam0) || *iter_newton > itmax) {
                if (unicode) { TRLIB_PRINTLN_1("%s%e%s%e", "Newton breakdown, d\u03bb = ", dlam, " \u03bb = ", *lam0) }
                else { TRLIB_PRINTLN_1("%s%e%s%e", "Newton breakdown, d\u03bb = ", dlam, " \u03bb = ", *lam0) }
                if(*iter_newton > itmax) { ret = TRLIB_TTR_ITMAX; break; }
                ret = TRLIB_TTR_NEWTON_BREAK; break;
            }
    
            // prepare next iteration
    
            // update lam
            *lam0 += dlam;
    
            // step (1) Factoring T0 + lam0 I = LDL^T
            TRLIB_DCOPY(&n0, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
            TRLIB_DAXPY(&n0, lam0, ones, &inc, diag_lam, &inc) // diag_lam <-- lam + diag_lam
            TRLIB_DCOPY(&n0, diag_lam, &inc, diag_fac0, &inc) // diag_fac <-- diag_lam
            TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac0, &inc) // offdiag_fac <-- offdiag
            TRLIB_DPTTRF(&n0, diag_fac0, offdiag_fac0, &info_fac) // compute factorization
            if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_FACTOR) }
    
            // step (2) Solving (T+lam I)*h = -lin
            TRLIB_DCOPY(&n0, neglin, &inc, sol0, &inc) // sol <-- neglin
            TRLIB_DPTTRS(&n0, &inc, diag_fac0, offdiag_fac0, sol0, &n0, &info_fac) // sol <-- (T+lam I)^-1 sol
            if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
            if (refine) { TRLIB_DPTRFS(&n0, &inc, diag_lam, offdiag, diag_fac0, offdiag_fac0, neglin, &n0, sol0, &n0, &ferr, &berr, work, &info_fac) }
            if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
            TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
    
            if (*iter_newton % 20 == 1) {
                if (unicode) { TRLIB_PRINTLN_1("%6s%14s%14s%14s", " iter ", "       \u03bb      ", "      d\u03bb      ", " \u2016h\u2080(\u03bb)\u2016-radius") }
                else { TRLIB_PRINTLN_1("%6s%14s%14s%14s", " iter ", "     lam      ", "     dlam     ", "  tr resdidual") }
            }
            TRLIB_PRINTLN_1("%6d%14e%14e%14e", *iter_newton, *lam0, dlam, norm_sol0-radius)
    
            // test for convergence
            if (norm_sol0 - radius <= tol_rel * radius) {
                // what if norm_sol < radius in a significant way?
                // theory tells this should not happen...
                    
                ret = TRLIB_TTR_CONV_BOUND; break;
            }
        }
    }

    *warm0 = 1;

    /* now in a situation were accurate lam0, h_0 exists to first irreducible block
     * invoke Theorem 5.8:
     * (i)  if lam0 >= -leftmost the pair lam0, h_0 solves the problem
     * (ii) if lam0 < -leftmost a solution has to be constructed to lam = -leftmost */

    // quick exit: only one irreducible block
    if (nirblk == 1) {
        *lam = *lam0; *warm = 1;
        TRLIB_DCOPY(&n0, sol0, &inc, sol, &inc) // sol <== sol0
        // compute objective. first store 2*gradient in w, then compute obj = .5*(sol, w)
        TRLIB_DCOPY(&n0, neglin, &inc, w, &inc) ferr = -2.0; TRLIB_DSCAL(&n0, &ferr, w, &inc) ferr = 1.0; // w <-- -2 neglin
        TRLIB_DLAGTM("N", &n0, &inc, &ferr, offdiag, diag, offdiag, sol, &n0, &ferr, w, &n0) // w <-- T*sol + w
        TRLIB_DDOT(dot, &n0, sol, &inc, w, &inc) *obj = 0.5*dot; // obj = .5*(sol, w)
        TRLIB_RETURN(ret)
    }

    // now that we have accurate lam, h_0 invoke Theorem 5.8
    // check if lam <= leftmost --> in that case the first block information describes everything
    TRLIB_PRINTLN_1("\nCheck if \u03bb\u2080 provides global solution, get leftmost ev for irred blocks")
    if(!*warm_leftmost) {
        *sub_fail = trlib_leftmost(nirblk, irblk, diag, offdiag, 0, leftmost[nirblk-1], 1000, TRLIB_EPS_POW_75, verbose-2, unicode, " LM ", fout, timing+10, ileftmost, leftmost);
        *warm_leftmost = 1;
    }
    TRLIB_PRINTLN_1("    leftmost = %e (block %d)", leftmost[*ileftmost], *ileftmost)
    if(*lam0 >= -leftmost[*ileftmost]) {
        if (unicode) { TRLIB_PRINTLN_1("  \u03bb\u2080 \u2265 -leftmost \u21d2 \u03bb = \u03bb\u2080, exit: h\u2080(\u03bb\u2080)") }
        else { TRLIB_PRINTLN_1("  lam0 >= -leftmost => lam = lam0, exit: h0(lam0)") }
        *lam = *lam0; *warm = 1;
        TRLIB_DCOPY(&n0, sol0, &inc, sol, &inc) // sol <== sol0
        // compute objective. first store 2*gradient in w, then compute obj = .5*(sol, w)
        TRLIB_DCOPY(&n0, neglin, &inc, w, &inc) ferr = -2.0; TRLIB_DSCAL(&n0, &ferr, w, &inc) ferr = 1.0; // w <-- -2 neglin
        TRLIB_DLAGTM("N", &n0, &inc, &ferr, offdiag, diag, offdiag, sol, &n0, &ferr, w, &n0) // w <-- T*sol + w
        TRLIB_DDOT(dot, &n0, sol, &inc, w, &inc) *obj = 0.5*dot; // obj = .5*(sol, w)
        TRLIB_RETURN(ret)
    }
    else {
        if (unicode) { TRLIB_PRINTLN_1("  -leftmost > \u03bb\u2080 \u21d2 \u03bb = -leftmost, exit: h\u2080(-leftmost) + \u03b1 u") }
        else  { TRLIB_PRINTLN_1("  -leftmost > lam0 => lam = -leftmost, exit: h0(-leftmost) + alpha u") }

        // Compute solution of (T0 - leftmost*I)*h0 = neglin
        *lam = -leftmost[*ileftmost]; *warm = 1;
        TRLIB_DCOPY(&n0, neglin, &inc, sol, &inc) // neglin <-- sol
        if(!*warm_fac){
            TRLIB_DCOPY(&n0, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
            TRLIB_DAXPY(&n0, lam, ones, &inc, diag_lam, &inc) // diag_lam <-- lam + diag_lam
            TRLIB_DCOPY(&n0, diag_lam, &inc, diag_fac, &inc) // diag_fac <-- diag_lam
            TRLIB_DCOPY(&nm0, offdiag, &inc, offdiag_fac, &inc) // offdiag_fac <-- offdiag
            TRLIB_DPTTRF(&n0, diag_fac, offdiag_fac, &info_fac) // compute factorization
            if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_FACTOR) } 
        }
        *warm_fac = 1;
        TRLIB_DPTTRS(&n0, &inc, diag_fac, offdiag_fac, sol, &n0, &info_fac) // sol <-- (T+lam I)^-1 sol
        if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
        if (refine) { TRLIB_DPTRFS(&n0, &inc, diag_lam, offdiag, diag_fac, offdiag_fac, neglin, &n0, sol, &n0, &ferr, &berr, work, &info_fac) }
        if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
        TRLIB_DNRM2(norm_sol0, &n0, sol, &inc)

        // compute normalized eigenvector u corresponding to leftmost of block ileftmost
        srand((unsigned) time(NULL));
        for( int kk = irblk[*ileftmost]; kk < irblk[*ileftmost+1]; ++kk ) { sol[kk] = ((double)rand()/(double)RAND_MAX); }
        nl = irblk[*ileftmost+1]-irblk[*ileftmost];
        *sub_fail = trlib_eigen_inverse(nl, diag+irblk[*ileftmost], offdiag+irblk[*ileftmost], 
                leftmost[*ileftmost], 10, TRLIB_EPS_POW_5, ones,
                diag_fac+irblk[*ileftmost], offdiag_fac+irblk[*ileftmost],
                sol+irblk[*ileftmost], 
                verbose-2, unicode, " EI", NULL, timing+11, &ferr, &berr, &jj); // can savely overwrite ferr, berr, jj with results. only interesting: eigenvector
        if (*sub_fail != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_EIG) }

        // solution is of form [h,0,...,0,alpha*u,0,...,0]
        // alpha = sqrt( radius^2 - ||h||^2 )
        ferr = sqrt( radius*radius - norm_sol0*norm_sol0 );
        TRLIB_DSCAL(&nl, &ferr, sol+irblk[*ileftmost], &inc)

        if (unicode) { TRLIB_PRINTLN_1("    with \u2016h\u2080(-leftmost)\u2016 = %e, \u03b1 = %e", norm_sol0, ferr) }
        else { TRLIB_PRINTLN_1("    with ||h0(-leftmost)|| = %e, alpha = %e", norm_sol0, ferr) }

        ret = TRLIB_TTR_HARD;
        
        // compute objective. first store 2*gradient in w, then compute obj = .5*(sol, w)
        *obj = 0.5*leftmost[*ileftmost]*ferr*ferr;
        TRLIB_DCOPY(&n0, neglin, &inc, w, &inc) ferr = -2.0; TRLIB_DSCAL(&n0, &ferr, w, &inc) ferr = 1.0; // w <-- -2 neglin
        TRLIB_DLAGTM("N", &n0, &inc, &ferr, offdiag, diag, offdiag, sol, &n0, &ferr, w, &n0) // w <-- T*sol + w
        TRLIB_DDOT(dot, &n0, sol, &inc, w, &inc) *obj = *obj+0.5*dot; // obj = .5*(sol, w)
        TRLIB_RETURN(ret);
    }
}

int trlib_leftmost(
        int nirblk, int *irblk, double *diag, double *offdiag,
        int warm, double leftmost_minor, int itmax, double tol_abs,
        int verbose, int unicode, char *prefix, FILE *fout,
        long *timing, int *ileftmost, double *leftmost) {
    int ret = 0; int curit;
    // FIXME: get pthreads running
    if(! warm) {
//        pthread_t* threads = malloc(nirblk*sizeof(pthread_t));
//        struct trlib_leftmost_data* lmd = malloc(nirblk*sizeof(struct trlib_leftmost_data));
        ret = 0; int curret;
        for(int ii = 0; ii < nirblk; ++ii) {
//            lmd[ii].n = irblk[ii+1]-irblk[ii];
//            lmd[ii].diag = diag+irblk[ii];
//            lmd[ii].offdiag = offdiag+irblk[ii];
//            lmd[ii].warm = 0;
//            lmd[ii].leftmost_minor = 0.0;
//            lmd[ii].itmax = itmax;
//            lmd[ii].tol_abs = tol_abs;
//            lmd[ii].verbose = verbose;
//            lmd[ii].unicode = unicode;
//            lmd[ii].prefix = prefix;
//            lmd[ii].fout = fout;
//            lmd[ii].timing = timing;
//            lmd[ii].leftmost = leftmost+ii;
//            lmd[ii].iter_pr = iter_pr+ii;
//            pthread_create(threads+ii, NULL, trlib_leftmost_irreducible_pthread, (void*) lmd+ii);
//            pthread_join(threads[ii], NULL);
            curret = trlib_leftmost_irreducible(irblk[ii+1]-irblk[ii], diag+irblk[ii], offdiag+irblk[ii], 0, 0.0, itmax,
                tol_abs, verbose, unicode, prefix, fout, timing, leftmost+ii, &curit);
            if (curret == 0) { ret = curret; }
        }
//        for(int ii = 0; ii < nirblk; ++ii) {
//            pthread_join(threads[ii], NULL);
//        }
        *ileftmost = 0;
        for(int ii = 1; ii < nirblk; ++ii) {
            if (leftmost[ii] < leftmost[*ileftmost]) { *ileftmost = ii; }
        }
        // free(threads); free(lmd);
    }
    else { 
        ret = trlib_leftmost_irreducible(irblk[nirblk] - irblk[nirblk-1], diag+irblk[nirblk-1], offdiag+irblk[nirblk-1],
                1, leftmost_minor, itmax, tol_abs, verbose, unicode, prefix, fout, timing, leftmost+nirblk-1, &curit);
        if (leftmost[nirblk-1] < leftmost[*ileftmost]) { *ileftmost = nirblk-1; }
    }
    return ret;
}

void *trlib_leftmost_irreducible_pthread( void *leftmost_data ) {
//    fprintf(stderr, "%s%ld\n", "Inside leftmost_irreducible_pthread, got address ", (long) leftmost_data);
    struct trlib_leftmost_data *lmd = (struct trlib_leftmost_data*) leftmost_data;
    int ret;
//    fprintf(stderr, "See n: %d\n", lmd->n);
    ret = trlib_leftmost_irreducible(lmd->n, lmd->diag, lmd->offdiag, lmd->warm, lmd->leftmost_minor, lmd->itmax,
            lmd->tol_abs, lmd->verbose, lmd->unicode, lmd->prefix, lmd->fout, lmd->timing, lmd->leftmost, lmd->iter_pr);
    pthread_exit( NULL );
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
            TRLIB_PRINTLN_1("%6s%8s%14s%14s%14s%14s%6s%6s", "  it  ", " action ", "     low      ", "   leftmost   ", "      up      ", "      prlp    ", " nneg ", "  br  ")
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
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%6d%6d", "", " bisecp ", low, *leftmost, up, prlp, n_neg_piv, jj)
            continue; 
        }

        // we have survived computing the Last-Pivot value without finding a zero pivot and at most one negative pivot

        // adapt bracket, no negative pivots encountered: leftmost provides new lower bound, otherwise upper bound
        if (n_neg_piv == 0) { low = *leftmost; }
        else { up = *leftmost; }

        // test if bracket interval is small or last pivot has converged to zero
        if (up-low <= tol_abs * fmax(1.0, fmax(fabs(low), fabs(up))) || fabs(prlp) <= tol_abs) { 
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%6d%6d", "", "  conv  ", low, *leftmost, up, prlp, n_neg_piv, jj)
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
        }
        else { leftmost_attempt = *leftmost - prlp/dprlp; } // Newton step

        // assess if we can use trial step
        if (low <= leftmost_attempt && leftmost_attempt <= up) { 
            if ( warm ) { TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%6d%6d", "", " qmodel ", low, *leftmost, up, prlp, n_neg_piv, jj) }
            else { TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%6d%6d", "", " newton ", low, *leftmost, up, prlp, n_neg_piv, jj) }
            *leftmost = leftmost_attempt;
        }
        else { 
            TRLIB_PRINTLN_1("%6s%8s%14e%14e%14e%14e%6d%6d", "", " bisecs ", low, *leftmost, up, prlp, n_neg_piv, jj)
            *leftmost = .5*(low+up);
        }
    }
}

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

    if ( *iter_inv == TRLIB_EIR_FAIL_FACTOR ) { TRLIB_RETURN(TRLIB_EIR_FAIL_FACTOR) }

    TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
    TRLIB_DSCAL(&n, &invnorm, eig, &inc) // normalize eig
    // perform inverse iteration
    while (1) {
        *iter_inv += 1;

        if ( *iter_inv > itmax ) { TRLIB_RETURN(TRLIB_EIR_ITMAX) }

        // solve (T - lam*I)*eig_new = eig_old
        TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, eig, &n, &info_fac)
        if( info_fac != 0 ) { TRLIB_RETURN(TRLIB_EIR_FAIL_LINSOLVE) }

        // normalize eig
        TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
        TRLIB_DSCAL(&n, &invnorm, eig, &inc)

        // check for convergence
        if (fabs( invnorm - *pert ) <= tol_abs ) { TRLIB_RETURN(TRLIB_EIR_CONV) }
    }
    
    TRLIB_RETURN(TRLIB_EIR_ITMAX)
};

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

