#include "trlib_krylov.h"

int trlib_krylov_min(
    int init, double radius, int equality, int itmax, int itmax_lanczos,
    double tol_rel_i, double tol_abs_i,
    double tol_rel_b, double tol_abs_b, double zero,
    int ctl_invariant, double g_dot_g, double v_dot_g, double p_dot_Hp,
    int *iwork, double *fwork, int refine,
    int verbose, int unicode, char *prefix, FILE *fout, long *timing,
    int *action, int *iter, int *ityp,
    double *flt1, double *flt2, double *flt3) {
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

    long *leftmost_timing = NULL;
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        leftmost_timing = timing + 1;
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
    int *nirblk = iwork + 14;
    int *irblk = iwork+15;

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
                *nirblk = 1;  // at start, there is one irreducible block
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
                if (v_dot_g <= 0.0 && g_dot_g > 0.0) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_TRIVIAL; returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
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
                    irblk[*nirblk] = *ii+1;
                    *exit_tri = trlib_tri_factor_min(
                        *nirblk, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                        warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                        &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                        h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                        leftmost_timing, obj, iter_tri, sub_fail_tri);

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
                // if g == 0: Krylov breakdown or convergence
                // if g != 0 and (v,g) <= 0 ---> preconditioner indefinite
                if(g_dot_g > 0.0 && v_dot_g <= 0.0) { if (*interior) {*action = TRLIB_CLA_TRIVIAL;} else {*ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF;} returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
                if (g_dot_g <= 0.0) { // Krylov iteration breaks down
                    if ( ctl_invariant <= TRLIB_CLC_NO_EXP_INV ) {
                        if ( *interior ) { *action = TRLIB_CLA_TRIVIAL; } else { *action = TRLIB_CLA_RETRANSF; }
                        *ityp = TRLIB_CLT_CG; returnvalue = TRLIB_CLR_FAIL_HARD; break;
                    }
                    else { 
                        // FIXME: add the case ctl_invariant == TRLIB_CLC_EXP_INV_GLO
                        /* decide if a new invariant Krylov subspace should be investigated
                           therefore compute actual gradient at current point and test for convergence */
                        if(*interior) { *action = TRLIB_CLA_TRIVIAL; } else { *action = TRLIB_CLA_RETRANSF; }
                        *ityp = TRLIB_CLT_CG; *status = TRLIB_CLS_CG_IF_IRBLK_P; returnvalue = TRLIB_CLR_CONTINUE; break;
                    }
                }

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
            case TRLIB_CLS_CG_IF_IRBLK_P:
                // compute convergence criterion
                *flt1 = *lam; *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_CONV_HARD;
                *status = TRLIB_CLS_CG_IF_IRBLK_C; returnvalue = TRLIB_CLR_CONTINUE; break;
            case TRLIB_CLS_CG_IF_IRBLK_C:
                // print iteration details
                // first print headline if necessary
                TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "||g(lam)||_iM", "   leftmost   ", "     lam      ")
                TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e", *ii, *iter_tri, "cg_h", *obj, v_dot_g, *leftmost, *lam) TRLIB_PRINTLN_2("%s", "")
                // check for convergence
                if (v_dot_g <= *stop_b) { *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_TRIVIAL; returnvalue = TRLIB_CLR_APPROX_HARD; break; }
                // if no convergence continue with next invariant Krylov subspace
                TRLIB_PRINTLN_2("No convergence within invariant subspace. Investigate next invariant subspace") TRLIB_PRINTLN_2("%s","") 
                *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_NEW_KRYLOV;
                *status = TRLIB_CLS_CG_IF_IRBLK_N; returnvalue = TRLIB_CLR_CONTINUE; break;
            case TRLIB_CLS_CG_IF_IRBLK_N:
                irblk[*nirblk] = *ii+1;
                (*nirblk)++;
                gamma[*ii] = sqrt(v_dot_g); // do not misinterpret this is as value of tridiagonal matrix, there it is 0
                *lanczos_switch = *ii;
                *ii += 1;
                *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_TRIVIAL; *status = TRLIB_CLS_L_UPDATE_P;
                returnvalue = TRLIB_CLR_CONTINUE; break;
            case TRLIB_CLS_HOTSTART:
                /* reentry with smaller trust region radius
                   we implement hotstart by not making use of the CG basis but rather the Lanczos basis
                   as this covers both cases: the interior and the boundary cases
                   the additional cost by doing this is neglible since we most likely will just do one iteration */
                // solve the corresponding tridiagonal problem, check for convergence and otherwise continue to iterate
                irblk[*nirblk] = *ii+1;
                *exit_tri = trlib_tri_factor_min(
                    *nirblk, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                    warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                    &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                    h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                    leftmost_timing, obj, iter_tri, sub_fail_tri);

                /* check for failure, beware: newton break is ok as this means most likely convergence
                   exit with error and ask the user to get (potentially invalid) solution candidate by backtransformation */
                if (*exit_tri < 0 && *exit_tri != TRLIB_TTR_NEWTON_BREAK) {
                    *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_FAIL_TTR; break;
                }

                ///* if tridiagonal problem cannot find suitable initial lambda it is most likely best to stop at this point
                // * since this means that there is severe ill-conditioning and the user should better present a
                // * better problem formulation. Continuing means most likely computing on garbage */
                if (*exit_tri == TRLIB_TTR_HARD_INIT_LAM) {
                    *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_HARD_INIT_LAM; break;
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
                    else { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_TRIVIAL; *status = TRLIB_CLS_L_NEW_ITER; break; }
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
                if ( fabs(p_dot_Hp) <= 0.0) { // Krylov iteration breaks down
                    if ( ctl_invariant <= TRLIB_CLC_EXP_INV_GLO ) {
                        *ityp = TRLIB_CLT_CG; *action = TRLIB_CLA_RETRANSF;
                        returnvalue = TRLIB_CLR_FAIL_HARD; break;
                        }
                }
                delta[*ii] = p_dot_Hp;
                /* solve tridiagonal reduction
                   first try to update factorization if available to start tridiagonal problem warmstarted */
                if( nirblk == 1) {
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
                }
                else {
                    // FIXME: implement proper warmstart
                    *warm_lam0 = 0; *warm_lam = 0; *warm_leftmost = 0;
                }
                /* call factor_min to solve tridiagonal problem, store solution candidate in h
                   the criterion to specify the maximum number of iterations is weird. it should not be dependent on problem size rather than condition of the hessian... */
                irblk[*nirblk] = *ii+1;
                *exit_tri = trlib_tri_factor_min(
                    *nirblk, irblk, delta, gamma, neglin, radius, 100+3*(*ii), TRLIB_EPS, *pos_def, equality,
                    warm_lam0, lam0, warm_lam, lam, warm_leftmost, ileftmost, leftmost,
                    &warm_fac0, delta_fac0, gamma_fac0, &warm_fac, delta_fac, gamma_fac,
                    h0, h, ones, fwork_tr, refine, verbose-1, unicode, " TR ", fout,
                    leftmost_timing, obj, iter_tri, sub_fail_tri);

                /* check for failure, beware: newton break is ok as this means most likely convergence
                   exit with error and ask the user to get (potentially invalid) solution candidate by backtransformation */
                if (*exit_tri < 0 && *exit_tri != TRLIB_TTR_NEWTON_BREAK) {
                    *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_FAIL_TTR; break;
                }

                ///* if tridiagonal problem cannot find suitable initial lambda it is most likely best to stop at this point
                // * since this means that there is severe ill-conditioning and the user should better present a
                // * better problem formulation. Continuing means most likely computing on garbage.
                // * Ill-conditioning is likely since we already are in Lanczos mode. */
                if (*exit_tri == TRLIB_TTR_HARD_INIT_LAM) {
                    *ityp = TRLIB_CLT_LANCZOS; *action = TRLIB_CLA_RETRANSF; returnvalue = TRLIB_CLR_HARD_INIT_LAM; break;
                }

                /* convergence check is logical at this position, *but* requires gamma(ii+1).
                   wait until gradient has been updated */
                // compute g^L(ii+1)
                *flt1 = -delta[*ii]/gamma[*ii-1]; *flt2 = -gamma[*ii-1]/gamma[*ii-2]; *flt3 = 1.0;
                // in the case that we just switched to Lanczos, we have to use different coefficients
                if (*ii == *lanczos_switch && *nirblk == 1) {
                    *flt1 = -delta[*ii]/sqrt(*v_g)*(*sigma); *flt2 = -gamma[*ii-1]*(*cgl); *flt3 = gamma[*ii-1]/sqrt(*v_g);
                    *cgl = 1.0; *cglm = 1.0;
                }
                // as well in the case that we have a new irreducible block
                if (*ii == irblk[*nirblk-1]) {
                    *flt1 = -delta[*ii]/gamma[*ii-1]; *flt2 = 0; *flt3 = 1.0;
                }
                *ityp = TRLIB_CLT_L;  *action = TRLIB_CLA_UPDATE_GRAD; *status = TRLIB_CLS_L_CMP_CONV;
                break;
            case TRLIB_CLS_L_CMP_CONV:
                // convergence check after new gradient has been computed
                if (v_dot_g <= 0.0) { if (*interior) {*action = TRLIB_CLA_TRIVIAL;} else {*ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF;} returnvalue = TRLIB_CLR_PCINDEF; break; } // exit if M^-1 indefinite
                gamma[*ii] = sqrt(v_dot_g);
                if (*nirblk == 1) {
                    // compute convergence indicator, store it in *v_g
                    *v_g = v_dot_g * h[*ii]*h[*ii];
                    *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_TRIVIAL; *status = TRLIB_CLS_L_CHK_CONV;
                }
                else { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_CONV_HARD; *status = TRLIB_CLS_L_CHK_CONV; }
                break;
            case TRLIB_CLS_L_CHK_CONV:
                // get convergence indicator in *v_g
                if (*nirblk > 1 ) { *v_g = v_dot_g; }
                // print some information
                // first print headline if necessary
                if (((*ii)-(*iter_last_head)) % 20 == 0 || *type_last_head != TRLIB_CLT_LANCZOS) {
                    if (unicode) { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "   \u03b3\u1d62\u208a\u2081|h\u1d62|   ", "   leftmost   ", "      \u03bb       ", "      \u03b3       ", "      \u03b4       ") }
                    else { TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6s%6s%6s%14s%14s%14s%14s%14s%14s", " iter ", "inewton", " type ", "   objective  ", "gam(i+1)|h(i)|", "   leftmost   ", "     lam      ", "    gamma     ", "    delta     ") }
                    *type_last_head = TRLIB_CLT_LANCZOS;
                    *iter_last_head = *ii;
                }
                TRLIB_PRINTLN_2("%s","") TRLIB_PRINTLN_1("%6d%6d%6s%14e%14e%14e%14e%14e%14e", *ii, *iter_tri, " lcz", *obj, sqrt(*v_g), *leftmost, *lam, *ii == 0 ? neglin[0] : gamma[*ii-1], delta[*ii]) TRLIB_PRINTLN_2("%s", "")

                // test for convergence
                if ( (*exit_tri != TRLIB_TTR_CONV_INTERIOR) && *v_g <= sqrt(*stop_b)) { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
                else if ( (*exit_tri == TRLIB_TTR_CONV_INTERIOR) && *v_g <= sqrt(*stop_i)) { *ityp = TRLIB_CLT_L; *action = TRLIB_CLA_RETRANSF; returnvalue = *exit_tri; break; }
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

int trlib_krylov_prepare_memory(int itmax, double *fwork) {
    for(int jj = 21+11*itmax; jj<22+12*itmax; ++jj) { *(fwork+jj) = 1.0; } // everything to 1.0 in ones
    memset(fwork+15+2*itmax, 0, itmax*sizeof(double)); // neglin = - gamma_0 e1, thus set neglin[1:] = 0
    return 0;
}

int trlib_krylov_memory_size(int itmax, int *iwork_size, int *fwork_size, int *h_pointer) {
    *iwork_size = 16+itmax;
    *fwork_size = 23+13*itmax+trlib_tri_factor_memory_size(itmax+1);
    *h_pointer = 17+4*itmax;
    return 0;
}

int trlib_krylov_timing_size() {
#if TRLIB_MEASURE_TIME
    return 1 + trlib_tri_timing_size();
#endif
    return 0;
}

