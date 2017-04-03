#include "trlib_test.h"

trlib_int_t trlib_test_malloc_qp(trlib_int_t qptype, trlib_int_t qpsolver, trlib_int_t n, trlib_int_t itmax, struct trlib_test_qp *qp) {
    qp->qptype = qptype;
    qp->qpsolver = qpsolver;
    qp->radius = 0.0;
    qp->prefix = calloc(1, sizeof(char));
    qp->verbose = 1;
    qp->unicode = 0;
    qp->stream = NULL;
    qp->itmax = itmax;
    qp->equality = 0;
    qp->reentry = 0;
    qp->refine = 1;
    qp->lam = 0.0;
    qp->tol_rel_i = TRLIB_EPS_POW_5;
    qp->tol_abs_i = 0.0;
    qp->tol_rel_b = TRLIB_EPS_POW_4;
    qp->tol_abs_b = 0.0;
    qp->ctl_invariant = TRLIB_CLC_NO_EXP_INV;
    if(qptype == TRLIB_TEST_DENSE_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_test_problem_dense));
        struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
        problem->n = n;
        problem->hess = calloc(n*n, sizeof(trlib_flt_t));
        problem->grad = calloc(n, sizeof(trlib_flt_t));
        problem->sol = malloc(n*sizeof(trlib_flt_t));
    }
    if(qptype == TRLIB_TEST_TRI_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_test_problem_tri));
        struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
        problem->n = n;
        problem->diag = calloc(n, sizeof(trlib_flt_t));
        problem->offdiag = calloc(n-1, sizeof(trlib_flt_t));
        problem->diag_fac = malloc(n*sizeof(trlib_flt_t));
        problem->diag_fac0 = malloc(n*sizeof(trlib_flt_t));
        problem->offdiag_fac = malloc((n-1)*sizeof(trlib_flt_t));
        problem->offdiag_fac0 = malloc((n-1)*sizeof(trlib_flt_t));
        problem->grad = calloc(n, sizeof(trlib_flt_t));
        problem->neggrad = calloc(n, sizeof(trlib_flt_t));
        problem->sol = malloc(n*sizeof(trlib_flt_t));
        problem->sol0 = malloc(n*sizeof(trlib_flt_t));
        problem->warm_fac = 0;
        problem->warm_fac0 = 0;
        problem->pos_def = 0;
    }
    if(qptype == TRLIB_TEST_OP_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_test_problem_op));
        struct trlib_test_problem_op* problem = (struct trlib_test_problem_op *)qp->problem;
        problem->n = n;
        problem->grad = calloc(n, sizeof(trlib_flt_t));
        problem->sol = malloc(n*sizeof(trlib_flt_t));
    }

    if(qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        qp->timing = malloc(trlib_krylov_timing_size()*sizeof(trlib_int_t));
        qp->work = (void *)malloc(sizeof(struct trlib_test_work_krylov));
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        trlib_int_t iwork_size, fwork_size, h_pointer;
        trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size, &h_pointer);
        work->iwork = malloc(iwork_size*sizeof(trlib_int_t));
        work->fwork = malloc(fwork_size*sizeof(trlib_flt_t));
        work->g = malloc(n*sizeof(trlib_flt_t));
        work->gm = malloc(n*sizeof(trlib_flt_t));
        work->p = malloc(n*sizeof(trlib_flt_t));
        work->Hp = malloc(n*sizeof(trlib_flt_t));
        work->Q = malloc((itmax+1)*n*sizeof(trlib_flt_t));
        work->orth_check = malloc((itmax+1)*(itmax+1)*sizeof(trlib_flt_t));
    }
    if(qpsolver == TRLIB_TEST_SOLVER_FACTOR) {
        qp->timing = malloc(trlib_tri_timing_size()*sizeof(trlib_int_t));
        qp->work = (void *)malloc(sizeof(struct trlib_test_work_factor));
        struct trlib_test_work_factor * work = (struct trlib_test_work_factor *)qp->work;
        work->fwork = malloc(4*n*sizeof(trlib_flt_t));
        work->ones = malloc(n*sizeof(trlib_flt_t));
        work->nirblk = 0;
        work->warm_lam = 0;
        work->warm_lam0 = 0;
        work->lam0 = 0.0;
        work->warm_leftmost = 0;
        work->ileftmost = 0;
        for(trlib_int_t ii = 0; ii < n; ++ii) { work->ones[ii] = 1.0; }
    }

    return 0;
}

trlib_int_t trlib_test_free_qp(struct trlib_test_qp *qp) {
    if(qp->prefix != NULL) { free(qp->prefix); }
    if(qp->timing != NULL) { free(qp->timing); }
    if(qp->qptype == TRLIB_TEST_DENSE_QP) {
        struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
        if(problem != NULL) {
            if(problem->hess != NULL) { free(problem->hess); }
            if(problem->grad != NULL) { free(problem->grad); }
            if(problem->sol != NULL) { free(problem->sol); }
            free(problem);
        }
    }
    if(qp->qptype == TRLIB_TEST_TRI_QP) {
        struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
        if(problem != NULL) {
            if(problem->diag != NULL) { free(problem->diag); }
            if(problem->offdiag != NULL) { free(problem->offdiag); }
            if(problem->diag_fac != NULL) { free(problem->diag_fac); }
            if(problem->diag_fac0 != NULL) { free(problem->diag_fac0); }
            if(problem->offdiag_fac != NULL) { free(problem->offdiag_fac); }
            if(problem->offdiag_fac0 != NULL) { free(problem->offdiag_fac0); }
            if(problem->grad != NULL) { free(problem->grad); }
            if(problem->neggrad != NULL) { free(problem->neggrad); }
            if(problem->sol != NULL) { free(problem->sol); }
            if(problem->sol0 != NULL) { free(problem->sol0); }
            free(problem);
        }
    }
    if(qp->qptype == TRLIB_TEST_OP_QP) {
        struct trlib_test_problem_op* problem = (struct trlib_test_problem_op *)qp->problem;
        if(problem != NULL) {
            if(problem->grad != NULL) { free(problem->grad); }
            if(problem->sol != NULL) { free(problem->sol); }
            problem->userdata = NULL;
            problem->hv = NULL;
            free(problem);
        }
    }

    if(qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        if(work != NULL) {
            if(work->iwork != NULL) { free(work->iwork); }
            if(work->fwork != NULL) { free(work->fwork); }
            if(work->g != NULL) { free(work->g); }
            if(work->gm != NULL) { free(work->gm); }
            if(work->p != NULL) { free(work->p); }
            if(work->Hp != NULL) { free(work->Hp); }
            if(work->Q != NULL) { free(work->Q); }
            if(work->orth_check != NULL) { free(work->orth_check); }
            free(work);
        }
    }
    if(qp->qpsolver == TRLIB_TEST_SOLVER_FACTOR) {
        struct trlib_test_work_factor * work = (struct trlib_test_work_factor *)qp->work;
        if(work != NULL) {
            if(work->fwork != NULL) { free(work->fwork); }
            if(work->ones != NULL) { free(work->ones); }
            if(work->irblk != NULL) { free(work->irblk); }
            if(work->leftmost != NULL) { free(work->leftmost); }
            free(work);
        }
    }

    return 0;
}

trlib_int_t trlib_test_solve_qp(struct trlib_test_qp *qp) { 
    if(qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        trlib_int_t n; trlib_flt_t *grad, *sol, *hess, *diag, *offdiag;
        void (*hv)(void *, trlib_int_t, trlib_flt_t *, trlib_flt_t *); trlib_flt_t *userdata;
        if(qp->qpsolver == TRLIB_TEST_DENSE_QP) { 
            struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; hess = problem->hess;
        }
        if(qp->qptype == TRLIB_TEST_TRI_QP) {
            struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; diag = problem->diag; offdiag = problem->offdiag;
        }
        if(qp->qptype == TRLIB_TEST_OP_QP) {
            struct trlib_test_problem_op* problem = (struct trlib_test_problem_op *)qp->problem;
            n = problem->n; grad = problem->grad; hv = problem->hv; userdata = problem->userdata; sol = problem->sol;
        }
        trlib_int_t init = 0, inc = 1, itp1 = 0;
        trlib_flt_t minus = -1.0, one = 1.0, z = 0.0;
        if(!qp->reentry) { init = TRLIB_CLS_INIT; trlib_krylov_prepare_memory(qp->itmax, work->fwork); }
        else { init = TRLIB_CLS_HOTSTART; }

        trlib_flt_t v_dot_g = 0.0, p_dot_Hp = 0.0, flt1, flt2, flt3;
        trlib_int_t action, ityp;

        trlib_int_t iwork_size, fwork_size, h_pointer;
        trlib_krylov_memory_size(qp->itmax, &iwork_size, &fwork_size, &h_pointer);

        while(1) {
            qp->ret = trlib_krylov_min(init, qp->radius, qp->equality, qp->itmax, 100,
                    qp->tol_rel_i, qp->tol_abs_i, qp->tol_rel_b, qp->tol_abs_b,
                    TRLIB_EPS*TRLIB_EPS, -1e20, qp->ctl_invariant, 0, 0, v_dot_g, v_dot_g, p_dot_Hp, work->iwork, work->fwork, 
                    qp->refine, qp->verbose, qp->unicode, qp->prefix, qp->stream, qp->timing,
                    &action, &(qp->iter), &ityp, &flt1, &flt2, &flt3);
            init = 0;
            //fprintf(stderr, "action: %ld, iter: %ld, ityp: %ld, flt1: %e, flt2: %e, flt3: %e\n",
            //        action, qp->iter, ityp, flt1, flt2, flt3);
            switch(action) {
                case TRLIB_CLA_INIT:
                    memset(sol, 0, n*sizeof(trlib_flt_t)); memset(work->gm, 0, n*sizeof(trlib_flt_t));
                    dcopy_(&n, grad, &inc, work->g, &inc);
                    v_dot_g = ddot_(&n, work->g, &inc, work->g, &inc);
                    dcopy_(&n, work->g, &inc, work->p, &inc); dscal_(&n, &minus, work->p, &inc); // p = -g
                    if(qp->qptype == TRLIB_TEST_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_OP_QP) {
                        hv(userdata, n, work->p, work->Hp); // Hp = H*p
                    }
                    p_dot_Hp = ddot_(&n, work->p, &inc, work->Hp, &inc);
                    dcopy_(&n, work->g, &inc, work->Q, &inc); // Q(0:n) = g
                    flt1 = 1.0/sqrt(v_dot_g); dscal_(&n, &flt1, work->Q, &inc); // Q(0:n) = g/sqrt(<g,g>)
                    break;
                case TRLIB_CLA_RETRANSF:
                    itp1 = qp->iter+1;
                    dgemv_("N", &n, &itp1, &one, work->Q, &n, work->fwork+h_pointer, &inc, &z, sol, &inc); // s = Q_i * h_i
                    break;
                case TRLIB_CLA_UPDATE_STATIO:
                    if (ityp == TRLIB_CLT_CG) { daxpy_(&n, &flt1, work->p, &inc, sol, &inc); }; // s += flt1*p
                    break;
                case TRLIB_CLA_UPDATE_GRAD:
                    if (ityp == TRLIB_CLT_CG) {
                        dcopy_(&n, work->g, &inc, work->Q+(qp->iter)*n, &inc); dscal_(&n, &flt2, work->Q+(qp->iter)*n, &inc); // Q(iter*n:(iter+1)*n) = flt2*g
                        dcopy_(&n, work->g, &inc, work->gm, &inc); daxpy_(&n, &flt1, work->Hp, &inc, work->g, &inc); // gm = g; g += flt1*Hp
                    }
                    if (ityp == TRLIB_CLT_L) {
                        dcopy_(&n, work->Hp, &inc, sol, &inc); daxpy_(&n, &flt1, work->g, &inc, sol, &inc); daxpy_(&n, &flt2, work->gm, &inc, sol, &inc); // s = Hp + flt1*g + flt2*gm
                        dcopy_(&n, work->g, &inc, work->gm, &inc); dscal_(&n, &flt3, work->gm, &inc); // gm = flt3*g
                        dcopy_(&n, sol, &inc, work->g, &inc); // g = s
                    }
                    v_dot_g = ddot_(&n, work->g, &inc, work->g, &inc);
                    break;
                case TRLIB_CLA_UPDATE_DIR:
                    if (ityp == TRLIB_CLT_CG) { dscal_(&n, &flt2, work->p, &inc); daxpy_(&n, &minus, work->g, &inc, work->p, &inc); } // p = -g + flt2 * p
                    if (ityp == TRLIB_CLT_L) { dcopy_(&n, work->g, &inc, work->p, &inc); dscal_(&n, &flt1, work->p, &inc); } // p = flt1*g
                    if(qp->qptype == TRLIB_TEST_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_OP_QP) {
                        hv(userdata, n, work->p, work->Hp); // Hp = H*p
                    }
                    p_dot_Hp = ddot_(&n, work->p, &inc, work->Hp, &inc);
                    if ( ityp == TRLIB_CLT_L) { dcopy_(&n, work->p, &inc, work->Q+(qp->iter)*n, &inc); } // Q(iter*n:(iter+1)*n) = p
                    break;
                case TRLIB_CLA_CONV_HARD:
                    itp1 = qp->iter+1;
                    trlib_flt_t *temp = malloc(n*sizeof(trlib_flt_t));
                    if(qp->qptype == TRLIB_TEST_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, sol, &inc, &z, temp, &inc); // temp = H*s
                    }
                    if(qp->qptype == TRLIB_TEST_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, sol, &inc, &z, temp, &inc); // temp = H*s
                    }
                    if(qp->qptype == TRLIB_TEST_OP_QP) {
                        hv(userdata, n, sol, temp); // temp = H*s
                    }
                    daxpy_(&n, &one, grad, &inc, temp, &inc); // temp = H*s + g
                    daxpy_(&n, &flt1, sol, &inc, temp, &inc); // temp = H*s + g + flt1*s
                    v_dot_g = ddot_(&n, temp, &inc, temp, &inc);
                    free(temp);
                    break;
                case TRLIB_CLA_NEW_KRYLOV:
                    // FIXME: implement proper reorthogonalization
                    memset(work->g, 0, n*sizeof(trlib_flt_t));
                    memset(work->gm, 0, n*sizeof(trlib_flt_t));
                    memset(work->p, 0, n*sizeof(trlib_flt_t));
                    (work->g)[0] = 1.0;
                    v_dot_g = ddot_(&n, work->g, &inc, work->g, &inc);
                    trlib_flt_t iv = 1.0/sqrt(v_dot_g);
                    daxpy_(&n, &iv, work->g, &inc, work->p, &inc);
                    if(qp->qptype == TRLIB_TEST_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_TEST_OP_QP) {
                        hv(userdata, n, work->p, work->Hp); // Hp = H*p
                    }
                    p_dot_Hp = ddot_(&n, work->p, &inc, work->Hp, &inc);
                    dcopy_(&n, work->p, &inc, work->Q+(qp->iter+1)*n, &inc); // Q(iter*n:(iter1)*n) = p
                    break;
                case TRLIB_CLA_OBJVAL:
                    // FIXME: implement this and add a test for convexification
                    break;
            }
            //fprintf(stderr, "<g,g> = %e, <v,g> = %e, <p,Hp> = %e\n", v_dot_g, v_dot_g, p_dot_Hp);
            if( qp->ret < 10 ) { break; }
        }

        qp->lam = work->fwork[7];
        qp->obj = work->fwork[8];
        qp->sub_fail = work->iwork[9];

    }

    if (qp->qpsolver == TRLIB_TEST_SOLVER_FACTOR) { 
        trlib_int_t inc = 1; trlib_flt_t minus = -1.0;
        struct trlib_test_work_factor * work = (struct trlib_test_work_factor *)qp->work;
        struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;

        dcopy_(&problem->n, problem->grad, &inc, problem->neggrad, &inc); dscal_(&problem->n, &minus, problem->neggrad, &inc);

        if(work->nirblk == 0) {
            // detect irreducible structure
            work->nirblk = 1;
            for(trlib_int_t ii = 0; ii < problem->n -1; ++ii) { if(problem->offdiag[ii] == 0.0) { work->nirblk++; } }
            work->irblk = malloc((work->nirblk+1)*sizeof(trlib_int_t));
            work->irblk[0] = 0; work->irblk[work->nirblk] = problem->n;
            trlib_int_t ircount = 1;
            for(trlib_int_t ii = 0; ii < problem->n -1; ++ii) { if(problem->offdiag[ii] == 0.0) { work->irblk[ircount] = ii+1; ircount++; } }
            work->leftmost = calloc(work->nirblk, sizeof(trlib_flt_t));
        }

        qp->ret = trlib_tri_factor_min(work->nirblk, work->irblk, 
            problem->diag, problem->offdiag, problem->neggrad,
            qp->radius, qp->itmax, TRLIB_EPS, TRLIB_EPS, problem->pos_def, qp->equality,
            &(work->warm_lam0), &(work->lam0), &(work->warm_lam), &(qp->lam),
            &(work->warm_leftmost), &(work->ileftmost), work->leftmost,
            &(problem->warm_fac0), problem->diag_fac0, problem->offdiag_fac0,
            &(problem->warm_fac), problem->diag_fac, problem->offdiag_fac,
            problem->sol0, problem->sol, work->ones, work->fwork, qp->refine,
            qp->verbose, qp->unicode, qp->prefix, qp->stream,
            qp->timing, &(qp->obj), &(qp->iter), &qp->sub_fail);
    }

    return 0;
}

trlib_int_t trlib_test_resolve_new_gradient(struct trlib_test_qp *qp) {
    if(qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        trlib_int_t n; trlib_flt_t *grad, *sol, *hess, *diag, *offdiag;
        void (*hv)(void *, trlib_int_t, trlib_flt_t *, trlib_flt_t *); trlib_flt_t *userdata;
        if(qp->qpsolver == TRLIB_TEST_DENSE_QP) { 
            struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; hess = problem->hess;
        }
        if(qp->qptype == TRLIB_TEST_TRI_QP) {
            struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; diag = problem->diag; offdiag = problem->offdiag;
        }
        if(qp->qptype == TRLIB_TEST_OP_QP) {
            struct trlib_test_problem_op* problem = (struct trlib_test_problem_op *)qp->problem;
            n = problem->n; grad = problem->grad; hv = problem->hv; userdata = problem->userdata; sol = problem->sol;
        }
        trlib_int_t init = 0, inc = 1, itp1 = 0;
        trlib_flt_t minus = -1.0, one = 1.0, z = 0.0;
        init = TRLIB_CLS_HOTSTART_G;
        trlib_flt_t v_dot_g = 0.0, p_dot_Hp = 0.0, flt1, flt2, flt3;
        trlib_int_t action, ityp;

        trlib_int_t gt_pointer;
        trlib_krylov_gt(qp->itmax, &gt_pointer);

        trlib_int_t iwork_size, fwork_size, h_pointer;
        trlib_krylov_memory_size(qp->itmax, &iwork_size, &fwork_size, &h_pointer);

        trlib_flt_t nl0 = (work->fwork)[gt_pointer];

        itp1 = qp->iter+1;
        dgemv_("T", &n, &itp1, &minus, work->Q, &n, grad, &inc, &z, work->fwork+gt_pointer, &inc); // neglin = - Q_i * grad

        while(1) {
            qp->ret = trlib_krylov_min(init, qp->radius, qp->equality, qp->itmax, 100,
                    qp->tol_rel_i, qp->tol_abs_i, qp->tol_rel_b, qp->tol_abs_b,
                    TRLIB_EPS*TRLIB_EPS, -1e20, qp->ctl_invariant, 0, 0, v_dot_g, v_dot_g, p_dot_Hp, work->iwork, work->fwork, 
                    qp->refine, qp->verbose, qp->unicode, qp->prefix, qp->stream, qp->timing,
                    &action, &(qp->iter), &ityp, &flt1, &flt2, &flt3);
            init = 0;

            if( qp->ret < 10 ) { break; }
        }

        itp1 = qp->iter+1;
        dgemv_("N", &n, &itp1, &one, work->Q, &n, work->fwork+h_pointer, &inc, &z, sol, &inc); // s = Q_i * h_i
        work->fwork[gt_pointer] = nl0;
        memset(work->fwork+gt_pointer+1, 0, qp->iter*sizeof(trlib_flt_t));

        qp->lam = work->fwork[7];
        qp->obj = work->fwork[8];
        qp->sub_fail = work->iwork[9];

    }

    return 0;
}

trlib_int_t trlib_test_check_optimality(struct trlib_test_qp *qp) { 
    trlib_int_t n, nn, nm1, jj; trlib_flt_t *sol, *grad, perturbed;
    trlib_flt_t *hess, *hess_lam;
    trlib_flt_t *diag, *offdiag, *diag_lam, *offdiag_lam;
    trlib_int_t inc = 1, ifail = 0; trlib_flt_t one = 1.0;
    qp->pos_def_res = 0.0; perturbed = qp->lam + qp->pos_def_res;
    if(qp->qptype == TRLIB_TEST_DENSE_QP) {
        struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
        n = problem->n; nn = problem->n*problem->n; sol = problem->sol; hess = problem->hess; grad = problem->grad;
        hess_lam = malloc(nn*sizeof(trlib_flt_t));
        dcopy_(&nn, hess, &inc, hess_lam, &inc);
        for(trlib_int_t ii = 0; ii < n; ++ii) { hess_lam[ii*n+ii] += perturbed; }
    }
    if(qp->qptype == TRLIB_TEST_TRI_QP) {
        struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
        n = problem->n; nm1 = problem->n-1; diag = problem->diag; offdiag = problem->offdiag; grad = problem->grad; sol = problem->sol;
        diag_lam = malloc(n*sizeof(trlib_flt_t)); offdiag_lam = malloc((n-1)*sizeof(trlib_flt_t));
        dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&nm1, offdiag, &inc, offdiag_lam, &inc);
        for(trlib_int_t ii = 0; ii < n; ++ii) { diag_lam[ii] += perturbed; }
    }
    jj = 0;
    while (1) {
        if(qp->qptype == TRLIB_TEST_DENSE_QP) { dpotrf_("L", &n, hess_lam, &n, &ifail); }
        if(qp->qptype == TRLIB_TEST_TRI_QP) { dpttrf_(&n, diag_lam, offdiag_lam, &ifail); }
        jj += 1;
        if (ifail == 0) { break; }
        if ( jj > 500 ) { break; }
        if ( qp->pos_def_res == 0.0 ) { qp->pos_def_res = TRLIB_EPS; } else { qp->pos_def_res = 2.0*qp->pos_def_res; }
        perturbed = qp->lam + qp->pos_def_res;
        if(qp->qptype == TRLIB_TEST_SOLVER_KRYLOV) {
            dcopy_(&nn, hess, &inc, hess_lam, &inc);
            for(trlib_int_t ii = 0; ii < n; ++ii) { hess_lam[ii*n+ii] += perturbed; }
        }
        if(qp->qptype == TRLIB_TEST_TRI_QP) {
            dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&nm1, offdiag, &inc, offdiag_lam, &inc);
            for(trlib_int_t ii = 0; ii < n; ++ii) { diag_lam[ii] += perturbed; }
        }
    }

    qp->tr_res = qp->radius - dnrm2_(&n, sol, &inc);
    trlib_flt_t *resv = malloc(n*sizeof(trlib_flt_t));
    if(qp->qptype == TRLIB_TEST_DENSE_QP) {
        dcopy_(&n, sol, &inc, resv, &inc);
        dgemv_("N", &n, &n, &one, hess, &n, sol, &inc, &(qp->lam), resv, &inc);
        daxpy_(&n, &one, grad, &inc, resv, &inc);
    }
    if(qp->qptype == TRLIB_TEST_TRI_QP) {
        dcopy_(&n, grad, &inc, resv, &inc);
        dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&nm1, offdiag, &inc, offdiag_lam, &inc);
        for(trlib_int_t ii = 0; ii < n; ++ii) { diag_lam[ii] += qp->lam; }
        dlagtm_("N", &n, &inc, &one, offdiag, diag_lam, offdiag, sol, &n, &one, resv, &n);
    }
    qp->kkt_res = dnrm2_(&n, resv, &inc);

    if(qp->qptype == TRLIB_TEST_DENSE_QP) {
        dcopy_(&n, grad, &inc, resv, &inc); perturbed = 2.0; dscal_(&n, &perturbed, resv, &inc); perturbed = 1.0; // w <-- 2 grad
        dgemv_("N", &n, &n, &one, hess, &n, sol, &inc, &one, resv, &inc); // w <-- H*sol + w
        qp->obj_check = ddot_(&n, sol, &inc, resv, &inc); qp->obj_check = 0.5*qp->obj_check; // obj = .5*(sol, w)
    }
    if(qp->qptype == TRLIB_TEST_TRI_QP) {
        dcopy_(&n, grad, &inc, resv, &inc); perturbed = 2.0; dscal_(&n, &perturbed, resv, &inc); perturbed = 1.0; // w <-- 2 grad
        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, sol, &n, &one, resv, &n); // w <-- T*sol + w
        qp->obj_check = ddot_(&n, sol, &inc, resv, &inc); qp->obj_check = 0.5*qp->obj_check; // obj = .5*(sol, w)
    }

    if(qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        for(trlib_int_t ii = 0; ii < qp->iter+1; ++ii) {
            for(trlib_int_t jj = 0; jj < ii+1; ++jj) {
                work->orth_check[ii*(qp->iter+1)+jj] = ddot_(&n, work->Q+ii*n, &inc, work->Q+jj*n, &inc);
                work->orth_check[jj*(qp->iter+1)+ii] = work->orth_check[ii*(qp->iter+1)+jj];
            }
            work->orth_check[ii*(qp->iter+1)+ii] = work->orth_check[ii*(qp->iter+1)+ii] - 1.0;
        }
        trlib_int_t lorth = (qp->iter+1)*(qp->iter+1);
        qp->orth_res = dnrm2_(&lorth, work->orth_check, &inc);
        qp->orth_res = qp->orth_res/(qp->iter+1);
    }

    if(qp->qptype == TRLIB_TEST_DENSE_QP) { free(hess_lam); }
    if(qp->qptype == TRLIB_TEST_TRI_QP) { free(diag_lam); free(offdiag_lam); }
    free(resv);

    return 0; 
}

// it seems that for some reason this construction is needed from cython
trlib_int_t trlib_test_problem_set_hvcb(struct trlib_test_problem_op* problem, void *userdata, void (*hv_cb)(void *, trlib_int_t, trlib_flt_t *, trlib_flt_t *)) {
    problem->userdata = userdata;
    problem->hv = hv_cb;
    return 0;
}

void trlib_test_solve_check_qp(struct trlib_test_qp *qp, char *name, trlib_flt_t tol, trlib_flt_t lanczos_tol) {
    qp->stream = stderr;
    trlib_test_solve_qp(qp);
    trlib_test_check_optimality(qp);
    trlib_test_print_result(qp, name, tol, lanczos_tol);
}

void trlib_test_print_result(struct trlib_test_qp *qp, char *name, trlib_flt_t tol, trlib_flt_t lanczos_tol) {
    printf("\n*************************************************************\n");
    printf("* Test Case   %-46s*\n", name);
    printf("*   Exit code:          %-2ld (%-2ld)%29s*\n", qp->ret, qp->sub_fail, "");
    printf("*   Objective:       %15e%15e%9s*\n", qp->obj, qp->obj_check, "");
    printf("*   TR radius:       %15e%24s*\n", qp->radius, "");
    printf("*   multiplier:      %15e%24s*\n", qp->lam, "");
    if (!qp->equality) { printf("*   TR residual:     %15e (inequality requested)%1s*\n", qp->tr_res, ""); }
    else { printf("*   TR resiudal:     %15e (equality requested)%3s*\n", qp->tr_res, ""); }
    printf("*   pos def perturb: %15e%24s*\n", qp->pos_def_res, "");
    printf("*   KKT residual:    %15e%24s*\n", qp->kkt_res, "");
    if (qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) { printf("*   ons residual:    %15e%24s*\n", qp->orth_res, ""); }
    printf("*************************************************************\n\n");

    // let us do a simple lanczos iteration ourselve and compare directions
    if (qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
        trlib_int_t n, inc = 1;
        struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
        trlib_flt_t igamma, gamma, delta, one, z;
        trlib_flt_t *g, *p, *pm, *Hp, *hess, *diag, *offdiag, *grad;
        one = 1.0; z = 0.0;
        if(qp->qptype == TRLIB_TEST_DENSE_QP) {
            struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense *)qp->problem;
            n = problem->n;
            hess = problem->hess;
            grad = problem->grad;
        }
        if(qp->qptype == TRLIB_TEST_TRI_QP) {
            struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri *)qp->problem;
            n = problem->n;
            diag = problem->diag;
            offdiag = problem->offdiag;
            grad = problem->grad;
        }
        g  = calloc(n, sizeof(trlib_flt_t));
        p  = calloc(n, sizeof(trlib_flt_t));
        Hp = calloc(n, sizeof(trlib_flt_t));
        pm = calloc(n, sizeof(trlib_flt_t));

        memcpy(g, grad, n*sizeof(trlib_flt_t));

        if(lanczos_tol >= 0) {
            for(trlib_int_t ii = 0; ii<qp->iter; ++ii) { 
                gamma = dnrm2_(&n, g, &inc); igamma = 1.0/gamma;
                memcpy(p, g, n*sizeof(trlib_flt_t)); dscal_(&n, &igamma, p, &inc);
                if(qp->qptype == TRLIB_TEST_DENSE_QP) { 
                    dgemv_("N", &n, &n, &one, hess, &n, p, &inc, &z, Hp, &inc); // Hp = H*p
                }
                if(qp->qptype == TRLIB_TEST_TRI_QP) { 
                    dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, p, &inc, &z, Hp, &inc); // Hp = H*p
                }
                delta = ddot_(&n, p, &inc, Hp, &inc);
                memcpy(g, Hp, n*sizeof(trlib_flt_t)); 
                igamma = -delta; daxpy_(&n, &igamma, p, &inc, g, &inc);
                igamma = -gamma; daxpy_(&n, &igamma, pm, &inc, g, &inc);
                memcpy(pm, p, n*sizeof(trlib_flt_t));
                for(trlib_int_t jj = 0; jj<n; ++jj){ ck_assert_msg( fabs(p[jj] - work->Q[jj+ii*n] ) <= lanczos_tol, "directions differ from those produced by stupid Lanczos, iterate %d, index %d, residual %e", ii, jj, p[jj] - work->Q[jj+ii*n]); }
            }
        }

        if (g != NULL) { free(g); }
        if (p != NULL) { free(p); }
        if (Hp != NULL) { free(Hp); }
        if (pm != NULL) { free(pm); }
    }

    #if TRLIB_TEST_PLOT
        if(qp->qpsolver == TRLIB_TEST_SOLVER_KRYLOV) {
            void *context = zmq_ctx_new ();
            void *requester = zmq_socket (context, ZMQ_REQ);
            zmq_connect (requester, "tcp://localhost:5678");

            void *buf;
            unsigned lenbuf;
            char recbuffer[10];

            struct trlib_test_work_krylov * work = (struct trlib_test_work_krylov *)qp->work;
            TrlibMatrixMessage orth_msg = TRLIB_MATRIX_MESSAGE__INIT;
            orth_msg.id = 0;
            orth_msg.m = qp->iter+1;
            orth_msg.n = qp->iter+1;
            orth_msg.n_data = orth_msg.m*orth_msg.n;
            orth_msg.data = work->orth_check;

            lenbuf = trlib_matrix_message__get_packed_size(&orth_msg);
            buf = malloc(lenbuf);
            trlib_matrix_message__pack(&orth_msg, buf);
            zmq_send(requester, buf, lenbuf, 0);
            zmq_recv(requester, recbuffer, 10, 0);
            free(buf);

            zmq_close(requester);
            zmq_ctx_destroy(context);
        }
    #endif

    ck_assert_msg(fabs(qp->pos_def_res) <= tol, "%s: Expected positive semidefinite regularized hessian, got multiplier %e, pertubation needed %e", name, qp->lam, qp->pos_def_res);
    if(qp->equality){
        ck_assert_msg(fabs(qp->tr_res) <= tol, "%s: Expected satisfaction of trust region constraint, residual %e", name, qp->tr_res);
    }
    else{
        ck_assert_msg(qp->tr_res >= -tol, "%s: Expected satisfaction of trust region constraint, residual %e", name, qp->tr_res);
        ck_assert_msg(fabs((qp->tr_res)*qp->lam) <= tol, "%s: Expected satisfaction of complementary, violation %e, trust region residual %e, multiplier %e", name, (qp->tr_res)*qp->lam, qp->tr_res, qp->lam);
    }
    ck_assert_msg(fabs(qp->kkt_res) <= tol, "%s: Expected satisfaction of KKT condition, residual %e", name, qp->kkt_res);
    ck_assert_msg(fabs(qp->obj - qp->obj_check) <= tol, "%s: Returned objective and computed objective mismatch: %e", name, qp->obj - qp->obj_check);

}

