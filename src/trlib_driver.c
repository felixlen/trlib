#include "trlib_driver.h"

int trlib_driver_malloc_qp(int qptype, int qpsolver, int n, int itmax, struct trlib_driver_qp *qp) {
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
    qp->timing = malloc(20*sizeof(long));
    qp->tol_rel_i = TRLIB_EPS_POW_5;
    qp->tol_abs_i = 0.0;
    qp->tol_rel_b = TRLIB_EPS_POW_4;
    qp->tol_abs_b = 0.0;
    if(qptype == TRLIB_DRIVER_DENSE_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_driver_problem_dense));
        struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense *)qp->problem;
        problem->n = n;
        problem->hess = calloc(n*n, sizeof(double));
        problem->grad = calloc(n, sizeof(double));
        problem->sol = malloc(n*sizeof(double));
    }
    if(qptype == TRLIB_DRIVER_TRI_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_driver_problem_tri));
        struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri *)qp->problem;
        problem->n = n;
        problem->diag = calloc(n, sizeof(double));
        problem->offdiag = calloc(n-1, sizeof(double));
        problem->diag_fac = malloc(n*sizeof(double));
        problem->diag_fac0 = malloc(n*sizeof(double));
        problem->offdiag_fac = malloc((n-1)*sizeof(double));
        problem->offdiag_fac0 = malloc((n-1)*sizeof(double));
        problem->grad = calloc(n, sizeof(double));
        problem->neggrad = calloc(n, sizeof(double));
        problem->sol = malloc(n*sizeof(double));
        problem->sol0 = malloc(n*sizeof(double));
        problem->warm_fac = 0;
        problem->warm_fac0 = 0;
        problem->pos_def = 0;
    }
    if(qptype == TRLIB_DRIVER_OP_QP) {
        qp->problem = (void *)malloc(sizeof(struct trlib_driver_problem_op));
        struct trlib_driver_problem_op* problem = (struct trlib_driver_problem_op *)qp->problem;
        problem->n = n;
        problem->grad = calloc(n, sizeof(double));
        problem->sol = malloc(n*sizeof(double));
    }

    if(qpsolver == TRLIB_DRIVER_SOLVER_KRYLOV) {
        qp->work = (void *)malloc(sizeof(struct trlib_driver_work_krylov));
        struct trlib_driver_work_krylov * work = (struct trlib_driver_work_krylov *)qp->work;
        int iwork_size, fwork_size, h_pointer;
        trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size, &h_pointer);
        work->iwork = malloc(iwork_size*sizeof(int));
        work->fwork = malloc(fwork_size*sizeof(double));
        work->g = malloc(n*sizeof(double));
        work->gm = malloc(n*sizeof(double));
        work->p = malloc(n*sizeof(double));
        work->Hp = malloc(n*sizeof(double));
        work->Q = malloc((itmax+1)*n*sizeof(double));
        work->orth_check = malloc((itmax+1)*(itmax+1)*sizeof(double));
    }
    if(qpsolver == TRLIB_DRIVER_SOLVER_FACTOR) {
        qp->work = (void *)malloc(sizeof(struct trlib_driver_work_factor));
        struct trlib_driver_work_factor * work = (struct trlib_driver_work_factor *)qp->work;
        work->fwork = malloc(4*n*sizeof(double));
        work->ones = malloc(n*sizeof(double));
        work->nirblk = 0;
        work->warm_lam = 0;
        work->warm_lam0 = 0;
        work->lam0 = 0.0;
        work->warm_leftmost = 0;
        work->ileftmost = 0;
        for(int ii = 0; ii < n; ++ii) { work->ones[ii] = 1.0; }
    }

    return 0;
}

int trlib_driver_free_qp(struct trlib_driver_qp *qp) {
    if(qp->prefix != NULL) { free(qp->prefix); }
    if(qp->timing != NULL) { free(qp->timing); }
    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) {
        struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense *)qp->problem;
        if(problem != NULL) {
            if(problem->hess != NULL) { free(problem->hess); }
            if(problem->grad != NULL) { free(problem->grad); }
            if(problem->sol != NULL) { free(problem->sol); }
            free(problem);
        }
    }
    if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
        struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri *)qp->problem;
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
    if(qp->qptype == TRLIB_DRIVER_OP_QP) {
        struct trlib_driver_problem_op* problem = (struct trlib_driver_problem_op *)qp->problem;
        if(problem != NULL) {
            if(problem->grad != NULL) { free(problem->grad); }
            if(problem->sol != NULL) { free(problem->sol); }
            problem->userdata = NULL;
            problem->hv = NULL;
            free(problem);
        }
    }

    if(qp->qpsolver == TRLIB_DRIVER_SOLVER_KRYLOV) {
        struct trlib_driver_work_krylov * work = (struct trlib_driver_work_krylov *)qp->work;
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
    if(qp->qpsolver == TRLIB_DRIVER_SOLVER_FACTOR) {
        struct trlib_driver_work_factor * work = (struct trlib_driver_work_factor *)qp->work;
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

int trlib_driver_solve_qp(struct trlib_driver_qp *qp) { 
    if(qp->qpsolver == TRLIB_DRIVER_SOLVER_KRYLOV) {
        struct trlib_driver_work_krylov * work = (struct trlib_driver_work_krylov *)qp->work;
        int n; double *grad; double *sol; double *hess; double *diag; double *offdiag;
        void (*hv)(void *, int, double *, double *); double *userdata;
        if(qp->qpsolver == TRLIB_DRIVER_DENSE_QP) { 
            struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; hess = problem->hess;
        }
        if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
            struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri *)qp->problem;
            n = problem->n; grad = problem->grad; sol = problem->sol; diag = problem->diag; offdiag = problem->offdiag;
        }
        if(qp->qptype == TRLIB_DRIVER_OP_QP) {
            struct trlib_driver_problem_op* problem = (struct trlib_driver_problem_op *)qp->problem;
            n = problem->n; grad = problem->grad; hv = problem->hv; userdata = problem->userdata; sol = problem->sol;
        }
        int init = 0; int inc = 1; int itp1 = 0;
        double minus = -1.0; double one = 1.0; double z = 0.0;
        if(!qp->reentry) { init = 1; trlib_krylov_prepare_memory(qp->itmax, work->fwork); }
        else { init = 2; }

        double v_dot_g = 0.0; double p_dot_Hp = 0.0; double flt1; double flt2; double flt3;
        int action; int ityp;

        int iwork_size, fwork_size, h_pointer;
        trlib_krylov_memory_size(qp->itmax, &iwork_size, &fwork_size, &h_pointer);

        while(1) {
            qp->ret = trlib_krylov_min(init, qp->radius, qp->equality, qp->itmax, 100,
                    qp->tol_rel_i, qp->tol_abs_i, qp->tol_rel_b, qp->tol_abs_b,
                    TRLIB_EPS, v_dot_g, v_dot_g, p_dot_Hp, work->iwork, work->fwork, qp->refine,
                    qp->verbose, qp->unicode, qp->prefix, qp->stream, qp->timing,
                    &action, &(qp->iter), &ityp, &flt1, &flt2, &flt3);
            init = 0;
            //fprintf(stderr, "action: %d, iter: %d, ityp: %d, flt1: %e, flt2: %e, flt3: %e\n",
            //        action, qp->iter, ityp, flt1, flt2, flt3);
            switch(action) {
                case TRLIB_CLA_INIT:
                    memset(sol, 0, n*sizeof(double)); memset(work->gm, 0, n*sizeof(double));
                    dcopy_(&n, grad, &inc, work->g, &inc);
                    v_dot_g = ddot_(&n, work->g, &inc, work->g, &inc);
                    dcopy_(&n, work->g, &inc, work->p, &inc); dscal_(&n, &minus, work->p, &inc); // p = -g
                    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_DRIVER_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_DRIVER_OP_QP) {
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
                    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) { 
                        dgemv_("N", &n, &n, &one, hess, &n, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_DRIVER_TRI_QP) { 
                        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, work->p, &inc, &z, work->Hp, &inc); // Hp = H*p
                    }
                    if(qp->qptype == TRLIB_DRIVER_OP_QP) {
                        hv(userdata, n, work->p, work->Hp); // Hp = H*p
                    }
                    p_dot_Hp = ddot_(&n, work->p, &inc, work->Hp, &inc);
                    if ( ityp == TRLIB_CLT_L) { dcopy_(&n, work->p, &inc, work->Q+(qp->iter)*n, &inc); } // Q(iter*n:(iter+1)*n) = p
                    break;
                case TRLIB_CLA_NEW_KRYLOV:
                    // FIXME: implement this, exit right now...
                    break;
            }
            if( qp->ret < 10 || action == TRLIB_CLA_NEW_KRYLOV ) { break; }
        }

        qp->lam = work->fwork[7];
        qp->obj = work->fwork[8];
        qp->sub_fail = work->iwork[9];

    }

    if (qp->qpsolver == TRLIB_DRIVER_SOLVER_FACTOR) { 
        int inc = 1; double minus = -1.0;
        struct trlib_driver_work_factor * work = (struct trlib_driver_work_factor *)qp->work;
        struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri *)qp->problem;

        dcopy_(&problem->n, problem->grad, &inc, problem->neggrad, &inc); dscal_(&problem->n, &minus, problem->neggrad, &inc);

        if(work->nirblk == 0) {
            // detect irreducible structure
            work->nirblk = 1;
            for(int ii = 0; ii < problem->n -1; ++ii) { if(problem->offdiag[ii] == 0.0) { work->nirblk++; } }
            work->irblk = malloc((work->nirblk+1)*sizeof(int));
            work->irblk[0] = 0; work->irblk[work->nirblk] = problem->n;
            int ircount = 1;
            for(int ii = 0; ii < problem->n -1; ++ii) { if(problem->offdiag[ii] == 0.0) { work->irblk[ircount] = ii+1; ircount++; } }
            work->leftmost = calloc(work->nirblk, sizeof(double));
        }

        qp->ret = trlib_tri_factor_min(work->nirblk, work->irblk, 
            problem->diag, problem->offdiag, problem->neggrad,
            qp->radius, qp->itmax, TRLIB_EPS, problem->pos_def, qp->equality,
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

int trlib_driver_check_optimality(struct trlib_driver_qp *qp) { 
    int n; int nn; int nm1; int jj; double *sol; double *grad; double perturbed;
    double *hess;  double *hess_lam;
    double *diag; double *offdiag; double *diag_lam; double *offdiag_lam;
    int inc = 1; int ifail = 0; double one = 1.0;
    qp->pos_def_res = 0.0; perturbed = qp->lam + qp->pos_def_res;
    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) {
        struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense *)qp->problem;
        n = problem->n; nn = problem->n*problem->n; sol = problem->sol; hess = problem->hess; grad = problem->grad;
        hess_lam = malloc(nn*sizeof(double));
        dcopy_(&nn, hess, &inc, hess_lam, &inc);
        for(int ii = 0; ii < n; ++ii) { hess_lam[ii*n+ii] += perturbed; }
    }
    if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
        struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri *)qp->problem;
        n = problem->n; nm1 = problem->n-1; diag = problem->diag; offdiag = problem->offdiag; grad = problem->grad; sol = problem->sol;
        diag_lam = malloc(n*sizeof(double)); offdiag_lam = malloc((n-1)*sizeof(double));
        dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&nm1, offdiag, &inc, offdiag_lam, &inc);
        for(int ii = 0; ii < n; ++ii) { diag_lam[ii] += perturbed; }
    }
    jj = 0;
    while (1) {
        if(qp->qptype == TRLIB_DRIVER_DENSE_QP) { dpotrf_("L", &n, hess_lam, &n, &ifail); }
        if(qp->qptype == TRLIB_DRIVER_TRI_QP) { dpttrf_(&n, diag_lam, offdiag_lam, &ifail); }
        jj += 1;
        if (ifail == 0) { break; }
        if ( jj > 500 ) { break; }
        if ( qp->pos_def_res == 0.0 ) { qp->pos_def_res = TRLIB_EPS; } else { qp->pos_def_res = 2.0*qp->pos_def_res; }
        perturbed = qp->lam + qp->pos_def_res;
        if(qp->qptype == TRLIB_DRIVER_SOLVER_KRYLOV) {
            dcopy_(&nn, hess, &inc, hess_lam, &inc);
            for(int ii = 0; ii < n; ++ii) { hess_lam[ii*n+ii] += perturbed; }
        }
        if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
            dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&nm1, offdiag, &inc, offdiag_lam, &inc);
            for(int ii = 0; ii < n; ++ii) { diag_lam[ii] += perturbed; }
        }
    }

    qp->tr_res = qp->radius - dnrm2_(&n, sol, &inc);
    double *resv = malloc(n*sizeof(double));
    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) {
        dcopy_(&n, sol, &inc, resv, &inc);
        dgemv_("N", &n, &n, &one, hess, &n, sol, &inc, &(qp->lam), resv, &inc);
        daxpy_(&n, &one, grad, &inc, resv, &inc);
    }
    if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
        dcopy_(&n, grad, &inc, resv, &inc);
        qp->kkt_res = dnrm2_(&n, resv, &inc);
        dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&n, offdiag, &inc, offdiag_lam, &inc);
        for(int ii = 0; ii < n; ++ii) { diag_lam[ii] += qp->lam; }
        dlagtm_("N", &n, &inc, &one, offdiag, diag_lam, offdiag, sol, &n, &one, resv, &n);
    }
    qp->kkt_res = dnrm2_(&n, resv, &inc);

    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) {
        dcopy_(&n, grad, &inc, resv, &inc); perturbed = 2.0; dscal_(&n, &perturbed, resv, &inc); perturbed = 1.0; // w <-- 2 grad
        dgemv_("N", &n, &n, &one, hess, &n, sol, &inc, &one, resv, &inc); // w <-- H*sol + w
        qp->obj_check = ddot_(&n, sol, &inc, resv, &inc); qp->obj_check = 0.5*qp->obj_check; // obj = .5*(sol, w)
    }
    if(qp->qptype == TRLIB_DRIVER_TRI_QP) {
        dcopy_(&n, grad, &inc, resv, &inc); perturbed = 2.0; dscal_(&n, &perturbed, resv, &inc); perturbed = 1.0; // w <-- 2 grad
        dlagtm_("N", &n, &inc, &one, offdiag, diag, offdiag, sol, &n, &one, resv, &n); // w <-- T*sol + w
        qp->obj_check = ddot_(&n, sol, &inc, resv, &inc); qp->obj_check = 0.5*qp->obj_check; // obj = .5*(sol, w)
    }

    if(qp->qpsolver == TRLIB_DRIVER_SOLVER_KRYLOV) {
        struct trlib_driver_work_krylov * work = (struct trlib_driver_work_krylov *)qp->work;
        for(int ii = 0; ii < qp->iter+1; ++ii) {
            for(int jj = 0; jj < ii+1; ++jj) {
                work->orth_check[ii*(qp->iter+1)+jj] = ddot_(&n, work->Q+ii*n, &inc, work->Q+jj*n, &inc);
                work->orth_check[jj*(qp->iter+1)+ii] = work->orth_check[ii*(qp->iter+1)+jj];
            }
            work->orth_check[ii*(qp->iter+1)+ii] = work->orth_check[ii*(qp->iter+1)+ii] - 1.0;
        }
        int lorth = (qp->iter+1)*(qp->iter+1);
        qp->orth_res = dnrm2_(&lorth, work->orth_check, &inc);
        qp->orth_res = qp->orth_res/(qp->iter+1);
    }

    if(qp->qptype == TRLIB_DRIVER_DENSE_QP) { free(hess_lam); }
    if(qp->qptype == TRLIB_DRIVER_TRI_QP) { free(diag_lam); free(offdiag_lam); }
    free(resv);

    return 0; 
}

// it seems that for some reason this construction is needed from cython
int trlib_driver_problem_set_hvcb(struct trlib_driver_problem_op* problem, void *userdata, void (*hv_cb)(void *, int, double *, double *)) {
    problem->userdata = userdata;
    problem->hv = hv_cb;
    return 0;
}
