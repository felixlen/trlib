#include "trlib_test.h"

START_TEST (test_simple)
{
    int n = 12; int nm1; nm1 = n-1; int nexpand; int nexpandm1;

    double *diag = malloc(n*sizeof(double));
    double *diag_fac0 = malloc(n*sizeof(double));
    double *diag_fac = malloc(n*sizeof(double));
    double *diag_lam = malloc(n*sizeof(double));
    double *offdiag = malloc((n-1)*sizeof(double));
    double *offdiag_fac0 = malloc((n-1)*sizeof(double));
    double *offdiag_fac = malloc((n-1)*sizeof(double));
    double *offdiag_lam = malloc((n-1)*sizeof(double));
    double *grad = calloc(n,sizeof(double));
    double *neglin = calloc(n,sizeof(double));
    double *sol = malloc(n*sizeof(double));
    double *sol0 = malloc(n*sizeof(double));
    double *ones = malloc(n*sizeof(double));
    for(int ii = 0; ii < n; ++ii) { ones[ii] = 1.0; }
    double *fwork = malloc(4*n*sizeof(double));
    int nirblk = 5;
    int *irblk = malloc((nirblk+1)*sizeof(int));
    int *irblk2 = malloc((nirblk+1)*sizeof(int));

    for(int ii = 0; ii < n; ++ii) { diag[ii] = 1.0 + (20.0*ii)/n; }
    for(int ii = 0; ii < n-1; ++ii) { offdiag[ii] = 1.0 - 0.5*ii; }
    irblk[0] = 0; irblk[1] = 3; irblk[2] = 6; irblk[3] = 9; irblk[4] = 10; irblk[5] = n;
    irblk2[0] = 0; irblk2[1] = 0; irblk2[2] = 0; irblk2[3] = 0; irblk2[4] = 0; irblk2[5] = 0;
    offdiag[2] = 0.0; offdiag[5] = 0.0; offdiag[8] = 0.0; offdiag[9] = 0.0;

    grad[0] = 1.0; neglin[0] = -1.0;

    int ret = 0;
    int jj = 0;
    int blkptr = 1;
    double radius = 1.0;
    int pos_def = 0;
    int equality = 0;
    int warm0 = 0;
    double lam0 = 0.0;
    int warm = 0;
    double lam = 0.0;
    int warm_leftmost = 0;
    int ileftmost = 0;
    double *leftmost = malloc(nirblk*sizeof(double));
    int warm_fac0 = 0;
    int warm_fac = 0;
    int refine = 1;
    long *timing = malloc(20*sizeof(long));
    double obj = 0.0;
    int iter_newton = 0;
    int sub_fail = 0;

    double *resv = malloc(n*sizeof(double));
    double pos_def_res, perturbed, tr_res, kkt_res, obj_check;
    double tol = TRLIB_EPS;

    for(int kk = 1; kk < n+1; ++kk) {
        
        if (irblk[blkptr]+1 == kk) {
            blkptr += 1;
        }

        irblk2[blkptr] = kk;

        for(int ll = 0; ll<nirblk+1; ++ll){ fprintf(stderr, "%d ", irblk2[ll]); } fprintf(stderr, "\n");

        ret = trlib_tri_factor_min(blkptr, irblk2, diag, offdiag, neglin, radius, 100, TRLIB_EPS,
                pos_def, equality, &warm0, &lam0, &warm, &lam, &warm_leftmost, &ileftmost,
                leftmost, &warm_fac0, diag_fac0, offdiag_fac0, &warm_fac, diag_fac, offdiag_fac,
                sol0, sol, ones, fwork, refine, 1, 1, "", stderr, timing,
                &obj, &iter_newton, &sub_fail);

        nexpand = irblk2[blkptr]; nexpandm1 = nexpand - 1;

        int inc = 1; int ifail = 0; double one = 1.0;
        pos_def_res = 0.0; perturbed = lam + pos_def_res;
        dcopy_(&nexpand, diag, &inc, diag_lam, &inc); dcopy_(&nexpandm1, offdiag, &inc, offdiag_lam, &inc);
        for(int ii = 0; ii < nexpand; ++ii) { diag_lam[ii] += perturbed; }
        jj = 0;
        while (1) {
            dpttrf_(&nexpand, diag_lam, offdiag_lam, &ifail);
            jj += 1;
            if (ifail == 0) { break; }
            if ( jj > 500 ) { break; }
            if ( pos_def_res == 0.0 ) { pos_def_res = TRLIB_EPS; } else { pos_def_res = 2.0*pos_def_res; }
            perturbed = lam + pos_def_res;
            dcopy_(&nexpand, diag, &inc, diag_lam, &inc); dcopy_(&nexpandm1, offdiag, &inc, offdiag_lam, &inc);
            for(int ii = 0; ii < nexpand; ++ii) { diag_lam[ii] += perturbed; }
        }

        tr_res = radius - dnrm2_(&nexpand, sol, &inc);
        dcopy_(&nexpand, grad, &inc, resv, &inc);
        kkt_res = dnrm2_(&nexpand, resv, &inc);
        dcopy_(&nexpand, diag, &inc, diag_lam, &inc); dcopy_(&nexpandm1, offdiag, &inc, offdiag_lam, &inc);
        for(int ii = 0; ii < nexpand; ++ii) { diag_lam[ii] += lam; }
        dlagtm_("N", &nexpand, &inc, &one, offdiag, diag_lam, offdiag, sol, &nexpand, &one, resv, &n);
        kkt_res = dnrm2_(&nexpand, resv, &inc);

        dcopy_(&nexpand, grad, &inc, resv, &inc); perturbed = 2.0; dscal_(&nexpand, &perturbed, resv, &inc); perturbed = 1.0; // w <-- 2 grad
        dlagtm_("N", &nexpand, &inc, &one, offdiag, diag, offdiag, sol, &nexpand, &one, resv, &n); // w <-- T*sol + w
        obj_check = ddot_(&nexpand, sol, &inc, resv, &inc); obj_check = 0.5*obj_check; // obj = .5*(sol, w)

        printf("\n*************************************************************\n");
        printf("* Test Case   %-46s*\n", "Expanding Tridiagonal");
        printf("*   Exit code:          %-2d (%-2d)%29s*\n", ret, sub_fail, "");
        printf("*   Objective:       %15e%15e%9s*\n", obj, obj_check, "");
        printf("*   TR radius:       %15e%24s*\n", radius, "");
        printf("*   multiplier:      %15e%24s*\n", lam, "");
        printf("*   TR residual:     %15e (inequality requested)%1s*\n", tr_res, "");
        printf("*   pos def perturb: %15e%24s*\n", pos_def_res, "");
        printf("*   KKT residual:    %15e%24s*\n", kkt_res, "");
        printf("*************************************************************\n\n");

        ck_assert_msg(fabs(pos_def_res) <= tol, "%s: Expected positive semidefinite regularized hessian, got multiplier %e, pertubation needed %e", "", lam, pos_def_res);
        ck_assert_msg(tr_res >= -tol, "%s: Expected satisfaction of trust region constraint, residual %e", "", tr_res);
        ck_assert_msg(fabs((tr_res)*lam) <= tol, "%s: Expected satisfaction of complementary, violation %e, trust region residual %e, multiplier %e", "", (tr_res)*lam, tr_res, lam);
        ck_assert_msg(fabs(kkt_res) <= tol, "%s: Expected satisfaction of KKT condition, residual %e", "", kkt_res);
        ck_assert_msg(fabs(obj - obj_check) <= tol, "%s: Returned objective and computed objective mismatch: %e", "", obj - obj_check);

    }

    free(diag); free(diag_fac0); free(diag_fac); free(diag_lam);
    free(offdiag); free(offdiag_fac); free(offdiag_fac0); free(offdiag_lam);
    free(grad); 
    free(neglin); free(sol); free(sol0);
    free(ones); free(fwork); free(irblk); free(irblk2); free(leftmost);
    free(timing); free(resv);
}
END_TEST

Suite *tri_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Tridiagonal TR Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_simple);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = tri_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
