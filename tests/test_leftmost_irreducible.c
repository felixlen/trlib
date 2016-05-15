#include "trlib_test.h"

trlib_flt_t factorization_zero(trlib_int_t n, trlib_flt_t lam, trlib_flt_t *diag, trlib_flt_t *offdiag) {
    trlib_int_t inc = 1, ifail = 0, jj;
    trlib_flt_t perturb = 0.0, perturbed;
    trlib_flt_t *diag_lam = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *ones = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *offdiag_lam = malloc((n-1)*sizeof(trlib_flt_t));
    for(trlib_int_t ii = 0; ii < n; ++ii) { ones[ii] = 1.0; }
    perturbed = -lam;
    dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&n, offdiag, &inc, offdiag_lam, &inc);
    daxpy_(&n, &perturbed, ones, &inc, diag_lam, &inc); dpttrf_(&n, diag_lam, offdiag_lam, &ifail);
    if (ifail != 0) { 
        perturb = -TRLIB_EPS; 
        jj = 0;
        while(ifail != 0 && jj < 500) {
            perturb = 2.0*perturb; perturbed = -lam - perturb; ++jj;
            dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&n, offdiag, &inc, offdiag_lam, &inc);
            daxpy_(&n, &perturbed, ones, &inc, diag_lam, &inc); dpttrf_(&n, diag_lam, offdiag_lam, &ifail);
        }
    } 
    else { 
        perturb = TRLIB_EPS; 
        jj = 0;
        while(ifail != 0 && jj < 500) {
            perturb = 2.0*perturb; perturbed = -lam - perturb; ++jj;
            dcopy_(&n, diag, &inc, diag_lam, &inc); dcopy_(&n, offdiag, &inc, offdiag_lam, &inc);
            daxpy_(&n, &perturbed, ones, &inc, diag_lam, &inc); dpttrf_(&n, diag_lam, offdiag_lam, &ifail);
        }
    }
    free(diag_lam); free(offdiag_lam); free(ones);
    return perturb;
}

START_TEST (test_trivial_exit)
{
    trlib_flt_t diag = 3.0;
    trlib_flt_t leftmost = 0.0;
    trlib_int_t *timing = malloc(trlib_leftmost_timing_size()*sizeof(trlib_int_t));
    trlib_int_t iter_pr = 0, ret = 0;
    ret = trlib_leftmost_irreducible(1, &diag, NULL, 0, 0.0, 1, TRLIB_EPS, 1, 1, "", stderr, timing,
           &leftmost, &iter_pr);
    ck_assert_msg(leftmost == diag, "One-Dimensional: Eigenvalue = Diagonal not satisfied, lam = %e, diag = %e", leftmost, diag);
    free(timing);
}
END_TEST

START_TEST (test_nontrivial)
{
    trlib_int_t n = 10, nm = n-1;
    trlib_flt_t *diag = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *offdiag = malloc((n-1)*sizeof(trlib_flt_t));
    trlib_int_t *timing = malloc(trlib_leftmost_timing_size()*sizeof(trlib_int_t));
    trlib_flt_t leftmost = 0.0, leftmost_minor, perturb;
    trlib_int_t iter_pr = 0, ret = 0;
    for(trlib_int_t ii = 0; ii < n; ++ii) {
        diag[ii] = 2.0+((trlib_flt_t)ii)/n;
        if ( ii < n-1 ) { offdiag[ii] = -1.0 - (2.0*ii)/n; }
    }
    trlib_leftmost_irreducible(nm, diag, offdiag, 0, 0.0, 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &leftmost, &iter_pr);
    perturb = factorization_zero(nm, leftmost, diag, offdiag);
    ck_assert_msg(fabs(perturb) < 500.0*TRLIB_EPS, "Cholesky factorization for T-lam*I fails on coldstart test, needed to perturb by %e", perturb);
    leftmost_minor = leftmost;

    trlib_leftmost_irreducible(n, diag, offdiag, 1, leftmost, 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &leftmost, &iter_pr);
    perturb = factorization_zero(nm, leftmost, diag, offdiag);
    ck_assert_msg(fabs(perturb) < 500.0*TRLIB_EPS, "Cholesky factorization for T-lam*I fails on coldstart test, needed to perturb by %e", perturb);

    free(diag); free(offdiag); free(timing);
}
END_TEST


Suite *leftmost_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Leftmost Irreducible Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_trivial_exit);
    tcase_add_test(tc_core, test_nontrivial);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = leftmost_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
