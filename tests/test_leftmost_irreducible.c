#include "trlib_test.h"

double factorization_zero(int n, double lam, double *diag, double *offdiag) {
    int inc = 1; int ifail = 0; int jj;
    double perturb = 0.0; double perturbed;
    double *diag_lam = malloc(n*sizeof(double));
    double *ones = malloc(n*sizeof(double));
    double *offdiag_lam = malloc((n-1)*sizeof(double));
    for(int ii = 0; ii < n; ++ii) { ones[ii] = 1.0; }
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
    #if TRLIB_TEST_OUTPUT
        char *bp; size_t size; FILE *stream = open_memstream(&bp, &size);
    #endif

    double diag = 3.0;
    double leftmost = 0.0;
    long *timing = malloc(8*sizeof(long));
    int iter_pr = 0; int ret = 0;
    ret = trlib_leftmost_irreducible(1, &diag, NULL, 0, 0.0, 1, TRLIB_EPS,
           TRLIB_TEST_VERBOSE(1), TRLIB_TEST_UNICODE, "", TRLIB_TEST_FOUT(stream), timing,
           &leftmost, &iter_pr);
    ck_assert_msg(leftmost == diag, "One-Dimensional: Eigenvalue = Diagonal not satisfied, lam = %e, diag = %e", leftmost, diag);
    TRLIB_TEST_SHOW_CLEANUP_STREAM()
    free(timing);
}
END_TEST

START_TEST (test_nontrivial)
{
    #if TRLIB_TEST_OUTPUT
        char *bp; size_t size; FILE *stream = open_memstream(&bp, &size);
    #endif

    int n = 10; int nm = n-1;
    double *diag = malloc(n*sizeof(double));
    double *offdiag = malloc((n-1)*sizeof(double));
    long *timing = malloc(8*sizeof(long));
    double leftmost = 0.0; double leftmost_minor; double perturb;
    int iter_pr = 0; int ret = 0;
    for(int ii = 0; ii < n; ++ii) {
        diag[ii] = 2.0+((double)ii)/n;
        if ( ii < n-1 ) { offdiag[ii] = -1.0 - (2.0*ii)/n; }
    }
    trlib_leftmost_irreducible(nm, diag, offdiag, 0, 0.0, 10*n, TRLIB_EPS,
           TRLIB_TEST_VERBOSE(1), TRLIB_TEST_UNICODE, "", TRLIB_TEST_FOUT(stream), timing,
           &leftmost, &iter_pr);
    perturb = factorization_zero(nm, leftmost, diag, offdiag);
    ck_assert_msg(fabs(perturb) < 500.0*TRLIB_EPS, "Cholesky factorization for T-lam*I fails on coldstart test, needed to perturb by %e", perturb);
    leftmost_minor = leftmost;

    trlib_leftmost_irreducible(n, diag, offdiag, 1, leftmost, 10*n, TRLIB_EPS,
           TRLIB_TEST_VERBOSE(1), TRLIB_TEST_UNICODE, "", TRLIB_TEST_FOUT(stream), timing,
           &leftmost, &iter_pr);
    perturb = factorization_zero(nm, leftmost, diag, offdiag);
    ck_assert_msg(fabs(perturb) < 500.0*TRLIB_EPS, "Cholesky factorization for T-lam*I fails on coldstart test, needed to perturb by %e", perturb);
    TRLIB_TEST_SHOW_CLEANUP_STREAM()

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
