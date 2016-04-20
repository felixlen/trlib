#include <check.h>
#include "trlib_leftmost.h"
#include "trlib_eigen_inverse.h"

START_TEST (test_trivial_exit)
{
    double diag = 3.0;
    double ones = 1.0;
    double diag_fac = 12.0;
    double eig = 7.0;
    long *timing = malloc(trlib_eigen_timing_size()*sizeof(long));
    double lam_pert = 0.0;
    double pert = 0.0;
    int iter_inv = 0;
    trlib_eigen_inverse(1, &diag, NULL, 3.0, 1, TRLIB_EPS, &ones, &diag_fac, NULL, &eig, 
            1, 1, "", stderr, timing,
            &lam_pert, &pert, &iter_inv);
    ck_assert_msg(eig != 0.0, "Failure: Received zero eigenvector");
    free(timing);
}
END_TEST

START_TEST (test_nontrivial)
{
    int n = 10; int inc = 1;
    double *diag = malloc(n*sizeof(double));
    double *diag_fac = malloc(n*sizeof(double));
    double *offdiag = malloc((n-1)*sizeof(double));
    double *offdiag_fac = malloc((n-1)*sizeof(double));
    double *ones = malloc(n*sizeof(double));
    double *eig = malloc(n*sizeof(double));
    double *leig = malloc(n*sizeof(double));
    double zero = 0.0;
    long *timing = malloc(trlib_eigen_timing_size()*sizeof(long));
    double lam_init = 0.0;
    double lam_pert = 0.0;
    double pert = 0.0;
    int iter_inv = 0;
    srand((unsigned) time(NULL));
    for(int ii = 0; ii < n; ++ii) {
        diag[ii] = 2.0+((double)ii)/n;
        ones[ii] = 1.0;
        eig[ii] = ((double)rand()/(double)RAND_MAX);
        leig[ii] = 0.0;
        if ( ii < n-1 ) { offdiag[ii] = -1.0 - (2.0*ii)/n; }
    }
    trlib_leftmost_irreducible(n, diag, offdiag, 0, 0.0, 10*n, TRLIB_EPS,
           1, 1, "", stderr, timing,
           &lam_init, &iter_inv);
    trlib_eigen_inverse(n, diag, offdiag, lam_init, 10, TRLIB_EPS, ones, diag_fac, offdiag_fac, eig,
           1, 1, "", stderr, timing,
           &lam_pert, &pert, &iter_inv);
    dlagtm_("N", &n, &inc, ones, offdiag, diag, offdiag, eig, &n, &zero, leig, &n); // leig <-- T*eig
    for(int ii = 0; ii < n; ++ii){ ck_assert_msg(fabs(lam_init*eig[ii] - leig[ii]) <= 5000.0*TRLIB_EPS, "Residual in eigenvector for component %d: %e", ii, lam_init*eig[ii] - leig[ii]); }
    free(diag); free(diag_fac); free(offdiag); free(offdiag_fac); free(ones); free(eig); free(timing);
}
END_TEST


Suite *eigen_inverse_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Eigen Inverse");
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
    s = eigen_inverse_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
