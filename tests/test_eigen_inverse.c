#include "trlib_test.h"
#include "trlib_leftmost.h"
#include "trlib_eigen_inverse.h"
#include <time.h>

START_TEST (test_trivial_exit)
{
    trlib_flt_t diag = 3.0;
    trlib_flt_t ones = 1.0;
    trlib_flt_t diag_fac = 12.0;
    trlib_flt_t eig = 7.0;
    trlib_int_t *timing = malloc(trlib_eigen_timing_size()*sizeof(trlib_int_t));
    trlib_flt_t lam_pert = 0.0;
    trlib_flt_t pert = 0.0;
    trlib_int_t iter_inv = 0;
    trlib_eigen_inverse(1, &diag, NULL, 3.0, 1, TRLIB_EPS, &ones, &diag_fac, NULL, &eig, 
            1, 1, "", stderr, timing,
            &lam_pert, &pert, &iter_inv);
    ck_assert_msg(eig != 0.0, "Failure: Received zero eigenvector");
    free(timing);
}
END_TEST

START_TEST (test_nontrivial)
{
    trlib_int_t n = 10, inc = 1;
    trlib_flt_t *diag = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *diag_fac = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *offdiag = malloc((n-1)*sizeof(trlib_flt_t));
    trlib_flt_t *offdiag_fac = malloc((n-1)*sizeof(trlib_flt_t));
    trlib_flt_t *ones = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *eig = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t *leig = malloc(n*sizeof(trlib_flt_t));
    trlib_flt_t zero = 0.0;
    trlib_int_t *timing = malloc(trlib_eigen_timing_size()*sizeof(trlib_int_t));
    trlib_flt_t lam_init = 0.0;
    trlib_flt_t lam_pert = 0.0;
    trlib_flt_t pert = 0.0;
    trlib_int_t iter_inv = 0;
    srand((unsigned) time(NULL));
    for(trlib_int_t ii = 0; ii < n; ++ii) {
        diag[ii] = 2.0+((trlib_flt_t)ii)/n;
        ones[ii] = 1.0;
        eig[ii] = ((trlib_flt_t)rand()/(trlib_flt_t)RAND_MAX);
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
    for(trlib_int_t ii = 0; ii < n; ++ii){ ck_assert_msg(fabs(lam_init*eig[ii] - leig[ii]) <= 5000.0*TRLIB_EPS, "Residual in eigenvector for component %d: %e", ii, lam_init*eig[ii] - leig[ii]); }
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
