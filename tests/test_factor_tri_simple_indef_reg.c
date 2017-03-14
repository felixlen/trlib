#include "trlib_test.h"
#include "trlib/trlib_tri_factor.h"

START_TEST (test_indef_reg)
{
    trlib_flt_t *diag = malloc(3*sizeof(double));
    trlib_flt_t *offdiag = malloc(2*sizeof(double));
    trlib_flt_t *grad = malloc(3*sizeof(double));
    trlib_flt_t *sol = malloc(3*sizeof(double));
    trlib_int_t *timing = malloc(20*sizeof(double));
 
    diag[0] = -3.0; diag[1] = 2.0; diag[2] = 1.0;
    offdiag[0] = -0.5; offdiag[1] = -0.75;

    grad[0] = 3.0; grad[1] = 4.0; grad[2] = 0.0;

    trlib_flt_t *ones = malloc(3*sizeof(double)); ones[0] = 1.0; ones[1] = 1.0; ones[2] = 1.0;
    trlib_flt_t *fwork = malloc(18*sizeof(double));
    trlib_flt_t ns = 0.0; trlib_int_t sf, ret;
    trlib_flt_t lam = 1.0;

    ret = trlib_tri_factor_get_regularization(3, diag, offdiag, grad, &lam, 3.0, 2.9, 3.1, sol, 
            ones, fwork, 1, 1, 1, "", stdout, timing, &ns, &sf);

    ck_assert_msg(lam <= 3.1*ns, "Expected 2.9 <= lam/ns = %e <= 3.1", lam/ns);
    ck_assert_msg(ns*2.9 <= lam, "Expected 2.9 <= lam/ns = %e <= 3.1", lam/ns);

    free(diag); free(offdiag); free(grad); free(sol);
    free(ones); free(fwork); free(timing);
}
END_TEST

Suite *tri_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Tridiagonal TR Regularization Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_indef_reg);
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
