#include "trlib_test.h"

START_TEST (test_narrow_lam)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_FACTOR, 24, 10*24, &qp);
    qp.verbose = 3;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;

    grad[0] = 1.4671888791e-06;
    diag[0] = 4.4795034695e+01;
    diag[1] = 1.2388436398e+03;
    diag[2] = 8.3037234317e+02;
    diag[3] = 1.1136317126e+03;
    diag[4] = 9.1432431753e+02;
    diag[5] = 1.0670316134e+03;
    diag[6] = 9.4933224266e+02;
    diag[7] = 1.0425765083e+03;
    diag[8] = 9.6905558346e+02;
    diag[9] = 1.0273540751e+03;
    diag[10] = 9.8186941286e+02;
    diag[11] = 1.0168954775e+03;
    diag[12] = 9.9091980189e+02;
    diag[13] = 1.0092475052e+03;
    diag[14] = 9.9766180553e+02;
    diag[15] = 1.0034199541e+03;
    diag[16] = 1.0028623022e+03;
    diag[17] = 9.9884713499e+02;
    diag[18] = 1.0077127690e+03;
    diag[19] = 9.9455603416e+02;
    diag[20] = 1.0108379109e+03;
    diag[21] = 9.9178868057e+02;
    diag[22] = 1.0133379725e+03;
    diag[23] = 9.9139932995e+02;
    offdiag[0] = 1.9832643560e+02;
    offdiag[1] = 5.0046346302e+02;
    offdiag[2] = 3.0923670888e+02;
    offdiag[3] = 4.6653417918e+02;
    offdiag[4] = 3.3894983744e+02;
    offdiag[5] = 4.5091891244e+02;
    offdiag[6] = 3.5287960168e+02;
    offdiag[7] = 4.4167511428e+02;
    offdiag[8] = 3.6120461802e+02;
    offdiag[9] = 4.3538890996e+02;
    offdiag[10] = 3.6691193225e+02;
    offdiag[11] = 4.3070808229e+02;
    offdiag[12] = 3.7119109749e+02;
    offdiag[13] = 4.2699366954e+02;
    offdiag[14] = 3.7460776588e+02;
    offdiag[15] = 4.2390572912e+02;
    offdiag[16] = 3.7746482832e+02;
    offdiag[17] = 4.2133019208e+02;
    offdiag[18] = 3.7998016687e+02;
    offdiag[19] = 4.1884332676e+02;
    offdiag[20] = 3.8218960538e+02;
    offdiag[21] = 4.1671561737e+02;
    offdiag[22] = 9.1771701683e+02;

    qp.radius = 27.61886142844956;
    trlib_test_solve_check_qp(&qp, "narrow suitable lam", 1e8*TRLIB_EPS, TRLIB_EPS);
    
    qp.radius = 10.0;
    trlib_test_solve_check_qp(&qp, "narrow suitable lam", 1e13*TRLIB_EPS, TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

Suite *tri_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Tridiagonal TR Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_narrow_lam);
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
