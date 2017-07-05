#include "trlib_test.h"

START_TEST (test_zero_hess)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 5;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;
    
    diag[0] = 0.0; diag[1] = 0.0; diag[2] = 0.0;
    offdiag[0] = 0.0; offdiag[1] = 0.0;

    grad[0] = 1.0; grad[1] = 1.0; grad[2] = 1.0;

    qp.radius = 1.0;
    trlib_test_solve_check_qp(&qp, "Coldstart simple 3D", 1e3*TRLIB_EPS, 1e1*TRLIB_EPS);
   
    qp.reentry = 1; 
    qp.radius = 0.5;
    //trlib_test_solve_check_qp(&qp, "Warmstart simple 3D", 1e3*TRLIB_EPS, 1e1*TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

Suite *zero_hess_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Zero Hessian TR Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_zero_hess);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = zero_hess_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
