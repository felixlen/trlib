#include "trlib_test.h"

START_TEST (test_expensive)
{
    struct trlib_driver_qp qp;
    trlib_driver_malloc_qp(TRLIB_DRIVER_TRI_QP, TRLIB_DRIVER_SOLVER_KRYLOV, 300, 10*300, &qp);
    qp.tol_rel_i = TRLIB_EPS;
    qp.tol_rel_b = TRLIB_EPS;
    qp.verbose = 1;

    struct trlib_driver_problem_tri* problem = (struct trlib_driver_problem_tri*) qp.problem;
    double *diag = problem->diag; double *offdiag = problem->offdiag; double *grad = problem->grad;
    
    diag[0] = 2.0; diag[1] = 2.0; diag[2] = 2.0;
    for(int ii = 0; ii < problem->n; ++ii) { grad[ii] = 1.0; }

    qp.radius = 3.2e3;
    trlib_test_solve_check_qp(&qp, "Coldstart diagonal with zeros", 1e7*TRLIB_EPS, -1.0);
    
    qp.reentry = 1;
    qp.radius = 1e3;
    trlib_test_solve_check_qp(&qp, "Warmstart diagonal with zeros", 1e7*TRLIB_EPS, -1.0);

    trlib_driver_free_qp(&qp);
}
END_TEST

Suite *tri_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Tridiagonal with zeros TR Suite");
    tc_core = tcase_create("Core");
    tcase_set_timeout(tc_core, 600.0);
    tcase_add_test(tc_core, test_expensive);
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
