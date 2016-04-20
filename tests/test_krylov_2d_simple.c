#include "trlib_test.h"

START_TEST (test_2d)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_DENSE_QP, TRLIB_TEST_SOLVER_KRYLOV, 2, 10*2, &qp);
    qp.verbose = 1;

    struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense*) qp.problem;
    double *hess = problem->hess; double *grad = problem->grad;

    hess[0] = 1.107272566595126e3; hess[1] = 4.701123595505616e2;
    hess[2] = 4.701123595505616e2; hess[3] = -1.222067920715109e-1;

    grad[0] = -4.637816414622023; grad[1] = 0.0; grad[2] = 4.0;  // easy case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "2D Simple Coldstart", 1e4*TRLIB_EPS, TRLIB_EPS);

    qp.radius = 0.5;
    qp.reentry = 1;
    trlib_test_solve_check_qp(&qp, "2D Simple Warmstart", 1e4*TRLIB_EPS, TRLIB_EPS);
    trlib_test_free_qp(&qp);
}
END_TEST

Suite *krylov_2d_simple_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Krylov Suite 2D Simple");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_2d);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = krylov_2d_simple_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
