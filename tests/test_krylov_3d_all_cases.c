#include "trlib_test.h"

START_TEST (test_3d_easy)
{
    struct trlib_driver_qp qp;
    trlib_driver_malloc_qp(TRLIB_DRIVER_DENSE_QP, TRLIB_DRIVER_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;

    struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense*) qp.problem;
    double *hess = problem->hess; double *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 5.0; grad[1] = 0.0; grad[2] = 4.0;  // easy case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Easy Case 3D Coldstart", 10*TRLIB_EPS, 1e1*TRLIB_EPS);

    qp.radius = 0.5;
    qp.reentry = 1;
    trlib_test_solve_check_qp(&qp, "Easy Case 3D Warmstart", 10*TRLIB_EPS, 1e1*TRLIB_EPS);
    trlib_driver_free_qp(&qp);
}
END_TEST

START_TEST (test_3d_near_hard)
{
    struct trlib_driver_qp qp;
    trlib_driver_malloc_qp(TRLIB_DRIVER_DENSE_QP, TRLIB_DRIVER_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;

    struct trlib_driver_problem_dense* problem = (struct trlib_driver_problem_dense*) qp.problem;
    double *hess = problem->hess; double *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 0.0; grad[1] = 2.0; grad[2] = 1e-6;  // near hard case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Near Hard 3D Coldstart", 1e7*TRLIB_EPS, 1e1*TRLIB_EPS);

    qp.radius = 0.5;
    qp.reentry = 1;
    trlib_test_solve_check_qp(&qp, "Near Hard 3D Warmstart", 1e7*TRLIB_EPS, 1e1*TRLIB_EPS);
    trlib_driver_free_qp(&qp);
}
END_TEST


Suite *krylov_3d_all_cases_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Krylov Suite 3D All Cases");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_3d_easy);
    tcase_add_test(tc_core, test_3d_near_hard);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = krylov_3d_all_cases_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
