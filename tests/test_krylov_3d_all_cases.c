#include "trlib_test.h"
#include "trlib/trlib_krylov.h"

START_TEST (test_3d_easy)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_DENSE_QP, TRLIB_TEST_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;

    struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense*) qp.problem;
    trlib_flt_t *hess = problem->hess, *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 5.0; grad[1] = 0.0; grad[2] = 4.0;  // easy case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Easy Case 3D Coldstart", 10*TRLIB_EPS, 1e1*TRLIB_EPS);

    qp.radius = 0.5;
    qp.reentry = 1;
    trlib_test_solve_check_qp(&qp, "Easy Case 3D Warmstart", 10*TRLIB_EPS, 1e1*TRLIB_EPS);
    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_3d_near_hard)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_DENSE_QP, TRLIB_TEST_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;

    struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense*) qp.problem;
    trlib_flt_t *hess = problem->hess, *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 0.0; grad[1] = 2.0; grad[2] = 1e-6;  // near hard case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Near Hard Case 3D Coldstart", 1e7*TRLIB_EPS, 1e1*TRLIB_EPS);

    qp.radius = 0.5;
    qp.reentry = 1;
    //trlib_test_solve_check_qp(&qp, "Near Hard Case 3D Warmstart", 1e7*TRLIB_EPS, 1e1*TRLIB_EPS);
    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_3d_hard)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_DENSE_QP, TRLIB_TEST_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;
    qp.ctl_invariant = TRLIB_CLC_EXP_INV_GLO;

    struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense*) qp.problem;
    trlib_flt_t *hess = problem->hess, *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 0.0; grad[1] = 2.0; grad[2] = 0.0;  // hard case
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Hard Case 3D Coldstart", 1e7*TRLIB_EPS, -1.0);

    qp.radius = 0.5;
    qp.reentry = 1;
    trlib_test_solve_check_qp(&qp, "Hard Case 3D Warmstart", 1e7*TRLIB_EPS, -1.0);
    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_3d_hard_as_resolve)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_DENSE_QP, TRLIB_TEST_SOLVER_KRYLOV, 3, 10*3, &qp);
    qp.verbose = 1;
    qp.ctl_invariant = TRLIB_CLC_EXP_INV_GLO;

    struct trlib_test_problem_dense* problem = (struct trlib_test_problem_dense*) qp.problem;
    trlib_flt_t *hess = problem->hess, *grad = problem->grad;

    hess[0] = 1.0; hess[1] = 0.0; hess[2] = 4.0;
    hess[3] = 0.0; hess[4] = 2.0; hess[5] = 0.0;
    hess[6] = 4.0; hess[7] = 0.0; hess[8] = 3.0;

    grad[0] = 1.0; grad[1] = 2.0; grad[2] = 3.0;  // something
    qp.radius = 1.0;

    trlib_test_solve_check_qp(&qp, "Easy Case 3D Coldstart build up for hard case", 1e2*TRLIB_EPS, 1e1*TRLIB_EPS);
    
    grad[0] = 0.0; grad[1] = 2.0; grad[2] = 0.0; // hard case

    qp.verbose = 2;

    trlib_test_resolve_new_gradient(&qp);
    trlib_test_check_optimality(&qp);
    trlib_test_print_result(&qp, "Hard Case 3D via warmstart", 1e5*TRLIB_EPS, -1.0);

    trlib_test_free_qp(&qp);
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
    tcase_add_test(tc_core, test_3d_hard);
    tcase_add_test(tc_core, test_3d_hard_as_resolve);
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
