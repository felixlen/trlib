#include "trlib_test.h"

START_TEST (test_1)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_KRYLOV, 300, 10*300, &qp);
    qp.verbose = 1;
    qp.tol_rel_i = 1e5*TRLIB_EPS;
    qp.tol_rel_b = 1e5*TRLIB_EPS;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;
    
    diag[0] = -1.0; diag[1] = 2.0; diag[2] = 2.0; diag[3] = 2.0;
    for(trlib_int_t ii = 0; ii < problem->n; ++ii) { grad[ii] = 1.0; }

    qp.radius = 3.2e3;
    trlib_test_solve_check_qp(&qp, "Coldstart diagonal with zeros", 1e9*TRLIB_EPS, 1e1*TRLIB_EPS);
    
    qp.reentry = 1;
    qp.radius = 1e3;
    trlib_test_solve_check_qp(&qp, "Warmstart diagonal with zeros", 1e9*TRLIB_EPS, 1e1*TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_2)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_KRYLOV, 300, 10*300, &qp);
    qp.verbose = 1;
    qp.tol_rel_i = 1e5*TRLIB_EPS;
    qp.tol_rel_b = 1e5*TRLIB_EPS;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;
    
    diag[0] = -1.0; diag[1] = 3.0; diag[2] = 2.0; diag[3] = 2.0;
    for(trlib_int_t ii = 0; ii < problem->n; ++ii) { grad[ii] = 1.0; }

    qp.radius = 3.2e3;
    trlib_test_solve_check_qp(&qp, "Coldstart diagonal with zeros", 1e9*TRLIB_EPS, 1e3*TRLIB_EPS);
    
    qp.reentry = 1;
    qp.radius = 1e3;
    trlib_test_solve_check_qp(&qp, "Warmstart diagonal with zeros", 1e9*TRLIB_EPS, 1e3*TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_3)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_KRYLOV, 300, 10*300, &qp);
    qp.verbose = 1;
    qp.tol_rel_i = 1e5*TRLIB_EPS;
    qp.tol_rel_b = 1e5*TRLIB_EPS;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;
    
    diag[0] = 3.0; diag[1] = 1.0; diag[2] = 2.0; diag[3] = 2.0;
    for(trlib_int_t ii = 0; ii < problem->n; ++ii) { grad[ii] = 1.0; }

    qp.radius = 3.2e3;
    trlib_test_solve_check_qp(&qp, "Coldstart diagonal with zeros", 1e9*TRLIB_EPS, 1e2*TRLIB_EPS);
    
    qp.reentry = 1;
    qp.radius = 1e3;
    trlib_test_solve_check_qp(&qp, "Warmstart diagonal with zeros", 1e9*TRLIB_EPS, 1e2*TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

START_TEST (test_4)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_KRYLOV, 300, 10*300, &qp);
    qp.verbose = 1;
    qp.tol_rel_i = 1e5*TRLIB_EPS;
    qp.tol_rel_b = 1e5*TRLIB_EPS;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    trlib_flt_t *diag = problem->diag, *offdiag = problem->offdiag, *grad = problem->grad;
    
    diag[0] = -1.0; diag[1] = -1.0; diag[2] = 2.0; diag[3] = 2.0;
    for(trlib_int_t ii = 0; ii < problem->n; ++ii) { grad[ii] = 1.0; }

    qp.radius = 3.2e3;
    trlib_test_solve_check_qp(&qp, "Coldstart diagonal with zeros", 1e9*TRLIB_EPS, 1e1*TRLIB_EPS);
    
    qp.reentry = 1;
    qp.radius = 1e3;
    trlib_test_solve_check_qp(&qp, "Warmstart diagonal with zeros", 1e9*TRLIB_EPS, 1e1*TRLIB_EPS);

    trlib_test_free_qp(&qp);
}
END_TEST

Suite *tri_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Tridiagonal Lanczos Trigger TR Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_1);
    tcase_add_test(tc_core, test_2);
    tcase_add_test(tc_core, test_3);
    tcase_add_test(tc_core, test_4);
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
