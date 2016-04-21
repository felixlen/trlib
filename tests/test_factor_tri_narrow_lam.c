#include "trlib_test.h"

START_TEST (test_narrow_lam)
{
    struct trlib_test_qp qp;
    trlib_test_malloc_qp(TRLIB_TEST_TRI_QP, TRLIB_TEST_SOLVER_FACTOR, 24, 10*24, &qp);
    qp.verbose = 3;

    struct trlib_test_problem_tri* problem = (struct trlib_test_problem_tri*) qp.problem;
    double *diag = problem->diag; double *offdiag = problem->offdiag; double *grad = problem->grad;

    diag[0] = 4.47950346952296243e+01;
    diag[1] = 1.23884363977749331e+03;
    diag[2] = 8.30372343168578254e+02;
    diag[3] = 1.11363171260887839e+03;
    diag[4] = 9.14324317530082681e+02;
    diag[5] = 1.06703161340074257e+03;
    diag[6] = 9.49332242656669450e+02;
    diag[7] = 1.04257650826223403e+03;
    diag[8] = 9.69055583464256415e+02;
    diag[9] = 1.02735407509631614e+03;
    diag[10] = 9.81869412858080523e+02;
    diag[11] = 1.01689547746230210e+03;
    diag[12] = 9.90919801892236137e+02;
    diag[13] = 1.00924750520936300e+03;
    diag[14] = 9.97661805526487569e+02;
    diag[15] = 1.00341995414418898e+03;
    diag[16] = 1.00286230221876963e+03;
    diag[17] = 9.98847134993543818e+02;
    diag[18] = 1.00771276904753631e+03;
    diag[19] = 9.94556034155226598e+02;
    diag[20] = 1.01083791086374208e+03;
    diag[21] = 9.91788680568126097e+02;
    diag[22] = 1.01333797254376259e+03;
    diag[23] = 9.91399329951191589e+02;
    offdiag[0] = 4.47950346952296243e+01;
    offdiag[1] = 1.23884363977749331e+03;
    offdiag[2] = 8.30372343168578254e+02;
    offdiag[3] = 1.11363171260887839e+03;
    offdiag[4] = 9.14324317530082681e+02;
    offdiag[5] = 1.06703161340074257e+03;
    offdiag[6] = 9.49332242656669450e+02;
    offdiag[7] = 1.04257650826223403e+03;
    offdiag[8] = 9.69055583464256415e+02;
    offdiag[9] = 1.02735407509631614e+03;
    offdiag[10] = 9.81869412858080523e+02;
    offdiag[11] = 1.01689547746230210e+03;
    offdiag[12] = 9.90919801892236137e+02;
    offdiag[13] = 1.00924750520936300e+03;
    offdiag[14] = 9.97661805526487569e+02;
    offdiag[15] = 1.00341995414418898e+03;
    offdiag[16] = 1.00286230221876963e+03;
    offdiag[17] = 9.98847134993543818e+02;
    offdiag[18] = 1.00771276904753631e+03;
    offdiag[19] = 9.94556034155226598e+02;
    offdiag[20] = 1.01083791086374208e+03;
    offdiag[21] = 9.91788680568126097e+02;
    offdiag[22] = 1.01333797254376259e+03;

    grad[0] = 1.46718887913599506e-06;

    qp.radius = 8.095430810031050583575e+00;
    trlib_test_solve_check_qp(&qp, "narrow suitable lam", 1e8*TRLIB_EPS, TRLIB_EPS);
    
    qp.radius = 0.5;
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
