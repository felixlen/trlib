#include <check.h>
#include "trlib_leftmost.h"

START_TEST (test_diagonal)
{
    int n = 10; int nirblk = n;
    double *diag = malloc(n*sizeof(double));
    double *offdiag = calloc((n-1), sizeof(double));
    double *leftmost = malloc(nirblk*sizeof(double));
    int *irblk = malloc((nirblk+1)*sizeof(int));
    long *timing = malloc(8*sizeof(long));
    int ileftmost; int ret = 0;
    for(int ii = 0; ii < nirblk+1; ++ii) { irblk[ii] = ii; }

    diag[0] = 10.0; diag[1] = 9.0; diag[2] = -10.0; diag[3] = 4.0; diag[4] = 0.0; diag[5] = 6.0; diag[6] = 0.0; diag[7] = -1.0; diag[8] = -10.0; diag[9] = 1.0;
    trlib_leftmost(nirblk, irblk, diag, offdiag, 0, 0.0, 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &ileftmost, leftmost);
    for(int ii = 0; ii < nirblk; ++ii) { ck_assert_msg(leftmost[ii] == diag[ii], "diagonal matrix: lefmost[%d] = %e should equal diag[%d] = %e", ii, leftmost[ii], ii, diag[ii]); }
    ck_assert_msg(ileftmost == 2, "block that corresponds to smallest eigenvalue has index 2, but received %d", ileftmost);

    free(diag); free(offdiag); free(leftmost); free(irblk); free(timing);
}
END_TEST

START_TEST (test_warm)
{
    int n = 3; int nirblk = 2;
    double *diag = malloc(n*sizeof(double));
    double *offdiag = calloc((n-1), sizeof(double));
    double *leftmost = malloc(nirblk*sizeof(double));
    int *irblk = malloc((nirblk+1)*sizeof(int));
    long *timing = malloc(8*sizeof(long));
    int ileftmost; int ret = 0;
    irblk[0] = 0; irblk[1] = 1; irblk[2] = 3;

    diag[0] = 1.0; diag[1] = 2.0; diag[2] = -1.75;
    offdiag[1] = 1.0;

    // just test principal 1x1 submatrix, should have leftmost eigenvalue 1.0
    nirblk = 1;
    trlib_leftmost(nirblk, irblk, diag, offdiag, 0, 0.0, 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &ileftmost, leftmost);
    ck_assert_msg((leftmost[0] == diag[0]), "first block diagonal, leftmost[0] = %e should equal diag[0] = %e", leftmost[0], diag[0]);
    ck_assert_msg(ileftmost == 0, "block that corresponds to smallest eigenvalue has index 0, but received %d", ileftmost);

    // just test principal 2x2 submatrix, should have leftmost eigenvalue 1.0
    nirblk = 2; irblk[nirblk] = 2;
    trlib_leftmost(nirblk, irblk, diag, offdiag, 1, 0.0, 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &ileftmost, leftmost);
    ck_assert_msg((leftmost[0] == diag[0]), "first block diagonal, leftmost[0] = %e should equal diag[0] = %e", leftmost[0], diag[0]);
    ck_assert_msg((leftmost[1] == diag[1]), "second block diagonal, leftmost[1] = %e should equal diag[1] = %e", leftmost[1], diag[1]);
    ck_assert_msg(ileftmost == 0, "block that corresponds to smallest eigenvalue has index 0, but received %d", ileftmost);

    // test complete 3x3 submatrix, should have leftmost eigenvalue 1.0
    nirblk = 2; irblk[nirblk] = 3;
    trlib_leftmost(nirblk, irblk, diag, offdiag, 1, diag[1], 10*n, TRLIB_EPS, 1, 1, "", stderr, timing,
           &ileftmost, leftmost);
    ck_assert_msg((leftmost[0] == diag[0]), "first block diagonal, leftmost[0] = %e should equal diag[0] = %e", leftmost[0], diag[0]);
    ck_assert_msg(fabs(leftmost[1] +2.0) <= 1.1*TRLIB_EPS, "second block leftmost[1] = -2.0, residue %e", leftmost[1]+2.0);
    ck_assert_msg(ileftmost == 1, "block that corresponds to smallest eigenvalue has index 1, but received %d", ileftmost);


    free(diag); free(offdiag); free(leftmost); free(irblk); free(timing);
}
END_TEST



Suite *leftmost_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Leftmost Reducible Suite");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_diagonal);
    tcase_add_test(tc_core, test_warm);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = leftmost_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
