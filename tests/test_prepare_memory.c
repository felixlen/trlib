#include "trlib_test.h"

START_TEST (test_prepare)
{
    int itmax = 314;
    double *fwork = malloc((27+17*itmax)*sizeof(double));
    trlib_prepare_memory(itmax, fwork);
    for(int jj = 21+11*itmax; jj<22+12*itmax; ++jj) { ck_assert_msg(fwork[jj] == 1.0, "Ones improperly initialized, %d --> %e", jj, fwork[jj]); }
    for(int jj = 15+2*itmax; jj<15+3*itmax; ++jj) { ck_assert_msg(fwork[jj] == 0.0, "Neglin improperly initialized, %d --> %e", jj, fwork[jj]); }
    free(fwork);
}
END_TEST

Suite *prepare_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Prepare Memory");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_prepare);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = prepare_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
