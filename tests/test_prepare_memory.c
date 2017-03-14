#include "trlib_test.h"
#include "trlib/trlib_krylov.h"

START_TEST (test_prepare)
{
    trlib_int_t itmax = 314;
    trlib_flt_t *fwork = malloc((29+17*itmax)*sizeof(trlib_flt_t));
    trlib_krylov_prepare_memory(itmax, fwork);
    for(trlib_int_t jj = 23+11*itmax; jj<24+12*itmax; ++jj) { ck_assert_msg(fwork[jj] == 1.0, "Ones improperly initialized, %d --> %e", jj, fwork[jj]); }
    for(trlib_int_t jj = 17+2*itmax; jj<17+3*itmax; ++jj) { ck_assert_msg(fwork[jj] == 0.0, "Neglin improperly initialized, %d --> %e", jj, fwork[jj]); }
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
