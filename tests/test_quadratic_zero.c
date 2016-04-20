#include <check.h>
#include "trlib_quadratic_zero.h"

START_TEST (test_well_behaved_seperated)
{
    double tol = TRLIB_EPS_POW_75;
    double c_lin = -5.0;
    double c_abs = 6.0;
    double t1 = 0.0;
    double t2 = 0.0;
    trlib_quadratic_zero(c_abs, c_lin, tol, 0, 0, "", NULL, &t1, &t2);
    ck_assert_msg(t1<=t2, "order failure: t1 > t2");
    ck_assert_msg((t1*(t1+c_lin)+c_abs)<=TRLIB_EPS, "t1 is not a zero");
    ck_assert_msg((t2*(t2+c_lin)+c_abs)<=TRLIB_EPS, "t2 is not a zero");
}
END_TEST

START_TEST (test_well_behaved_tiny_seperation)
{
    double tol = TRLIB_EPS_POW_75;
    double c_lin = -1.0-1.0-10.0*TRLIB_EPS;
    double c_abs = 1.0+10.0*TRLIB_EPS;
    double t1 = 0.0;
    double t2 = 0.0;
    trlib_quadratic_zero(c_abs, c_lin, tol, 0, 0, "", NULL, &t1, &t2);
    ck_assert_msg(t1<=t2, "order failure: t1 > t2");
    ck_assert_msg((t1*(t1+c_lin)+c_abs)<=TRLIB_EPS, "t1 is not a zero");
    ck_assert_msg((t2*(t2+c_lin)+c_abs)<=TRLIB_EPS, "t2 is not a zero");
}
END_TEST

START_TEST (test_ill_behaved)
{
    double tol = TRLIB_EPS_POW_75;
    double c_lin = 1.0;
    double c_abs = 0.5*tol;
    double t1 = 0.0;
    double t2 = 0.0;
    trlib_quadratic_zero(c_abs, c_lin, tol, 0, 0, "", NULL, &t1, &t2);
    ck_assert_msg(t1<=t2, "order failure: t1 > t2");
    ck_assert_msg((t1*(t1+c_lin)+c_abs)<=TRLIB_EPS, "t1 is not a zero");
    ck_assert_msg((t2*(t2+c_lin)+c_abs)<=TRLIB_EPS, "t2 is not a zero");
}
END_TEST

Suite *quadratic_zero_suite(void)
{
    Suite *s;
    TCase *tc_core;
    s = suite_create("Quadratic Zero");
    tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_well_behaved_seperated);
    tcase_add_test(tc_core, test_well_behaved_tiny_seperation);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = quadratic_zero_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed==0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
