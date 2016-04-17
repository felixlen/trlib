cimport libc.stdio

cdef extern from "trlib.h":
    int trlib_prepare_memory(int itmax, double *fwork)
    int trlib_krylov_min(
        int init, double radius, int equality, int itmax, int itmax_lanczos,
        double tol_rel_i, double tol_abs_i, double tol_rel_b, double tol_abs_b, double zero,
        double v_dot_g, double p_dot_Hp , int *iwork, double *fwork, int refine,
        int verbose, int unicode, char *prefix, libc.stdio.FILE *fout,  long *timing, int *action,
        int *iter, int *ityp, double *flt1, double *flt2, double *flt3)

cdef extern from "trlib_driver.h":
    struct trlib_driver_qp:
        int qptype
        int qpsolver
        void* problem
        void *work
        int itmax
        char *prefix
        int verbose
        int unicode
        libc.stdio.FILE *stream
        double radius
        int equality
        int reentry
        int refine
        long *timing
        double lam
        double tol_rel_i
        double tol_abs_i
        double tol_rel_b
        double tol_abs_b
        int ret
        int sub_fail
        int iter
        double obj
        double obj_check
        double pos_def_res
        double kkt_res
        double tr_res
        double orth_res

    struct trlib_driver_problem_op:
        int n
        double *grad
        double *sol
        void *userdata
        void (*hv)(void *, int, double *, double *)

    int trlib_driver_malloc_qp(int qptype, int qpsolver, int n, int itmax, trlib_driver_qp *qp)
    int trlib_driver_free_qp(trlib_driver_qp *qp)
    int trlib_driver_solve_qp(trlib_driver_qp *qp)
    int trlib_driver_problem_set_hvcb(trlib_driver_problem_op* problem, void *userdata, void (*hv_cb)(void *, int, double *, double *))

