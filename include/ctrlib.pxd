cimport libc.stdio

cdef extern from "trlib_krylov.h":
    int trlib_krylov_prepare_memory(int itmax, double *fwork)
    int trlib_krylov_memory_size(int itmax, int *iwork_size, int *fwork_size, int *h_pointer)
    int trlib_krylov_min(
        int init, double radius, int equality, int itmax, int itmax_lanczos,
        double tol_rel_i, double tol_abs_i,
        double tol_rel_b, double tol_abs_b, double zero,
        int ctl_invariant, double g_dot_g, double v_dot_g, double p_dot_Hp,
        int *iwork, double *fwork, int refine,
        int verbose, int unicode, char *prefix, libc.stdio.FILE *fout, long *timing,
        int *action, int *iter, int *ityp,
        double *flt1, double *flt2, double *flt3)
    int trlib_krylov_timing_size()
