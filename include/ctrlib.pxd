cimport libc.stdio

cdef extern from "trlib_krylov.h":
    long trlib_krylov_prepare_memory(long itmax, double *fwork)
    long trlib_krylov_memory_size(long itmax, long *iwork_size, long *fwork_size, long *h_pointer)
    long trlib_krylov_min(
        long init, double radius, long equality, long itmax, long itmax_lanczos,
        double tol_rel_i, double tol_abs_i,
        double tol_rel_b, double tol_abs_b, double zero,
        long ctl_invariant, double g_dot_g, double v_dot_g, double p_dot_Hp,
        long *iwork, double *fwork, int refine,
        long verbose, long unicode, char *prefix, libc.stdio.FILE *fout, long *timing,
        long *action, long *iter, long *ityp,
        double *flt1, double *flt2, double *flt3)
    int trlib_krylov_timing_size()
