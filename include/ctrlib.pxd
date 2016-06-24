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
    long trlib_krylov_timing_size()

cdef extern from "trlib_tri_factor.h":
    long trlib_tri_factor_min(
        long nirblk, long *irblk, double *diag, double *offdiag,
        double *neglin, double radius, 
        long itmax, double tol_rel, long pos_def, long equality,
        long *warm0, double *lam0, long *warm, double *lam,
        long *warm_leftmost, long *ileftmost, double *leftmost,
        long *warm_fac0, double *diag_fac0, double *offdiag_fac0,
        long *warm_fac, double *diag_fac, double *offdiag_fac,
        double *sol0, double *sol, double *ones, double *fwork,
        long refine,
        long verbose, long unicode, char *prefix, libc.stdio.FILE *fout,
        long *timing, double *obj, long *iter_newton, long *sub_fail)

cdef extern from "trlib_leftmost.h":
    long trlib_leftmost_irreducible(
        long n, double *diag, double *offdiag,
        long warm, double leftmost_minor, long itmax, double tol_abs,
        long verbose, long unicode, char *prefix, libc.stdio.FILE *fout,
        long *timing, double *leftmost, long *iter_pr)

    long trlib_tri_timing_size()
    long trlib_tri_factor_memory_size(long)

