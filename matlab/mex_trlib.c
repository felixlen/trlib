#include "mex.h"
#include "trlib.h"
#include <assert.h>

const char *trlib_fields [] = {
    "init", "radius", "equality", "itmax", "itmax_lanczos",
    "tol_rel_i", "tol_abs_i",
    "tol_rel_b", "tol_abs_b", "zero",
    "ctl_invariant", "g_dot_g", "v_dot_g", "p_dot_Hp",
    "iwork", "fwork", "refine",
    "verbose", "action", "iter", "ityp", "flt", "krylov_min_retval"
};

mxArray *mxCreateInt64Scalar (trlib_int_t value)
{
    mxArray *res = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL);
    trlib_int_t *res_val = mxGetData (res);
    *res_val = value;
    return res;
}

void call_krylov_min (mxArray *TR)
{
    trlib_int_t *init = mxGetData (mxGetField (TR, 0, "init"));
    trlib_flt_t *radius = mxGetData (mxGetField (TR, 0, "radius"));
    trlib_int_t *equality = mxGetData (mxGetField (TR, 0, "equality"));
    trlib_int_t *itmax = mxGetData (mxGetField (TR, 0, "itmax"));
    trlib_int_t *itmax_lanczos = mxGetData (mxGetField (TR, 0, "itmax_lanczos"));
    trlib_flt_t *tol_rel_i = mxGetData (mxGetField (TR, 0, "tol_rel_i"));
    trlib_flt_t *tol_rel_b = mxGetData (mxGetField (TR, 0, "tol_rel_b"));
    trlib_flt_t *tol_abs_i = mxGetData (mxGetField (TR, 0, "tol_abs_i"));
    trlib_flt_t *tol_abs_b = mxGetData (mxGetField (TR, 0, "tol_abs_b"));
    trlib_flt_t *zero = mxGetData (mxGetField (TR, 0, "zero"));
    trlib_int_t *ctl_invariant = mxGetData (mxGetField (TR, 0, "ctl_invariant"));
    trlib_flt_t *g_dot_g = mxGetData (mxGetField (TR, 0, "g_dot_g"));
    trlib_flt_t *g_dot_v = mxGetData (mxGetField (TR, 0, "g_dot_v"));
    trlib_flt_t *p_dot_Hp = mxGetData (mxGetField (TR, 0, "p_dot_Hp"));
    trlib_int_t *iwork = mxGetData (mxGetField (TR, 0, "iwork"));
    trlib_flt_t *fwork = mxGetData (mxGetField (TR, 0, "fwork"));
    trlib_int_t *refine = mxGetData (mxGetField (TR, 0, "refine"));
    trlib_int_t *verbose = mxGetData (mxGetField (TR, 0, "verbose"));
    trlib_int_t *action = mxGetData (mxGetField (TR, 0, "action"));
    trlib_int_t *iter = mxGetData (mxGetField (TR, 0, "iter"));
    trlib_int_t *ityp = mxGetData (mxGetField (TR, 0, "ityp"));
    trlib_flt_t *flt = mxGetData (mxGetField (TR, 0, "flt"));
    trlib_int_t *krylov_min_retval = mxGetData (mxGetField (TR, 0, "krylov_min_retval"));

    *krylov_min_retval = trlib_krylov_min (*init, *radius, *equality, *itmax,
            *itmax_lanczos, *tol_rel_i, *tol_rel_b, *tol_abs_i, *tol_abs_b,
            *zero, *ctl_invariant, *g_dot_g, *g_dot_v, *p_dot_Hp, iwork, fwork,
            *refine, *verbose, 0, "mex_trlib", stdout, NULL, action, iter, ityp,
            flt, flt+1, flt+2);
}

/* The gateway function */
void mexFunction (int nlhs, mxArray *plhs [], int nrhs, const mxArray *prhs [])
{
    assert (sizeof (trlib_int_t) == 8);
    char command [2];
    mwSize dims [2];

    if (nrhs < 1)
        mexErrMsgTxt ("mex_trlib requires a command string as its first argument");

    mxGetString (prhs [0], command, 2);
    command [1] = 0;
    if (strcmp (command, "s") == 0) {
        // solve
    }
    else if (strcmp (command, "i") == 0) {
        // initialize
        if (nrhs != 2)
            mexErrMsgTxt ("mex_trlib (""i"", itmax) needs exactly two arguments");
        if (nlhs != 1)
            mexErrMsgTxt ("mex_trlib (""i"", itmax) needs exactly two outputs");
        if (!mxIsClass (prhs [1], "int64"))
            mexErrMsgTxt ("second argument must be int64");

        trlib_int_t iwork_size, fwork_size, h_pointer;
        trlib_int_t *itmax = (trlib_int_t *) mxGetData (prhs [1]);
        if (!itmax)
            mexErrMsgTxt ("itmax pointer is null");
        trlib_krylov_memory_size (*itmax, &iwork_size, &fwork_size, &h_pointer);

        dims [0] = iwork_size;
        dims [1] = 1;
        mxArray *iwork = mxCreateNumericArray (2, dims, mxINT64_CLASS, mxREAL);
        dims [0] = fwork_size;
        mxArray *fwork = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);

        trlib_krylov_prepare_memory (*itmax, mxGetPr (fwork));

        dims [0] = 1;
        plhs [0] = mxCreateStructArray (2, dims, sizeof (trlib_fields) / sizeof (char *), 
                trlib_fields);
        mxSetField (plhs [0], 0, "init", mxCreateInt64Scalar (1));
        mxSetField (plhs [0], 0, "radius", mxCreateDoubleScalar (1.0));
        mxSetField (plhs [0], 0, "equality", mxCreateInt64Scalar (0));
        mxSetField (plhs [0], 0, "itmax", mxCreateInt64Scalar (*itmax));
        mxSetField (plhs [0], 0, "itmax_lanczos", mxCreateInt64Scalar (100));
        mxSetField (plhs [0], 0, "tol_rel_i", mxCreateDoubleScalar (1e-8));
        mxSetField (plhs [0], 0, "tol_rel_b", mxCreateDoubleScalar (1e-5));
        mxSetField (plhs [0], 0, "tol_abs_i", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "tol_abs_b", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "zero", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "ctl_invariant", mxCreateInt64Scalar (0));
        mxSetField (plhs [0], 0, "iwork", iwork);
        mxSetField (plhs [0], 0, "fwork", fwork);
        mxSetField (plhs [0], 0, "refine", mxCreateInt64Scalar (1));
        mxSetField (plhs [0], 0, "action", mxCreateInt64Scalar (0));
        mxSetField (plhs [0], 0, "iter", mxCreateInt64Scalar (0));
        mxSetField (plhs [0], 0, "ityp", mxCreateInt64Scalar (0));
        mxSetField (plhs [0], 0, "flt", mxCreateDoubleMatrix (3, 1, mxREAL));
        mxSetField (plhs [0], 0, "krylov_min_retval", mxCreateInt64Scalar (0));

        //call_krylov_min (plhs [0]);
    }
    else
        mexErrMsgTxt ("Unknown trlib command");
}

