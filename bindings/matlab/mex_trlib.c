/* MIT License
 *
 * Copyright (c) 2016--2017 Andreas Potschka, Felix Lenders
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "mex.h"
#include "trlib.h"
#include <assert.h>
#include <string.h>

const char *trlib_fields [] = {
    "init", "radius", "equality", "itmax", "itmax_lanczos",
    "tol_rel_i", "tol_abs_i",
    "tol_rel_b", "tol_abs_b", "zero", "obj_lo",
    "ctl_invariant", "convexify", "earlyterm",
    "g_dot_g", "v_dot_g", "p_dot_Hp",
    "iwork", "fwork", "h_pointer", "refine",
    "verbose", "action", "iter", "ityp", "flt", "krylov_min_retval"
};

mxArray *mxCreateIntScalar (trlib_int_t value)
{
    mxArray *res;
    if ( sizeof(mwSignedIndex) == 8 )
        res = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL);
    else
        res = mxCreateNumericMatrix (1, 1, mxINT32_CLASS, mxREAL);
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
    trlib_flt_t *obj_lo = mxGetData (mxGetField (TR, 0, "obj_lo"));
    trlib_int_t *ctl_invariant = mxGetData (mxGetField (TR, 0, "ctl_invariant"));
    trlib_int_t *convexify = mxGetData (mxGetField (TR, 0, "convexify"));
    trlib_int_t *earlyterm = mxGetData (mxGetField (TR, 0, "earlyterm"));
    trlib_flt_t *g_dot_g = mxGetData (mxGetField (TR, 0, "g_dot_g"));
    trlib_flt_t *v_dot_g = mxGetData (mxGetField (TR, 0, "v_dot_g"));
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
            *itmax_lanczos, *tol_rel_i, *tol_abs_i, *tol_rel_b, *tol_abs_b,
            *zero, *obj_lo, *ctl_invariant, *convexify, *earlyterm,
            *g_dot_g, *v_dot_g, *p_dot_Hp, iwork, fwork,
            *refine, *verbose, 0, "mex_trlib: ", stdout, NULL, action, iter, ityp,
            flt, flt+1, flt+2);
    *init = 0;
}

/* The gateway function */
void mexFunction (int nlhs, mxArray *plhs [], int nrhs, const mxArray *prhs [])
{
    assert (sizeof (trlib_int_t) == 8);
    char command [2];
    mwSize dims [2];

    if (nlhs != 1)
        mexErrMsgTxt ("mex_trlib requires exactly one output");
    if (nrhs < 1)
        mexErrMsgTxt ("mex_trlib requires a command string as its first argument");

    mxGetString (prhs [0], command, 2);
    command [1] = 0;
    if (strcmp (command, "t") == 0) {
        /* type of integer */
        if (nrhs != 1)
            mexErrMsgTxt ("mex_trlib (""t"", TR) needs exactly one argument");
        plhs [0] = mxCreateIntScalar(sizeof(mwSignedIndex));
    }
    else if (strcmp (command, "s") == 0) {
        /* solve */
        if (nrhs != 2)
            mexErrMsgTxt ("mex_trlib (""s"", TR) needs exactly two arguments");
        // this seems to work in Linux and Mac OS if compiled with gcc running by CMake
        // plhs [0] = (mxArray *) prhs [1];
        // if running Windows or MATLAB MEX Compiler is used, we have to deep copy
        plhs [0] = mxDuplicateArray(prhs[1]);
        call_krylov_min (plhs [0]);
    }
    else if (strcmp (command, "i") == 0) {
        /* initialize */
        if (nrhs != 2)
            mexErrMsgTxt ("mex_trlib (""i"", itmax) needs exactly two arguments");
        if( sizeof(mwSignedIndex) == 8 && !mxIsClass (prhs [1], "int64"))
            mexErrMsgTxt ("second argument must be int64");
        else if( sizeof(mwSignedIndex) == 4 && !mxIsClass (prhs [1], "int32"))
            mexErrMsgTxt ("second argument must be int32");

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
        mxSetField (plhs [0], 0, "init", mxCreateIntScalar (1));
        mxSetField (plhs [0], 0, "radius", mxCreateDoubleScalar (1.0));
        mxSetField (plhs [0], 0, "equality", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "itmax", mxCreateIntScalar (*itmax));
        mxSetField (plhs [0], 0, "itmax_lanczos", mxCreateIntScalar (100));
        mxSetField (plhs [0], 0, "tol_rel_i", mxCreateDoubleScalar (-2.0));
        mxSetField (plhs [0], 0, "tol_abs_i", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "tol_rel_b", mxCreateDoubleScalar (-3.0));
        mxSetField (plhs [0], 0, "tol_abs_b", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "zero", mxCreateDoubleScalar (2.22e-16));
        mxSetField (plhs [0], 0, "obj_lo", mxCreateDoubleScalar (-1e20));
        mxSetField (plhs [0], 0, "ctl_invariant", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "convexify", mxCreateIntScalar (1));
        mxSetField (plhs [0], 0, "earlyterm", mxCreateIntScalar (1));
        mxSetField (plhs [0], 0, "g_dot_g", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "v_dot_g", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "p_dot_Hp", mxCreateDoubleScalar (0.0));
        mxSetField (plhs [0], 0, "iwork", iwork);
        mxSetField (plhs [0], 0, "fwork", fwork);
        mxSetField (plhs [0], 0, "h_pointer", mxCreateIntScalar (h_pointer+1));
        mxSetField (plhs [0], 0, "refine", mxCreateIntScalar (1));
        mxSetField (plhs [0], 0, "verbose", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "action", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "iter", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "ityp", mxCreateIntScalar (0));
        mxSetField (plhs [0], 0, "flt", mxCreateDoubleMatrix (3, 1, mxREAL));
        mxSetField (plhs [0], 0, "krylov_min_retval", mxCreateIntScalar (0));
    }
    else
        mexErrMsgTxt ("Unknown trlib command");
}

