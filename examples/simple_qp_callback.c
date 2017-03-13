#include <stdlib.h>
#include "math.h"
#include "string.h"
#include "trlib.h"
#include "trlib_krylov.h"

// blas
void daxpy_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
void dscal_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx);
void dcopy_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
trlib_flt_t dnrm2_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx);
trlib_flt_t ddot_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
// lapack
void dgemv_(char *trans, trlib_int_t *m, trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *a, trlib_int_t *lda, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *beta, trlib_flt_t *y, trlib_int_t *incy);

/* Simple Driver Program that solves a trust region problem
 *
 * min 1/2 x^T H x + g^T x
 * s.t. || x || <= Delta
 *
 * where H is provided in the form of a callback function */

struct trlib_qpdata {
    trlib_int_t n;                     ///< dimension of problem
    trlib_int_t maxiter;               ///< maximum number of Krylov subspace iterations
    trlib_int_t *iwork;                ///< integer work space
    trlib_flt_t *fwork;                ///< floating point workspace
    trlib_flt_t *gradient;             ///< gradient of QP
    trlib_int_t hotstart;              ///< flag that determines if hotstarted or not
    void (*hv_cb)(double *, double *); ///< callback to compute hessian vector product
    trlib_int_t iter;                  ///< iteration counter
    trlib_flt_t *g;                    ///< gradient of Krylov iteration
    trlib_flt_t *gm;                   ///< previous gradient of Krylov iteration
    trlib_flt_t *p;                    ///< direction
    trlib_flt_t *Hp;                   ///< hessian product
    trlib_flt_t *Q;                    ///< matrix with Lanczos directions
};

int prepare_qp(trlib_int_t n, trlib_int_t maxiter, double *gradient, void (*hv_cb)(double *, double *), struct trlib_qpdata *data) {
    data->n = n;
    data->maxiter = maxiter;
    data->gradient = gradient;
    data->hv_cb = hv_cb;
    data->hotstart = 0;
    trlib_int_t iwork_size, fwork_size, h_pointer;
    trlib_krylov_memory_size(maxiter, &iwork_size, &fwork_size, &h_pointer);
    data->iwork = malloc(iwork_size*sizeof(trlib_int_t));
    data->fwork = malloc(fwork_size*sizeof(trlib_flt_t));
    data->g = malloc(n*sizeof(double));
    data->gm = malloc(n*sizeof(double));
    data->p = malloc(n*sizeof(double));
    data->Hp = malloc(n*sizeof(double));
    data->Q = malloc((maxiter+1)*n*sizeof(double));
    data->iter = 0;
    return 0;
}

int destroy_qp(struct trlib_qpdata *data) {
    free(data->iwork);
    free(data->fwork);
    free(data->g);
    free(data->gm);
    free(data->p);
    free(data->Hp);
    free(data->Q);
    return 0;
}

int solve_qp(struct trlib_qpdata *data, trlib_flt_t radius, double *sol, double *lam) {

    // some default settings
    trlib_int_t equality = 0;
    trlib_int_t maxlanczos = 100;
    trlib_int_t ctl_invariant = 0;
    trlib_int_t refine = 1;
    trlib_int_t verbose = 1;
    trlib_int_t unicode = 0;
    trlib_flt_t tol_rel_i = 1e-8;
    trlib_flt_t tol_rel_b = 1e-5;
    trlib_flt_t tol_abs_i = 0.0;
    trlib_flt_t tol_abs_b = 0.0;


    trlib_int_t ret = 0;

    trlib_int_t n = data->n;
    trlib_int_t init = 0, inc = 1, itp1 = 0;
    trlib_flt_t minus = -1.0, one = 1.0, z = 0.0;
    if(!data->hotstart) { init = TRLIB_CLS_INIT; trlib_krylov_prepare_memory(data->maxiter, data->fwork); }
    else { init = TRLIB_CLS_HOTSTART; }

    trlib_flt_t v_dot_g = 0.0, p_dot_Hp = 0.0, flt1, flt2, flt3;
    trlib_int_t action, ityp;

    trlib_int_t iwork_size, fwork_size, h_pointer;
    trlib_krylov_memory_size(data->maxiter, &iwork_size, &fwork_size, &h_pointer);

    while(1) {
        ret = trlib_krylov_min(init, radius, equality, data->maxiter, maxlanczos,
                tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b,
                TRLIB_EPS*TRLIB_EPS, ctl_invariant, v_dot_g, v_dot_g, p_dot_Hp, data->iwork, data->fwork, 
                refine, verbose, unicode, "", stdout, NULL,
                &action, &(data->iter), &ityp, &flt1, &flt2, &flt3);
        init = 0;
        switch(action) {
            case TRLIB_CLA_INIT:
                memset(sol, 0, n*sizeof(trlib_flt_t)); memset(data->gm, 0, n*sizeof(trlib_flt_t));
                dcopy_(&n, data->gradient, &inc, data->g, &inc);
                v_dot_g = ddot_(&n, data->g, &inc, data->g, &inc);
                dcopy_(&n, data->g, &inc, data->p, &inc); dscal_(&n, &minus, data->p, &inc); // p = -g
                data->hv_cb(data->p, data->Hp); // Hp = H*p
                p_dot_Hp = ddot_(&n, data->p, &inc, data->Hp, &inc);
                dcopy_(&n, data->g, &inc, data->Q, &inc); // Q(0:n) = g
                flt1 = 1.0/sqrt(v_dot_g); dscal_(&n, &flt1, data->Q, &inc); // Q(0:n) = g/sqrt(<g,g>)
                break;
            case TRLIB_CLA_RETRANSF:
                itp1 = data->iter+1;
                dgemv_("N", &n, &itp1, &one, data->Q, &n, data->fwork+h_pointer, &inc, &z, sol, &inc); // s = Q_i * h_i
                break;
            case TRLIB_CLA_UPDATE_STATIO:
                if (ityp == TRLIB_CLT_CG) { daxpy_(&n, &flt1, data->p, &inc, sol, &inc); }; // s += flt1*p
                break;
            case TRLIB_CLA_UPDATE_GRAD:
                if (ityp == TRLIB_CLT_CG) {
                    dcopy_(&n, data->g, &inc, data->Q+(data->iter)*n, &inc); dscal_(&n, &flt2, data->Q+(data->iter)*n, &inc); // Q(iter*n:(iter+1)*n) = flt2*g
                    dcopy_(&n, data->g, &inc, data->gm, &inc); daxpy_(&n, &flt1, data->Hp, &inc, data->g, &inc); // gm = g; g += flt1*Hp
                }
                if (ityp == TRLIB_CLT_L) {
                    dcopy_(&n, data->Hp, &inc, sol, &inc); daxpy_(&n, &flt1, data->g, &inc, sol, &inc); daxpy_(&n, &flt2, data->gm, &inc, sol, &inc); // s = Hp + flt1*g + flt2*gm
                    dcopy_(&n, data->g, &inc, data->gm, &inc); dscal_(&n, &flt3, data->gm, &inc); // gm = flt3*g
                    dcopy_(&n, sol, &inc, data->g, &inc); // g = s
                }
                v_dot_g = ddot_(&n, data->g, &inc, data->g, &inc);
                break;
            case TRLIB_CLA_UPDATE_DIR:
                if (ityp == TRLIB_CLT_CG) { dscal_(&n, &flt2, data->p, &inc); daxpy_(&n, &minus, data->g, &inc, data->p, &inc); } // p = -g + flt2 * p
                if (ityp == TRLIB_CLT_L) { dcopy_(&n, data->g, &inc, data->p, &inc); dscal_(&n, &flt1, data->p, &inc); } // p = flt1*g
                data->hv_cb(data->p, data->Hp); // Hp = H*p
                p_dot_Hp = ddot_(&n, data->p, &inc, data->Hp, &inc);
                if ( ityp == TRLIB_CLT_L) { dcopy_(&n, data->p, &inc, data->Q+(data->iter)*n, &inc); } // Q(iter*n:(iter+1)*n) = p
                break;
            case TRLIB_CLA_CONV_HARD:
                itp1 = data->iter+1;
                trlib_flt_t *temp = malloc(n*sizeof(trlib_flt_t));
                data->hv_cb(sol, temp); // temp = H*s
                daxpy_(&n, &one, data->gradient, &inc, temp, &inc); // temp = H*s + g
                daxpy_(&n, &flt1, sol, &inc, temp, &inc); // temp = H*s + g + flt1*s
                v_dot_g = ddot_(&n, temp, &inc, temp, &inc);
                free(temp);
                break;
            case TRLIB_CLA_NEW_KRYLOV:
                printf("Hit invariant Krylov subspace. Please implement proper reorthogonalization!");
                break;
        }
        if( ret < 10 ) { break; }
    }
    *lam = data->fwork[7];

    if(!data->hotstart) { data->hotstart = 1; }
    return ret;
}

/* Test the driver program to solve a 3D problem with two different radii
 */

void hessvec(double *d, double *Hd) {
    Hd[0] = d[0] + 4.0*d[2];
    Hd[1] = 2.0*d[1];
    Hd[2] = 4.0*d[0] + 3.0*d[2];
}

int main () {
    
    trlib_int_t n = 3;        // number of variables
    trlib_int_t maxiter = 10*n; // maximum number of CG iterations

    // gradient of QP
    double *g = malloc(n*sizeof(double));
    g[0] = 5.0; g[1] = 0.0; g[2] = 4.0;

    // get datastructure for QP solution and prepare it
    struct trlib_qpdata data;
    prepare_qp(n, maxiter, g, &hessvec, &data);

    // allocate memory for solution
    double *sol = malloc(n*sizeof(double));
    double lam = 0.0;

    // solve QP with trust region radius 2.0
    double radius = 2.0;
    printf("Attempting to solve trust region problem with radius %f\n", radius);
    solve_qp(&data, radius, sol, &lam);
    printf("Got lagrange multiplier %f and solution vector [%f %f %f]\n\n", radius, sol[0], sol[1], sol[2]);

    // resolve QP with trust region radius 1.0
    radius = 1.0;
    printf("Attempting to solve trust region problem with radius %f\n", radius);
    solve_qp(&data, radius, sol, &lam);
    printf("Got lagrange multiplier %f and solution vector [%f %f %f]\n\n", radius, sol[0], sol[1], sol[2]);

    // clean up
    destroy_qp(&data);
    free(g); free(sol);

    return 0;

}
