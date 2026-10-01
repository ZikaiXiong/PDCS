// The grid/few-block mapping needs one-thread global wrappers, while the
// moderate, sufficient, and massive mappings call the device functions
// directly.  Keep all EXP projection mathematics in exp_proj.cu so standard,
// diagonal-primal, and diagonal-initial (dual/slack) paths cannot drift.
#include "exp_proj.cu"

__global__ void exponent_proj_kernel(
    double *v, double *t_warm_start, double abs_tol, double rel_tol) {
  exponent_proj(v, t_warm_start, abs_tol, rel_tol);
}

__global__ void exponent_proj_diagonal_kernel(
    double *v, double *D, double *t_warm_start,
    double abs_tol, double rel_tol) {
  exponent_proj_diagonal(v, D, t_warm_start, abs_tol, rel_tol);
}

__global__ void dualExponent_proj_diagonal_kernel(
    double *v, double *D, double *temp, double *t_warm_start,
    double abs_tol, double rel_tol) {
  dualExponent_proj_diagonal(
      v, D, temp, t_warm_start, abs_tol, rel_tol);
}

__global__ void dualExponent_proj_kernel(
    double *v, double *t_warm_start, double abs_tol, double rel_tol) {
  dualExponent_proj(v, t_warm_start, abs_tol, rel_tol);
}
