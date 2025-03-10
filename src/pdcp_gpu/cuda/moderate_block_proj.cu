#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
// #define proj_abs_tol_squared 1e-32
#define MAX_ITER 100000
// suitable for very much block number

#include "exp_proj.cu"

// n is the length of the vector, including the first element
// len is the length of the vector, not including the first element or the top two elements

// BLAS functions
__device__ double nrm2(const long* __restrict__ n, const double* __restrict__ x, long *thread_idx, long *blk_dim)
{
  __shared__ double partial_norm[1024];
  partial_norm[*thread_idx] = 0.0;
  // calculate the norm of the vector x
  double val = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    val = x[j];
    partial_norm[*thread_idx] += val * val;
  }
  __syncthreads();
  for (int stride = *blk_dim / 2; stride > 0; stride /= 2)
  {
    if (*thread_idx < stride)
    {
      partial_norm[*thread_idx] += partial_norm[*thread_idx + stride];
    }
    __syncthreads();
  }
  return sqrt(partial_norm[0]);
}

__device__ double nrm2_squared(const long* __restrict__ n, const double* __restrict__ x, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // calculate the norm of the vector x
  __shared__ double partial_norm[1024];
  partial_norm[*thread_idx] = 0.0;
  double val = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    val = x[j];
    partial_norm[*thread_idx] += val * val;
  }
  __syncthreads();
  for (int stride = *blk_dim / 2; stride > 0; stride /= 2)
  {
    if (*thread_idx < stride)
    {
      partial_norm[*thread_idx] += partial_norm[*thread_idx + stride];
    }
    __syncthreads();
  }
  return partial_norm[0];
}

__device__ void mem_copy(const long* __restrict__ n, double* __restrict__ dst, const double* __restrict__ src, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // dst = src
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    dst[j] = src[j];
  }
  // __syncthreads();
}

__device__ void scal(const long* __restrict__ n, const double* __restrict__ sx, const double* __restrict__ sa,  double* __restrict__ sy, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the scalar sa,
  // sy = sx * sa
  if (*sa == 1.0)
  {
    mem_copy(n, sy, sx, thread_idx, blk_dim);
    return;
  }
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    sy[j] = sx[j] * sa[0];
  }
  // __syncthreads();
}

__device__ void scal_inplace(const long* __restrict__ n, const double* __restrict__ sa, double* __restrict__ sx, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the scalar sa,
  // sy = sx * sa
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    sx[j] *= sa[0];
  }
  // __syncthreads();
}

__device__ void rscl(const long* __restrict__ n, const double* __restrict__ sx, const double* __restrict__ sa,  double* __restrict__ sy, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the scalar sa,
  // sy = sx / sa
  if (*sa == 1.0)
  {
    mem_copy(n, sy, sx, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    sy[j] = sx[j] / sa[0];
  }
  // __syncthreads();
}

__device__ void rscl_inplace(const long* __restrict__ n, const double* __restrict__ sa, double* __restrict__ sx, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the scalar sa,
  // sy = sx / sa
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    sx[j] /= sa[0];
  }
  // __syncthreads();
}

__device__ void vvscal(const long* __restrict__ n, const double* __restrict__ s, const double* __restrict__ x, double* __restrict__ y, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the vector s,
  // y = s * x
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    y[j] = x[j] * s[j];
  }
  // __syncthreads();
}

__device__ void vvscal_inplace(const long* __restrict__ n, const double* __restrict__ s, double* __restrict__ x, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the vector s,
  // x = s * x
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    x[j] *= s[j];
  }
  // __syncthreads();
}

__device__ void vvrscl(const long* __restrict__ n, const double* __restrict__ x, const double* __restrict__ s,  double* __restrict__ y, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the vector s,
  // y = x / s
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    y[j] = x[j] / s[j];
  }
  // __syncthreads();
}

__device__ void vvrscl_inplace(const long* __restrict__ n, const double* __restrict__ s, double* __restrict__ x, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // scale the vector x by the vector s,
  // x = x / s
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    x[j] /= s[j];
  }
  // __syncthreads();
}

__device__ double diff_norm(const long* __restrict__ n, const double* __restrict__ x, const double* __restrict__ y, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  // y = x - y
  __shared__ double partial_diff[1024];
  partial_diff[*thread_idx] = 0.0;
  double diff = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    diff = x[j] - y[j];
    partial_diff[*thread_idx] += diff * diff;
  }
  __syncthreads();
  for (int stride = *blk_dim / 2; stride > 0; stride /= 2)
  {
    if (*thread_idx < stride)
    {
      partial_diff[*thread_idx] += partial_diff[*thread_idx + stride];
    }
    __syncthreads();
  }
  return sqrt(partial_diff[0]);
}

// END OF BLAS FUNCTIONS

// AUXILIARY FUNCTIONS
__device__ double f(double *a4, double *a3, double *a2, double *a1, double *a0, double x) {
    return a4[0] * x * x * x * x + a3[0] * x * x * x + a2[0] * x * x + a1[0] * x + a0[0];
}

__device__ double df(double *a4, double *a3, double *a2, double *a1, double *a0, double x) {
    return 4 * a4[0] * x * x * x + 3 * a3[0] * x * x + 2 * a2[0] * x + a1[0];
}

__device__ double solve_quartic(double *a4, double *a3, double *a2, double *a1, double *a0) {
    // solve the polynomial equation a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0
    // using Newton's method to search for the root
    double tolerance = 1e-10; // tolerance
    double step = 1e-6;      // search step
    double x = sqrt(-a0[0]);        // search starting point (can be adjusted according to the scenario)
    int found_roots = 0;     // number of real roots found
    int iter = 0;
    while (x <= 1.5 * sqrt(-a0[0]) && iter < 10000000) {
        // calculate f(x) and f'(x)
        double fx = f(a4, a3, a2, a1, a0, x);
        double dfx = df(a4, a3, a2, a1, a0, x);

        if (fabs(fx) < tolerance) { // check if x is a root
            double fx_small = f(a4, a3, a2, a1, a0, x - step);
            double fx_large = f(a4, a3, a2, a1, a0, x + step);
            if (fx_small < 0 && fx_large > 0) {
                return x;
            }
            else {
                x += 1.5 * sqrt(-a0[0]);
            }
        } else if (dfx != 0) { // use Newton's method to iterate
            x = x - fx / dfx;
        }
    }

    if (found_roots == 0) {
      printf("No real roots found in the range: %f.\n", 1.5 * sqrt(-a0[0]));
      return x;
    }
    return 0.0;
}

// projection functions
__device__ void box_proj(double *sol, const double *bl, const double *bu, long *n, long *thread_idx, long *blk_dim)
{
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    // printf("sol[%ld]: %f, bl[%ld]: %f, bu[%ld]: %f\n", j, sol[j], j, bl[j], j, bu[j]);
    sol[j] = fmin(fmax(sol[j], bl[j]), bu[j]);
  }
}

__device__ void soc_proj(double* __restrict__ sol, long* __restrict__ n, long *thread_idx, long *blk_dim)
{
  long len = *n - 1;
  double norm = nrm2(&len, &sol[1], thread_idx, blk_dim);
  double t = sol[0];
  // if (*thread_idx == 0) {
  //   printf("moderate_block_proj soc_proj, norm: %f, t: %f\n", norm, t);
  // }
  if (norm + t <= 0)
  {
    // printf("moderate_block_proj soc_proj 0\n");
    for (long j = *thread_idx; j < *n; j += *blk_dim)
    {
      sol[j] = 0.0;
    }
  }
  else if (norm <= t)
  {
    // Do nothing, continue with the next iteration
    // printf("moderate_block_proj soc_proj 1\n");
  }
  else
  {
    double c = (1.0 + t / norm) / 2.0;
    if (*thread_idx == 0)
    {
      // printf("moderate_block_proj soc_proj 2\n");
      sol[0] = norm * c;
    }
    for (long j = *thread_idx + 1; j < *n; j += *blk_dim)
    {
      sol[j] *= c;
    }
  }
}

__device__ void positive_proj(double *sol, long *n)
{
  for (long j = 0; j < *n; ++j)
  {
    sol[j] = fmax(sol[j], 0.0);
  }
}

__device__ void process_lambd1(double *x0, double *y0, double *C, double *x, double *y)
{
  // solving 0.5 * (x - x0)^2 + 0.5 * (y - y0)^2 s.t. x >= 0, y >= 0, x * y = C
  if (*C == 0)
  {
    // case 1: x = 0, y = max(y0, 0)
    x[0] = 0.0;
    y[0] = fmax(*y0, 0.0);
    double diff = y[0] - *y0;
    double obj1 = 0.5 * (*x0) * (*x0) + 0.5 * diff * diff;
    // case 2: x = max(x0, 0), y = 0
    x[0] = fmax(*x0, 0.0);
    y[0] = 0.0;
    diff = x[0] - *x0;
    double obj2 = 0.5 * diff * diff + 0.5 * (*y0) * (*y0);
    if (obj1 < obj2)
    {
      x[0] = 0.0;
      y[0] = fmax(*y0, 0.0);
      return;
    }
    else
    {
      x[0] = fmax(*x0, 0.0);
      y[0] = 0.0;
      return;
    }
  }
  else
  {
    // case 3: min 0.5 * (x - x0)^2 + 0.5 * (y - y0)^2 s.t. x >= 0, y >= 0, x * y = C
    double a4 = 1.0;
    double a3 = -x0[0];
    double a2 = 0.0;
    double a1 = C[0] * y0[0];
    double a0 = -C[0] * C[0];
    double root = solve_quartic(&a4, &a3, &a2, &a1, &a0);
    x[0] = root;
    y[0] = C[0] / root;
  }
}

__device__ int solve_quadratic(double *a, double *b, double *c, double *roots) {
  double discriminant = b[0] * b[0] - 4 * a[0] * c[0];
  if (discriminant < 0) {
    printf("No real roots found in the range: %f.\n", 1.5 * sqrt(-a[0]));
    return 0;
  }
  else if (discriminant == 0) {
    roots[0] = -b[0] / (2 * a[0]);
    return 1;
  }
  else {
    roots[0] = (-b[0] + sqrt(discriminant)) / (2 * a[0]);
    roots[1] = (-b[0] - sqrt(discriminant)) / (2 * a[0]);
    return 2;
  }
}

// END OF AUXILIARY FUNCTIONS

__device__ void rsoc_proj(double *sol, long *n, double *temp1, double *temp2, long* __restrict__ thread_idx, long* __restrict__ blk_dim){
  double minVal = 1e-3;
  rscl(n, sol, &minVal, sol, thread_idx, blk_dim);
  __syncthreads();
  double x0y0 = sol[0] * sol[1];
  double x0Squr = sol[0] * sol[0];
  double y0Squr = sol[1] * sol[1];
  long len = *n - 2;
  double z0NrmSqur = nrm2_squared(&len, &sol[2], thread_idx, blk_dim);
  if (2 * x0y0 > z0NrmSqur && sol[0] >= 0 && sol[1] >= 0) {
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
  if (sol[0] <= 0 && sol[1] <= 0 && 2 * x0y0 >= z0NrmSqur) {
    // thrust::fill(sol, sol + n[0], 0.0);
    for (long j = *thread_idx; j < *n; j += *blk_dim){
      sol[j] = 0.0;
    }
    // __syncthreads();
    return;
  }
  if (fabs(sol[0] + sol[1]) < positive_zero) {
    long len = *n - 2;
    double s = 2;
    rscl(&len, &sol[2], &s, &sol[2], thread_idx, blk_dim);
    double C = nrm2_squared(&len, &sol[2], thread_idx, blk_dim);
    if (*thread_idx == 0) {
      process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
    }
    __syncthreads();
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
  double alpha = z0NrmSqur - 2 * x0y0;
  double beta = -2 * (z0NrmSqur + x0Squr + y0Squr);
  double roots[2];
  int rootNum = solve_quadratic(&alpha, &beta, &alpha, roots);
  if (rootNum == 1) {
    double lambd = roots[0];
    if (fabs(lambd - 1) < positive_zero) {
      long len = *n - 2;
      double s = 2;
      rscl(&len, &sol[2], &s, &sol[2], thread_idx, blk_dim); // sol[2:end] /= 2
      double C = nrm2_squared(&len, &sol[2], thread_idx, blk_dim);
      if (*thread_idx == 0) {
        process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
      }
      __syncthreads();
      scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
      return;
    }
    double denominator = (1 - lambd * lambd);
    double xNew = (sol[0] + lambd * sol[1]) / denominator;
    double yNew = (sol[1] + lambd * sol[0]) / denominator;
    sol[0] = xNew;
    sol[1] = yNew;
    long len = *n - 2;
    rscl(&len, &sol[2], &denominator, &sol[2], thread_idx, blk_dim); // sol[2:end] /= denominator
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
  else if (rootNum == 2) {
    // two roots
    // case 1: xNew1 > 0, yNew1 > 0, xNew2 > 0, yNew2 > 0
    double lambd1 = roots[0];
    double denominator1 = (1 - lambd1 * lambd1);
    double xNew1 = (sol[0] + lambd1 * sol[1]) / denominator1;
    double yNew1 = (sol[1] + lambd1 * sol[0]) / denominator1;
    // case 2: xNew1 > 0, yNew1 > 0, xNew2 <= 0, yNew2 <= 0
    double lambd2 = roots[1];
    double denominator2 = (1 - lambd2 * lambd2);
    double xNew2 = (sol[0] + lambd2 * sol[1]) / denominator2;
    double yNew2 = (sol[1] + lambd2 * sol[0]) / denominator2;
    if (xNew1 > 0 && yNew1 > 0) {
      if (xNew2 > 0 && yNew2 > 0) {
        // two points are feasible
        temp1[0] = xNew1;
        temp1[1] = yNew1;
        temp2[0] = xNew2;
        temp2[1] = yNew2;
        denominator1 = 1 + lambd1;
        denominator2 = 1 + lambd2;
        rscl(&len, &sol[2], &denominator1, &temp1[2], thread_idx, blk_dim); // sol[2:end] /= denominator1
        rscl(&len, &sol[2], &denominator2, &temp2[2], thread_idx, blk_dim); // sol[2:end] /= denominator2
        double norm1 = diff_norm(n, temp1, sol, thread_idx, blk_dim);
        double norm2 = diff_norm(n, temp2, sol, thread_idx, blk_dim);
        if (norm1 < norm2) {
          mem_copy(n, sol, temp1, thread_idx, blk_dim);
        }
        else {
          mem_copy(n, sol, temp2, thread_idx, blk_dim);
        }
        scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
        return;
      }
      else {
        // only one point is feasible
        if (*thread_idx == 0) {
          sol[0] = xNew1;
          sol[1] = yNew1;
        }
        __syncthreads();
        denominator1 = 1 + lambd1;
        rscl(&len, &sol[2], &denominator1, &sol[2], thread_idx, blk_dim); // sol[2:end] /= denominator1
        scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
        return;
      }
    }
    else if (xNew2 > 0 && yNew2 > 0) {
      if (*thread_idx == 0) {
        sol[0] = xNew2;
        sol[1] = yNew2;
      }
      __syncthreads();
      denominator2 = 1 + lambd2;
      rscl(&len, &sol[2], &denominator2, &sol[2], thread_idx, blk_dim); // sol[2:end] /= denominator2
      scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
      return;
    }
    else {
      // thrust::fill(sol, sol + n[0], 0.0);
      for (long j = *thread_idx; j < *n; j += *blk_dim){
        sol[j] = 0.0;
      }
      // __syncthreads();
    }
  }
}


__device__ double oracle_soc_f_sqrt(double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  // len not including the first element
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  // __syncthreads();
  return nrm2(len, temp_part, thread_idx, blk_dim) - (x[0] / (1 - 2 * xi[0]));
}

__device__ void oracle_soc_h(double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  // len not including the first element
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  double left = nrm2_squared(len, temp_part, thread_idx, blk_dim);
  double right = (x[0] / (1 - 2 * xi[0]));
  right = right * right;
  *f = left - right;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = temp_part[j] / sqrt(fmax(2 * xi[0] + D_scaled_squared_part[j], 1e-16));
  }
  right = right / (1 - 2 * xi[0]);
  *h = -4 * (nrm2_squared(len, temp_part, thread_idx, blk_dim) + right);
}

// __device__ void newton_soc_rootsearch(double *xiLeft, double *xiRight, double *xi, double *sol, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len) {
//   for (int i = 0; i < 20; ++i) {
//     double f, h;
//     oracle_soc_h(xi, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part, len, &f, &h);
//     if (f < 0) {
//       *xiRight = *xi;
//     }
//     else {
//       *xiLeft = *xi;
//     }
//     if (*xiRight <= *xiLeft) {
//       break;
//     }
//     if (fabs(f) <= abs_tol) {
//       break;
//     }
//     *xi = fmin(fmax(*xi, *xiLeft + rel_tol), *xiRight - rel_tol);
//   }
// }

__device__ void soc_proj_diagonal_recover(double *sol, long *n, double *xi, double *minVal, double *t_warm_start, double *D_scaled_squared, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  if (*thread_idx == 0) {
    t_warm_start[0] = xi[0];
    sol[0] = sol[0] / (1 - 2 * t_warm_start[0]) * minVal[0];
  }
  for (long j = 1 + *thread_idx; j < *n; j += *blk_dim) {
    sol[j] = sol[j] / (1 + 2 *  xi[0] * D_scaled_squared[j]) * minVal[0];
  }
  // __syncthreads();
  return;
}

__device__ void soc_proj_decreasing_newton_step(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  *t_warm_start_val_test -= fmax(fmin(*oracleVal / *oracleVal_h, 0.001), -0.001);
  *t_warm_start_val_test = fmax(fmin(*t_warm_start_val_test, *xiRight - rel_tol), *xiLeft + rel_tol);
  oracle_soc_h(t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, thread_idx, blk_dim);
  if (fabs(*oracleVal) < abs_tol * abs_tol){
    *xi = *t_warm_start_val_test;
    soc_proj_diagonal_recover(sol, n, xi, minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    return;
  }
  if (*oracleVal < 0){
    *xiRight = *t_warm_start_val_test;
  }
  else {
    *xiLeft = *t_warm_start_val_test;
  }
}

__device__ void decreasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  if (*oracleVal < 0){
    *xiRight = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001) {
      for (int k = 0; k < 2; ++k) {
        soc_proj_decreasing_newton_step(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001) {
      for (int k = 0; k < 2; ++k) {
        soc_proj_decreasing_newton_step(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void soc_proj_decreasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol)
{
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_soc_rootsearch(&xiLeft, &xiRight, &xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    *xi = (*xiRight + *xiLeft) / 2;
    *oracleVal = oracle_soc_f_sqrt(xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
    count++;
    if(count > MAX_ITER){
      break;
    }
    if (*oracleVal < 0){
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}

__device__ void soc_increasing_newton_step(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  *t_warm_start_val_test -= fmax(fmin(*oracleVal / *oracleVal_h, 0.001), -0.001);
  *t_warm_start_val_test = fmax(fmin(*t_warm_start_val_test, *xiRight - rel_tol), *xiLeft + rel_tol);
  // *oracleVal = oracle_soc_f_sqrt(t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
  oracle_soc_h(t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, thread_idx, blk_dim);
  if (fabs(*oracleVal) < abs_tol * abs_tol){
    *xi = *t_warm_start_val_test;
    *xiLeft = *t_warm_start_val_test;
    *xiRight = *t_warm_start_val_test;
    soc_proj_diagonal_recover(sol, n, xi, minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    return;
  }
  if (*oracleVal > 0){
    *xiRight = *t_warm_start_val_test;
  }
  else {
    *xiLeft = *t_warm_start_val_test;
  }
}

__device__ void increasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) 
{
  if (*oracleVal > 0){
    *xiRight = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001) {
      for (int k = 0; k < 2; ++k) {
        soc_increasing_newton_step(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001) {
      for (int k = 0; k < 2; ++k) {
        soc_increasing_newton_step(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void soc_proj_increasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol)
{
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_soc_rootsearch(&xiLeft, &xiRight, &xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    *xi = (*xiRight + *xiLeft) / 2;
    *oracleVal = oracle_soc_f_sqrt(xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
    count++;
    if (count > MAX_ITER){
      break;
    }
    if (*oracleVal > 0){
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}


__device__ void soc_proj_diagonal(double* __restrict__ sol, long* __restrict__ n, double* __restrict__ D_scaled, double* __restrict__ D_scaled_squared, double* __restrict__ D_scaled_mul_x, double* __restrict__ temp, double* __restrict__ t_warm_start, long* __restrict__ i, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  double minVal = 1e-3;
  rscl_inplace(n, &minVal, sol, thread_idx, blk_dim);
  __syncthreads();
  double t = sol[0];
  long len = *n - 1;
  double *x2end = sol + 1;
  double *D_scaled_part = D_scaled + 1;
  double *temp_part = temp + 1;
  double *D_scaled_mul_x_part = D_scaled_mul_x + 1;
  double *D_scaled_squared_part = D_scaled_squared + 1;
  double xi = 0.0;
  double xiLeft = 0.0;
  double xiRight = 1.0;
  double oracleVal = 1.0;
  double oracleVal_h = 1.0;
  double t_warm_start_val_test = t_warm_start[0];
  // temp_part = x2end ./ D_scaled_part
  vvrscl(&len, x2end, D_scaled_part, temp_part, thread_idx, blk_dim);
  __syncthreads();
  if (nrm2(&len, temp_part, thread_idx, blk_dim) <= -sol[0] && sol[0] <= 0) {
    for (long j = *thread_idx; j < *n; j += *blk_dim){
      sol[j] = 0.0;
    }
    return;
  }
  vvscal(&len, D_scaled_part, x2end, D_scaled_mul_x_part, thread_idx, blk_dim);
  if (nrm2(&len, D_scaled_mul_x_part, thread_idx, blk_dim) <= sol[0]) {
    if (*thread_idx == 0) {
      sol[0] = fmax(sol[0], 0.0);
    }
    __syncthreads();
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
  if (t > rel_tol) {
    xiRight = 0.5;
    xiLeft = 0.0;
    oracleVal = 1.0;
    if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight){
      // oracleVal = oracle_soc_f_sqrt(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim);
      oracle_soc_h(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        xi = t_warm_start[0];
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        return;
      }
      decreasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    soc_proj_decreasing_binary_search(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
    soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    return;
  }
  else if (t < -rel_tol) {
    xiRight = 1.0;
    xiLeft = 0.5;
    if (t_warm_start_val_test > xiLeft){
      // oracleVal = oracle_soc_f_sqrt(&t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim);
      oracle_soc_h(&t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        xi = t_warm_start_val_test;
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        return;
      }
      increasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    while (oracle_soc_f_sqrt(&xiRight, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
      xiLeft = xiRight;
      xiRight *= 2;
    }
    soc_proj_increasing_binary_search(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
    soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  else {
    for (long j = 1 + *thread_idx; j < *n; j += *blk_dim) {
      sol[j] = sol[j] / (1 + D_scaled_squared[j]) * minVal;
      temp[j] = D_scaled[j] * sol[j];
    }
    // __syncthreads();

    double temp_val = nrm2(&len, temp + 1, thread_idx, blk_dim); // has multiply minVal
    if (*thread_idx == 0) {
      sol[0] = temp_val;
    }
    // __syncthreads();
    return;
  }
}


__device__ double oracle_rsoc_f_sqrt(double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_squared_part, double *temp_part, long *len, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = x_mul_d_part[j] / (1 + xi[0] * D_scaled_squared_part[j]);
  }
  double xi_sqr = xi[0] * xi[0];
  double xi_sqr_one = xi_sqr - 1;
  double xi_sqr_one_sqr = xi_sqr_one * xi_sqr_one;
  return nrm2(len, temp_part, thread_idx, blk_dim) - sqrt(2 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) / xi_sqr_one_sqr);
}

__device__ void oracle_rsoc_h(double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = x_mul_d_part[j] / (1 + xi[0] * D_scaled_squared_part[j]);
  }
  double xi_sqr = xi[0] * xi[0];
  double xi_sqr_one = xi_sqr - 1;
  double xi_sqr_one_sqr = xi_sqr_one * xi_sqr_one;
  double left = nrm2_squared(len, temp_part, thread_idx, blk_dim);
  double right = 2 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) / xi_sqr_one_sqr;
  *f = left - right;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = temp_part[j] / sqrt(1 + xi[0] * D_scaled_squared_part[j]) * D_scaled_part[j];
  }
  double h_left = -2 * nrm2_squared(len, temp_part, thread_idx, blk_dim);
  double h_right1 = 2 * (2 * x0y0[0] * xi[0] + x0_sqr[0] + y0_sqr[0]) / (1 - 2 * xi_sqr);
  double h_right2 = 8 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) * xi_sqr_one * xi[0] / xi_sqr_one_sqr;
  *h = h_left - h_right1 + h_right2;
}

// __device__ void newton_rsoc_rootsearch(double *xiLeft, double *xiRight, double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len) {
//   for (int i = 0; i < 20; ++i) {
//     double f, h;
//     oracle_rsoc_h(xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_part, D_scaled_squared_part, temp_part, len, &f, &h);
//     if (f < 0) {
//       *xiRight = *xi;
//     }
//     else {
//       *xiLeft = *xi;
//     }
//     if (*xiRight <= *xiLeft) {
//       break;
//     }
//     if (f < 1e+32 && f > -1e+32 && h < -rel_tol) {
//       *xi = *xi - f / h;
//     }
//     else {
//       break;
//     }
//     if (fabs(f) <= abs_tol) {
//       break;
//     }
//     *xi = fmin(fmax(*xi, *xiLeft + rel_tol), *xiRight - rel_tol);
//   }
// }

__device__ void recover_sol_rsoc(double *sol, long *n, double *minVal, double *xi, double *t_warm_start, double *D_scaled_squared, long* __restrict__ thread_idx, long* __restrict__ blk_dim){
  if (*thread_idx == 0){
    t_warm_start[0] = *xi;
    double xNew = (sol[0] + sol[1] * xi[0]) / (1 - xi[0] * xi[0] + positive_zero) * minVal[0];
    double yNew = (sol[1] + sol[0] * xi[0]) / (1 - xi[0] * xi[0] + positive_zero) * minVal[0];
    sol[0] = xNew;
    sol[1] = yNew;
  }
  __syncthreads();
  for (long j = 2 + *thread_idx; j < *n; j += *blk_dim) {
    sol[j] = sol[j] / (1 + xi[0] * D_scaled_squared[j]) * minVal[0];
  }
}

__device__ void rsoc_decreasing_newton_step(double *sol, long *n, double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  *t_warm_start_val_test -= fmax(fmin((*oracleVal)/((*oracleVal_h)), 0.001), -0.001);
  *t_warm_start_val_test = fmax(fmin(*t_warm_start_val_test, *xiRight - rel_tol), *xiLeft + rel_tol);
  // *oracleVal = oracle_rsoc_f_sqrt(t_warm_start_val_test, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
  oracle_rsoc_h(t_warm_start_val_test, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, thread_idx, blk_dim);
  if (fabs(*oracleVal) < abs_tol * abs_tol) {
    *xiRight = *t_warm_start_val_test;
    *xi = *t_warm_start_val_test;
    recover_sol_rsoc(sol, n, minVal, t_warm_start_val_test, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  if (*oracleVal < 0){
    *xiRight = *t_warm_start_val_test;
  }
  else {
    *xiLeft = *t_warm_start_val_test;
  }
}

__device__ void decreasing_binary_rsoc_proj_init(double *sol, long *n, double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  if (*oracleVal < 0) {
    *xiRight = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k){
        rsoc_decreasing_newton_step(sol, n, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k){  
        rsoc_decreasing_newton_step(sol, n, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void rsoc_proj_decreasing_binary_search(double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  *xi = (*xiRight + *xiLeft) / 2;
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    *xi = (*xiRight + *xiLeft) / 2;
    *oracleVal = oracle_rsoc_f_sqrt(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
    count++;
    if (count > MAX_ITER){
      break;
    }
    if (*oracleVal < 0) {
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}

__device__ void rsoc_increasing_newton_step(double *sol, long *n, double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  *t_warm_start_val_test -= fmax(fmin((*oracleVal)/((*oracleVal_h)), 0.001), -0.001);
  *t_warm_start_val_test = fmax(fmin(*t_warm_start_val_test, *xiRight - rel_tol), *xiLeft + rel_tol);
  oracle_rsoc_h(t_warm_start_val_test, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, thread_idx, blk_dim);
  if (fabs(*oracleVal) < abs_tol * abs_tol) {
    *xiRight = *t_warm_start_val_test;
    *xiLeft = *t_warm_start_val_test;
    recover_sol_rsoc(sol, n, minVal, t_warm_start_val_test, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  if (*oracleVal > 0){
    *xiRight = *t_warm_start_val_test;
  }
  else {
    *xiLeft = *t_warm_start_val_test;
  }
}

__device__ void increasing_binary_rsoc_proj_init(double *sol, long *n, double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  if (*oracleVal > 0) {
    *xiRight = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k){
        rsoc_increasing_newton_step(sol, n, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = t_warm_start[0];
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k){
        rsoc_increasing_newton_step(sol, n, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, minVal, t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void rsoc_proj_increasing_binary_search(double *x0_sqr, double *y0_sqr, double *x0y0, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol){
  *xi = (*xiRight + *xiLeft) / 2;
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    *xi = (*xiRight + *xiLeft) / 2;
    count++;
    if (count > MAX_ITER){
      break;
    }
    *oracleVal = oracle_rsoc_f_sqrt(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len, thread_idx, blk_dim);
    if (*oracleVal > 0) {
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}


__device__ void rsoc_proj_diagonal(double *sol, long *n, double *D_scaled, double *D_scaled_squared, double *D_scaled_mul_x, double *temp, double *t_warm_start, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  double minVal = 1e-3;
  if (fabs(sol[0]) > 1e+5){
    minVal = 1e+3;
  }
  rscl(n, sol, &minVal, sol, thread_idx, blk_dim);
  __syncthreads();
  long len = *n - 2;
  double *z = sol + 2;
  double *D_scaled_part = D_scaled + 2;
  double *temp_part = temp + 2;
  double *D_scaled_mul_x_part = D_scaled_mul_x + 2;
  double *D_scaled_squared_part = D_scaled_squared + 2;
  double xiLeft = 0.0;
  double xiRight = 0.0;
  double oracleVal = 0.0;
  double oracleVal_h = 0.0;
  double xi = t_warm_start[0];

  vvscal(&len, D_scaled_part, z, D_scaled_mul_x_part, thread_idx, blk_dim);
  double z0NrmSqur = nrm2_squared(&len, D_scaled_mul_x_part, thread_idx, blk_dim);
  if (2 * sol[0] * sol[1] >= z0NrmSqur && sol[0] >= 0 && sol[1] >= 0) {
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  vvrscl(&len, z, D_scaled_part, temp_part, thread_idx, blk_dim);
  double val = nrm2_squared(&len, temp_part, thread_idx, blk_dim);
  if (sol[0] <= 0 && sol[1] <= 0 && 2 * sol[0] * sol[1] > val) {
    for (long j = *thread_idx; j < *n; j += *blk_dim){
      sol[j] = 0.0;
    }
    // __syncthreads();
    return;
  }
  if (fabs(sol[0] + sol[1]) < positive_zero) {
    for (long j = *thread_idx; j < len; j += *blk_dim) {
      z[j] = z[j] / (1 + D_scaled_squared_part[j]);
      temp_part[j] = D_scaled_part[j] * z[j];
    }
    double C = nrm2_squared(&len, temp_part, thread_idx, blk_dim);
    if (*thread_idx == 0){
      process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
    }
    __syncthreads();
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
  double x0_sqr = sol[0] * sol[0];
  double y0_sqr = sol[1] * sol[1];
  double x0y0 = sol[0] * sol[1];
  double t_warm_start_val_test = t_warm_start[0];
  if (sol[0] > 0 && sol[1] > 0) {
    xiRight = 1.0;
    xiLeft = 0.0;
    oracleVal = 1.0;
    xi = (xiRight + xiLeft) / 2;
    if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
      oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol) {
        xi = t_warm_start[0];
        recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        // __syncthreads();
        return;
      }
      decreasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    rsoc_proj_decreasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
    xi = (xiRight + xiLeft) / 2;
    recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  else if (sol[0] < 0 && sol[1] < 0) {
    vvrscl(&len, z, D_scaled_part, temp_part, thread_idx, blk_dim);
    double val = nrm2_squared(&len, temp_part, thread_idx, blk_dim);
    if (2 * sol[0] * sol[1] > val) {
      // thrust::fill(sol, sol + n[0], 0.0);
      for (long j = *thread_idx; j < *n; j += *blk_dim){
        sol[j] = 0.0;
      }
      // __syncthreads();
      return;
    }
    xiRight = 2.0;
    xiLeft = 1.0;
    if (t_warm_start[0] > xiLeft) {
      xiRight = t_warm_start[0];
      oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol) {
        recover_sol_rsoc(sol, n, &minVal, &t_warm_start_val_test, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        // __syncthreads();
        return;
      }
      decreasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
      xiLeft = xiRight;
      xiRight *= 2;
    }
    rsoc_proj_decreasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
    xi = (xiRight + xiLeft) / 2;
    recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    // __syncthreads();
    return;
  }
  else{
    if (sol[0] < 0 && sol[1] > 0 && (sol[0] + sol[1] < 0 || sol[0] + sol[1] == 0)) {
      xiRight = -sol[0] / sol[1];
      xiLeft = 1.0;
      if (sol[1] == 0){
        xiRight = 1.0;
        if (t_warm_start[0] > xiLeft) {
          xiRight = t_warm_start[0];
          oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
          if (fabs(oracleVal) < abs_tol * abs_tol) {
            xi = t_warm_start[0];
            recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
            // __syncthreads();
            return;
          }
          increasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
        }
        while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
          xiLeft = xiRight;
          xiRight *= 2;
        }
      }
      rsoc_proj_increasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
      // __syncthreads();
      return;
    }
    else if (sol[0] < 0 && sol[1] > 0 && sol[0] + sol[1] >= 0) {
      xiRight = 1.0;
      xiLeft = -sol[0] / sol[1];
      if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
          xi = t_warm_start[0];
          recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
          // __syncthreads();
          return;
        }
        decreasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
      rsoc_proj_decreasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
      // __syncthreads();
      return;
    }
    else if (sol[0] >= 0 && sol[1] <= 0 && sol[0] + sol[1] <= 0) {
      xiLeft = 1.0;
      xiRight = -sol[1] / sol[0];
      if (sol[0] == 0){
        xiRight = 1.0;
        if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
          oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
          if (fabs(oracleVal) < abs_tol * abs_tol) {
            xi = t_warm_start[0];
            recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
            // __syncthreads();
            return;
          }
        }
        increasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
      while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
        xiLeft = xiRight;
        xiRight *= 2;
      }
      rsoc_proj_increasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
      // __syncthreads();
      return;
    }
    else if (sol[0] >= 0 && sol[1] <= 0 && sol[0] + sol[1] >= 0) {
      xiRight = 1.0;
      xiLeft = -sol[1] / sol[0];
      if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
          xi = t_warm_start[0];
          recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
          // __syncthreads();
          return;
        }
        decreasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
      rsoc_proj_decreasing_binary_search(&x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
      // __syncthreads();
      return;
    }
  }
}


// function for setting function pointers
//     0: dual_free_proj!
//     1: dual_free_proj_diagonal!
//     2: con_zero_proj!
//     3: dual_positive_proj!
//     4: dual_positive_proj_diagonal!
//     5: dual_soc_proj!
//     6: dual_soc_proj_diagonal!
//     7: dual_soc_proj_const_scale_diagonal!
//     8: dual_rsoc_proj!
//     9: dual_rsoc_proj_diagonal!
//     10: dual_rsoc_proj_const_scale_diagonal!
//     11: dual_EXP_proj!
//     12: dual_EXP_proj_diagonal!
//     13: con_EXP_proj!
//     14: dual_DUALEXP_proj!
//     15: dual_DUALEXP_proj_diagonal!
//     16: con_DUALEXP_proj!
//     17: box_proj!
//     18: box_proj_diagonal!
//     19: slack_box_proj!
//     20: soc_cone_proj!
//     21: soc_cone_proj_const_scale!
//     22: soc_cone_proj_diagonal!
//     23: rsoc_cone_proj!
//     24: rsoc_cone_proj_const_scale!
//     25: rsoc_cone_proj_diagonal!
//     26: EXP_proj!
//     27: EXP_proj_diagonal!
//     28: DUALEXP_proj!
//     29: DUALEXPonent_proj_diagonal!


extern "C" __global__ void
moderate_block_proj(double* arr, double* bl, double* bu, double* D_scaled, double* D_scaled_squared,  double* D_scaled_mul_x, double* temp, double* t_warm_start, const long* gpu_head_start, const long* ns, int blkNum, long* proj_type, double abs_tol, double rel_tol)
{
  long blk_idx = blockIdx.x;
  long thread_idx = threadIdx.x;
  long blk_dim = blockDim.x;
  long total_thread = gridDim.x * blk_dim;
  // one block per cone projection
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // one thread per cone projection
  if (proj_type[0] == 17 || proj_type[0] == 19 || proj_type[0] == 18){
    // all threads for box projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    double *sub_bl = bl + gpu_head_start[0];
    double *sub_bu = bu + gpu_head_start[0];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = min(max(sol[i], sub_bl[i]), sub_bu[i]);
    }
  }
  if (proj_type[0] == 2){
    // all threads for zero projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = 0.0;
    }
  }
  if (proj_type[0] == 3 || proj_type[0] == 4){
    // all threads for positive projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = fmax(sol[i], 0.0);
    }
  }
  if (proj_type[1] == 3 || proj_type[1] == 4){
    // all threads for positive projection
    long n = ns[1];
    double *sol = arr + gpu_head_start[1];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = fmax(sol[i], 0.0);
    }
  }
  if (proj_type[2] == 3 || proj_type[2] == 4){
    // all threads for positive projection
    long n = ns[2];
    double *sol = arr + gpu_head_start[2];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = fmax(sol[i], 0.0);
    }
  }
  if (blk_idx < blkNum)
  // if (i == 0)
  {
    long n = ns[blk_idx];
    double *sol = arr + gpu_head_start[blk_idx];
    double *sub_D_scaled = D_scaled + gpu_head_start[blk_idx];
    double *sub_D_scaled_squared = D_scaled_squared + gpu_head_start[blk_idx];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + gpu_head_start[blk_idx];
    double *sub_temp = temp + gpu_head_start[blk_idx];
    // double *sub_bl = bl + gpu_head_start[blk_idx];
    // double *sub_bu = bu + gpu_head_start[blk_idx];
    if (proj_type[blk_idx] == 0 || proj_type[blk_idx] == 1){
      // dual_free_proj
      ;
    }
    // else if (proj_type[blk_idx] == 17 || proj_type[blk_idx] == 19 || proj_type[blk_idx] == 18){
    //   // box_proj
    //   box_proj(sol, sub_bl, sub_bu, &n, &thread_idx, &blk_dim);
    // }
    // else if (proj_type[blk_idx] == 2){
    //   // thrust::fill(sol, sol + n, 0.0);
    //   for (long j = thread_idx; j < n; j += blk_dim){
    //     sol[j] = 0.0;
    //   }
    // }
    // else if (proj_type[blk_idx] == 3 || proj_type[blk_idx] == 4){
    //   // dual_positive_proj
    //   for (long j = thread_idx; j < n; j += blk_dim){
    //     sol[j] = fmax(sol[j], 0.0);
    //   }
    // }
    else if (proj_type[blk_idx] == 5 || proj_type[blk_idx] == 7 || proj_type[blk_idx] == 20 || proj_type[blk_idx] == 21){
      soc_proj(sol, &n, &thread_idx, &blk_dim);
    }
    else if (proj_type[blk_idx] == 6 || proj_type[blk_idx] == 22){
      soc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[blk_idx], &blk_idx, &thread_idx, &blk_dim, abs_tol, rel_tol);     
    }
    else if (proj_type[blk_idx] == 8 || proj_type[blk_idx] == 10 || proj_type[blk_idx] == 23 || proj_type[blk_idx] == 24){
      rsoc_proj(sol, &n, sub_D_scaled_mul_x, sub_temp, &thread_idx, &blk_dim);
    }
    else if (proj_type[blk_idx] == 9 || proj_type[blk_idx] == 25){
      rsoc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[blk_idx], &thread_idx, &blk_dim, abs_tol, rel_tol);
    }
    else if (proj_type[blk_idx] == 11 || proj_type[blk_idx] == 16 || proj_type[blk_idx] == 28){
      // dualExponent_proj
      if (thread_idx == 0){
        dualExponent_proj(sol, &t_warm_start[blk_idx], abs_tol, rel_tol);
      }
    }
    else if (proj_type[blk_idx] == 14 || proj_type[blk_idx] == 13 || proj_type[blk_idx] == 26 ){
      // exponent_proj
      if (thread_idx == 0){
        exponent_proj(sol, &t_warm_start[blk_idx], abs_tol, rel_tol);
      }
    }
    else if (proj_type[blk_idx] == 12 || proj_type[blk_idx] == 29){
      // dualExponent_proj_diagonal
      if (thread_idx == 0){
        // printf("cuda dualExponent_proj_diagonal sol: %f, %f, %f, sub_D_scaled: %f, %f, %f, sub_temp: %f, %f, %f\n", sol[0], sol[1], sol[2], sub_D_scaled[0], sub_D_scaled[1], sub_D_scaled[2], sub_temp[0], sub_temp[1], sub_temp[2]);
        dualExponent_proj_diagonal(sol, sub_D_scaled, sub_temp, &t_warm_start[blk_idx], abs_tol, rel_tol);
      }
    }
    else if (proj_type[blk_idx] == 15 || proj_type[blk_idx] == 27){
      // exponent_proj_diagonal
      if (thread_idx == 0){
        double sub_D_scaled_inv[3];
        sub_D_scaled_inv[0] = 1.0 / sub_D_scaled[0];
        sub_D_scaled_inv[1] = 1.0 / sub_D_scaled[1];
        sub_D_scaled_inv[2] = 1.0 / sub_D_scaled[2];
        // printf("cuda exponent_proj_diagonal sol: %f, %f, %f, sub_D_scaled_inv: %f, %f, %f\n", sol[0], sol[1], sol[2], sub_D_scaled_inv[0], sub_D_scaled_inv[1], sub_D_scaled_inv[2]);
        exponent_proj_diagonal(sol, sub_D_scaled_inv, &t_warm_start[blk_idx], abs_tol, rel_tol);
      }
    }
  }
}