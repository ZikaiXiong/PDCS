#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
// #define proj_abs_tol_squared 1e-32
#define MAX_ITER 100000
#ifndef PDCS_ENABLE_SAFEGUARDED_NEWTON
#define PDCS_ENABLE_SAFEGUARDED_NEWTON 1
#endif
#ifndef PDCS_ENABLE_COLD_SOC_NEWTON
// Experimental: initialize safeguarded Newton at the midpoint of the known
// positive-SOC bracket when no usable warm start is available.
#define PDCS_ENABLE_COLD_SOC_NEWTON 0
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_ORACLE
// Evaluate the SOC root value (and, for Newton, its derivative) while the
// vector is resident in registers.  The derivative path reduces a pair of
// accumulators once instead of materializing and reducing the vector twice.
#define PDCS_ENABLE_FUSED_SOC_ORACLE 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
// Compute ||x ./ D|| and ||D .* x|| in one vector traversal and one paired
// block reduction.  The D .* x vector is retained for the root oracle.
#define PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS 1
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_ROOT
#define PDCS_ENABLE_BOUNDED_SOC_ROOT 1
#endif
#ifndef PDCS_SOC_BOUNDED_NEWTON_STEPS
#define PDCS_SOC_BOUNDED_NEWTON_STEPS 8
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
// On the block-wise path the shorter Illinois search increases register and
// scalar-control cost more than it saves at the measured cone sizes.
#define PDCS_ENABLE_BOUNDED_SOC_ILLINOIS 0
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_HALLEY
#define PDCS_ENABLE_BOUNDED_SOC_HALLEY 0
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
#define PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT 1
#endif
#ifndef PDCS_SOC_LOGIT_STEPS
#define PDCS_SOC_LOGIT_STEPS 48
#endif
#ifndef PDCS_ENABLE_SHUFFLE_BLOCK_REDUCTION
#define PDCS_ENABLE_SHUFFLE_BLOCK_REDUCTION 0
#endif
#ifndef PDCS_SOC_COORDINATE_MODE
// 0: original xi coordinate; 1: shifted-log; 2: smooth-logit on (0, 0.5)
// and shifted-log on (0.5, infinity); 3: use mode 2 only while a bracket
// spans a large multiplicative range, then automatically return to xi.
#define PDCS_SOC_COORDINATE_MODE 0
#endif
#ifndef PDCS_SOC_LOG_RATIO_THRESHOLD
#define PDCS_SOC_LOG_RATIO_THRESHOLD 64.0
#endif
#ifndef PDCS_ENABLE_EXPONENT_EXPANSION
#define PDCS_ENABLE_EXPONENT_EXPANSION 0
#endif
#ifndef PDCS_ENABLE_PERTURBATION_BRACKET
#define PDCS_ENABLE_PERTURBATION_BRACKET 0
#endif
#ifndef PDCS_PERTURBATION_TRIGGER_RATIO
#define PDCS_PERTURBATION_TRIGGER_RATIO 4096.0
#endif
#ifndef PDCS_SOC_NEWTON_STEPS
#define PDCS_SOC_NEWTON_STEPS 2
#endif
#include "bounded_soc_root_step.cuh"
#include "bounded_soc_logit_root.cuh"
// suitable for very much block number

#include "exp_proj.cu"

#ifndef PDCS_BLOCK_REDUCTION_CAPACITY
// Production historically reserves room for the maximum CUDA block size.
// Diagnostic launches are fixed at 256 threads and may override this with
// 256 to avoid combining 48 KiB of static shared memory with counter writes.
// The active reduction order is unchanged because every loop is bounded by
// blk_dim rather than this storage capacity.
#define PDCS_BLOCK_REDUCTION_CAPACITY 1024
#endif

// n is the length of the vector, including the first element
// len is the length of the vector, not including the first element or the top two elements

// Reduce through registers inside each warp and use shared memory only for
// the warp partials.  The previous tree reduction reserved 1024 doubles per
// call site and executed one block barrier per tree level even though the
// production launch uses 256 threads.
__device__ double block_reduce_sum(
    double value, long thread_idx, long blk_dim) {
#if PDCS_ENABLE_SHUFFLE_BLOCK_REDUCTION
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  __shared__ double warp_sums[32];
  int lane = thread_idx & (warpSize - 1);
  int warp = thread_idx / warpSize;
  int warp_count = (blk_dim + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  double block_sum = (warp == 0 && lane < warp_count) ? warp_sums[lane] : 0.0;
  if (warp == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      block_sum += __shfl_down_sync(mask, block_sum, offset);
    }
    if (lane == 0) warp_sums[0] = block_sum;
  }
  __syncthreads();
  return warp_sums[0];
#else
  __shared__ double partial[PDCS_BLOCK_REDUCTION_CAPACITY];
  partial[thread_idx] = value;
  __syncthreads();
  for (int stride = blk_dim / 2; stride > 0; stride /= 2) {
    if (thread_idx < stride) partial[thread_idx] += partial[thread_idx + stride];
    __syncthreads();
  }
  return partial[0];
#endif
}

__device__ void block_reduce_pair(
    double value, double derivative, long thread_idx, long blk_dim,
    double *value_sum, double *derivative_sum) {
#if PDCS_ENABLE_SHUFFLE_BLOCK_REDUCTION
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
    derivative += __shfl_down_sync(mask, derivative, offset);
  }
  __shared__ double warp_values[32];
  __shared__ double warp_derivatives[32];
  int lane = thread_idx & (warpSize - 1);
  int warp = thread_idx / warpSize;
  int warp_count = (blk_dim + warpSize - 1) / warpSize;
  if (lane == 0) {
    warp_values[warp] = value;
    warp_derivatives[warp] = derivative;
  }
  __syncthreads();
  double block_value =
      (warp == 0 && lane < warp_count) ? warp_values[lane] : 0.0;
  double block_derivative =
      (warp == 0 && lane < warp_count) ? warp_derivatives[lane] : 0.0;
  if (warp == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      block_value += __shfl_down_sync(mask, block_value, offset);
      block_derivative +=
          __shfl_down_sync(mask, block_derivative, offset);
    }
    if (lane == 0) {
      warp_values[0] = block_value;
      warp_derivatives[0] = block_derivative;
    }
  }
  __syncthreads();
  *value_sum = warp_values[0];
  *derivative_sum = warp_derivatives[0];
#else
  __shared__ double partial_values[PDCS_BLOCK_REDUCTION_CAPACITY];
  __shared__ double partial_derivatives[PDCS_BLOCK_REDUCTION_CAPACITY];
  partial_values[thread_idx] = value;
  partial_derivatives[thread_idx] = derivative;
  __syncthreads();
  for (int stride = blk_dim / 2; stride > 0; stride /= 2) {
    if (thread_idx < stride) {
      partial_values[thread_idx] += partial_values[thread_idx + stride];
      partial_derivatives[thread_idx] += partial_derivatives[thread_idx + stride];
    }
    __syncthreads();
  }
  *value_sum = partial_values[0];
  *derivative_sum = partial_derivatives[0];
#endif
}

#if PDCS_ENABLE_BOUNDED_SOC_HALLEY || PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
__device__ void block_reduce_triple(
    double value, double derivative, double second,
    long thread_idx, long blk_dim, double *value_sum,
    double *derivative_sum, double *second_sum) {
#if PDCS_ENABLE_SHUFFLE_BLOCK_REDUCTION
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
    derivative += __shfl_down_sync(mask, derivative, offset);
    second += __shfl_down_sync(mask, second, offset);
  }
  __shared__ double warp_values[32];
  __shared__ double warp_derivatives[32];
  __shared__ double warp_seconds[32];
  int lane = thread_idx & (warpSize - 1);
  int warp = thread_idx / warpSize;
  int warp_count = (blk_dim + warpSize - 1) / warpSize;
  if (lane == 0) {
    warp_values[warp] = value;
    warp_derivatives[warp] = derivative;
    warp_seconds[warp] = second;
  }
  __syncthreads();
  double block_value =
      (warp == 0 && lane < warp_count) ? warp_values[lane] : 0.0;
  double block_derivative =
      (warp == 0 && lane < warp_count) ? warp_derivatives[lane] : 0.0;
  double block_second =
      (warp == 0 && lane < warp_count) ? warp_seconds[lane] : 0.0;
  if (warp == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      block_value += __shfl_down_sync(mask, block_value, offset);
      block_derivative += __shfl_down_sync(mask, block_derivative, offset);
      block_second += __shfl_down_sync(mask, block_second, offset);
    }
    if (lane == 0) {
      warp_values[0] = block_value;
      warp_derivatives[0] = block_derivative;
      warp_seconds[0] = block_second;
    }
  }
  __syncthreads();
  *value_sum = warp_values[0];
  *derivative_sum = warp_derivatives[0];
  *second_sum = warp_seconds[0];
#else
  __shared__ double partial_values[PDCS_BLOCK_REDUCTION_CAPACITY];
  __shared__ double partial_derivatives[PDCS_BLOCK_REDUCTION_CAPACITY];
  __shared__ double partial_seconds[PDCS_BLOCK_REDUCTION_CAPACITY];
  partial_values[thread_idx] = value;
  partial_derivatives[thread_idx] = derivative;
  partial_seconds[thread_idx] = second;
  __syncthreads();
  for (int stride = blk_dim / 2; stride > 0; stride /= 2) {
    if (thread_idx < stride) {
      partial_values[thread_idx] += partial_values[thread_idx + stride];
      partial_derivatives[thread_idx] += partial_derivatives[thread_idx + stride];
      partial_seconds[thread_idx] += partial_seconds[thread_idx + stride];
    }
    __syncthreads();
  }
  *value_sum = partial_values[0];
  *derivative_sum = partial_derivatives[0];
  *second_sum = partial_seconds[0];
#endif
}
#endif

// BLAS functions
__device__ double nrm2(const long* __restrict__ n, const double* __restrict__ x, long *thread_idx, long *blk_dim)
{
  PDCS_PROFILE_VV_REDUCTION();
  double local_norm = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    double val = x[j];
    local_norm += val * val;
  }
  return sqrt(block_reduce_sum(local_norm, *thread_idx, *blk_dim));
}

__device__ double nrm2_squared(const long* __restrict__ n, const double* __restrict__ x, long* __restrict__ thread_idx, long* __restrict__ blk_dim)
{
  PDCS_PROFILE_VV_REDUCTION();
  double local_norm = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    double val = x[j];
    local_norm += val * val;
  }
  return block_reduce_sum(local_norm, *thread_idx, *blk_dim);
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
  PDCS_PROFILE_VV_REDUCTION();
  double local_diff = 0.0;
  #pragma unroll
  for (long j = *thread_idx; j < *n; j += *blk_dim)
  {
    double diff = x[j] - y[j];
    local_diff += diff * diff;
  }
  return sqrt(block_reduce_sum(local_diff, *thread_idx, *blk_dim));
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
  PDCS_PROFILE_ORACLE();
  // len not including the first element
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  PDCS_PROFILE_VV_REDUCTION();
  double local_value = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    double y = D_scaled_part_mul_x_part[j] /
               (1.0 + (2.0 * xi[0]) * D_scaled_squared_part[j]);
    local_value += y * y;
  }
  double value_sum = block_reduce_sum(local_value, *thread_idx, *blk_dim);
  return sqrt(value_sum) - (x[0] / (1 - 2 * xi[0]));
#else
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  // __syncthreads();
  return nrm2(len, temp_part, thread_idx, blk_dim) - (x[0] / (1 - 2 * xi[0]));
#endif
}

__device__ void oracle_soc_h(double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  // len not including the first element
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  PDCS_PROFILE_VV_REDUCTION();
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + (2.0 * xi[0]) * q;
    double y = D_scaled_part_mul_x_part[j] / denominator;
    double y_squared = y * y;
    local_value += y_squared;
    local_derivative += y_squared * q / denominator;
  }
  double value_sum;
  double derivative_sum;
  block_reduce_pair(local_value, local_derivative, *thread_idx, *blk_dim,
                    &value_sum, &derivative_sum);
  double denominator = 1.0 - 2.0 * xi[0];
  double right = (x[0] / denominator) * (x[0] / denominator);
  *f = value_sum - right;
  *h = -4.0 * (derivative_sum + right / denominator);
#else
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  double left = nrm2_squared(len, temp_part, thread_idx, blk_dim);
  double right = (x[0] / (1 - 2 * xi[0]));
  right = right * right;
  *f = left - right;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + 2.0 * xi[0] * q;
    temp_part[j] *= sqrt(fmax(q / denominator, 0.0));
  }
  right = right / (1 - 2 * xi[0]);
  *h = -4 * (nrm2_squared(len, temp_part, thread_idx, blk_dim) + right);
#endif
}

__device__ double oracle_soc_bounded_u_f(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_VV_REDUCTION();
  double local_value = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    const double a = D_scaled_part_mul_x_part[j];
    const double c = D_scaled_squared_part[j];
    const double ratio = increasing ? u / (c + 1.0 - u)
                                    : (1.0 - u) / (1.0 + c * u);
    const double value = a * ratio;
    local_value += value * value;
  }
  const double value_sum = block_reduce_sum(
      local_value, *thread_idx, *blk_dim);
  return value_sum - t * t;
}

__device__ void oracle_soc_bounded_u_h(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len, double *f, double *h,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    const double a = D_scaled_part_mul_x_part[j];
    const double a_squared = a * a;
    const double c = D_scaled_squared_part[j];
    if (increasing) {
      const double denominator = c + 1.0 - u;
      const double denominator_squared = denominator * denominator;
      local_value += a_squared * u * u / denominator_squared;
      local_derivative += 2.0 * a_squared * u * (1.0 + c) /
                          (denominator_squared * denominator);
    } else {
      const double one_minus_u = 1.0 - u;
      const double denominator = 1.0 + c * u;
      const double denominator_squared = denominator * denominator;
      local_value += a_squared * one_minus_u * one_minus_u /
                     denominator_squared;
      local_derivative -= 2.0 * a_squared * one_minus_u * (1.0 + c) /
                          (denominator_squared * denominator);
    }
  }
  double value_sum;
  double derivative_sum;
  block_reduce_pair(local_value, local_derivative, *thread_idx, *blk_dim,
                    &value_sum, &derivative_sum);
  *f = value_sum - t * t;
  *h = derivative_sum;
}

#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
__device__ void oracle_soc_bounded_u_h2(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double *f, double *h, double *h2,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double local_value = 0.0;
  double local_derivative = 0.0;
  double local_second = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    const double a = D_scaled_part_mul_x_part[j];
    const double a_squared = a * a;
    const double c = D_scaled_squared_part[j];
    const double q = 1.0 + c;
    if (increasing) {
      const double denominator = q - u;
      const double denominator_squared = denominator * denominator;
      const double denominator_fourth = denominator_squared * denominator_squared;
      local_value += a_squared * u * u / denominator_squared;
      local_derivative += 2.0 * a_squared * u * q /
                          (denominator_squared * denominator);
      local_second += 2.0 * a_squared * q * (q + 2.0 * u) /
                      denominator_fourth;
    } else {
      const double one_minus_u = 1.0 - u;
      const double denominator = 1.0 + c * u;
      const double denominator_squared = denominator * denominator;
      const double denominator_fourth = denominator_squared * denominator_squared;
      local_value += a_squared * one_minus_u * one_minus_u /
                     denominator_squared;
      local_derivative -= 2.0 * a_squared * one_minus_u * q /
                          (denominator_squared * denominator);
      local_second += 2.0 * a_squared * q *
                      (1.0 + 3.0 * c - 2.0 * c * u) /
                      denominator_fourth;
    }
  }
  double value_sum;
  double derivative_sum;
  double second_sum;
  block_reduce_triple(local_value, local_derivative, local_second,
                      *thread_idx, *blk_dim, &value_sum,
                      &derivative_sum, &second_sum);
  *f = value_sum - t * t;
  *h = derivative_sum;
  *h2 = second_sum;
}
#endif

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
__device__ double oracle_soc_logit_z_f(
    double z, double t, bool negative_branch,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_VV_REDUCTION();
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double local_value = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    local_value += pdcs_soc_logit_term_f(
        D_scaled_part_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch);
  }
  return block_reduce_sum(local_value, *thread_idx, *blk_dim) - t * t;
}

__device__ void oracle_soc_logit_z_h2(
    double z, double t, bool negative_branch,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double *f, double *h, double *h2,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double local_value = 0.0;
  double local_derivative = 0.0;
  double local_second = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    double value;
    double derivative;
    double second;
    pdcs_soc_logit_term(
        D_scaled_part_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch, &value, &derivative, &second);
    local_value += value;
    local_derivative += derivative;
    local_second += second;
  }
  double value_sum;
  double derivative_sum;
  double second_sum;
  block_reduce_triple(local_value, local_derivative, local_second,
                      *thread_idx, *blk_dim, &value_sum,
                      &derivative_sum, &second_sum);
  *f = value_sum - t * t;
  *h = derivative_sum;
  *h2 = second_sum;
}

__device__ void oracle_soc_logit_z_h(
    double z, double t, bool negative_branch,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double *f, double *h,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    double value;
    double derivative;
    pdcs_soc_logit_term_h(
        D_scaled_part_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch, &value, &derivative);
    local_value += value;
    local_derivative += derivative;
  }
  double value_sum;
  double derivative_sum;
  block_reduce_pair(local_value, local_derivative, *thread_idx, *blk_dim,
                    &value_sum, &derivative_sum);
  *f = value_sum - t * t;
  *h = derivative_sum;
}

__device__ double soc_logit_z_solve(
    double t, bool negative_branch, double warm_z,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim,
    double endpoint_norm, double abs_tol, double rel_tol) {
  double left = -700.0;
  double right = 700.0;
  const double t_squared = t * t;
  double left_f = -t_squared;
  double right_f = endpoint_norm * endpoint_norm - t_squared;
  const bool valid_warm =
      isfinite(warm_z) && warm_z > left && warm_z < right &&
      warm_z != PDCS_SOC_WARM_START_SENTINEL;
  const bool use_halley =
      !PDCS_ENABLE_ADAPTIVE_LOGIT_NEWTON || !valid_warm;
  double z = valid_warm ? warm_z : 0.0;
  if (valid_warm) PDCS_PROFILE_WARM_ATTEMPT();

  double f;
  double h;
  double h2;
  if (use_halley) {
    oracle_soc_logit_z_h2(
        z, t, negative_branch, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &f, &h, &h2, thread_idx, blk_dim);
  } else {
    oracle_soc_logit_z_h(
        z, t, negative_branch, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &f, &h, thread_idx, blk_dim);
    h2 = 0.0;
  }
  if (valid_warm && pdcs_soc_logit_converged(
          f, h, z, left, right, t, abs_tol, rel_tol)) {
    PDCS_PROFILE_WARM_ACCEPT();
    return z;
  }
  if (f > 0.0) {
    right = z;
    right_f = f;
  } else {
    left = z;
    left_f = f;
  }

  for (int iter = 0;
       iter < PDCS_SOC_LOGIT_STEPS && PDCS_ENABLE_SAFEGUARDED_NEWTON;
       ++iter) {
    if (pdcs_soc_logit_converged(
            f, h, z, left, right, t, abs_tol, rel_tol)) return z;
    double candidate;
    if (!pdcs_soc_logit_candidate(
            z, f, h, h2, left, right, left_f, right_f, use_halley,
            &candidate)) break;
    PDCS_PROFILE_NEWTON_ATTEMPT();
    double candidate_f;
    double candidate_h;
    double candidate_h2;
    if (use_halley) {
      oracle_soc_logit_z_h2(
          candidate, t, negative_branch, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len, &candidate_f, &candidate_h,
          &candidate_h2, thread_idx, blk_dim);
    } else {
      oracle_soc_logit_z_h(
          candidate, t, negative_branch, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len, &candidate_f, &candidate_h,
          thread_idx, blk_dim);
      candidate_h2 = 0.0;
    }
    if (candidate_f > 0.0) {
      right = candidate;
      right_f = candidate_f;
    } else {
      left = candidate;
      left_f = candidate_f;
    }
    if (!isfinite(candidate_f)) break;
    PDCS_PROFILE_NEWTON_ACCEPT();
    z = candidate;
    f = candidate_f;
    h = candidate_h;
    h2 = candidate_h2;
  }
  if (pdcs_soc_logit_converged(
          f, h, z, left, right, t, abs_tol, rel_tol)) return z;

  int count = 0;
#if PDCS_ENABLE_SOC_LOGIT_ILLINOIS
  int last_updated_side = 0;
#endif
  while (right - left > rel_tol && count <= MAX_ITER) {
    PDCS_PROFILE_BISECTION();
    ++count;
#if PDCS_ENABLE_SOC_LOGIT_ILLINOIS
    const double denominator = right_f - left_f;
    z = left - left_f * (right - left) / denominator;
    const double guard = fmax(64.0 * 2.220446049250313e-16,
                              1e-6 * (right - left));
    if (!isfinite(z) || !isfinite(denominator) ||
        fabs(denominator) <= 1e-300 || z <= left + guard ||
        z >= right - guard) {
      z = 0.5 * (left + right);
      last_updated_side = 0;
    }
#else
    z = 0.5 * (left + right);
#endif
    f = oracle_soc_logit_z_f(
        z, t, negative_branch, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, thread_idx, blk_dim);
    if (!isfinite(f) || f == 0.0) break;
    if (f > 0.0) {
      right = z;
      right_f = f;
#if PDCS_ENABLE_SOC_LOGIT_ILLINOIS
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
#endif
    } else {
      left = z;
      left_f = f;
#if PDCS_ENABLE_SOC_LOGIT_ILLINOIS
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
#endif
    }
  }
  if (count > MAX_ITER) PDCS_PROFILE_MAX_ITER();
  if (f != 0.0) z = 0.5 * (left + right);
  PDCS_PROFILE_RESIDUAL(f);
  PDCS_PROFILE_BRACKET(left, right);
  return fmin(fmax(z, -700.0), 700.0);
}

__device__ void soc_logit_z_recover(
    double *sol, long *n, double z, bool negative_branch,
    double minVal, double *t_warm_start, double *D_scaled_squared,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  if (*thread_idx == 0) {
    t_warm_start[0] = z;
    sol[0] = negative_branch ? -sol[0] * v / s * minVal
                             : sol[0] / s * minVal;
  }
  for (long j = 1 + *thread_idx; j < *n; j += *blk_dim) {
    const double c = D_scaled_squared[j];
    sol[j] = negative_branch ? sol[j] * v / (c + v) * minVal
                             : sol[j] / (1.0 + c * v) * minVal;
  }
  __syncthreads();
}
#endif

__device__
double soc_bounded_u_solve(
    double t, bool increasing, double warm_u,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim,
    double endpoint_norm, double abs_tol, double rel_tol) {
  double left = 0.0;
  double right = 1.0;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS || PDCS_ENABLE_BOUNDED_SOC_HALLEY
  const double t_squared = t * t;
  const double endpoint_f = endpoint_norm * endpoint_norm - t_squared;
  double left_f = increasing ? -t_squared : endpoint_f;
  double right_f = increasing ? endpoint_f : -t_squared;
#endif
  const bool valid_warm = isfinite(warm_u) && warm_u > 0.0 && warm_u < 1.0;
  double u = valid_warm ? warm_u : 0.5;
  if (valid_warm) PDCS_PROFILE_WARM_ATTEMPT();

  double f;
  double h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
  double h2;
  oracle_soc_bounded_u_h2(
      u, t, increasing, D_scaled_part_mul_x_part,
      D_scaled_squared_part, len, &f, &h, &h2, thread_idx, blk_dim);
#else
  oracle_soc_bounded_u_h(
      u, t, increasing, D_scaled_part_mul_x_part,
      D_scaled_squared_part, len, &f, &h, thread_idx, blk_dim);
#endif
  if (valid_warm && pdcs_bounded_soc_projection_converged(
          f, h, u, t, abs_tol, rel_tol)) {
    PDCS_PROFILE_WARM_ACCEPT();
    return u;
  }
  if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
    right = u;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS || PDCS_ENABLE_BOUNDED_SOC_HALLEY
    right_f = f;
#endif
  } else {
    left = u;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS || PDCS_ENABLE_BOUNDED_SOC_HALLEY
    left_f = f;
#endif
  }

  for (int iter = 0;
       iter < PDCS_SOC_BOUNDED_NEWTON_STEPS &&
       PDCS_ENABLE_SAFEGUARDED_NEWTON;
       ++iter) {
    if (pdcs_bounded_soc_projection_converged(
            f, h, u, t, abs_tol, rel_tol) ||
        pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) return u;
    double candidate;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    if (!pdcs_bounded_soc_candidate(
            u, f, h, h2, left, right, left_f, right_f, &candidate)) break;
#else
    if (!pdcs_bounded_soc_newton_candidate(
            u, f, h, left, right, &candidate)) break;
#endif

    PDCS_PROFILE_NEWTON_ATTEMPT();
    double candidate_f;
    double candidate_h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    double candidate_h2;
    oracle_soc_bounded_u_h2(
        candidate, t, increasing, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &candidate_f, &candidate_h,
        &candidate_h2, thread_idx, blk_dim);
#else
    oracle_soc_bounded_u_h(
        candidate, t, increasing, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &candidate_f, &candidate_h,
        thread_idx, blk_dim);
#endif
    if ((increasing && candidate_f > 0.0) ||
        (!increasing && candidate_f < 0.0)) {
      right = candidate;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS || PDCS_ENABLE_BOUNDED_SOC_HALLEY
      right_f = candidate_f;
#endif
    } else {
      left = candidate;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS || PDCS_ENABLE_BOUNDED_SOC_HALLEY
      left_f = candidate_f;
#endif
    }
    if (!isfinite(candidate_f)) break;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    if (pdcs_bounded_soc_projection_converged(
            candidate_f, candidate_h, candidate, t,
            abs_tol, rel_tol)) {
      PDCS_PROFILE_NEWTON_ACCEPT();
      return candidate;
    }
    // The candidate is bracket-safe; do not confuse a small/stagnating
    // Newton step with convergence near a transformed-coordinate endpoint.
#else
    if (fabs(candidate_f) >= fabs(f)) break;
#endif
    PDCS_PROFILE_NEWTON_ACCEPT();
    u = candidate;
    f = candidate_f;
    h = candidate_h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    h2 = candidate_h2;
#endif
    if (pdcs_bounded_soc_projection_converged(
            f, h, u, t, abs_tol, rel_tol) ||
        pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) return u;
  }

  int count = 0;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
  int last_updated_side = 0;
#endif
  while (!pdcs_bounded_soc_bracket_converged(
             left, right, rel_tol) && count <= MAX_ITER) {
    PDCS_PROFILE_BISECTION();
    ++count;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
    const double denominator = right_f - left_f;
    u = left - left_f * (right - left) / denominator;
    const double guard = fmax(
        pdcs_bounded_soc_coordinate_resolution(0.5 * (left + right)),
        1e-6 * (right - left));
    if (!isfinite(u) || !isfinite(denominator) ||
        fabs(denominator) <= 1e-300 || u <= left + guard ||
        u >= right - guard) {
      u = pdcs_bounded_soc_bisection_midpoint(left, right);
      last_updated_side = 0;
    }
#else
    u = pdcs_bounded_soc_bisection_midpoint(left, right);
#endif
    if (pdcs_bounded_soc_near_endpoint(u, rel_tol)) {
      oracle_soc_bounded_u_h(
          u, t, increasing, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len, &f, &h, thread_idx, blk_dim);
    } else {
      f = oracle_soc_bounded_u_f(
          u, t, increasing, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len, thread_idx, blk_dim);
    }
    if (!isfinite(f) || pdcs_bounded_soc_projection_converged(
            f, h, u, t, abs_tol, rel_tol)) break;
    const bool update_right =
        (increasing && f > 0.0) || (!increasing && f < 0.0);
    if (update_right) {
      right = u;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
      right_f = f;
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
#endif
    } else {
      left = u;
#if PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
      left_f = f;
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
#endif
    }
  }
  if (count > MAX_ITER) PDCS_PROFILE_MAX_ITER();
  if (!pdcs_bounded_soc_projection_converged(
          f, h, u, t, abs_tol, rel_tol)) {
    u = pdcs_bounded_soc_bisection_midpoint(left, right);
  }
  PDCS_PROFILE_RESIDUAL(f);
  PDCS_PROFILE_BRACKET(left, right);
  return fmin(fmax(u, 64.0 * 2.220446049250313e-16),
              1.0 - 64.0 * 2.220446049250313e-16);
}

__device__ void soc_bounded_u_recover(
    double *sol, long *n, double u, bool increasing,
    double minVal, double *t_warm_start, double *D_scaled_squared,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  if (*thread_idx == 0) {
    t_warm_start[0] = u;
    sol[0] = increasing ? -sol[0] * (1.0 - u) / u * minVal
                        : sol[0] / (1.0 - u) * minVal;
  }
  for (long j = 1 + *thread_idx; j < *n; j += *blk_dim) {
    const double c = D_scaled_squared[j];
    sol[j] = increasing ? sol[j] * (1.0 - u) / (c + 1.0 - u) * minVal
                        : sol[j] / (1.0 + c * u) * minVal;
  }
  __syncthreads();
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

// Coordinate maps used by the safeguarded root search.  The small positive
// shift makes the maps finite at xi=0 and xi=0.5.  Mode 2 resolves both ends
// of the bounded positive-t branch with a smooth logit, while the unbounded
// negative-t branch uses log(xi-0.5+s).
__device__ double soc_coordinate_shift(double rel_tol) {
  return fmax(rel_tol, 32.0 * 2.220446049250313e-16);
}

__device__ double soc_root_to_coordinate(
    double xi, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return xi;
#elif PDCS_SOC_COORDINATE_MODE == 1
  double anchor = increasing ? 0.5 : 0.0;
  return log(fmax(xi - anchor + shift, shift));
#else
  if (increasing) {
    return log(fmax(xi - 0.5 + shift, shift));
  }
  double numerator = fmax(xi + shift, shift);
  double denominator = fmax(0.5 - xi + shift, shift);
  return log(numerator / denominator);
#endif
}

__device__ double soc_coordinate_to_root(
    double u, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return u;
#elif PDCS_SOC_COORDINATE_MODE == 1
  double anchor = increasing ? 0.5 : 0.0;
  return anchor + exp(u) - shift;
#else
  if (increasing) {
    return 0.5 + exp(u) - shift;
  }
  double sigmoid;
  if (u >= 0.0) {
    sigmoid = 1.0 / (1.0 + exp(-u));
  }
  else {
    double exp_u = exp(u);
    sigmoid = exp_u / (1.0 + exp_u);
  }
  return sigmoid * (0.5 + 2.0 * shift) - shift;
#endif
}

__device__ double soc_root_coordinate_derivative(
    double xi, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return 1.0;
#elif PDCS_SOC_COORDINATE_MODE == 1
  double anchor = increasing ? 0.5 : 0.0;
  return fmax(xi - anchor + shift, shift);
#else
  if (increasing) {
    return fmax(xi - 0.5 + shift, shift);
  }
  return fmax((xi + shift) * (0.5 - xi + shift) /
              (0.5 + 2.0 * shift), shift);
#endif
}

__device__ bool soc_bracket_needs_log_coordinate(
    double xi_left, double xi_right, bool increasing, double shift) {
  if (increasing) {
    double left_q = fmax(xi_left - 0.5 + shift, shift);
    double right_q = fmax(xi_right - 0.5 + shift, left_q);
    return isfinite(right_q) &&
           right_q / left_q >= PDCS_SOC_LOG_RATIO_THRESHOLD;
  }
  double zero_ratio = (xi_right + shift) /
                      fmax(xi_left + shift, shift);
  double half_ratio = (0.5 - xi_left + shift) /
                      fmax(0.5 - xi_right + shift, shift);
  return (isfinite(zero_ratio) &&
          zero_ratio >= PDCS_SOC_LOG_RATIO_THRESHOLD) ||
         (isfinite(half_ratio) &&
          half_ratio >= PDCS_SOC_LOG_RATIO_THRESHOLD);
}

__device__ bool soc_newton_needs_log_coordinate(
    double xi, double xi_left, double xi_right, bool increasing,
    double shift) {
  if (increasing) {
    double left_q = fmax(xi_left - 0.5 + shift, shift);
    double x_q = fmax(xi - 0.5 + shift, shift);
    double right_q = fmax(xi_right - 0.5 + shift, x_q);
    return right_q / x_q >= PDCS_SOC_LOG_RATIO_THRESHOLD ||
           x_q / left_q >= PDCS_SOC_LOG_RATIO_THRESHOLD;
  }
  double zero_ratio = (xi_right + shift) / fmax(xi + shift, shift);
  double half_ratio = (0.5 - xi_left + shift) /
                      fmax(0.5 - xi + shift, shift);
  return zero_ratio >= PDCS_SOC_LOG_RATIO_THRESHOLD ||
         half_ratio >= PDCS_SOC_LOG_RATIO_THRESHOLD;
}

// Locate the scale of the root displacement from the warm start by binary
// searching the exponent in epsilon*2^k.  This takes log-log queries in the
// ratio between the available branch width and the tolerance-sized epsilon.
// It is optional until the workload A/B establishes that the saved Newton or
// bisection reductions outweigh these additional scalar-oracle evaluations.
__device__ void soc_perturbation_exponent_bracket(
    double *sol, double *D_scaled_mul_x_part,
    double *D_scaled_squared_part, double *temp_part, long *len,
    double warm_x, double warm_f, double *xiLeft, double *xiRight,
    bool increasing, long* __restrict__ thread_idx,
    long* __restrict__ blk_dim, double rel_tol) {
#if PDCS_ENABLE_PERTURBATION_BRACKET
  bool root_right = increasing ? warm_f < 0.0 : warm_f > 0.0;
  double boundary_distance = root_right ? *xiRight - warm_x
                                        : warm_x - *xiLeft;
  double epsilon = fmax(rel_tol * (1.0 + fabs(warm_x)),
                        16.0 * 2.220446049250313e-16 * (1.0 + fabs(warm_x)));
  double boundary_guard = fmax(epsilon, 1e-14 * (1.0 + fabs(warm_x)));
  double usable_distance = boundary_distance - boundary_guard;
  if (!(usable_distance > epsilon) || !isfinite(usable_distance)) return;

  int max_exponent = 0;
  double radius = epsilon;
  while (radius < usable_distance && max_exponent < 60) {
    radius *= 2.0;
    ++max_exponent;
  }
  radius = fmin(radius, usable_distance);
  double candidate = root_right ? warm_x + radius : warm_x - radius;
  PDCS_PROFILE_EXPANSION();
  double candidate_f = oracle_soc_f_sqrt(
      &candidate, sol, D_scaled_mul_x_part, D_scaled_squared_part,
      temp_part, len, thread_idx, blk_dim);
  bool sign_changed = (warm_f < 0.0 && candidate_f >= 0.0) ||
                      (warm_f > 0.0 && candidate_f <= 0.0);
  if (!sign_changed) return;

  int low_exponent = -1;  // displacement zero: the warm-start sign
  int high_exponent = max_exponent;
  while (high_exponent - low_exponent > 1) {
    int mid_exponent = (low_exponent + high_exponent) / 2;
    double mid_radius = ldexp(epsilon, mid_exponent);
    mid_radius = fmin(mid_radius, usable_distance);
    double mid = root_right ? warm_x + mid_radius : warm_x - mid_radius;
    PDCS_PROFILE_EXPANSION();
    double mid_f = oracle_soc_f_sqrt(
        &mid, sol, D_scaled_mul_x_part, D_scaled_squared_part,
        temp_part, len, thread_idx, blk_dim);
    bool mid_changed = (warm_f < 0.0 && mid_f >= 0.0) ||
                       (warm_f > 0.0 && mid_f <= 0.0);
    if (mid_changed) high_exponent = mid_exponent;
    else low_exponent = mid_exponent;
  }

  double low_radius = low_exponent < 0 ? 0.0 :
                      fmin(ldexp(epsilon, low_exponent), usable_distance);
  double high_radius = fmin(ldexp(epsilon, high_exponent), usable_distance);
  double same_sign_x = root_right ? warm_x + low_radius
                                  : warm_x - low_radius;
  double opposite_sign_x = root_right ? warm_x + high_radius
                                      : warm_x - high_radius;
  *xiLeft = fmin(same_sign_x, opposite_sign_x);
  *xiRight = fmax(same_sign_x, opposite_sign_x);
#endif
}

// Refine the bracket from a warm start with safeguarded Newton steps.  The
// squared oracle and its derivative are already available from the warm-start
// check.  A step is accepted only when it remains strictly inside the current
// bracket.  If Newton becomes unreliable, the caller retains a valid bracket
// and falls back to the cheaper scalar-oracle bisection below.
__device__ void soc_safeguarded_newton(
    double *sol, double *D_scaled_mul_x_part,
    double *D_scaled_squared_part, double *temp_part, long *len,
    double *oracleVal, double *oracleVal_h, double *xiLeft,
    double *xiRight, double *xi, double *warm_x, bool increasing,
    long* __restrict__ thread_idx, long* __restrict__ blk_dim,
    double abs_tol, double rel_tol) {
  const int max_newton_steps = PDCS_SOC_NEWTON_STEPS;
  double x = *warm_x;
  double f = *oracleVal;
  double h = *oracleVal_h;

#if !PDCS_ENABLE_SAFEGUARDED_NEWTON
  if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
    *xiRight = x;
  } else {
    *xiLeft = x;
  }
  *xi = x;
  *oracleVal = copysign(1.0, f);
  return;
#endif

  for (int k = 0; k < max_newton_steps; ++k) {
    // Update the bracket with the oracle value at x before proposing a step.
    if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
      *xiRight = x;
    }
    else {
      *xiLeft = x;
    }

    if (fabs(f) <= abs_tol * abs_tol) {
      *xi = x;
      *xiLeft = x;
      *xiRight = x;
      *oracleVal = f;
      *oracleVal_h = h;
      return;
    }

    double width = *xiRight - *xiLeft;
    if (!isfinite(f) || !isfinite(h) || fabs(h) <= 1e-18 || width <= 0.0) {
      break;
    }

#if PDCS_SOC_COORDINATE_MODE == 0
    double candidate = x - f / h;
    double guard = fmax(rel_tol, 1e-8 * width);
    if (!isfinite(candidate) || candidate <= *xiLeft + guard ||
        candidate >= *xiRight - guard) {
      break;
    }
#else
    double shift = soc_coordinate_shift(rel_tol);
    bool use_log_coordinate = true;
#if PDCS_SOC_COORDINATE_MODE == 3
    use_log_coordinate = soc_newton_needs_log_coordinate(
        x, *xiLeft, *xiRight, increasing, shift);
#endif
    double candidate;
    if (use_log_coordinate) {
      double u = soc_root_to_coordinate(x, increasing, shift);
      double u_left = soc_root_to_coordinate(*xiLeft, increasing, shift);
      double u_right = soc_root_to_coordinate(*xiRight, increasing, shift);
      double dx_du = soc_root_coordinate_derivative(x, increasing, shift);
      double derivative_u = h * dx_du;
      double candidate_u = u - f / derivative_u;
      candidate = soc_coordinate_to_root(candidate_u, increasing, shift);
      double u_guard = 1e-8 * (u_right - u_left);
      if (!isfinite(candidate_u) || !isfinite(candidate) ||
          !isfinite(derivative_u) || fabs(derivative_u) <= 1e-18 ||
          candidate_u <= u_left + u_guard ||
          candidate_u >= u_right - u_guard ||
          candidate <= *xiLeft || candidate >= *xiRight) {
        break;
      }
    }
    else {
      candidate = x - f / h;
      double guard = fmax(rel_tol, 1e-8 * width);
      if (!isfinite(candidate) || candidate <= *xiLeft + guard ||
          candidate >= *xiRight - guard) {
        break;
      }
    }
#endif

    double old_abs_f = fabs(f);
    double candidate_f;
    double candidate_h;
    PDCS_PROFILE_NEWTON_ATTEMPT();
    oracle_soc_h(&candidate, sol, D_scaled_mul_x_part,
                 D_scaled_squared_part, temp_part, len, &candidate_f,
                 &candidate_h, thread_idx, blk_dim);

    x = candidate;
    f = candidate_f;
    h = candidate_h;

    // Keep the useful bracket reduction from this evaluation, but do not spend
    // another two reductions if Newton is not decreasing the residual.
    if (fabs(f) >= old_abs_f) {
      if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
        *xiRight = x;
      }
      else {
        *xiLeft = x;
      }
      break;
    }
    PDCS_PROFILE_NEWTON_ACCEPT();
  }

  *xi = x;
  *warm_x = x;
  *oracleVal_h = h;
  // The binary-search oracle uses the unsquared residual.  Force its first
  // iteration unless Newton met the squared-oracle convergence test above.
  *oracleVal = copysign(1.0, f);
}

__device__ void decreasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) {
  soc_safeguarded_newton(sol, D_scaled_mul_x_part,
                         D_scaled_squared_part, temp_part, len, oracleVal,
                         oracleVal_h, xiLeft, xiRight, xi,
                         t_warm_start_val_test, false, thread_idx, blk_dim,
                         abs_tol, rel_tol);
  double normalized_width = (*xiRight - *xiLeft) /
                            (1.0 + *xiRight + *xiLeft);
  if (normalized_width > PDCS_PERTURBATION_TRIGGER_RATIO * rel_tol) {
    soc_perturbation_exponent_bracket(
        sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len,
        *xi, *oracleVal, xiLeft, xiRight, false, thread_idx, blk_dim,
        rel_tol);
  }
}

// Bisection in exactly the same coordinate used by Newton.  In shifted-log
// mode this is sqrt((L+s)(R+s))-s; in smooth-logit mode it is the midpoint of
// the transformed bracket and remains finite at both endpoints.
__device__ double soc_coordinate_bisection_midpoint(
    double xiLeft, double xiRight, bool increasing, double rel_tol) {
  double shift = soc_coordinate_shift(rel_tol);
#if PDCS_SOC_COORDINATE_MODE == 3
  if (!soc_bracket_needs_log_coordinate(
          xiLeft, xiRight, increasing, shift)) {
    return 0.5 * (xiLeft + xiRight);
  }
#endif
  double u_left = soc_root_to_coordinate(xiLeft, increasing, shift);
  double u_right = soc_root_to_coordinate(xiRight, increasing, shift);
  double midpoint = soc_coordinate_to_root(
      0.5 * (u_left + u_right), increasing, shift);
  if (!isfinite(midpoint) || midpoint <= xiLeft || midpoint >= xiRight) {
    midpoint = 0.5 * (xiLeft + xiRight);
  }
  return midpoint;
}

__device__ void soc_proj_decreasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol)
{
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_soc_rootsearch(&xiLeft, &xiRight, &xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    *xi = soc_coordinate_bisection_midpoint(
        *xiLeft, *xiRight, false, rel_tol);
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

__device__ void increasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double *t_warm_start_val_test, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol) 
{
  soc_safeguarded_newton(sol, D_scaled_mul_x_part,
                         D_scaled_squared_part, temp_part, len, oracleVal,
                         oracleVal_h, xiLeft, xiRight, xi,
                         t_warm_start_val_test, true, thread_idx, blk_dim,
                         abs_tol, rel_tol);
  double normalized_width = (*xiRight - *xiLeft) /
                            (1.0 + *xiRight + *xiLeft);
  if (normalized_width > PDCS_PERTURBATION_TRIGGER_RATIO * rel_tol) {
    soc_perturbation_exponent_bracket(
        sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len,
        *xi, *oracleVal, xiLeft, xiRight, true, thread_idx, blk_dim,
        rel_tol);
  }
}

// Establish an unbounded negative-t bracket by searching the binary exponent:
// q = xi-0.5, q_k = q_0 * 2^(2^k).  Once a sign change is found, bisection on
// the integer exponent leaves adjacent powers of two around an arbitrary
// continuous root.  Thus the dynamic-range query count is O(log log q*), not
// O(log q*) as in ordinary doubling.
__device__ bool soc_exponent_expansion_bracket(
    double *sol, double *D_scaled_mul_x_part,
    double *D_scaled_squared_part, double *temp_part, long *len,
    double *xiLeft, double *xiRight, long* __restrict__ thread_idx,
    long* __restrict__ blk_dim, double abs_tol) {
#if !PDCS_ENABLE_EXPONENT_EXPANSION
  return false;
#else
  double right_f = oracle_soc_f_sqrt(
      xiRight, sol, D_scaled_mul_x_part, D_scaled_squared_part,
      temp_part, len, thread_idx, blk_dim);
  if (!isfinite(right_f)) return false;
  if (fabs(right_f) <= abs_tol) {
    *xiLeft = *xiRight;
    return true;
  }
  if (right_f >= 0.0) return true;

  double base_q = *xiRight - 0.5;
  if (!(base_q > 0.0) || !isfinite(base_q)) return false;
  int low_exponent = 0;
  int high_exponent = 1;
  bool found = false;
  while (high_exponent <= 1020) {
    double candidate_q = ldexp(base_q, high_exponent);
    double candidate = 0.5 + candidate_q;
    if (!isfinite(candidate)) break;
    PDCS_PROFILE_EXPANSION();
    double candidate_f = oracle_soc_f_sqrt(
        &candidate, sol, D_scaled_mul_x_part, D_scaled_squared_part,
        temp_part, len, thread_idx, blk_dim);
    if (!isfinite(candidate_f)) break;
    if (fabs(candidate_f) <= abs_tol) {
      *xiLeft = candidate;
      *xiRight = candidate;
      return true;
    }
    if (candidate_f >= 0.0) {
      found = true;
      break;
    }
    low_exponent = high_exponent;
    if (high_exponent > 510) break;
    high_exponent *= 2;
  }
  if (!found) return false;

  while (high_exponent - low_exponent > 1) {
    int mid_exponent = low_exponent +
                       (high_exponent - low_exponent) / 2;
    double candidate = 0.5 + ldexp(base_q, mid_exponent);
    PDCS_PROFILE_EXPANSION();
    double candidate_f = oracle_soc_f_sqrt(
        &candidate, sol, D_scaled_mul_x_part, D_scaled_squared_part,
        temp_part, len, thread_idx, blk_dim);
    if (!isfinite(candidate_f)) return false;
    if (fabs(candidate_f) <= abs_tol) {
      *xiLeft = candidate;
      *xiRight = candidate;
      return true;
    }
    if (candidate_f >= 0.0) high_exponent = mid_exponent;
    else low_exponent = mid_exponent;
  }
  *xiLeft = 0.5 + ldexp(base_q, low_exponent);
  *xiRight = 0.5 + ldexp(base_q, high_exponent);
  return true;
#endif
}

__device__ void soc_proj_increasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *xiLeft, double *xiRight, double *xi, long* __restrict__ thread_idx, long* __restrict__ blk_dim, double abs_tol, double rel_tol)
{
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_soc_rootsearch(&xiLeft, &xiRight, &xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    *xi = soc_coordinate_bisection_midpoint(
        *xiLeft, *xiRight, true, rel_tol);
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

__device__ void soc_initial_norm_pair(
    const double *x, const double *D_scaled, double *D_scaled_mul_x,
    long len, long thread_idx, long blk_dim, double *polar_norm,
    double *weighted_norm) {
  PDCS_PROFILE_VV_REDUCTION();
  double local_polar = 0.0;
  double local_weighted = 0.0;
  for (long j = thread_idx; j < len; j += blk_dim) {
    double xj = x[j];
    double dj = D_scaled[j];
    double divided = xj / dj;
    double weighted = dj * xj;
    D_scaled_mul_x[j] = weighted;
    local_polar += divided * divided;
    local_weighted += weighted * weighted;
  }
  double polar_squared;
  double weighted_squared;
  block_reduce_pair(local_polar, local_weighted, thread_idx, blk_dim,
                    &polar_squared, &weighted_squared);
  *polar_norm = sqrt(polar_squared);
  *weighted_norm = sqrt(weighted_squared);
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
#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double polar_norm;
  double weighted_norm;
  soc_initial_norm_pair(x2end, D_scaled_part, D_scaled_mul_x_part, len,
                        *thread_idx, *blk_dim, &polar_norm, &weighted_norm);
  if (polar_norm <= -sol[0] && sol[0] <= 0) {
#else
  // temp_part = x2end ./ D_scaled_part
  vvrscl(&len, x2end, D_scaled_part, temp_part, thread_idx, blk_dim);
  __syncthreads();
  if (nrm2(&len, temp_part, thread_idx, blk_dim) <= -sol[0] && sol[0] <= 0) {
#endif
    PDCS_PROFILE_BRANCH(1);
    for (long j = *thread_idx; j < *n; j += *blk_dim){
      sol[j] = 0.0;
    }
    return;
  }
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  vvscal(&len, D_scaled_part, x2end, D_scaled_mul_x_part, thread_idx, blk_dim);
  double weighted_norm = nrm2(&len, D_scaled_mul_x_part, thread_idx, blk_dim);
#endif
  if (weighted_norm <= sol[0]) {
    PDCS_PROFILE_BRANCH(0);
    if (*thread_idx == 0) {
      sol[0] = fmax(sol[0], 0.0);
    }
    __syncthreads();
    scal_inplace(n, &minVal, sol, thread_idx, blk_dim);
    return;
  }
#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
  if (*n >= PDCS_SOC_LOGIT_MIN_DIMENSION &&
      (t > rel_tol || t < -rel_tol)) {
    const bool negative_branch = t < 0.0;
    PDCS_PROFILE_BRANCH(negative_branch ? 3 : 2);
    const double z = soc_logit_z_solve(
        t, negative_branch, t_warm_start_val_test, D_scaled_mul_x_part,
        D_scaled_squared_part, &len, thread_idx, blk_dim,
        negative_branch ? polar_norm : weighted_norm, abs_tol, rel_tol);
    soc_logit_z_recover(
        sol, n, z, negative_branch, minVal, t_warm_start,
        D_scaled_squared, thread_idx, blk_dim);
    return;
  }
#endif
#if PDCS_ENABLE_BOUNDED_SOC_ROOT
  if (t > rel_tol || t < -rel_tol) {
    const bool increasing = t < 0.0;
    PDCS_PROFILE_BRANCH(increasing ? 3 : 2);
    const double u = soc_bounded_u_solve(
        t, increasing, t_warm_start_val_test, D_scaled_mul_x_part,
        D_scaled_squared_part, &len, thread_idx, blk_dim,
        increasing ? polar_norm : weighted_norm, abs_tol, rel_tol);
    soc_bounded_u_recover(
        sol, n, u, increasing, minVal, t_warm_start,
        D_scaled_squared, thread_idx, blk_dim);
    return;
  }
#endif
  if (t > rel_tol) {
    PDCS_PROFILE_BRANCH(2);
    xiRight = 0.5;
    xiLeft = 0.0;
    oracleVal = 1.0;
    if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight){
      PDCS_PROFILE_WARM_ATTEMPT();
      // oracleVal = oracle_soc_f_sqrt(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim);
      oracle_soc_h(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        PDCS_PROFILE_WARM_ACCEPT();
        xi = t_warm_start[0];
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        return;
      }
      decreasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
#if PDCS_ENABLE_COLD_SOC_NEWTON
    else {
      t_warm_start_val_test = 0.5 * (xiLeft + xiRight);
      oracle_soc_h(&t_warm_start_val_test, sol, D_scaled_mul_x_part,
                   D_scaled_squared_part, temp_part, &len, &oracleVal,
                   &oracleVal_h, thread_idx, blk_dim);
      decreasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part,
          D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h,
          &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp,
          D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test,
          thread_idx, blk_dim, abs_tol, rel_tol);
    }
#endif
    soc_proj_decreasing_binary_search(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &xiLeft, &xiRight, &xi, thread_idx, blk_dim, abs_tol, rel_tol);
    soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
    return;
  }
  else if (t < -rel_tol) {
    PDCS_PROFILE_BRANCH(3);
    xiRight = 1.0;
    xiLeft = 0.5;
    if (t_warm_start_val_test > xiLeft){
      // oracleVal = oracle_soc_f_sqrt(&t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim);
      oracle_soc_h(&t_warm_start_val_test, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        PDCS_PROFILE_WARM_ACCEPT();
        xi = t_warm_start_val_test;
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        return;
      }
      increasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    bool exponent_bracketed = soc_exponent_expansion_bracket(
        sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len,
        &xiLeft, &xiRight, thread_idx, blk_dim, abs_tol);
    if (!exponent_bracketed) {
      while (oracle_soc_f_sqrt(&xiRight, sol, D_scaled_mul_x_part,
                               D_scaled_squared_part, temp_part, &len,
                               thread_idx, blk_dim) < 0) {
        PDCS_PROFILE_EXPANSION();
        xiLeft = xiRight;
        xiRight *= 2;
      }
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
  PDCS_PROFILE_ORACLE();
  for (long j = *thread_idx; j < *len; j += *blk_dim) {
    temp_part[j] = x_mul_d_part[j] / (1 + xi[0] * D_scaled_squared_part[j]);
  }
  double xi_sqr = xi[0] * xi[0];
  double xi_sqr_one = xi_sqr - 1;
  double xi_sqr_one_sqr = xi_sqr_one * xi_sqr_one;
  return nrm2(len, temp_part, thread_idx, blk_dim) - sqrt(2 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) / xi_sqr_one_sqr);
}

__device__ void oracle_rsoc_h(double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h, long* __restrict__ thread_idx, long* __restrict__ blk_dim) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
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
  PDCS_PROFILE_NEWTON_ATTEMPT();
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
    PDCS_PROFILE_BISECTION();
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
  PDCS_PROFILE_NEWTON_ATTEMPT();
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
    PDCS_PROFILE_BISECTION();
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
      PDCS_PROFILE_WARM_ATTEMPT();
      oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
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
      PDCS_PROFILE_WARM_ATTEMPT();
      xiRight = t_warm_start[0];
      oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
      if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
        recover_sol_rsoc(sol, n, &minVal, &t_warm_start_val_test, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
        // __syncthreads();
        return;
      }
      decreasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
    }
    while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
      PDCS_PROFILE_EXPANSION();
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
      PDCS_PROFILE_WARM_ATTEMPT();
          xiRight = t_warm_start[0];
          oracle_rsoc_h(&t_warm_start_val_test, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
          if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
            xi = t_warm_start[0];
            recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
            // __syncthreads();
            return;
          }
          increasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
        }
        while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
      PDCS_PROFILE_EXPANSION();
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
      PDCS_PROFILE_WARM_ATTEMPT();
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
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
      PDCS_PROFILE_WARM_ATTEMPT();
          oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
          if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
            xi = t_warm_start[0];
            recover_sol_rsoc(sol, n, &minVal, &xi, t_warm_start, D_scaled_squared, thread_idx, blk_dim);
            // __syncthreads();
            return;
          }
        }
        increasing_binary_rsoc_proj_init(sol, n, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, &t_warm_start_val_test, thread_idx, blk_dim, abs_tol, rel_tol);
      }
      while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, thread_idx, blk_dim) < 0) {
      PDCS_PROFILE_EXPANSION();
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
      PDCS_PROFILE_WARM_ATTEMPT();
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, thread_idx, blk_dim);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
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
  if (blkNum > 1 && (proj_type[1] == 3 || proj_type[1] == 4)){
    // all threads for positive projection
    long n = ns[1];
    double *sol = arr + gpu_head_start[1];
    for (int i = global_thread_idx; i < n; i += total_thread){
      sol[i] = fmax(sol[i], 0.0);
    }
  }
  if (blkNum > 2 && (proj_type[2] == 3 || proj_type[2] == 4)){
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

// One-launch heterogeneous block-wise entry point. Native structured cones
// occupy one cooperative block, large simple cones occupy one block, and EXP
// or tiny simple cones are packed one per thread in the tail of the same grid.
// Tiny SOC cones are handled by the separate massive indexed kernel only when
// there are enough of them to amortize that second launch.
extern "C" __global__ void
moderate_block_proj_indexed(
    double* arr, double* bl, double* bu, double* D_scaled,
    double* D_scaled_squared, double* D_scaled_mul_x, double* temp,
    double* t_warm_start, const long* gpu_head_start, const long* ns,
    const long* native_indices, long native_count,
    const long* simple_indices, long simple_count,
    const long* serial_indices, long serial_count, long* proj_type,
    double abs_tol, double rel_tol) {
  long work_block = blockIdx.x;
  long thread_idx = threadIdx.x;
  long blk_dim = blockDim.x;

  if (work_block < native_count) {
    long cone_idx = native_indices[work_block];
    long n = ns[cone_idx];
    double *sol = arr + gpu_head_start[cone_idx];
    double *sub_D_scaled = D_scaled + gpu_head_start[cone_idx];
    double *sub_D_scaled_squared = D_scaled_squared + gpu_head_start[cone_idx];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + gpu_head_start[cone_idx];
    double *sub_temp = temp + gpu_head_start[cone_idx];
    long code = proj_type[cone_idx];

    if (code == 5 || code == 7 || code == 20 || code == 21) {
      soc_proj(sol, &n, &thread_idx, &blk_dim);
    }
    else if (code == 6 || code == 22) {
      soc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared,
                        sub_D_scaled_mul_x, sub_temp,
                        &t_warm_start[cone_idx], &cone_idx, &thread_idx,
                        &blk_dim, abs_tol, rel_tol);
    }
    else if (code == 8 || code == 10 || code == 23 || code == 24) {
      rsoc_proj(sol, &n, sub_D_scaled_mul_x, sub_temp, &thread_idx,
                &blk_dim);
    }
    else if (code == 9 || code == 25) {
      rsoc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared,
                         sub_D_scaled_mul_x, sub_temp,
                         &t_warm_start[cone_idx], &thread_idx, &blk_dim,
                         abs_tol, rel_tol);
    }
    return;
  }

  work_block -= native_count;
  if (work_block < simple_count) {
    long cone_idx = simple_indices[work_block];
    long n = ns[cone_idx];
    double *sol = arr + gpu_head_start[cone_idx];
    double *sub_bl = bl + gpu_head_start[cone_idx];
    double *sub_bu = bu + gpu_head_start[cone_idx];
    long code = proj_type[cone_idx];
    if (code == 17 || code == 18 || code == 19) {
      for (long j = thread_idx; j < n; j += blk_dim) {
        sol[j] = fmax(fmin(sol[j], sub_bu[j]), sub_bl[j]);
      }
    }
    else if (code == 2) {
      for (long j = thread_idx; j < n; j += blk_dim) sol[j] = 0.0;
    }
    else if (code == 3 || code == 4) {
      for (long j = thread_idx; j < n; j += blk_dim) {
        sol[j] = fmax(sol[j], 0.0);
      }
    }
    return;
  }

  work_block -= simple_count;
  long list_idx = work_block * blk_dim + thread_idx;
  if (list_idx >= serial_count) return;
  long cone_idx = serial_indices[list_idx];
  long n = ns[cone_idx];
  double *sol = arr + gpu_head_start[cone_idx];
  double *sub_D_scaled = D_scaled + gpu_head_start[cone_idx];
  double *sub_temp = temp + gpu_head_start[cone_idx];
  double *sub_bl = bl + gpu_head_start[cone_idx];
  double *sub_bu = bu + gpu_head_start[cone_idx];
  long code = proj_type[cone_idx];

  if (code == 17 || code == 18 || code == 19) {
    for (long j = 0; j < n; ++j) {
      sol[j] = fmax(fmin(sol[j], sub_bu[j]), sub_bl[j]);
    }
  }
  else if (code == 2) {
    for (long j = 0; j < n; ++j) sol[j] = 0.0;
  }
  else if (code == 3 || code == 4) {
    for (long j = 0; j < n; ++j) sol[j] = fmax(sol[j], 0.0);
  }
  else if (code == 11 || code == 16 || code == 28) {
    dualExponent_proj(sol, &t_warm_start[cone_idx], abs_tol, rel_tol);
  }
  else if (code == 13 || code == 14 || code == 26) {
    exponent_proj(sol, &t_warm_start[cone_idx], abs_tol, rel_tol);
  }
  else if (code == 12 || code == 29) {
    dualExponent_proj_diagonal(sol, sub_D_scaled, sub_temp,
                               &t_warm_start[cone_idx], abs_tol, rel_tol);
  }
  else if (code == 15 || code == 27) {
    double sub_D_scaled_inv[3];
    sub_D_scaled_inv[0] = 1.0 / sub_D_scaled[0];
    sub_D_scaled_inv[1] = 1.0 / sub_D_scaled[1];
    sub_D_scaled_inv[2] = 1.0 / sub_D_scaled[2];
    exponent_proj_diagonal(sol, sub_D_scaled_inv,
                           &t_warm_start[cone_idx], abs_tol, rel_tol);
  }
}
