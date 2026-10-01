#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
// #define proj_abs_tol_squared 1e-32
#define MAX_ITER 10000
#ifndef PDCS_ENABLE_SAFEGUARDED_NEWTON
#define PDCS_ENABLE_SAFEGUARDED_NEWTON 1
#endif
#ifndef PDCS_ENABLE_COLD_SOC_NEWTON
#define PDCS_ENABLE_COLD_SOC_NEWTON 0
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_ORACLE
#define PDCS_ENABLE_FUSED_SOC_ORACLE 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
#define PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS 1
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_ROOT
#define PDCS_ENABLE_BOUNDED_SOC_ROOT 1
#endif
#ifndef PDCS_SOC_BOUNDED_NEWTON_STEPS
#define PDCS_SOC_BOUNDED_NEWTON_STEPS 8
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_ILLINOIS
#define PDCS_ENABLE_BOUNDED_SOC_ILLINOIS 1
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_HALLEY
#define PDCS_ENABLE_BOUNDED_SOC_HALLEY 0
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
#define PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT 0
#endif
#ifndef PDCS_SOC_LOGIT_STEPS
#define PDCS_SOC_LOGIT_STEPS 48
#endif
#include "soc_root_coordinate.cuh"
#include "bounded_soc_root_step.cuh"
#include "bounded_soc_logit_root.cuh"


#include "exp_proj.cu"

// n is the length of the vector, including the first element
// len is the length of the vector, not including the first element or the top two elements

// BLAS functions
__device__ double nrm2(const long* __restrict__ n, const double* __restrict__ x)
{
  PDCS_PROFILE_VV_REDUCTION();
  // calculate the norm of the vector x
  double norm = 0.0;
  double val = 0.0;
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    val = x[j];
    norm += val * val;
  }
  return sqrt(norm);
}

__device__ double nrm2_squared(const long* __restrict__ n, const double* __restrict__ x)
{
  PDCS_PROFILE_VV_REDUCTION();
  // calculate the norm of the vector x
  double norm = 0.0;
  double val = 0.0;
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    val = x[j];
    norm += val * val;
  }
  return norm;
}

__device__ void mem_copy(const long* __restrict__ n, double* __restrict__ dst, const double* __restrict__ src)
{
  // dst = src
  for (long j = 0; j < *n; ++j)
  {
    dst[j] = src[j];
  }
}

__device__ void scal_inplace(const long* __restrict__ n, const double* __restrict__ sa, double* __restrict__ sx)
{
  // scale the vector x by the scalar sa,
  // sy = sx * sa
  if (*sa == 1.0)
  {
    return;
  }
  for (long j = 0; j < *n; ++j)
  {
    sx[j] *= sa[0];
  }
}

__device__ void rscl(const long* __restrict__ n, const double* __restrict__ sx, const double* __restrict__ sa,  double* __restrict__ sy)
{
  // scale the vector x by the scalar sa,
  // sy = sx / sa
  if (*sa == 1.0)
  {
    mem_copy(n, sy, sx);
    return;
  }
  for (long j = 0; j < *n; ++j)
  {
    sy[j] = sx[j] / sa[0];
  }
}

__device__ void rscl_inplace(const long* __restrict__ n, const double* __restrict__ sa, double* __restrict__ sx)
{
  // scale the vector x by the scalar sa,
  // sy = sx / sa
  if (*sa == 1.0)
  {
    return;
  }
  for (long j = 0; j < *n; ++j)
  {
    sx[j] /= sa[0];
  }
}

__device__ void vvscal(const long* __restrict__ n, const double* __restrict__ s, const double* __restrict__ x, double* __restrict__ y)
{
  // scale the vector x by the vector s,
  // y = s * x
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    y[j] = x[j] * s[j];
  }
}

__device__ void vvscal_inplace(const long* __restrict__ n, const double* __restrict__ s, double* __restrict__ x)
{
  // scale the vector x by the vector s,
  // x = s * x
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    x[j] *= s[j];
  }
}

__device__ void vvrscl(const long* __restrict__ n, const double* __restrict__ x, const double* __restrict__ s,  double* __restrict__ y)
{
  // scale the vector x by the vector s,
  // y = x / s
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    y[j] = x[j] / s[j];
  }
}

__device__ void vvrscl_inplace(const long* __restrict__ n, const double* __restrict__ s, double* __restrict__ x)
{
  // scale the vector x by the vector s,
  // x = x / s
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    x[j] /= s[j];
  }
}

__device__ double diff_norm(const long* __restrict__ n, const double* __restrict__ x, const double* __restrict__ y)
{
  PDCS_PROFILE_VV_REDUCTION();
  // y = x - y
  double norm = 0.0;
  double diff = 0.0;
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    diff = x[j] - y[j];
    norm += diff * diff;
  }
  return sqrt(norm);
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
    if (a0[0] >= 0) {
      printf("No real roots found in the range: %f.\n", 1.5 * sqrt(-a0[0]));
      return 0.0;
    }
    double x = sqrt(-a0[0]);        // search starting point (can be adjusted according to the scenario)
    int found_roots = 0;     // number of real roots found
    int iter = 0;
    if (a0[0] >= 0) {
      printf("No real roots found in the range: %f.\n", 1.5 * sqrt(-a0[0]));
      return 0.0;
    }
    while (x <= 1.5 * sqrt(-a0[0]) && iter < 100000) {
        // calculate f(x) and f'(x)
        double fx = f(a4, a3, a2, a1, a0, x);
        double dfx = df(a4, a3, a2, a1, a0, x);

        if (fabs(fx) < tolerance) { // check if x is a root
            double fx_small = f(a4, a3, a2, a1, a0, x - step);
            double fx_large = f(a4, a3, a2, a1, a0, x + step);
            if (fx_small < 0 && fx_large > 0) {
                found_roots = 1;
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
__device__ void box_proj(double *sol, const double *bl, const double *bu, long *n)
{
  #pragma unroll
  for (long j = 0; j < *n; ++j)
  {
    // printf("sol[%ld]: %f, bl[%ld]: %f, bu[%ld]: %f\n", j, sol[j], j, bl[j], j, bu[j]);
    sol[j] = min(max(sol[j], bl[j]), bu[j]);
  }
}

__device__ void soc_proj(double* __restrict__ sol, long* __restrict__ n)
{
  long len = *n - 1;
  double norm = nrm2(&len, &sol[1]);
  double t = sol[0];
  if (norm + t <= 0)
  {
    for (long j = 0; j < *n; ++j)
    {
      sol[j] = 0.0;
    }
    // thrust::fill(sol, sol + n[0], 0.0);
  }
  else if (norm <= t)
  {
    // Do nothing, continue with the next iteration
  }
  else
  {
    double c = (1.0 + t / norm) / 2.0;
    sol[0] = norm * c;
    for (long j = 1; j < *n; ++j)
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

__device__ void rsoc_proj(double *sol, long *n, double *temp1, double *temp2){
  double minVal = 1e-3;
  rscl(n, sol, &minVal, sol);
  double x0y0 = sol[0] * sol[1];
  double x0Squr = sol[0] * sol[0];
  double y0Squr = sol[1] * sol[1];
  long len = *n - 2;
  double z0NrmSqur = nrm2_squared(&len, &sol[2]);
  if (2 * x0y0 > z0NrmSqur && sol[0] >= 0 && sol[1] >= 0) {
    scal_inplace(n, &minVal, sol);
    return;
  }
  if (sol[0] <= 0 && sol[1] <= 0 && 2 * x0y0 >= z0NrmSqur) {
    // thrust::fill(sol, sol + n[0], 0.0);
    for (long j = 0; j < *n; ++j){
      sol[j] = 0.0;
    }
    return;
  }
  if (fabs(sol[0] + sol[1]) < positive_zero) {
    long len = *n - 2;
    double s = 2;
    rscl(&len, &sol[2], &s, &sol[2]);
    double C = nrm2_squared(&len, &sol[2]);
    process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
    scal_inplace(n, &minVal, sol);
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
      rscl(&len, &sol[2], &s, &sol[2]); // sol[2:end] /= 2
      double C = nrm2_squared(&len, &sol[2]);
      process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
      scal_inplace(n, &minVal, sol);
      return;
    }
    double denominator = (1 - lambd * lambd);
    double xNew = (sol[0] + lambd * sol[1]) / denominator;
    double yNew = (sol[1] + lambd * sol[0]) / denominator;
    sol[0] = xNew;
    sol[1] = yNew;
    long len = *n - 2;
    rscl(&len, &sol[2], &denominator, &sol[2]); // sol[2:end] /= denominator
    scal_inplace(n, &minVal, sol);
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
        rscl(&len, &sol[2], &denominator1, &temp1[2]); // sol[2:end] /= denominator1
        rscl(&len, &sol[2], &denominator2, &temp2[2]); // sol[2:end] /= denominator2
        double norm1 = diff_norm(n, temp1, sol);
        double norm2 = diff_norm(n, temp2, sol);
        if (norm1 < norm2) {
          mem_copy(n, sol, temp1);
        }
        else {
          mem_copy(n, sol, temp2);
        }
        scal_inplace(n, &minVal, sol);
        return;
      }
      else {
        // only one point is feasible
        sol[0] = xNew1;
        sol[1] = yNew1;
        denominator1 = 1 + lambd1;
        rscl(&len, &sol[2], &denominator1, &sol[2]); // sol[2:end] /= denominator1
        scal_inplace(n, &minVal, sol);
        return;
      }
    }
    else if (xNew2 > 0 && yNew2 > 0) {
      sol[0] = xNew2;
      sol[1] = yNew2;
      denominator2 = 1 + lambd2;
      rscl(&len, &sol[2], &denominator2, &sol[2]); // sol[2:end] /= denominator2
      scal_inplace(n, &minVal, sol);
      return;
    }
    else {
      // thrust::fill(sol, sol + n[0], 0.0);
      for (long j = 0; j < *n; ++j){
        sol[j] = 0.0;
      }
    }
  }
}


__device__ double oracle_soc_f_sqrt(double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len) {
  PDCS_PROFILE_ORACLE();
  // len not including the first element
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  PDCS_PROFILE_VV_REDUCTION();
  double value_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    double y = D_scaled_part_mul_x_part[j] /
               (1.0 + (2.0 * xi[0]) * D_scaled_squared_part[j]);
    value_sum += y * y;
  }
  return sqrt(value_sum) - (x[0] / (1 - 2 * xi[0]));
#else
  for (long j = 0; j < *len; ++j) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  return nrm2(len, temp_part) - (x[0] / (1 - 2 * xi[0]));
#endif
}

__device__ void oracle_soc_h(double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  // len not including the first element
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  PDCS_PROFILE_VV_REDUCTION();
  double value_sum = 0.0;
  double derivative_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + (2.0 * xi[0]) * q;
    double y = D_scaled_part_mul_x_part[j] / denominator;
    double y_squared = y * y;
    value_sum += y_squared;
    derivative_sum += y_squared * q / denominator;
  }
  double denominator = 1.0 - 2.0 * xi[0];
  double right = (x[0] / denominator) * (x[0] / denominator);
  *f = value_sum - right;
  *h = -4.0 * (derivative_sum + right / denominator);
#else
  for (long j = 0; j < *len; ++j) {
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
  double left = nrm2_squared(len, temp_part);
  double temp = 1 - 2 * xi[0];
  double right = (x[0] / temp) * (x[0] / temp);
  *f = left - right;
  for (long j = 0; j < *len; ++j) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + 2.0 * xi[0] * q;
    temp_part[j] *= sqrt(fmax(q / denominator, 0.0));
  }
  right = right / temp;
  *h = -4 * (nrm2_squared(len, temp_part) + right);
#endif
}

__device__ double oracle_soc_bounded_u_f(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_VV_REDUCTION();
  double value_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    const double a = D_scaled_part_mul_x_part[j];
    const double c = D_scaled_squared_part[j];
    const double ratio = increasing ? u / (c + 1.0 - u)
                                    : (1.0 - u) / (1.0 + c * u);
    const double value = a * ratio;
    value_sum += value * value;
  }
  return value_sum - t * t;
}

__device__ void oracle_soc_bounded_u_h(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len, double *f, double *h) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double value_sum = 0.0;
  double derivative_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    const double a = D_scaled_part_mul_x_part[j];
    const double a_squared = a * a;
    const double c = D_scaled_squared_part[j];
    if (increasing) {
      const double denominator = c + 1.0 - u;
      const double denominator_squared = denominator * denominator;
      value_sum += a_squared * u * u / denominator_squared;
      derivative_sum += 2.0 * a_squared * u * (1.0 + c) /
                        (denominator_squared * denominator);
    } else {
      const double one_minus_u = 1.0 - u;
      const double denominator = 1.0 + c * u;
      const double denominator_squared = denominator * denominator;
      value_sum += a_squared * one_minus_u * one_minus_u /
                   denominator_squared;
      derivative_sum -= 2.0 * a_squared * one_minus_u * (1.0 + c) /
                        (denominator_squared * denominator);
    }
  }
  *f = value_sum - t * t;
  *h = derivative_sum;
}

#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
__device__ void oracle_soc_bounded_u_h2(
    double u, double t, bool increasing,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double *f, double *h, double *h2) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double value_sum = 0.0;
  double derivative_sum = 0.0;
  double second_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    const double a = D_scaled_part_mul_x_part[j];
    const double a_squared = a * a;
    const double c = D_scaled_squared_part[j];
    const double q = 1.0 + c;
    if (increasing) {
      const double denominator = q - u;
      const double denominator_squared = denominator * denominator;
      const double denominator_fourth = denominator_squared * denominator_squared;
      value_sum += a_squared * u * u / denominator_squared;
      derivative_sum += 2.0 * a_squared * u * q /
                        (denominator_squared * denominator);
      second_sum += 2.0 * a_squared * q * (q + 2.0 * u) /
                    denominator_fourth;
    } else {
      const double one_minus_u = 1.0 - u;
      const double denominator = 1.0 + c * u;
      const double denominator_squared = denominator * denominator;
      const double denominator_fourth = denominator_squared * denominator_squared;
      value_sum += a_squared * one_minus_u * one_minus_u /
                   denominator_squared;
      derivative_sum -= 2.0 * a_squared * one_minus_u * q /
                        (denominator_squared * denominator);
      second_sum += 2.0 * a_squared * q *
                    (1.0 + 3.0 * c - 2.0 * c * u) /
                    denominator_fourth;
    }
  }
  *f = value_sum - t * t;
  *h = derivative_sum;
  *h2 = second_sum;
}
#endif

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
__device__ double oracle_soc_logit_z_f(
    double z, double t, bool negative_branch,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_VV_REDUCTION();
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double value_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    value_sum += pdcs_soc_logit_term_f(
        D_scaled_part_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch);
  }
  return value_sum - t * t;
}

__device__ void oracle_soc_logit_z_h2(
    double z, double t, bool negative_branch,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double *f, double *h, double *h2) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  PDCS_PROFILE_VV_REDUCTION();
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double value_sum = 0.0;
  double derivative_sum = 0.0;
  double second_sum = 0.0;
  for (long j = 0; j < *len; ++j) {
    double value;
    double derivative;
    double second;
    pdcs_soc_logit_term(
        D_scaled_part_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch, &value, &derivative, &second);
    value_sum += value;
    derivative_sum += derivative;
    second_sum += second;
  }
  *f = value_sum - t * t;
  *h = derivative_sum;
  *h2 = second_sum;
}

__device__ double soc_logit_z_solve(
    double t, bool negative_branch, double warm_z,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
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
  oracle_soc_logit_z_h2(
      z, t, negative_branch, D_scaled_part_mul_x_part,
      D_scaled_squared_part, len, &f, &h, &h2);
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
    oracle_soc_logit_z_h2(
        candidate, t, negative_branch, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &candidate_f, &candidate_h,
        &candidate_h2);
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
  int last_updated_side = 0;
  while (right - left > rel_tol && count <= MAX_ITER) {
    PDCS_PROFILE_BISECTION();
    ++count;
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
    f = oracle_soc_logit_z_f(
        z, t, negative_branch, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len);
    if (!isfinite(f) || f == 0.0) break;
    if (f > 0.0) {
      right = z;
      right_f = f;
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
    } else {
      left = z;
      left_f = f;
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
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
    double minVal, double *t_warm_start, double *D_scaled_squared) {
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  t_warm_start[0] = z;
  sol[0] = negative_branch ? -sol[0] * v / s * minVal
                           : sol[0] / s * minVal;
  for (long j = 1; j < *n; ++j) {
    const double c = D_scaled_squared[j];
    sol[j] = negative_branch ? sol[j] * v / (c + v) * minVal
                             : sol[j] / (1.0 + c * v) * minVal;
  }
}
#endif

__device__ double soc_bounded_u_solve(
    double t, bool increasing, double warm_u,
    double *D_scaled_part_mul_x_part,
    double *D_scaled_squared_part, long *len,
    double endpoint_norm, double abs_tol, double rel_tol) {
  double left = 0.0;
  double right = 1.0;
  const double t_squared = t * t;
  const double endpoint_f = endpoint_norm * endpoint_norm - t_squared;
  double left_f = increasing ? -t_squared : endpoint_f;
  double right_f = increasing ? endpoint_f : -t_squared;
  const bool valid_warm = isfinite(warm_u) && warm_u > 0.0 && warm_u < 1.0;
  double u = valid_warm ? warm_u : 0.5;
  if (valid_warm) PDCS_PROFILE_WARM_ATTEMPT();

  double f;
  double h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
  double h2;
  oracle_soc_bounded_u_h2(
      u, t, increasing, D_scaled_part_mul_x_part,
      D_scaled_squared_part, len, &f, &h, &h2);
#else
  oracle_soc_bounded_u_h(
      u, t, increasing, D_scaled_part_mul_x_part,
      D_scaled_squared_part, len, &f, &h);
#endif
  if (valid_warm && pdcs_bounded_soc_projection_converged(
          f, h, u, t, abs_tol, rel_tol)) {
    PDCS_PROFILE_WARM_ACCEPT();
    return u;
  }
  if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
    right = u;
    right_f = f;
  } else {
    left = u;
    left_f = f;
  }

  for (int iter = 0;
       iter < PDCS_SOC_BOUNDED_NEWTON_STEPS &&
       PDCS_ENABLE_SAFEGUARDED_NEWTON;
       ++iter) {
    if (pdcs_bounded_soc_projection_converged(
            f, h, u, t, abs_tol, rel_tol) ||
        pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) {
      return u;
    }
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
        &candidate_h2);
#else
    oracle_soc_bounded_u_h(
        candidate, t, increasing, D_scaled_part_mul_x_part,
        D_scaled_squared_part, len, &candidate_f, &candidate_h);
#endif
    if ((increasing && candidate_f > 0.0) ||
        (!increasing && candidate_f < 0.0)) {
      right = candidate;
      right_f = candidate_f;
    } else {
      left = candidate;
      left_f = candidate_f;
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
  int last_updated_side = 0;
  while (!pdcs_bounded_soc_bracket_converged(
             left, right, rel_tol) && count <= MAX_ITER) {
    PDCS_PROFILE_BISECTION();
    ++count;
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
    if (pdcs_bounded_soc_near_endpoint(u, rel_tol)) {
      oracle_soc_bounded_u_h(
          u, t, increasing, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len, &f, &h);
    } else {
      f = oracle_soc_bounded_u_f(
          u, t, increasing, D_scaled_part_mul_x_part,
          D_scaled_squared_part, len);
    }
    if (!isfinite(f) || pdcs_bounded_soc_projection_converged(
            f, h, u, t, abs_tol, rel_tol)) break;
    const bool update_right =
        (increasing && f > 0.0) || (!increasing && f < 0.0);
    if (update_right) {
      right = u;
      right_f = f;
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
    } else {
      left = u;
      left_f = f;
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
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
    double minVal, double *t_warm_start, double *D_scaled_squared) {
  t_warm_start[0] = u;
  if (increasing) {
    sol[0] = -sol[0] * (1.0 - u) / u * minVal;
    for (long j = 1; j < *n; ++j) {
      const double c = D_scaled_squared[j];
      sol[j] = sol[j] * (1.0 - u) / (c + 1.0 - u) * minVal;
    }
  } else {
    sol[0] = sol[0] / (1.0 - u) * minVal;
    for (long j = 1; j < *n; ++j) {
      sol[j] = sol[j] / (1.0 + u * D_scaled_squared[j]) * minVal;
    }
  }
}

__device__ void newton_soc_rootsearch(double *xiLeft, double *xiRight, double *xi, double *sol, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double abs_tol, double rel_tol) {
  for (int i = 0; i < 20; ++i) {
    double f, h;
    oracle_soc_h(xi, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part, len, &f, &h);
    if (f < 0) {
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
    if (*xiRight <= *xiLeft) {
      break;
    }
    if (fabs(f) <= abs_tol * abs_tol) {
      break;
    }
    *xi = fmin(fmax(*xi, *xiLeft + rel_tol), *xiRight - rel_tol);
  }
}

__device__ void soc_proj_diagonal_recover(double *sol, long *n, double *xi, double *minVal, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *temp_part, double *D_scaled_mul_x_part, double *D_scaled_squared_part, long *len) {
  t_warm_start[0] = *xi;
  sol[0] = sol[0] / (1 - 2 * *xi) * minVal[0];
  for (long j = 1; j < *n; ++j) {
    sol[j] = sol[j] / (1 + 2 * *xi * D_scaled_squared[j]) * minVal[0];
  }
}

__device__ void soc_proj_decreasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double abs_tol, double rel_tol) {
  *xi = (*xiRight + *xiLeft) / 2;
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    count++;
    if (count > MAX_ITER){
      PDCS_PROFILE_MAX_ITER();
      break;
    }
    *xi = pdcs_soc_bisection_midpoint(
        *xiLeft, *xiRight, false, rel_tol);
    *oracleVal = oracle_soc_f_sqrt(xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len);
    if (*oracleVal < 0){
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
#ifdef PDCS_PROFILE_ROOT_SEARCH
  double profile_xi = *xi;
  double profile_residual = oracle_soc_f_sqrt(&profile_xi, sol, D_scaled_mul_x_part,
      D_scaled_squared_part, temp_part, len);
  PDCS_PROFILE_RESIDUAL(profile_residual);
  PDCS_PROFILE_BRACKET(*xiLeft, *xiRight);
  PDCS_PROFILE_TERMINATION(count > MAX_ITER ? 3 :
      (fabs(profile_residual) <= abs_tol ? 1 : 2));
#endif
}

__device__ void soc_safeguarded_newton(
    double *sol, double *D_scaled_mul_x_part,
    double *D_scaled_squared_part, double *temp_part, long *len,
    double *oracleVal, double *oracleVal_h, double *xiLeft,
    double *xiRight, double *xi, double *warm_x, bool increasing,
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
    double shift = pdcs_soc_coordinate_shift(rel_tol);
    bool use_log_coordinate = true;
#if PDCS_SOC_COORDINATE_MODE == 3
    use_log_coordinate = pdcs_soc_newton_needs_log_coordinate(
        x, *xiLeft, *xiRight, increasing, shift);
#endif
    double candidate;
    if (use_log_coordinate) {
      double u = pdcs_soc_root_to_coordinate(x, increasing, shift);
      double u_left = pdcs_soc_root_to_coordinate(*xiLeft, increasing, shift);
      double u_right = pdcs_soc_root_to_coordinate(*xiRight, increasing, shift);
      double derivative_u = h * pdcs_soc_root_coordinate_derivative(
          x, increasing, shift);
      double candidate_u = u - f / derivative_u;
      candidate = pdcs_soc_coordinate_to_root(
          candidate_u, increasing, shift);
      double u_guard = 1e-8 * (u_right - u_left);
      if (!isfinite(candidate_u) || !isfinite(candidate) ||
          !isfinite(derivative_u) || fabs(derivative_u) <= 1e-18 ||
          candidate_u <= u_left + u_guard ||
          candidate_u >= u_right - u_guard ||
          candidate <= *xiLeft || candidate >= *xiRight) break;
    }
    else {
      candidate = x - f / h;
      double guard = fmax(rel_tol, 1e-8 * width);
      if (!isfinite(candidate) || candidate <= *xiLeft + guard ||
          candidate >= *xiRight - guard) break;
    }
#endif

    double old_abs_f = fabs(f);
    double candidate_f;
    double candidate_h;
    PDCS_PROFILE_NEWTON_ATTEMPT();
    oracle_soc_h(&candidate, sol, D_scaled_mul_x_part,
                 D_scaled_squared_part, temp_part, len, &candidate_f,
                 &candidate_h);
    x = candidate;
    f = candidate_f;
    h = candidate_h;
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
  *oracleVal = copysign(1.0, f);
}

__device__ void decreasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double abs_tol, double rel_tol) {
  *xi = *t_warm_start;
  soc_safeguarded_newton(sol, D_scaled_mul_x_part,
                         D_scaled_squared_part, temp_part, len, oracleVal,
                         oracleVal_h, xiLeft, xiRight, xi, xi, false,
                         abs_tol, rel_tol);
}


__device__ void soc_proj_increasing_binary_search(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double abs_tol, double rel_tol) {
  *xi = (*xiRight + *xiLeft) / 2;
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    count++;
    if (count > MAX_ITER){
      PDCS_PROFILE_MAX_ITER();
      break;
    }
    *xi = pdcs_soc_bisection_midpoint(
        *xiLeft, *xiRight, true, rel_tol);
    *oracleVal = oracle_soc_f_sqrt(xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len);
    if (*oracleVal > 0){
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
#ifdef PDCS_PROFILE_ROOT_SEARCH
  double profile_xi = *xi;
  double profile_residual = oracle_soc_f_sqrt(&profile_xi, sol, D_scaled_mul_x_part,
      D_scaled_squared_part, temp_part, len);
  PDCS_PROFILE_RESIDUAL(profile_residual);
  PDCS_PROFILE_BRACKET(*xiLeft, *xiRight);
  PDCS_PROFILE_TERMINATION(count > MAX_ITER ? 3 :
      (fabs(profile_residual) <= abs_tol ? 1 : 2));
#endif
}

__device__ void increasing_binary_soc_proj_init(double *sol, long *n, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double *D_scaled_squared, double *temp, double *D_scaled_mul_x, double *D_scaled_part, double *minVal, double abs_tol, double rel_tol) {
  *xi = *t_warm_start;
  soc_safeguarded_newton(sol, D_scaled_mul_x_part,
                         D_scaled_squared_part, temp_part, len, oracleVal,
                         oracleVal_h, xiLeft, xiRight, xi, xi, true,
                         abs_tol, rel_tol);
}

__device__ bool soc_exponent_expansion_bracket(
    double *sol, double *D_scaled_mul_x_part,
    double *D_scaled_squared_part, double *temp_part, long *len,
    double *xiLeft, double *xiRight, double abs_tol) {
#if !PDCS_ENABLE_EXPONENT_EXPANSION
  return false;
#else
  double right_f = oracle_soc_f_sqrt(
      xiRight, sol, D_scaled_mul_x_part, D_scaled_squared_part,
      temp_part, len);
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
    double candidate = 0.5 + ldexp(base_q, high_exponent);
    if (!isfinite(candidate)) break;
    PDCS_PROFILE_EXPANSION();
    double candidate_f = oracle_soc_f_sqrt(
        &candidate, sol, D_scaled_mul_x_part, D_scaled_squared_part,
        temp_part, len);
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
        temp_part, len);
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



__device__ void soc_initial_norm_pair(
    const double *x, const double *D_scaled, double *D_scaled_mul_x,
    long len, double *polar_norm, double *weighted_norm) {
  PDCS_PROFILE_VV_REDUCTION();
  double polar_squared = 0.0;
  double weighted_squared = 0.0;
  #pragma unroll
  for (long j = 0; j < len; ++j) {
    double xj = x[j];
    double dj = D_scaled[j];
    double divided = xj / dj;
    double weighted = dj * xj;
    D_scaled_mul_x[j] = weighted;
    polar_squared += divided * divided;
    weighted_squared += weighted * weighted;
  }
  *polar_norm = sqrt(polar_squared);
  *weighted_norm = sqrt(weighted_squared);
}

__device__ void soc_proj_diagonal(double* __restrict__ sol, long* __restrict__ n, double* __restrict__ D_scaled, double* __restrict__ D_scaled_squared, double* __restrict__ D_scaled_mul_x, double* __restrict__ temp, double* __restrict__ t_warm_start, int* __restrict__ i, double abs_tol, double rel_tol) {
  double minVal = 1e-3;
  rscl_inplace(n, &minVal, sol);
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
#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double polar_norm;
  double weighted_norm;
  soc_initial_norm_pair(x2end, D_scaled_part, D_scaled_mul_x_part, len,
                        &polar_norm, &weighted_norm);
  if (polar_norm <= -sol[0] && sol[0] <= 0) {
#else
  // temp_part = x2end ./ D_scaled_part
  vvrscl(&len, x2end, D_scaled_part, temp_part);
  if (nrm2(&len, temp_part) <= -sol[0] && sol[0] <= 0) {
#endif
    PDCS_PROFILE_BRANCH(1);
    // thrust::fill(sol, sol + n[0], 0.0);
    for (long j = 0; j < *n; ++j){
      sol[j] = 0.0;
    }
    // printf("soc_proj_diagonal 0\n");
    return;
  }
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  // D_scaled_mul_x_part = D_scaled_part .* x2end
  vvscal(&len, D_scaled_part, x2end, D_scaled_mul_x_part);
  double weighted_norm = nrm2(&len, D_scaled_mul_x_part);
#endif
  if (weighted_norm <= sol[0]) {
    PDCS_PROFILE_BRANCH(0);
    sol[0] = fmax(sol[0], 0.0);
    scal_inplace(n, &minVal, sol);
    // printf("soc_proj_diagonal 1\n");
    return;
  }
#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
  if (PDCS_ENABLE_THREAD_SOC_LOGIT_ROOT &&
      *n >= PDCS_SOC_LOGIT_MIN_DIMENSION &&
      (t > rel_tol || t < -rel_tol)) {
    const bool negative_branch = t < 0.0;
    PDCS_PROFILE_BRANCH(negative_branch ? 3 : 2);
    const double z = soc_logit_z_solve(
        t, negative_branch, t_warm_start[0], D_scaled_mul_x_part,
        D_scaled_squared_part, &len,
        negative_branch ? polar_norm : weighted_norm, abs_tol, rel_tol);
    soc_logit_z_recover(
        sol, n, z, negative_branch, minVal, t_warm_start,
        D_scaled_squared);
    return;
  }
#endif
#if PDCS_ENABLE_BOUNDED_SOC_ROOT
  if (t > rel_tol || t < -rel_tol) {
    const bool increasing = t < 0.0;
    PDCS_PROFILE_BRANCH(increasing ? 3 : 2);
    const double u = soc_bounded_u_solve(
        t, increasing, t_warm_start[0], D_scaled_mul_x_part,
        D_scaled_squared_part, &len,
        increasing ? polar_norm : weighted_norm, abs_tol, rel_tol);
    soc_bounded_u_recover(
        sol, n, u, increasing, minVal, t_warm_start, D_scaled_squared);
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
      // oracleVal = oracle_soc_f_sqrt(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
      oracle_soc_h(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        PDCS_PROFILE_WARM_ACCEPT();
        xi = t_warm_start[0];
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, temp_part, D_scaled_mul_x_part, D_scaled_squared_part, &len);
        // printf("soc_proj_diagonal 2\n");
        return;
      }
      decreasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, abs_tol, rel_tol);
    }
#if PDCS_ENABLE_COLD_SOC_NEWTON
    else {
      double cold_start = 0.5 * (xiLeft + xiRight);
      oracle_soc_h(&cold_start, sol, D_scaled_mul_x_part,
                   D_scaled_squared_part, temp_part, &len, &oracleVal,
                   &oracleVal_h);
      decreasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part,
          D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h,
          &xiLeft, &xiRight, &xi, &cold_start, D_scaled_squared, temp,
          D_scaled_mul_x, D_scaled_part, &minVal, abs_tol, rel_tol);
    }
#endif
    // newton_soc_rootsearch(&xiLeft, &xiRight, &xi, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
    soc_proj_decreasing_binary_search(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
    PDCS_PROFILE_RESIDUAL(oracleVal);
    soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, temp_part, D_scaled_mul_x_part, D_scaled_squared_part, &len);
    return;
  }
  else if (t < -rel_tol) {
    PDCS_PROFILE_BRANCH(3);
    xiRight = 1.0;
    xiLeft = 0.5;
    if (t_warm_start[0] > xiLeft){
      PDCS_PROFILE_WARM_ATTEMPT();
      // oracleVal = oracle_soc_f_sqrt(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
      oracle_soc_h(t_warm_start, sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
      if (fabs(oracleVal) < abs_tol * abs_tol){
        PDCS_PROFILE_WARM_ACCEPT();
        xi = t_warm_start[0];
        soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, temp_part, D_scaled_mul_x_part, D_scaled_squared_part, &len);
        return;
      }
      increasing_binary_soc_proj_init(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, &minVal, abs_tol, rel_tol);
    }
    bool exponent_bracketed = soc_exponent_expansion_bracket(
        sol, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len,
        &xiLeft, &xiRight, abs_tol);
    if (!exponent_bracketed) {
      while (oracle_soc_f_sqrt(&xiRight, sol, D_scaled_mul_x_part,
                               D_scaled_squared_part, temp_part, &len) < 0) {
        PDCS_PROFILE_EXPANSION();
        xiLeft = xiRight;
        xiRight *= 2;
      }
    }
    soc_proj_increasing_binary_search(sol, n, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
    PDCS_PROFILE_RESIDUAL(oracleVal);
    t_warm_start[0] = xi;
    soc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared, temp, D_scaled_mul_x, D_scaled_part, temp_part, D_scaled_mul_x_part, D_scaled_squared_part, &len);
    return;
  }
  else {
    for (long j = 1; j < *n; ++j) {
      sol[j] = sol[j] / (1 + D_scaled_squared[j]) * minVal;
      temp[j] = D_scaled[j] * sol[j];
    }
    sol[0] = nrm2(&len, temp + 1); // has multiply minVal
    // printf("soc_proj_diagonal 5\n");
    return;
  }
}

__device__ double oracle_rsoc_f_sqrt(double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_squared_part, double *temp_part, long *len) {
  PDCS_PROFILE_ORACLE();
  for (long j = 0; j < *len; ++j) {
    temp_part[j] = x_mul_d_part[j] / (1 + xi[0] * D_scaled_squared_part[j]);
  }
  double xi_sqr = xi[0] * xi[0];
  double xi_sqr_one = xi_sqr - 1;
  double xi_sqr_one_sqr = xi_sqr_one * xi_sqr_one;
  return nrm2(len, temp_part) - sqrt(2 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) / xi_sqr_one_sqr);
}

__device__ void oracle_rsoc_h(double *xi, double *x0_sqr, double *y0_sqr, double *x0y0, double *x_mul_d_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *f, double *h) {
  PDCS_PROFILE_ORACLE();
  PDCS_PROFILE_GRADIENT();
  for (long j = 0; j < *len; ++j) {
    temp_part[j] = x_mul_d_part[j] / (1 + xi[0] * D_scaled_squared_part[j]);
  }
  double xi_sqr = xi[0] * xi[0];
  double xi_sqr_one = xi_sqr - 1;
  double xi_sqr_one_sqr = xi_sqr_one * xi_sqr_one;
  double left = nrm2_squared(len, temp_part);
  double right = 2 * (x0y0[0] + (x0_sqr[0] + y0_sqr[0]) * xi[0] + x0y0[0] * xi_sqr) / xi_sqr_one_sqr;
  *f = left - right;
  for (long j = 0; j < *len; ++j) {
    temp_part[j] = temp_part[j] / sqrt(1 + xi[0] * D_scaled_squared_part[j]) * D_scaled_part[j];
  }
  double h_left = -2 * nrm2_squared(len, temp_part);
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
//     if (f < 1e+32 && f > -1e+32 && h < -proj_rel_tol) {
//       *xi = *xi - f / h;
//     }
//     else {
//       break;
//     }
//     if (fabs(f) <= proj_abs_tol) {
//       break;
//     }
//     *xi = fmin(fmax(*xi, *xiLeft + proj_rel_tol), *xiRight - proj_rel_tol);
//   }
// }

__device__ void rsoc_proj_diagonal_recover(double *sol, long *n, double *xi, double *minVal, double *t_warm_start, double *D_scaled_squared){
  t_warm_start[0] = *xi;
  double xNew = (sol[0] + sol[1] * xi[0]) / (1 - xi[0] * xi[0] + positive_zero) * minVal[0];
  double yNew = (sol[1] + sol[0] * xi[0]) / (1 - xi[0] * xi[0] + positive_zero) * minVal[0];
  sol[0] = xNew;
  sol[1] = yNew;
  for (long j = 2; j < *n; ++j) {
    sol[j] = sol[j] / (1 + xi[0] * D_scaled_squared[j]) * minVal[0];
  }
}

__device__ void rsoc_proj_decreasing_newton_step(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double abs_tol, double rel_tol){
  *xi -= fmax(fmin(*oracleVal / *oracleVal_h, 0.001), -0.001);
  *xi = fmax(fmin(*xi, *xiRight - rel_tol), *xiLeft + rel_tol);
  oracle_rsoc_h(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h);
  if (fabs(*oracleVal) < abs_tol * abs_tol){
    rsoc_proj_diagonal_recover(sol, n, xi, minVal, t_warm_start, D_scaled_squared);
    return;
  }
  if (*oracleVal < 0){
    *xiRight = *xi;
  }
  else {
    *xiLeft = *xi;
  }
}

__device__ void decreasing_binary_rsoc_proj_init(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double abs_tol, double rel_tol){
  *xi = *t_warm_start;
  if (*oracleVal < 0) {
    *xiRight = *t_warm_start;
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k) {
        rsoc_proj_decreasing_newton_step(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, minVal, x0_sqr, y0_sqr, x0y0, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = *t_warm_start;
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k) {
        rsoc_proj_decreasing_newton_step(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, minVal, x0_sqr, y0_sqr, x0y0, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void rsoc_proj_decreasing_binary_search(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double abs_tol, double rel_tol){
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_rsoc_rootsearch(&xiLeft, &xiRight, &xi, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    count++;
    if (count > MAX_ITER){
      break;
    }
    *xi = (*xiRight + *xiLeft) / 2;
    *oracleVal = oracle_rsoc_f_sqrt(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len);
    // printf("cuda x0 > 0 && y0 > 0 oracleVal: %.20e, xi: %.20e\n", oracleVal, xi);
    if (*oracleVal < 0) {
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}

__device__ void rsoc_proj_increasing_newton_step(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double abs_tol, double rel_tol){
  *xi -= fmax(fmin(*oracleVal / *oracleVal_h, 0.001), -0.001);
  *xi = fmax(fmin(*xi, *xiRight - rel_tol), *xiLeft + rel_tol);
  oracle_rsoc_h(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, oracleVal, oracleVal_h);
  if (fabs(*oracleVal) < abs_tol * abs_tol){
    *xiLeft = *xi;
    *xiRight = *xi;
    rsoc_proj_diagonal_recover(sol, n, xi, minVal, t_warm_start, D_scaled_squared);
    return;
  }
  if (*oracleVal > 0){
    *xiRight = *xi;
  }
  else {
    *xiLeft = *xi;
  }
}

__device__ void increasing_binary_rsoc_proj_init(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double *t_warm_start, double abs_tol, double rel_tol){
  if (*oracleVal > 0) {
    *xiRight = *t_warm_start;
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k) {
        rsoc_proj_increasing_newton_step(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, minVal, x0_sqr, y0_sqr, x0y0, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, abs_tol, rel_tol);
      }
    }
  }
  else {
    *xiLeft = *t_warm_start;
    if (fabs(*oracleVal) < 0.001){
      for (int k = 0; k < 2; ++k) {
        rsoc_proj_increasing_newton_step(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, len, minVal, x0_sqr, y0_sqr, x0y0, oracleVal, oracleVal_h, xiLeft, xiRight, xi, t_warm_start, abs_tol, rel_tol);
      }
    }
  }
}

__device__ void rsoc_proj_increasing_binary_search(double *sol, long *n, double *D_scaled_squared, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len, double *minVal, double *x0_sqr, double *y0_sqr, double *x0y0, double *oracleVal, double *oracleVal_h, double *xiLeft, double *xiRight, double *xi, double abs_tol, double rel_tol){
  *xi = (*xiRight + *xiLeft) / 2;
  // newton_rsoc_rootsearch(&xiLeft, &xiRight, &xi, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len);
  int count = 0;
  while ((*xiRight - *xiLeft) / (1 + *xiRight + *xiLeft) > rel_tol && fabs(*oracleVal) > abs_tol) {
    PDCS_PROFILE_BISECTION();
    count++;
    if (count > MAX_ITER){
      break;
    }
    *xi = (*xiRight + *xiLeft) / 2;
    *oracleVal = oracle_rsoc_f_sqrt(xi, x0_sqr, y0_sqr, x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len);
    // printf("cuda x0 > 0 && y0 > 0 oracleVal: %.20e, xi: %.20e\n", oracleVal, xi);
    if (*oracleVal > 0) {
      *xiRight = *xi;
    }
    else {
      *xiLeft = *xi;
    }
  }
}

__device__ void rsoc_proj_diagonal(double *sol, long *n, double *D_scaled, double *D_scaled_squared, double *D_scaled_mul_x, double *temp, double *t_warm_start, double abs_tol, double rel_tol) {
  double minVal = 1e-3;
  rscl(n, sol, &minVal, sol);
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
  double xi = 0.0;

  vvscal(&len, D_scaled_part, z, D_scaled_mul_x_part);
  double z0NrmSqur = nrm2_squared(&len, D_scaled_mul_x_part);
  if (2 * sol[0] * sol[1] >= z0NrmSqur && sol[0] >= 0 && sol[1] >= 0) {
    scal_inplace(n, &minVal, sol);
    return;
  }
  vvrscl(&len, z, D_scaled_part, temp_part);
  double val = nrm2_squared(&len, temp_part);
  if (sol[0] <= 0 && sol[1] <= 0 && 2 * sol[0] * sol[1] > val) {
    // thrust::fill(sol, sol + n[0], 0.0);
    for (long j = 0; j < *n; ++j){
      sol[j] = 0.0;
    }
    return;
  }
  if (fabs(sol[0] + sol[1]) < positive_zero) {
    for (long j = 0; j < len; ++j) {
      z[j] = z[j] / (1 + D_scaled_squared_part[j]);
      temp_part[j] = D_scaled_part[j] * z[j];
    }
    double C = nrm2_squared(&len, temp_part);
    process_lambd1(&sol[0], &sol[1], &C, &sol[0], &sol[1]);
    scal_inplace(n, &minVal, sol);
    return;
  }
  double x0_sqr = sol[0] * sol[0];
  double y0_sqr = sol[1] * sol[1];
  double x0y0 = sol[0] * sol[1];
  if (sol[0] > 0 && sol[1] > 0) {
    xiRight = 1.0;
    xiLeft = 0.0;
    oracleVal = 1.0;
    xi = (xiRight + xiLeft) / 2;
    if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
      PDCS_PROFILE_WARM_ATTEMPT();
      // oracleVal = oracle_rsoc_f_sqrt(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
      oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
      if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
        xi = t_warm_start[0];
        rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
        return;
      }
      decreasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
    }
    rsoc_proj_decreasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
    xi = (xiRight + xiLeft) / 2;
    rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
    return;
  }
  else if (sol[0] < 0 && sol[1] < 0) {
    vvrscl(&len, z, D_scaled_part, temp_part);
    double val = nrm2_squared(&len, temp_part);
    if (2 * sol[0] * sol[1] > val) {
      for (long j = 0; j < *n; ++j){
        sol[j] = 0.0;
      }
      return;
    }
    xiRight = 2.0;
    xiLeft = 1.0;
    if (t_warm_start[0] > xiLeft) {
      PDCS_PROFILE_WARM_ATTEMPT();
      xiRight = t_warm_start[0];
      // oracleVal = oracle_rsoc_f_sqrt(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
      oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
      increasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
    }
    while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len) < 0) {
      PDCS_PROFILE_EXPANSION();
      xiLeft = xiRight;
      xiRight *= 2;
    }
    rsoc_proj_increasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
    xi = (xiRight + xiLeft) / 2;
    rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
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
          // oracleVal = oracle_rsoc_f_sqrt(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
          oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
          increasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
        }
        while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len) < 0) {
      PDCS_PROFILE_EXPANSION();
          xiLeft = xiRight;
          xiRight *= 2;
        }
      }
      rsoc_proj_increasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
      return;
    }
    else if (sol[0] < 0 && sol[1] > 0 && sol[0] + sol[1] >= 0) {
      xiRight = 1.0;
      xiLeft = -sol[0] / sol[1];
      oracleVal = 1.0;
      if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
      PDCS_PROFILE_WARM_ATTEMPT();
        // oracleVal = oracle_rsoc_f_sqrt(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
          xi = t_warm_start[0];
          rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
          return;
        }
        decreasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
      }
      rsoc_proj_decreasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
      return;
    }
    else if (sol[0] >= 0 && sol[1] <= 0 && sol[0] + sol[1] <= 0) {
      xiLeft = 1.0;
      xiRight = -sol[1] / sol[0];
      if (sol[0] == 0){
        xiRight = 1.0;
        if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
      PDCS_PROFILE_WARM_ATTEMPT();
          oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
          if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
            xi = t_warm_start[0];
            rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
            return;
          }
          increasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
        }
      }
      while (oracle_rsoc_f_sqrt(&xiRight, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len) < 0) {
      PDCS_PROFILE_EXPANSION();
        xiLeft = xiRight;
        xiRight *= 2;
      }
      rsoc_proj_increasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
      return;
    }
    else if (sol[0] >= 0 && sol[1] <= 0 && sol[0] + sol[1] >= 0) {
      xiRight = 1.0;
      xiLeft = -sol[1] / sol[0];
      oracleVal = 1.0;
      if (t_warm_start[0] > xiLeft && t_warm_start[0] < xiRight) {
      PDCS_PROFILE_WARM_ATTEMPT();
        // oracleVal = oracle_rsoc_f_sqrt(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len);
        oracle_rsoc_h(t_warm_start, &x0_sqr, &y0_sqr, &x0y0, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &oracleVal, &oracleVal_h);
        if (fabs(oracleVal) < abs_tol * abs_tol) {
        PDCS_PROFILE_WARM_ACCEPT();
          xi = t_warm_start[0];
          rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
          return;
        }
        decreasing_binary_rsoc_proj_init(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, t_warm_start, abs_tol, rel_tol);
      }
      rsoc_proj_decreasing_binary_search(sol, n, D_scaled_squared, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, &len, &minVal, &x0_sqr, &y0_sqr, &x0y0, &oracleVal, &oracleVal_h, &xiLeft, &xiRight, &xi, abs_tol, rel_tol);
      xi = (xiRight + xiLeft) / 2;
      rsoc_proj_diagonal_recover(sol, n, &xi, &minVal, t_warm_start, D_scaled_squared);
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
massive_block_proj(double* arr, double* bl, double* bu, double* D_scaled, double* D_scaled_squared,  double* D_scaled_mul_x, double* temp, double* t_warm_start, const long* gpu_head_start, const long* ns, int blkNum, long* proj_type, double abs_tol, double rel_tol)
{
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = blockDim.x * gridDim.x;
  // one thread per cone projection
  if (proj_type[0] == 17 || proj_type[0] == 19 || proj_type[0] == 18){
    // all threads for box projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    double *sub_bl = bl + gpu_head_start[0];
    double *sub_bu = bu + gpu_head_start[0];
    for (int i = global_idx; i < n; i += global_size){
      sol[i] = min(max(sol[i], sub_bl[i]), sub_bu[i]);
    }
  }
  if (proj_type[0] == 2){
    // all threads for zero projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (int i = global_idx; i < n; i += global_size){
      sol[i] = 0.0;
    }
  }
  if (proj_type[0] == 3 || proj_type[0] == 4){
    // all threads for positive projection
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (int i = global_idx; i < n; i += global_size){
      sol[i] = fmax(sol[i], 0.0);
    }
  }
  if (blkNum > 1 && (proj_type[1] == 3 || proj_type[1] == 4)){
    // all threads for positive projection
    long n = ns[1];
    double *sol = arr + gpu_head_start[1];
    for (int i = global_idx; i < n; i += global_size){
      sol[i] = fmax(sol[i], 0.0);
    }
  }

  // if (global_idx < blkNum)
  // if (i == 0)
  for (int i = global_idx; i < blkNum; i += global_size)
  {
    long n = ns[i];
    double *sol = arr + gpu_head_start[i];
    double *sub_D_scaled = D_scaled + gpu_head_start[i];
    double *sub_D_scaled_squared = D_scaled_squared + gpu_head_start[i];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + gpu_head_start[i];
    double *sub_temp = temp + gpu_head_start[i];
    double *sub_bl = bl + gpu_head_start[i];
    double *sub_bu = bu + gpu_head_start[i];
    if (proj_type[i] == 0 || proj_type[i] == 1){
      // dual_free_proj
      ;
    }
    else if ((proj_type[i] == 17 || proj_type[i] == 18 ||
              proj_type[i] == 19) && i != 0) {
      box_proj(sol, sub_bl, sub_bu, &n);
    }
    else if (proj_type[i] == 2 && i != 0) {
      for (long j = 0; j < n; ++j) sol[j] = 0.0;
    }
    else if ((proj_type[i] == 3 || proj_type[i] == 4) && i > 1) {
      positive_proj(sol, &n);
    }
    // else if (proj_type[i] == 17 || proj_type[i] == 19 || proj_type[i] == 18){
    //   // box_proj
    //   // printf("box_proj, proj_type[i]: %ld\n", proj_type[i]);
    //   // box_proj(sol, sub_bl, sub_bu, &n);
    //   for (long j = 0; j < n; ++j)
    //   {
    //     // printf("sol[%ld]: %f, bl[%ld]: %f, bu[%ld]: %f\n", j, sol[j], j, sub_bl[j], j, sub_bu[j]);
    //     sol[j] = min(max(sol[j], sub_bl[j]), sub_bu[j]);
    //   }
    // }
    // else if (proj_type[i] == 2){
    //   // thrust::fill(sol, sol + n, 0.0);
    //   for (long j = 0; j < n; ++j){
    //     sol[j] = 0.0;
    //   }
    // }
    // else if (proj_type[i] == 3 || proj_type[i] == 4){
    //   // dual_positive_proj
    //   for (long j = 0; j < n; ++j){
    //     sol[j] = fmax(sol[j], 0.0);
    //   }
    // }
    else if (proj_type[i] == 5 || proj_type[i] == 7 || proj_type[i] == 20 || proj_type[i] == 21){
      soc_proj(sol, &n);
    }
    else if (proj_type[i] == 6 || proj_type[i] == 22){
      soc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], &i, abs_tol, rel_tol);
    }
    else if (proj_type[i] == 8 || proj_type[i] == 10 || proj_type[i] == 23 || proj_type[i] == 24){
      rsoc_proj(sol, &n, sub_D_scaled_mul_x, sub_temp);
    }
    else if (proj_type[i] == 9 || proj_type[i] == 25){
      rsoc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (proj_type[i] == 11 || proj_type[i] == 16 || proj_type[i] == 28){
      // dualExponent_proj
      dualExponent_proj(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (proj_type[i] == 14 || proj_type[i] == 13 || proj_type[i] == 26 ){
      // exponent_proj
      exponent_proj(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (proj_type[i] == 12 || proj_type[i] == 29){
      // dualExponent_proj_diagonal
      dualExponent_proj_diagonal(sol, sub_D_scaled, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (proj_type[i] == 15 || proj_type[i] == 27){
      // exponent_proj_diagonal
      double sub_D_scaled_inv[3];
      sub_D_scaled_inv[0] = 1.0 / sub_D_scaled[0];
      sub_D_scaled_inv[1] = 1.0 / sub_D_scaled[1];
      sub_D_scaled_inv[2] = 1.0 / sub_D_scaled[2];
      // printf("enter exp diagonal proj!");
      exponent_proj_diagonal(sol, sub_D_scaled_inv, &t_warm_start[i], abs_tol, rel_tol);
    }
  }
}

// SOC/RSOC-only specialization for thread-wise layouts. Keeping EXP root
// search out of this entry point materially reduces per-thread register
// pressure on instances containing tens of thousands of tiny SOC cones.
extern "C" __global__ void
massive_soc_block_proj(
    double* arr, double* bl, double* bu, double* D_scaled,
    double* D_scaled_squared, double* D_scaled_mul_x, double* temp,
    double* t_warm_start, const long* gpu_head_start, const long* ns,
    int blkNum, long* proj_type, double abs_tol, double rel_tol) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = blockDim.x * gridDim.x;

  if (proj_type[0] == 17 || proj_type[0] == 18 || proj_type[0] == 19) {
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    double *sub_bl = bl + gpu_head_start[0];
    double *sub_bu = bu + gpu_head_start[0];
    for (long j = global_idx; j < n; j += global_size) {
      sol[j] = fmax(fmin(sol[j], sub_bu[j]), sub_bl[j]);
    }
  }
  if (proj_type[0] == 2) {
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (long j = global_idx; j < n; j += global_size) sol[j] = 0.0;
  }
  if (proj_type[0] == 3 || proj_type[0] == 4) {
    long n = ns[0];
    double *sol = arr + gpu_head_start[0];
    for (long j = global_idx; j < n; j += global_size) {
      sol[j] = fmax(sol[j], 0.0);
    }
  }
  if (blkNum > 1 && (proj_type[1] == 3 || proj_type[1] == 4)) {
    long n = ns[1];
    double *sol = arr + gpu_head_start[1];
    for (long j = global_idx; j < n; j += global_size) {
      sol[j] = fmax(sol[j], 0.0);
    }
  }

  for (int i = global_idx; i < blkNum; i += global_size) {
    long n = ns[i];
    double *sol = arr + gpu_head_start[i];
    double *sub_D_scaled = D_scaled + gpu_head_start[i];
    double *sub_D_scaled_squared = D_scaled_squared + gpu_head_start[i];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + gpu_head_start[i];
    double *sub_temp = temp + gpu_head_start[i];
    long code = proj_type[i];
    if (code == 5 || code == 7 || code == 20 || code == 21) {
      soc_proj(sol, &n);
    }
    else if (code == 6 || code == 22) {
      soc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared,
                        sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], &i,
                        abs_tol, rel_tol);
    }
    else if (code == 8 || code == 10 || code == 23 || code == 24) {
      rsoc_proj(sol, &n, sub_D_scaled_mul_x, sub_temp);
    }
    else if (code == 9 || code == 25) {
      rsoc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared,
                         sub_D_scaled_mul_x, sub_temp, &t_warm_start[i],
                         abs_tol, rel_tol);
    }
  }
}

// Compact thread-wise launch used by the heterogeneous dispatcher.  The
// ordinary kernel maps every cone slot to a thread; this entry point maps only
// the cone indices whose dimension/type is actually suitable for one thread.
extern "C" __global__ void
massive_block_proj_indexed(
    double* arr, double* bl, double* bu, double* D_scaled,
    double* D_scaled_squared, double* D_scaled_mul_x, double* temp,
    double* t_warm_start, const long* gpu_head_start, const long* ns,
    const long* cone_indices, int cone_count, long* proj_type,
    double abs_tol, double rel_tol) {
  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= cone_count) return;

  int i = (int)cone_indices[list_idx];
  long n = ns[i];
  double *sol = arr + gpu_head_start[i];
  double *sub_D_scaled = D_scaled + gpu_head_start[i];
  double *sub_D_scaled_squared = D_scaled_squared + gpu_head_start[i];
  double *sub_D_scaled_mul_x = D_scaled_mul_x + gpu_head_start[i];
  double *sub_temp = temp + gpu_head_start[i];
  double *sub_bl = bl + gpu_head_start[i];
  double *sub_bu = bu + gpu_head_start[i];
  long code = proj_type[i];

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
  else if (code == 5 || code == 7 || code == 20 || code == 21) {
    soc_proj(sol, &n);
  }
  else if (code == 6 || code == 22) {
    soc_proj_diagonal(sol, &n, sub_D_scaled, sub_D_scaled_squared,
                      sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], &i,
                      abs_tol, rel_tol);
  }
  else if (code == 11 || code == 16 || code == 28) {
    dualExponent_proj(sol, &t_warm_start[i], abs_tol, rel_tol);
  }
  else if (code == 14 || code == 13 || code == 26) {
    exponent_proj(sol, &t_warm_start[i], abs_tol, rel_tol);
  }
  else if (code == 12 || code == 29) {
    dualExponent_proj_diagonal(sol, sub_D_scaled, sub_temp,
                               &t_warm_start[i], abs_tol, rel_tol);
  }
  else if (code == 15 || code == 27) {
    double sub_D_scaled_inv[3];
    sub_D_scaled_inv[0] = 1.0 / sub_D_scaled[0];
    sub_D_scaled_inv[1] = 1.0 / sub_D_scaled[1];
    sub_D_scaled_inv[2] = 1.0 / sub_D_scaled[2];
    exponent_proj_diagonal(sol, sub_D_scaled_inv, &t_warm_start[i],
                           abs_tol, rel_tol);
  }
}

// Compact handling for zero/positive/box/free blocks when every structured
// cone has been moved to an indexed hierarchy-specific launch.  One CUDA
// block cooperates on one simple vector, so large box cones remain parallel.
extern "C" __global__ void
simple_block_proj_indexed(
    double* arr, double* bl, double* bu, const long* gpu_head_start,
    const long* ns, const long* cone_indices, int cone_count,
    long* proj_type) {
  int list_idx = blockIdx.x;
  if (list_idx >= cone_count) return;
  int i = (int)cone_indices[list_idx];
  long n = ns[i];
  double *sol = arr + gpu_head_start[i];
  double *sub_bl = bl + gpu_head_start[i];
  double *sub_bu = bu + gpu_head_start[i];
  long code = proj_type[i];

  if (code == 17 || code == 18 || code == 19) {
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
      sol[j] = fmax(fmin(sol[j], sub_bu[j]), sub_bl[j]);
    }
  }
  else if (code == 2) {
    for (long j = threadIdx.x; j < n; j += blockDim.x) sol[j] = 0.0;
  }
  else if (code == 3 || code == 4) {
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
      sol[j] = fmax(sol[j], 0.0);
    }
  }
}
