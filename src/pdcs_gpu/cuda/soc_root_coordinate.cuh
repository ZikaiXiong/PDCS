#pragma once

#ifndef PDCS_SOC_COORDINATE_MODE
// 0: xi; 1: shifted-log; 2: smooth-logit/shifted-log; 3: adaptive mode 2.
#define PDCS_SOC_COORDINATE_MODE 0
#endif
#ifndef PDCS_SOC_LOG_RATIO_THRESHOLD
#define PDCS_SOC_LOG_RATIO_THRESHOLD 64.0
#endif
#ifndef PDCS_ENABLE_EXPONENT_EXPANSION
#define PDCS_ENABLE_EXPONENT_EXPANSION 0
#endif
#ifndef PDCS_SOC_NEWTON_STEPS
#define PDCS_SOC_NEWTON_STEPS 2
#endif

__host__ __device__ __forceinline__ double pdcs_soc_coordinate_shift(double rel_tol) {
  return fmax(rel_tol, 32.0 * 2.220446049250313e-16);
}

__host__ __device__ __forceinline__ double pdcs_soc_root_to_coordinate(
    double xi, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return xi;
#elif PDCS_SOC_COORDINATE_MODE == 1
  double anchor = increasing ? 0.5 : 0.0;
  return log(fmax(xi - anchor + shift, shift));
#else
  if (increasing) return log(fmax(xi - 0.5 + shift, shift));
  return log(fmax(xi + shift, shift) /
             fmax(0.5 - xi + shift, shift));
#endif
}

__host__ __device__ __forceinline__ double pdcs_soc_coordinate_to_root(
    double u, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return u;
#elif PDCS_SOC_COORDINATE_MODE == 1
  return (increasing ? 0.5 : 0.0) + exp(u) - shift;
#else
  if (increasing) return 0.5 + exp(u) - shift;
  double sigmoid;
  if (u >= 0.0) sigmoid = 1.0 / (1.0 + exp(-u));
  else {
    double exp_u = exp(u);
    sigmoid = exp_u / (1.0 + exp_u);
  }
  return sigmoid * (0.5 + 2.0 * shift) - shift;
#endif
}

__host__ __device__ __forceinline__ double pdcs_soc_root_coordinate_derivative(
    double xi, bool increasing, double shift) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return 1.0;
#elif PDCS_SOC_COORDINATE_MODE == 1
  return fmax(xi - (increasing ? 0.5 : 0.0) + shift, shift);
#else
  if (increasing) return fmax(xi - 0.5 + shift, shift);
  return fmax((xi + shift) * (0.5 - xi + shift) /
              (0.5 + 2.0 * shift), shift);
#endif
}

__host__ __device__ __forceinline__ bool pdcs_soc_bracket_needs_log_coordinate(
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

__host__ __device__ __forceinline__ bool pdcs_soc_newton_needs_log_coordinate(
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

__host__ __device__ __forceinline__ double pdcs_soc_bisection_midpoint(
    double xi_left, double xi_right, bool increasing, double rel_tol) {
#if PDCS_SOC_COORDINATE_MODE == 0
  return 0.5 * (xi_left + xi_right);
#else
  double shift = pdcs_soc_coordinate_shift(rel_tol);
#if PDCS_SOC_COORDINATE_MODE == 3
  if (!pdcs_soc_bracket_needs_log_coordinate(
          xi_left, xi_right, increasing, shift)) {
    return 0.5 * (xi_left + xi_right);
  }
#endif
  double u_left = pdcs_soc_root_to_coordinate(xi_left, increasing, shift);
  double u_right = pdcs_soc_root_to_coordinate(xi_right, increasing, shift);
  double midpoint = pdcs_soc_coordinate_to_root(
      0.5 * (u_left + u_right), increasing, shift);
  if (!isfinite(midpoint) || midpoint <= xi_left || midpoint >= xi_right) {
    midpoint = 0.5 * (xi_left + xi_right);
  }
  return midpoint;
#endif
}
