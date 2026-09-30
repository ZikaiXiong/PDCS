#pragma once

#ifndef PDCS_EXP_COORDINATE_MODE
// 0: rho; 1: asinh(rho/s); 2: adaptive asinh for large signed brackets.
#define PDCS_EXP_COORDINATE_MODE 0
#endif
#ifndef PDCS_STANDARD_EXP_COORDINATE_MODE
// Keep the legacy global switch as a compatibility default, but allow the
// standard and diagonal EXP roots to select coordinates independently.
#define PDCS_STANDARD_EXP_COORDINATE_MODE PDCS_EXP_COORDINATE_MODE
#endif
#ifndef PDCS_DIAGONAL_EXP_COORDINATE_MODE
#define PDCS_DIAGONAL_EXP_COORDINATE_MODE PDCS_EXP_COORDINATE_MODE
#endif
#ifndef PDCS_EXP_COORDINATE_SCALE
#define PDCS_EXP_COORDINATE_SCALE 1.0
#endif
#ifndef PDCS_EXP_LOG_THRESHOLD
#define PDCS_EXP_LOG_THRESHOLD 16.0
#endif

__host__ __device__ __forceinline__ bool pdcs_exp_bracket_needs_log_coordinate(
    double low, double high) {
  return fmax(fabs(low), fabs(high)) >=
         PDCS_EXP_LOG_THRESHOLD * PDCS_EXP_COORDINATE_SCALE;
}

template <int CoordinateMode>
__host__ __device__ __forceinline__ double pdcs_exp_root_to_coordinate_mode(
    double rho) {
  if (CoordinateMode == 0) return rho;
  return asinh(rho / PDCS_EXP_COORDINATE_SCALE);
}

template <int CoordinateMode>
__host__ __device__ __forceinline__ double pdcs_exp_coordinate_to_root_mode(
    double u) {
  if (CoordinateMode == 0) return u;
  return PDCS_EXP_COORDINATE_SCALE * sinh(u);
}

template <int CoordinateMode>
__host__ __device__ __forceinline__ double
pdcs_exp_root_coordinate_derivative_mode(double rho) {
  if (CoordinateMode == 0) return 1.0;
  return hypot(PDCS_EXP_COORDINATE_SCALE, rho);
}

template <int CoordinateMode>
__host__ __device__ __forceinline__ double
pdcs_exp_bisection_midpoint_mode(double low, double high) {
  if (CoordinateMode == 0) return 0.5 * (low + high);
  if (CoordinateMode == 2 &&
      !pdcs_exp_bracket_needs_log_coordinate(low, high)) {
    return 0.5 * (low + high);
  }
  double midpoint = pdcs_exp_coordinate_to_root_mode<CoordinateMode>(
      0.5 * (pdcs_exp_root_to_coordinate_mode<CoordinateMode>(low) +
             pdcs_exp_root_to_coordinate_mode<CoordinateMode>(high)));
  if (!isfinite(midpoint) || midpoint <= low || midpoint >= high) {
    midpoint = 0.5 * (low + high);
  }
  return midpoint;
}

template <int CoordinateMode>
__host__ __device__ __forceinline__ double pdcs_exp_newton_candidate_mode(
    double rho, double f, double df, double low, double high) {
  if (CoordinateMode == 0) return rho - f / df;
  bool use_log_coordinate = true;
  if (CoordinateMode == 2) {
    use_log_coordinate = pdcs_exp_bracket_needs_log_coordinate(low, high);
  }
  if (!use_log_coordinate) return rho - f / df;
  double u = pdcs_exp_root_to_coordinate_mode<CoordinateMode>(rho);
  double derivative_u =
      df * pdcs_exp_root_coordinate_derivative_mode<CoordinateMode>(rho);
  if (!isfinite(derivative_u) || fabs(derivative_u) <= 1e-18) {
    return HUGE_VAL;
  }
  return pdcs_exp_coordinate_to_root_mode<CoordinateMode>(
      u - f / derivative_u);
}

__host__ __device__ __forceinline__ double
pdcs_standard_exp_bisection_midpoint(double low, double high) {
  return pdcs_exp_bisection_midpoint_mode<PDCS_STANDARD_EXP_COORDINATE_MODE>(
      low, high);
}

__host__ __device__ __forceinline__ double
pdcs_diagonal_exp_bisection_midpoint(double low, double high) {
  return pdcs_exp_bisection_midpoint_mode<PDCS_DIAGONAL_EXP_COORDINATE_MODE>(
      low, high);
}

__host__ __device__ __forceinline__ double
pdcs_standard_exp_newton_candidate(double rho, double f, double df,
                                   double low, double high) {
  return pdcs_exp_newton_candidate_mode<PDCS_STANDARD_EXP_COORDINATE_MODE>(
      rho, f, df, low, high);
}

__host__ __device__ __forceinline__ double
pdcs_diagonal_exp_newton_candidate(double rho, double f, double df,
                                   double low, double high) {
  return pdcs_exp_newton_candidate_mode<PDCS_DIAGONAL_EXP_COORDINATE_MODE>(
      rho, f, df, low, high);
}

// Compatibility wrappers for callers that intentionally use the legacy
// shared coordinate policy.
__host__ __device__ __forceinline__ double pdcs_exp_bisection_midpoint(
    double low, double high) {
  return pdcs_exp_bisection_midpoint_mode<PDCS_EXP_COORDINATE_MODE>(low, high);
}

__host__ __device__ __forceinline__ double pdcs_exp_newton_candidate(
    double rho, double f, double df, double low, double high) {
  return pdcs_exp_newton_candidate_mode<PDCS_EXP_COORDINATE_MODE>(
      rho, f, df, low, high);
}
