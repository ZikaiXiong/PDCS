#pragma once

#ifndef PDCS_ENABLE_ADAPTIVE_LOGIT_NEWTON
// Use Halley to find a root from a cold start and ordinary Newton once a
// genuine warm start is available. This avoids evaluating F'' after PDHG has
// supplied a usable previous root.
#define PDCS_ENABLE_ADAPTIVE_LOGIT_NEWTON 1
#endif

#ifndef PDCS_LOGIT_REQUIRE_RESIDUAL_WITH_CORRECTION
// Use the Newton correction test together with the strict bracket test.
// Setting this to one additionally requires the residual test; residual OR
// correction without the bracket guard is unsafe on highly conditioned cones.
#define PDCS_LOGIT_REQUIRE_RESIDUAL_WITH_CORRECTION 1
#endif

#ifndef PDCS_SOC_WARM_START_SENTINEL
// Cone warm-start arrays are historically initialized to 1.0. That is
// outside the old bounded-u interval but inside the logit interval.
#define PDCS_SOC_WARM_START_SENTINEL 1.0
#endif

#ifndef PDCS_SOC_LOGIT_MIN_DIMENSION
// The logit root is most valuable when each oracle traverses a large vector.
// Tiny cones are launch/arithmetic bound and retain the faster bounded-u
// implementation.  This threshold counts the SOC head coordinate as well.
#define PDCS_SOC_LOGIT_MIN_DIMENSION 16384
#endif

#ifndef PDCS_ENABLE_GRID_SOC_LOGIT_ROOT
// The host/cuBLAS grid mapping pays different launch and synchronization
// costs; keep its established bounded-u solver unless explicitly profiling
// the all-logit experiment.
#define PDCS_ENABLE_GRID_SOC_LOGIT_ROOT 0
#endif

#ifndef PDCS_ENABLE_THREAD_SOC_LOGIT_ROOT
// A single thread traversing a long cone makes the per-coordinate logistic
// arithmetic dominate. Block and warp mappings benefit from fewer traversals;
// the thread mapping does not on the measured large mixed layouts.
#define PDCS_ENABLE_THREAD_SOC_LOGIT_ROOT 0
#endif

#ifndef PDCS_ENABLE_SOC_LOGIT_ILLINOIS
// Keep the logit fallback policy independent from the bounded-u fallback.
// This lets small/medium cones reuse Newton endpoint values through Illinois
// without silently changing the separately tuned large-cone logit path.
#define PDCS_ENABLE_SOC_LOGIT_ILLINOIS 0
#endif

// Stable logit coordinate for the two generic diagonal-SOC branches.
//
//   s = sigmoid(z), v = 1 - s = sigmoid(-z)
//
// For t > 0 choose s = 1 - 2 lambda and for t < 0 choose
// s = 1 - (2 lambda)^(-1).  Both equations are strictly increasing:
//
//   t > 0: sum a_i^2 [s / (1 + c_i v)]^2 - t^2 = 0
//   t < 0: sum a_i^2 [s / (c_i + v)]^2 - t^2 = 0.
//
// Computing s and v from exp(-abs(z)) avoids cancellation at both endpoints.

__host__ __device__ __forceinline__ void pdcs_soc_logistic_pair(
    double z, double *s, double *v) {
  const double exponential = exp(-fabs(z));
  const double inverse_denominator = 1.0 / (1.0 + exponential);
  if (z >= 0.0) {
    *s = inverse_denominator;
    *v = exponential * inverse_denominator;
  } else {
    *s = exponential * inverse_denominator;
    *v = inverse_denominator;
  }
}

__host__ __device__ __forceinline__ void pdcs_soc_logit_term(
    double a, double c, double s, double v, bool negative_branch,
    double *value, double *derivative, double *second) {
  const double a_squared = a * a;
  const double q = 1.0 + c;
  const double alpha = negative_branch ? c : 1.0;
  const double beta = negative_branch ? 1.0 : c;
  const double denominator = alpha + beta * v;
  const double denominator_squared = denominator * denominator;
  const double term = a_squared * s * s / denominator_squared;
  const double term_derivative =
      2.0 * a_squared * s * s * v * q /
      (denominator_squared * denominator);
  *value = term;
  *derivative = term_derivative;
  *second = term_derivative *
      (2.0 * v - s + 3.0 * beta * s * v / denominator);
}

// First-order-only term for a genuine warm start. Ordinary Newton never uses
// F'', so avoid computing it for every cone coordinate.
__host__ __device__ __forceinline__ void pdcs_soc_logit_term_h(
    double a, double c, double s, double v, bool negative_branch,
    double *value, double *derivative) {
  const double a_squared = a * a;
  const double q = 1.0 + c;
  const double alpha = negative_branch ? c : 1.0;
  const double beta = negative_branch ? 1.0 : c;
  const double denominator = alpha + beta * v;
  const double inverse_denominator = 1.0 / denominator;
  const double ratio = s * inverse_denominator;
  const double term = a_squared * ratio * ratio;
  *value = term;
  *derivative = 2.0 * term * v * q * inverse_denominator;
}

// Function-value-only term for safeguarded fallback iterations. Once a
// bracket is known, midpoint/Illinois selection needs F but not F' or F''.
__host__ __device__ __forceinline__ double pdcs_soc_logit_term_f(
    double a, double c, double s, double v, bool negative_branch) {
  const double denominator = negative_branch ? c + v : 1.0 + c * v;
  const double ratio = s / denominator;
  return a * a * ratio * ratio;
}

__host__ __device__ __forceinline__ bool pdcs_soc_logit_converged(
    double f, double h, double z, double left, double right,
    double t, double abs_tol, double rel_tol) {
  if (right - left <= rel_tol) return true;
  if (!isfinite(f) || !isfinite(h) || fabs(h) <= 1e-300) return false;
  const bool residual_ok = fabs(f) <= abs_tol * (1.0 + t * t);
  const bool correction_ok = fabs(f / h) <= rel_tol;
#if PDCS_LOGIT_REQUIRE_RESIDUAL_WITH_CORRECTION
  return residual_ok && correction_ok;
#else
  return correction_ok;
#endif
}

__host__ __device__ __forceinline__ bool pdcs_soc_logit_valid_candidate(
    double candidate, double left, double right, double guard) {
  return isfinite(candidate) && candidate > left + guard &&
         candidate < right - guard;
}

// Select one point without another vector traversal.  When derivative steps
// cannot safely advance, doubling |z| moves through exponent space; because z
// is logarithmic this locates the original multiplicative scale in O(log log)
// evaluations.  Illinois and midpoint retain an unconditional strict bracket.
__host__ __device__ __forceinline__ bool pdcs_soc_logit_candidate(
    double z, double f, double h, double h2,
    double left, double right, double left_f, double right_f,
    bool use_halley, double *candidate) {
  const double width = right - left;
  if (!(width > 0.0) || !isfinite(f)) return false;
  const double guard = fmax(64.0 * 2.220446049250313e-16,
                            1e-8 * width);

  if (use_halley && isfinite(h) && isfinite(h2)) {
    const double denominator = 2.0 * h * h - f * h2;
    if (isfinite(denominator) && fabs(denominator) > 1e-300) {
      const double halley = z - 2.0 * f * h / denominator;
      if (pdcs_soc_logit_valid_candidate(halley, left, right, guard)) {
        *candidate = halley;
        return true;
      }
    }
  }

  if (isfinite(h) && fabs(h) > 1e-300) {
    const double newton = z - f / h;
    if (pdcs_soc_logit_valid_candidate(newton, left, right, guard)) {
      *candidate = newton;
      return true;
    }
  }

  const double direction = f < 0.0 ? 1.0 : -1.0;
  const double expansion_step = fmax(1.0, fabs(z));
  const double exponent_candidate = z + direction * expansion_step;
  if (pdcs_soc_logit_valid_candidate(
          exponent_candidate, left, right, guard)) {
    *candidate = exponent_candidate;
    return true;
  }

  const double denominator = right_f - left_f;
  if (isfinite(left_f) && isfinite(right_f) && isfinite(denominator) &&
      fabs(denominator) > 1e-300) {
    const double illinois = left - left_f * width / denominator;
    if (pdcs_soc_logit_valid_candidate(illinois, left, right, guard)) {
      *candidate = illinois;
      return true;
    }
  }

  const double midpoint = 0.5 * (left + right);
  if (pdcs_soc_logit_valid_candidate(
          midpoint, left, right, 64.0 * 2.220446049250313e-16)) {
    *candidate = midpoint;
    return true;
  }
  return false;
}
