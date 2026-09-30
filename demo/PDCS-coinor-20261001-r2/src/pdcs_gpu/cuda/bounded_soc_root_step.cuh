#pragma once

#ifndef PDCS_ENABLE_BOUNDED_SOC_SMALL_STEP_EXIT
// The production bounded-u Newton path used this inexpensive correction test
// before the experimental logit/Halley work.  Keep it independently
// switchable so a candidate cannot silently change the legacy default.
#define PDCS_ENABLE_BOUNDED_SOC_SMALL_STEP_EXIT 1
#endif

#ifndef PDCS_ENABLE_BOUNDED_SOC_EXPONENTIAL_BISECTION
// Experimental fallback: bisect the odds u/(1-u), rather than u itself,
// when the remaining bracket spans a large multiplicative range.  Warm-start
// Newton remains the first choice; this only changes fallback points.
#define PDCS_ENABLE_BOUNDED_SOC_EXPONENTIAL_BISECTION 1
#endif

#ifndef PDCS_BOUNDED_SOC_EXPONENTIAL_RATIO
#define PDCS_BOUNDED_SOC_EXPONENTIAL_RATIO 4096.0
#endif

// Candidate selection for the bounded SOC root u in (0, 1).  The caller owns
// the strict bracket and evaluates exactly one selected point.  This helper is
// deliberately scalar: Halley, ordinary Newton, logit-Newton, Illinois, and
// midpoint candidates are tried without another vector traversal.

__host__ __device__ __forceinline__ bool pdcs_bounded_soc_valid_candidate(
    double candidate, double left, double right, double guard) {
  return isfinite(candidate) && candidate > left + guard &&
         candidate < right - guard;
}

// Absolute spacing of the stored u coordinate.  A fixed eps() floor is much
// too coarse close to zero (where Float64 retains relative resolution), while
// this scale also correctly reflects the loss of complement resolution near
// one.
__host__ __device__ __forceinline__ double
pdcs_bounded_soc_coordinate_resolution(double u) {
  const double machine_epsilon = 2.220446049250313e-16;
  const double minimum_normal = 2.2250738585072014e-308;
  return 64.0 * machine_epsilon * fmax(fabs(u), minimum_normal);
}

__host__ __device__ __forceinline__ bool
pdcs_bounded_soc_small_step_converged(
    double previous, double candidate, double rel_tol) {
#if PDCS_ENABLE_BOUNDED_SOC_SMALL_STEP_EXIT
  const double endpoint_distance = fmax(
      64.0 * 2.2250738585072014e-308,
      fmin(fabs(candidate), fabs(1.0 - candidate)));
  const double tolerance = fmax(
      pdcs_bounded_soc_coordinate_resolution(candidate),
      rel_tol * endpoint_distance);
  return isfinite(previous) && isfinite(candidate) &&
         fabs(candidate - previous) <= tolerance;
#else
  return false;
#endif
}

// The scalar bounded-u equation is only an auxiliary representation of the
// metric projection.  Its recovery map contains 1/u on the negative branch
// and 1/(1-u) on the positive branch, so an absolute root tolerance is not
// forward-error safe near either endpoint.  The Newton correction |F/F'|
// estimates the remaining root error; scaling it by the distance to the
// nearest endpoint controls the relative error of both singular factors.
__host__ __device__ __forceinline__ double
pdcs_bounded_soc_endpoint_distance(double u) {
  return fmax(64.0 * 2.2250738585072014e-308,
              fmin(fabs(u), fabs(1.0 - u)));
}

// Away from an endpoint, the bounded-u recovery is well conditioned and the
// established residual/bracket tests are sufficient.  Only enter the more
// expensive forward-error path once the singular recovery factor exceeds
// 1e3 (or the tolerance-dependent threshold is more conservative).
__host__ __device__ __forceinline__ bool
pdcs_bounded_soc_near_endpoint(double u, double rel_tol) {
  const double machine_epsilon = 2.220446049250313e-16;
  const double threshold =
      fmax(1e-3, sqrt(fmax(rel_tol, machine_epsilon)));
  return pdcs_bounded_soc_endpoint_distance(u) <= threshold;
}

__host__ __device__ __forceinline__ bool
pdcs_bounded_soc_projection_converged(
    double f, double h, double u, double t,
    double abs_tol, double rel_tol) {
  if (!isfinite(f) || !isfinite(u)) return false;
  const bool residual_ok =
      fabs(f) <= abs_tol * (1.0 + t * t);
  if (!residual_ok) return false;
  if (!pdcs_bounded_soc_near_endpoint(u, rel_tol)) return true;
  if (!isfinite(h) || fabs(h) <= 1e-300) return false;
  const double correction = fabs(f / h);
  const double forward_safe_tolerance =
      fmax(pdcs_bounded_soc_coordinate_resolution(u),
           rel_tol * pdcs_bounded_soc_endpoint_distance(u));
  return residual_ok && isfinite(correction) &&
         correction <= forward_safe_tolerance;
}

__host__ __device__ __forceinline__ bool
pdcs_bounded_soc_bracket_converged(
    double left, double right, double rel_tol) {
  if (!isfinite(left) || !isfinite(right) || !(right >= left)) {
    return false;
  }
  const double midpoint = 0.5 * (left + right);
  if (!pdcs_bounded_soc_near_endpoint(midpoint, rel_tol)) {
    return right - left <= rel_tol;
  }
  return right - left <= fmax(
      pdcs_bounded_soc_coordinate_resolution(midpoint),
      rel_tol * pdcs_bounded_soc_endpoint_distance(midpoint));
}

// Safeguarded Newton in u away from the endpoints, with a logit-coordinate
// correction when the direct step loses multiplicative endpoint resolution.
// This helper does not evaluate the oracle and therefore adds no vector work.
__host__ __device__ __forceinline__ bool
pdcs_bounded_soc_newton_candidate(
    double u, double f, double h, double left, double right,
    double *candidate) {
  if (!isfinite(u) || !isfinite(f) || !isfinite(h) ||
      fabs(h) <= 1e-300 || !(right > left)) {
    return false;
  }
  const double machine_guard = 64.0 * 2.220446049250313e-16;
  const double width = right - left;
  const double guard = fmax(
      pdcs_bounded_soc_coordinate_resolution(0.5 * (left + right)),
      1e-8 * width);
  const double direct = u - f / h;
  if (pdcs_bounded_soc_valid_candidate(direct, left, right, guard)) {
    *candidate = direct;
    return true;
  }

  const double clamped_u = fmin(fmax(u, machine_guard),
                                1.0 - machine_guard);
  const double derivative_logit =
      h * clamped_u * (1.0 - clamped_u);
  if (!isfinite(derivative_logit) || fabs(derivative_logit) <= 1e-300) {
    return false;
  }
  const double z = log(clamped_u) - log1p(-clamped_u);
  const double z_candidate = z - f / derivative_logit;
  double transformed;
  if (z_candidate >= 36.0) {
    transformed = 1.0 - machine_guard;
  } else if (z_candidate <= -36.0) {
    transformed = machine_guard;
  } else if (z_candidate >= 0.0) {
    const double exponential = exp(-z_candidate);
    transformed = 1.0 / (1.0 + exponential);
  } else {
    const double exponential = exp(z_candidate);
    transformed = exponential / (1.0 + exponential);
  }
  if (pdcs_bounded_soc_valid_candidate(
          transformed, left, right,
          pdcs_bounded_soc_coordinate_resolution(transformed))) {
    *candidate = transformed;
    return true;
  }
  return false;
}

// Midpoint in the logarithm of the odds.  The stable logistic recovery avoids
// overflow, while clamping keeps a strict finite bracket.  For ordinary
// brackets the arithmetic midpoint is cheaper and is retained.
__host__ __device__ __forceinline__ double
pdcs_bounded_soc_bisection_midpoint(
    double left, double right) {
  const double arithmetic = 0.5 * (left + right);
#if !PDCS_ENABLE_BOUNDED_SOC_EXPONENTIAL_BISECTION
  return arithmetic;
#else
  const double machine_guard = 64.0 * 2.220446049250313e-16;
  const double safe_left = fmin(
      fmax(left, machine_guard), 1.0 - machine_guard);
  const double safe_right = fmin(
      fmax(right, machine_guard), 1.0 - machine_guard);
  if (!(safe_right > safe_left)) return arithmetic;

  const double left_log_odds =
      log(safe_left) - log1p(-safe_left);
  const double right_log_odds =
      log(safe_right) - log1p(-safe_right);
  const double log_ratio = right_log_odds - left_log_odds;
  if (!isfinite(log_ratio) ||
      log_ratio < log(PDCS_BOUNDED_SOC_EXPONENTIAL_RATIO)) {
    return arithmetic;
  }

  const double z = 0.5 * (left_log_odds + right_log_odds);
  double candidate;
  if (z >= 0.0) {
    const double exponential = exp(-z);
    candidate = 1.0 / (1.0 + exponential);
  } else {
    const double exponential = exp(z);
    candidate = exponential / (1.0 + exponential);
  }
  return pdcs_bounded_soc_valid_candidate(
             candidate, left, right,
             pdcs_bounded_soc_coordinate_resolution(candidate)) ?
         candidate : arithmetic;
#endif
}

__host__ __device__ __forceinline__ bool pdcs_bounded_soc_candidate(
    double u, double f, double h, double h2,
    double left, double right, double left_f, double right_f,
    double *candidate) {
  const double width = right - left;
  if (!(width > 0.0) || !isfinite(f)) return false;
  const double guard = fmax(
      pdcs_bounded_soc_coordinate_resolution(0.5 * (left + right)),
      1e-8 * width);

  // Halley's method is cubic near a simple root.  F, F', and F'' are produced
  // by one fused vector traversal/reduction in the enabled implementation.
  if (isfinite(h) && isfinite(h2)) {
    const double denominator = 2.0 * h * h - f * h2;
    if (isfinite(denominator) && fabs(denominator) > 1e-300) {
      const double halley = u - 2.0 * f * h / denominator;
      if (pdcs_bounded_soc_valid_candidate(halley, left, right, guard)) {
        *candidate = halley;
        return true;
      }
    }
  }

  // Ordinary Newton remains the best inexpensive alternative away from the
  // endpoints.  The shared helper retries the same correction in logit
  // coordinates when direct-u arithmetic loses endpoint resolution.
  if (pdcs_bounded_soc_newton_candidate(
          u, f, h, left, right, candidate)) {
    return true;
  }

  // If all derivative steps are rejected, use the current strict bracket.
  // Illinois interpolation usually improves substantially on a midpoint but
  // the midpoint remains the unconditional finite fallback.
  const double denominator = right_f - left_f;
  if (isfinite(left_f) && isfinite(right_f) && isfinite(denominator) &&
      fabs(denominator) > 1e-300) {
    const double illinois = left - left_f * width / denominator;
    if (pdcs_bounded_soc_valid_candidate(illinois, left, right, guard)) {
      *candidate = illinois;
      return true;
    }
  }
  const double midpoint = 0.5 * (left + right);
  if (pdcs_bounded_soc_valid_candidate(midpoint, left, right,
          pdcs_bounded_soc_coordinate_resolution(midpoint))) {
    *candidate = midpoint;
    return true;
  }
  return false;
}
