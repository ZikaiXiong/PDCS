#pragma once
#include <stdint.h>

// Fixed-width diagnostic ABI. A value of -1 means that a counter was not
// emitted by the selected diagnostic path; consumers must never treat -1 as 0.
struct RootProfileRecord {
  int32_t branch_code;
  int32_t interval_expansion_iterations;
  int32_t newton_attempts;
  int32_t newton_accepts;
  int32_t bisection_iterations;
  int32_t oracle_evaluations;
  // Number of oracle evaluations that also formed a derivative. Every such
  // call is included in oracle_evaluations; this separate counter lets the
  // rebuttal distinguish function-only bisection from fused F/F' work.
  int32_t gradient_evaluations;
  int32_t warm_start_attempted;
  int32_t warm_start_accepted;
  int32_t max_iter_reached;
  int32_t termination_reason; // 1 residual, 2 bracket width, 3 cap, -1 unknown
  int32_t output_finite;
  // One full-cone traversal followed by a reduction.  A fused reduction may
  // return a value/derivative pair but still scans and synchronizes once.
  // This field deliberately reuses the former reserved ABI slot.
  int32_t vector_vector_reductions;
  double final_residual;
  double final_bracket_left;
  double final_bracket_right;
  double normalized_bracket_width;
};

static_assert(sizeof(RootProfileRecord) == 88, "RootProfileRecord ABI changed");

__device__ RootProfileRecord* pdcs_root_profile_records = nullptr;
__device__ long pdcs_root_profile_count = 0;

__device__ inline long pdcs_root_profile_index() {
#if defined(PDCS_PROFILE_MAPPING_THREAD)
  return (long)blockIdx.x * blockDim.x + threadIdx.x;
#elif defined(PDCS_PROFILE_MAPPING_WARP)
  return ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
#elif defined(PDCS_PROFILE_MAPPING_BLOCK)
  return (long)blockIdx.x;
#else
  return -1;
#endif
}

__device__ inline bool pdcs_root_profile_leader() {
#if defined(PDCS_PROFILE_MAPPING_THREAD)
  return true;
#elif defined(PDCS_PROFILE_MAPPING_WARP)
  return (threadIdx.x & 31) == 0;
#elif defined(PDCS_PROFILE_MAPPING_BLOCK)
  return threadIdx.x == 0;
#else
  return false;
#endif
}

#define PDCS_PROFILE_ADD(field, amount) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].field += (int)(amount); \
} while (0)
#define PDCS_PROFILE_BISECTION() PDCS_PROFILE_ADD(bisection_iterations, 1)
#define PDCS_PROFILE_EXPANSION() PDCS_PROFILE_ADD(interval_expansion_iterations, 1)
#define PDCS_PROFILE_ORACLE() PDCS_PROFILE_ADD(oracle_evaluations, 1)
#define PDCS_PROFILE_GRADIENT() PDCS_PROFILE_ADD(gradient_evaluations, 1)
#define PDCS_PROFILE_VV_REDUCTION() PDCS_PROFILE_ADD(vector_vector_reductions, 1)
#define PDCS_PROFILE_NEWTON_ATTEMPT() PDCS_PROFILE_ADD(newton_attempts, 1)
#define PDCS_PROFILE_NEWTON_ACCEPT() PDCS_PROFILE_ADD(newton_accepts, 1)
#define PDCS_PROFILE_MAX_ITER() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].max_iter_reached = 1; \
} while (0)
#define PDCS_PROFILE_RESIDUAL(value) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].final_residual = (double)(value); \
} while (0)
#define PDCS_PROFILE_TERMINATION(code) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].termination_reason = (int32_t)(code); \
} while (0)
#define PDCS_PROFILE_BRACKET(left, right) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) { \
    double _pdcs_l = (double)(left); \
    double _pdcs_r = (double)(right); \
    pdcs_root_profile_records[_pdcs_i].final_bracket_left = _pdcs_l; \
    pdcs_root_profile_records[_pdcs_i].final_bracket_right = _pdcs_r; \
    pdcs_root_profile_records[_pdcs_i].normalized_bracket_width = \
      (_pdcs_r-_pdcs_l)/(1.0+_pdcs_r+_pdcs_l); \
  } \
} while (0)
#define PDCS_PROFILE_OUTPUT_FINITE(value) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].output_finite = (int32_t)(value); \
} while (0)
#define PDCS_PROFILE_BRANCH(code) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].branch_code = (int32_t)(code); \
} while (0)
#define PDCS_PROFILE_WARM_ATTEMPT() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].warm_start_attempted = 1; \
} while (0)
#define PDCS_PROFILE_WARM_ACCEPT() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].warm_start_accepted = 1; \
} while (0)

__device__ inline RootProfileRecord pdcs_empty_root_record() {
  RootProfileRecord r;
  r.branch_code = -1;
  r.interval_expansion_iterations = 0;
  r.newton_attempts = 0;
  r.newton_accepts = 0;
  r.bisection_iterations = 0;
  r.oracle_evaluations = 0;
  r.gradient_evaluations = 0;
  r.warm_start_attempted = 0;
  r.warm_start_accepted = 0;
  r.max_iter_reached = 0;
  r.termination_reason = -1;
  r.output_finite = -1;
  r.vector_vector_reductions = 0;
  r.final_residual = 0.0/0.0;
  r.final_bracket_left = 0.0/0.0;
  r.final_bracket_right = 0.0/0.0;
  r.normalized_bracket_width = 0.0/0.0;
  return r;
}

extern "C" __global__ void pdcs_root_profile_initialize(
    RootProfileRecord* records, long count) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    pdcs_root_profile_records = records;
    pdcs_root_profile_count = count;
  }
  if (i < count) records[i] = pdcs_empty_root_record();
}
