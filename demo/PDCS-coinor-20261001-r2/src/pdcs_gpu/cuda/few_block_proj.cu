#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

// Version the Julia/C ABI explicitly.  Returning a checked status from
// few_block_proj is ABI version 2; older void-returning artifacts must not be
// loaded by a newer wrapper.
static constexpr int PDCS_GRIDWISE_ABI_VERSION = 2;
static constexpr int PDCS_GRIDWISE_INVALID_ARGUMENT = 1;
static constexpr int PDCS_GRIDWISE_CUDA_STATUS_BASE = 1000;
static constexpr int PDCS_GRIDWISE_CUBLAS_STATUS_BASE = 2000;

static thread_local int pdcs_gridwise_cuda_status = 0;
static thread_local int pdcs_gridwise_cublas_status = 0;

static inline cudaError_t pdcs_record_cuda(cudaError_t status) {
  if (status != cudaSuccess && pdcs_gridwise_cuda_status == 0) {
    pdcs_gridwise_cuda_status = static_cast<int>(status);
  }
  return status;
}

static inline cublasStatus_t pdcs_record_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS && pdcs_gridwise_cublas_status == 0) {
    pdcs_gridwise_cublas_status = static_cast<int>(status);
  }
  return status;
}

extern "C" int pdcs_gridwise_abi_version() {
  return PDCS_GRIDWISE_ABI_VERSION;
}


#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
#define minVal 1e-3
#define minVal_inv 1e+3
#ifndef PDCS_ENABLE_SAFEGUARDED_NEWTON
#define PDCS_ENABLE_SAFEGUARDED_NEWTON 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_ORACLE
#define PDCS_ENABLE_FUSED_SOC_ORACLE 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
#define PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS 1
#endif
#ifndef PDCS_ENABLE_GRID_SOC_FASTPATH
// Grid-wise diagonal SOC projection is host-orchestrated.  Reuse its existing
// cone workspace and make the two cheap cone decisions on the host instead of
// allocating three device scalars on every projection call.
#define PDCS_ENABLE_GRID_SOC_FASTPATH 1
#endif
#ifndef PDCS_PROFILE_GRID_ROOT_SEARCH
#define PDCS_PROFILE_GRID_ROOT_SEARCH 0
#endif
#ifndef PDCS_ENABLE_BOUNDED_SOC_ROOT
// Work in the bounded variables from Proposition soc_bisection:
//   u = 2 lambda                    for t > 0,
//   u = 1 - (2 lambda)^(-1)         for t < 0.
// Both roots lie strictly in (0, 1), so the negative branch no longer needs
// an outward/doubling search to construct a finite lambda bracket.
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
#ifndef PDCS_SOC_POST_BRACKET_NEWTON_STEPS
// The unbounded negative branch cannot safely use Newton until its upper
// endpoint has been found.  Once that strict bracket exists, fused F/F' costs
// one vector traversal, so a handful of safeguarded steps is substantially
// cheaper than dozens of full-vector bisection evaluations.
#define PDCS_SOC_POST_BRACKET_NEWTON_STEPS 8
#endif
#include "soc_root_coordinate.cuh"
#include "bounded_soc_root_step.cuh"
#include "bounded_soc_logit_root.cuh"

#include "exp_proj_kernel.cu"

// A cuBLAS handle is only valid in the cuBLAS runtime that created it.  This
// shared object may be linked against a different CUDA installation than the
// one selected internally by CUDA.jl, so create, configure, query, and destroy
// the grid-wise handle here, next to all of its consumers.
extern "C" void create_cublas_handle_inner(cublasHandle_t* handle) {
  *handle = nullptr;
  cublasStatus_t status = cublasCreate_v2(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "few_block_proj: cublasCreate_v2 failed (%d)\n",
            static_cast<int>(status));
  }
}

extern "C" int configure_cublas_handle_inner(cublasHandle_t handle,
                                                int reproducible) {
  cublasStatus_t status = cublasSetStream_v2(handle, nullptr);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  if (reproducible) {
    cublasMath_t mode = static_cast<cublasMath_t>(
        CUBLAS_PEDANTIC_MATH |
        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    status = cublasSetMathMode(handle, mode);
  }
  return static_cast<int>(status);
}

extern "C" int cublas_handle_configuration_inner(cublasHandle_t handle,
                                                    int* atomics_mode,
                                                    int* math_mode) {
  cublasAtomicsMode_t atomics;
  cublasMath_t math;
  cublasStatus_t status = cublasGetAtomicsMode(handle, &atomics);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasGetMathMode(handle, &math);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  *atomics_mode = static_cast<int>(atomics);
  *math_mode = static_cast<int>(math);
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" void destroy_cublas_handle_inner(cublasHandle_t handle) {
  if (handle == nullptr) return;
  cublasStatus_t status = cublasDestroy_v2(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "few_block_proj: cublasDestroy_v2 failed (%d)\n",
            static_cast<int>(status));
  }
}

#if PDCS_PROFILE_GRID_ROOT_SEARCH
struct GridSocProfileCounters {
  uint64_t projection_events;
  uint64_t polar_events;
  uint64_t interior_events;
  uint64_t root_events;
  uint64_t positive_root_events;
  uint64_t negative_root_events;
  uint64_t warm_start_attempts;
  uint64_t warm_start_direct_accepts;
  uint64_t newton_attempts;
  uint64_t newton_accepts;
  uint64_t newton_converged_events;
  uint64_t bisection_events;
  uint64_t bisection_iterations;
  uint64_t function_evaluations;
  uint64_t gradient_evaluations;
  uint64_t vector_reductions;
};

static GridSocProfileCounters grid_soc_profile = {};

extern "C" void pdcs_grid_soc_profile_reset() {
  grid_soc_profile = {};
}

extern "C" void pdcs_grid_soc_profile_read(uint64_t* output, int length) {
  const uint64_t values[] = {
      grid_soc_profile.projection_events,
      grid_soc_profile.polar_events,
      grid_soc_profile.interior_events,
      grid_soc_profile.root_events,
      grid_soc_profile.positive_root_events,
      grid_soc_profile.negative_root_events,
      grid_soc_profile.warm_start_attempts,
      grid_soc_profile.warm_start_direct_accepts,
      grid_soc_profile.newton_attempts,
      grid_soc_profile.newton_accepts,
      grid_soc_profile.newton_converged_events,
      grid_soc_profile.bisection_events,
      grid_soc_profile.bisection_iterations,
      grid_soc_profile.function_evaluations,
      grid_soc_profile.gradient_evaluations,
      grid_soc_profile.vector_reductions,
  };
  int count = length < 16 ? length : 16;
  for (int i = 0; i < count; ++i) output[i] = values[i];
}

#define GRID_SOC_PROFILE_ADD(field, amount) \
  do { grid_soc_profile.field += (uint64_t)(amount); } while (0)
#else
#define GRID_SOC_PROFILE_ADD(field, amount) do {} while (0)
#endif

// n is the length of the vector, including the first element
// len is the length of the vector, not including the first element or the top two elements

__global__ void box_proj(double *sol, const double *bl, const double *bu, long *n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n) {
      sol[idx] = min(max(sol[idx], bl[idx]), bu[idx]);
    }
}

__global__ void positive_proj(double *sol, long *n){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n){
        sol[idx] = max(sol[idx], 0.0);
    }
}

// Keep the grid-wise diagonal-SOC normalization on the same CUDA launch
// stream as the root-search and recovery kernels. Issuing this operation via
// a cuBLAS handle created by another shared-library instance can leave the
// Dscal unordered with respect to the CUDA C++ kernels, which makes recovery
// apply minVal without the matching minVal_inv normalization.
__global__ void scale_vector(
    double* __restrict__ values, long length, double factor) {
  long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < length) values[index] *= factor;
}

// `exponent_proj_diagonal_kernel` consumes the inverse diagonal. The PTX
// block/warp/thread-wise kernels perform this conversion before calling their
// device function; grid-wise must use the same convention.
__global__ void invert_exp_diagonal(const double* D, double* D_inv){
    int index = threadIdx.x;
    if (index < 3){
        D_inv[index] = 1.0 / D[index];
    }
}



__global__ void soc_proj_scale_kernel(double* sol, double* temp, long* n){
  double t = sol[0];
  double *norm = temp;
  if (*norm + t <= 0.0)
  {
    *norm = 0.0;
  }
  else if (*norm <= t)
  {
    *norm = 1.0;
  }
  else
  {
    sol[0] = *norm;
    *norm = (1.0 + t / *norm) / 2.0;
  }
}

extern void soc_proj(cublasHandle_t handle, double* __restrict__ sol, long* __restrict__ n_cpu, long* __restrict__ n_gpu, long* __restrict__ len_cpu, double* __restrict__ temp, int ThreadPerBlock, int nBlock)
{
  // Keep the cuBLAS reduction, scalar branch, and cuBLAS scaling in one
  // stream.  The handle is owned by Julia and is not guaranteed to use the
  // CUDA C++ translation unit's implicit default stream.  Launching the
  // scalar kernel without this explicit stream made it observe the previous
  // projection's workspace value on H100.
  cudaStream_t stream = nullptr;
  if (pdcs_record_cublas(cublasGetStream_v2(handle, &stream)) !=
      CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "few_block_proj: cublasGetStream_v2 failed\n");
    return;
  }
  pdcs_record_cublas(
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  // temp for storing the norm of the vector
  pdcs_record_cublas(cublasDnrm2(handle, *len_cpu, sol + 1, 1, temp));
  soc_proj_scale_kernel<<<1, 1, 0, stream>>>(sol, temp, n_gpu);
  pdcs_record_cublas(cublasDscal(handle, *n_cpu, temp, sol, 1));


  // create a new handle
  // cublasHandle_t handle_temp;
  // cublasCreate(&handle_temp);

  // cublasSetPointerMode(handle_temp, CUBLAS_POINTER_MODE_DEVICE);
  // // temp for storing the norm of the vector
  // cublasDnrm2(handle_temp, *len_cpu, sol + 1, 1, temp);
  // soc_proj_scale_kernel<<<1, 1>>>(sol, temp, n_gpu);
  // cublasDscal(handle_temp, *n_cpu, temp, sol, 1);
  // cublasDestroy(handle_temp);


}





__global__ void vvrscl(long* __restrict__ len, double* __restrict__ x, double* __restrict__ y, double* __restrict__ z) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    z[j] = x[j] / y[j];
  }
}

__global__ void soc_cone_dual(double* __restrict__ sol_gpu, long* __restrict__ n_gpu, double* __restrict__ temp_gpu, bool* __restrict__ d_return_flag) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j == 0){
    if (temp_gpu[0] <= -sol_gpu[0] && sol_gpu[0] <= 0){
      *d_return_flag = true;
    }
  }
  __syncthreads();
  if (*d_return_flag){
    for (long j = 0; j < *n_gpu; ++j){
      sol_gpu[j] = 0.0;
    }
  }
}

__global__ void vvscal(long* __restrict__ len, double* __restrict__ x, double* __restrict__ y, double* __restrict__ z) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    z[j] = x[j] * y[j];
  }
}

__global__ void soc_cone_heuristic(double* __restrict__ sol_gpu, double* __restrict__ temp_gpu, bool* __restrict__ d_return_flag) {
  if (temp_gpu[0] <= sol_gpu[0])
  {
    *d_return_flag = true;
  }
  if (*d_return_flag)
  {
    sol_gpu[0] = max(sol_gpu[0], 0.0);
  }
}

// __global__ void determine_case(int* __restrict__ case_flag, double *__restrict__ sol) {
//   long j = threadIdx.x + blockIdx.x * blockDim.x;
//   if (j == 0){
//     if (sol[0] > proj_rel_tol){
//       *case_flag = 0;
//     }
//     else if (sol[0] < -proj_rel_tol){
//       *case_flag = 1;
//     }
//     else {
//       *case_flag = 2;
//     }
//   }
// }

__global__ void initialize_case0(double* __restrict__ xiRight_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ oracleVal_gpu) {
    *xiRight_gpu = 0.5;
    *xiLeft_gpu = 0.0;
    *oracleVal_gpu = 1.0;
}

__global__ void initialize_case1(double* __restrict__ xiRight_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ oracleVal_gpu) {
    *xiRight_gpu = 1.0;
    *xiLeft_gpu = 0.5;
    *oracleVal_gpu = 1.0;
}

__global__ void check_t_range_case0(double* __restrict__ t_warm_start_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_auxiliary_flag) {
    xiLeft_gpu[0] = 0.0;
    xiRight_gpu[0] = 0.5;
    oracleVal_gpu[0] = 1.0;
    if (t_warm_start_gpu[0] > xiLeft_gpu[0] && t_warm_start_gpu[0] < xiRight_gpu[0]){
      *d_auxiliary_flag = true;
    }
}

__global__ void oracle_soc_f_sqrt_kernel(double* __restrict__ xi, double* __restrict__ x, double* __restrict__ D_scaled_part_mul_x_part, double* __restrict__ D_scaled_squared_part, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu) {
  // len not including the first element
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len){
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
}

__global__ void oracle_soc_f_sqrt_value_kernel(
    double xi, double* __restrict__ D_scaled_part_mul_x_part,
    double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < len) {
    temp_part[j] = D_scaled_part_mul_x_part[j] /
        (1.0 + (2.0 * xi) * D_scaled_squared_part[j]);
  }
}

__global__ void oracle_soc_f_sqrt_final_case0(double *__restrict__ xi, double* __restrict__ x, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
    *oracleVal_gpu -= (x[0] / (1 - 2 * xi[0]));
    if (fabs(*oracleVal_gpu) < abs_tol){
      *d_return_flag = true;
    }else{
      *d_return_flag = false;
    }
    if (*oracleVal_gpu < 0.0){
      *d_auxiliary_flag = true;
    }else{
      *d_auxiliary_flag = false;
    }
}

__global__ void oracle_soc_f_sqrt_final_case1(double *__restrict__ xi, double* __restrict__ x, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
    *oracleVal_gpu -= (x[0] / (1 - 2 * xi[0]));
    if (fabs(*oracleVal_gpu) < abs_tol){
      *d_return_flag = true;
    }else{
      *d_return_flag = false;
    }
    if (*oracleVal_gpu < 0.0){
      *d_auxiliary_flag = true;
    }else{
      *d_auxiliary_flag = false;
    }
}

extern "C" void oracle_soc_f_sqrt_case0(cublasHandle_t handle, double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len_cpu, long *len_gpu, int nThread, int nBlock, double* __restrict__ oracleVal_gpu,  bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
  oracle_soc_f_sqrt_kernel<<<nBlock, nThread>>>(xi, x, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part, len_gpu, oracleVal_gpu);
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu));
  oracle_soc_f_sqrt_final_case0<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
}

extern "C" void oracle_soc_f_sqrt_case1(cublasHandle_t handle, double *xi, double *x, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len_cpu, long *len_gpu, int nThread, int nBlock,  double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
  oracle_soc_f_sqrt_kernel<<<nBlock, nThread>>>(xi, x, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_gpu, oracleVal_gpu);
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu));
  oracle_soc_f_sqrt_final_case1<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
}

__global__ void oracle_soc_h_scale_kernel(
    double xi, double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long* __restrict__ len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + 2.0 * xi * q;
    temp_part[j] *= sqrt(fmax(q / denominator, 0.0));
  }
}

__global__ void oracle_soc_h_scale_value_kernel(
    double xi, double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < len) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + 2.0 * xi * q;
    temp_part[j] *= sqrt(fmax(q / denominator, 0.0));
  }
}

#if PDCS_ENABLE_FUSED_SOC_ORACLE
// One direct reduction avoids writing `temp_part`, and the Newton variant
// reduces the value and derivative accumulators in the same kernel launch.
// `result` points at the cone's temporary array and has at least two doubles
// for every SOC handled by the grid-wise path.
__global__ void oracle_soc_f_reduce_kernel(
    double xi, const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double local_value = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double y = D_scaled_mul_x_part[j] /
               (1.0 + (2.0 * xi) * D_scaled_squared_part[j]);
    local_value += y * y;
  }
  partial[threadIdx.x] = local_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(result, partial[0]);
}

__global__ void oracle_soc_h_reduce_kernel(
    double xi, const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double* partial_value = partial;
  double* partial_derivative = partial + blockDim.x;
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    const double q = D_scaled_squared_part[j];
    const double denominator = 1.0 + (2.0 * xi) * q;
    double y = D_scaled_mul_x_part[j] / denominator;
    double y_squared = y * y;
    local_value += y_squared;
    // If y_j = a_j / (1 + 2 xi q_j), then
    // d(y_j^2)/d xi = -4 y_j^2 q_j / (1 + 2 xi q_j).
    local_derivative += y_squared * q / denominator;
  }
  partial_value[threadIdx.x] = local_value;
  partial_derivative[threadIdx.x] = local_derivative;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_value[threadIdx.x] += partial_value[threadIdx.x + stride];
      partial_derivative[threadIdx.x] +=
          partial_derivative[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial_value[0]);
    atomicAdd(result + 1, partial_derivative[0]);
  }
}

// The bounded-coordinate equations use a_j = sqrt(c_j) x_j, where
// c_j = hat(d)_j^(-2).  `D_scaled_mul_x_part` already stores a_j and
// `D_scaled_squared_part` stores c_j.  The value and derivative variants below
// therefore need only one traversal/reduction, just like the lambda oracle.
__global__ void oracle_soc_bounded_u_f_reduce_kernel(
    double u, bool increasing,
    const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double local_value = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    const double a = D_scaled_mul_x_part[j];
    const double c = D_scaled_squared_part[j];
    double ratio;
    if (increasing) {
      // t < 0: F_-(u), increasing on (0, 1).
      ratio = u / (c + 1.0 - u);
    } else {
      // t > 0: F_+(u), decreasing on (0, 1).
      ratio = (1.0 - u) / (1.0 + c * u);
    }
    const double value = a * ratio;
    local_value += value * value;
  }
  partial[threadIdx.x] = local_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(result, partial[0]);
}

__global__ void oracle_soc_bounded_u_h_reduce_kernel(
    double u, bool increasing,
    const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double* partial_value = partial;
  double* partial_derivative = partial + blockDim.x;
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    const double a = D_scaled_mul_x_part[j];
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
  partial_value[threadIdx.x] = local_value;
  partial_derivative[threadIdx.x] = local_derivative;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_value[threadIdx.x] += partial_value[threadIdx.x + stride];
      partial_derivative[threadIdx.x] +=
          partial_derivative[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial_value[0]);
    atomicAdd(result + 1, partial_derivative[0]);
  }
}

#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
__global__ void oracle_soc_bounded_u_h2_reduce_kernel(
    double u, bool increasing,
    const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double* partial_value = partial;
  double* partial_derivative = partial + blockDim.x;
  double* partial_second = partial + 2 * blockDim.x;
  double local_value = 0.0;
  double local_derivative = 0.0;
  double local_second = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    const double a = D_scaled_mul_x_part[j];
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
  partial_value[threadIdx.x] = local_value;
  partial_derivative[threadIdx.x] = local_derivative;
  partial_second[threadIdx.x] = local_second;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_value[threadIdx.x] += partial_value[threadIdx.x + stride];
      partial_derivative[threadIdx.x] +=
          partial_derivative[threadIdx.x + stride];
      partial_second[threadIdx.x] += partial_second[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial_value[0]);
    atomicAdd(result + 1, partial_derivative[0]);
    atomicAdd(result + 2, partial_second[0]);
  }
}
#endif
#endif

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
__global__ void oracle_soc_logit_z_f_reduce_kernel(
    double z, bool negative_branch,
    const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double local_value = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    local_value += pdcs_soc_logit_term_f(
        D_scaled_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch);
  }
  partial[threadIdx.x] = local_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(result, partial[0]);
}

__global__ void oracle_soc_logit_z_h2_reduce_kernel(
    double z, bool negative_branch,
    const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double* partial_value = partial;
  double* partial_derivative = partial + blockDim.x;
  double* partial_second = partial + 2 * blockDim.x;
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  double local_value = 0.0;
  double local_derivative = 0.0;
  double local_second = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double value;
    double derivative;
    double second;
    pdcs_soc_logit_term(
        D_scaled_mul_x_part[j], D_scaled_squared_part[j], s, v,
        negative_branch, &value, &derivative, &second);
    local_value += value;
    local_derivative += derivative;
    local_second += second;
  }
  partial_value[threadIdx.x] = local_value;
  partial_derivative[threadIdx.x] = local_derivative;
  partial_second[threadIdx.x] = local_second;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_value[threadIdx.x] += partial_value[threadIdx.x + stride];
      partial_derivative[threadIdx.x] +=
          partial_derivative[threadIdx.x + stride];
      partial_second[threadIdx.x] += partial_second[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial_value[0]);
    atomicAdd(result + 1, partial_derivative[0]);
    atomicAdd(result + 2, partial_second[0]);
  }
}
#endif

#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
// Form D .* x and evaluate both cheap cone tests in a single memory traversal.
// Warp partials keep the shared-memory footprint independent of block size.
__global__ void soc_initial_norm_pair_kernel(
    const double* __restrict__ x, const double* __restrict__ D_scaled,
    double* __restrict__ D_scaled_mul_x, long len,
    double* __restrict__ result) {
  extern __shared__ double warp_partial[];
  double* polar_partial = warp_partial;
  double* weighted_partial = warp_partial + ((blockDim.x + 31) / 32);
  double polar_squared = 0.0;
  double weighted_squared = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double xj = x[j];
    double dj = D_scaled[j];
    double divided = xj / dj;
    double weighted = dj * xj;
    D_scaled_mul_x[j] = weighted;
    polar_squared += divided * divided;
    weighted_squared += weighted * weighted;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    polar_squared += __shfl_down_sync(0xffffffffu, polar_squared, offset);
    weighted_squared +=
        __shfl_down_sync(0xffffffffu, weighted_squared, offset);
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) / 32;
  if (lane == 0) {
    polar_partial[warp] = polar_squared;
    weighted_partial[warp] = weighted_squared;
  }
  __syncthreads();
  if (warp == 0) {
    polar_squared = lane < warp_count ? polar_partial[lane] : 0.0;
    weighted_squared = lane < warp_count ? weighted_partial[lane] : 0.0;
    for (int offset = 16; offset > 0; offset /= 2) {
      polar_squared += __shfl_down_sync(0xffffffffu, polar_squared, offset);
      weighted_squared +=
          __shfl_down_sync(0xffffffffu, weighted_squared, offset);
    }
    if (lane == 0) {
      atomicAdd(result, polar_squared);
      atomicAdd(result + 1, weighted_squared);
    }
  }
}
#endif

static double oracle_soc_f_host(
    cublasHandle_t handle, double xi, double sol0,
    double* xi_gpu, double* D_scaled_mul_x_part,
    double* D_scaled_squared_part, double* temp_part,
    long* len_cpu, long* len_gpu, int nThread, int nBlock,
    double* oracle_gpu) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, sizeof(double)));
  oracle_soc_f_reduce_kernel<<<nBlock, nThread,
      nThread * sizeof(double)>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, *len_cpu, oracle_gpu);
  double value_sum;
  pdcs_record_cuda(cudaMemcpy(
      &value_sum, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));
  return sqrt(fmax(value_sum, 0.0)) - sol0 / (1.0 - 2.0 * xi);
#else
  oracle_soc_f_sqrt_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, *len_cpu);
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu));
  double norm;
  pdcs_record_cuda(cudaMemcpy(
      &norm, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));
  return norm - sol0 / (1.0 - 2.0 * xi);
#endif
}

static void oracle_soc_h_host(
    cublasHandle_t handle, double xi, double sol0,
    double* xi_gpu, double* D_scaled_mul_x_part,
    double* D_scaled_squared_part, double* temp_part,
    long* len_cpu, long* len_gpu, int nThread, int nBlock,
    double* oracle_gpu, double* f, double* h) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(gradient_evaluations, 1);
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
#else
  GRID_SOC_PROFILE_ADD(vector_reductions, 2);
#endif
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, 2 * sizeof(double)));
  oracle_soc_h_reduce_kernel<<<nBlock, nThread,
      2 * nThread * sizeof(double)>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, *len_cpu, oracle_gpu);
  double sums[2];
  pdcs_record_cuda(cudaMemcpy(
      sums, oracle_gpu, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  double denominator = 1.0 - 2.0 * xi;
  double right = (sol0 / denominator) * (sol0 / denominator);
  *f = sums[0] - right;
  *h = -4.0 * (sums[1] + right / denominator);
#else
  oracle_soc_f_sqrt_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, *len_cpu);
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu));
  double norm1;
  pdcs_record_cuda(cudaMemcpy(
      &norm1, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));

  oracle_soc_h_scale_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_squared_part, temp_part, *len_cpu);
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu));
  double norm2;
  pdcs_record_cuda(cudaMemcpy(
      &norm2, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));

  double denominator = 1.0 - 2.0 * xi;
  double right = (sol0 / denominator) * (sol0 / denominator);
  *f = norm1 * norm1 - right;
  *h = -4.0 * (norm2 * norm2 + right / denominator);
#endif
}

static double oracle_soc_bounded_u_f_host(
    double u, double sol0, bool increasing,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, sizeof(double)));
  oracle_soc_bounded_u_f_reduce_kernel<<<nBlock, nThread,
      nThread * sizeof(double)>>>(
      u, increasing, D_scaled_mul_x_part, D_scaled_squared_part, len,
      oracle_gpu);
  double value_sum;
  pdcs_record_cuda(cudaMemcpy(
      &value_sum, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));
  return value_sum - sol0 * sol0;
}

static void oracle_soc_bounded_u_h_host(
    double u, double sol0, bool increasing,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu,
    double* f, double* h) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(gradient_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, 2 * sizeof(double)));
  oracle_soc_bounded_u_h_reduce_kernel<<<nBlock, nThread,
      2 * nThread * sizeof(double)>>>(
      u, increasing, D_scaled_mul_x_part, D_scaled_squared_part, len,
      oracle_gpu);
  double sums[2];
  pdcs_record_cuda(cudaMemcpy(
      sums, oracle_gpu, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  *f = sums[0] - sol0 * sol0;
  *h = sums[1];
}

#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
static void oracle_soc_bounded_u_h2_host(
    double u, double sol0, bool increasing,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu,
    double* f, double* h, double* h2) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(gradient_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, 3 * sizeof(double)));
  oracle_soc_bounded_u_h2_reduce_kernel<<<nBlock, nThread,
      3 * nThread * sizeof(double)>>>(
      u, increasing, D_scaled_mul_x_part, D_scaled_squared_part, len,
      oracle_gpu);
  double sums[3];
  pdcs_record_cuda(cudaMemcpy(
      sums, oracle_gpu, 3 * sizeof(double), cudaMemcpyDeviceToHost));
  *f = sums[0] - sol0 * sol0;
  *h = sums[1];
  *h2 = sums[2];
}
#endif

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
static double oracle_soc_logit_z_f_host(
    double z, double sol0, bool negative_branch,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, sizeof(double)));
  oracle_soc_logit_z_f_reduce_kernel<<<nBlock, nThread,
      nThread * sizeof(double)>>>(
      z, negative_branch, D_scaled_mul_x_part, D_scaled_squared_part,
      len, oracle_gpu);
  double value_sum;
  pdcs_record_cuda(cudaMemcpy(
      &value_sum, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost));
  return value_sum - sol0 * sol0;
}

static void oracle_soc_logit_z_h2_host(
    double z, double sol0, bool negative_branch,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu,
    double* f, double* h, double* h2) {
  GRID_SOC_PROFILE_ADD(function_evaluations, 1);
  GRID_SOC_PROFILE_ADD(gradient_evaluations, 1);
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle_gpu, 0, 3 * sizeof(double)));
  oracle_soc_logit_z_h2_reduce_kernel<<<nBlock, nThread,
      3 * nThread * sizeof(double)>>>(
      z, negative_branch, D_scaled_mul_x_part, D_scaled_squared_part,
      len, oracle_gpu);
  double sums[3];
  pdcs_record_cuda(cudaMemcpy(
      sums, oracle_gpu, 3 * sizeof(double), cudaMemcpyDeviceToHost));
  *f = sums[0] - sol0 * sol0;
  *h = sums[1];
  *h2 = sums[2];
}

static double soc_logit_z_solve_host(
    double sol0, bool negative_branch, double warm_z,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu,
    double endpoint_norm, double abs_tol, double rel_tol) {
  double left = -700.0;
  double right = 700.0;
  const double t_squared = sol0 * sol0;
  double left_f = -t_squared;
  double right_f = endpoint_norm * endpoint_norm - t_squared;
  const bool valid_warm =
      isfinite(warm_z) && warm_z > left && warm_z < right &&
      warm_z != PDCS_SOC_WARM_START_SENTINEL;
  const bool use_halley =
      !PDCS_ENABLE_ADAPTIVE_LOGIT_NEWTON || !valid_warm;
  double z = valid_warm ? warm_z : 0.0;
  if (valid_warm) GRID_SOC_PROFILE_ADD(warm_start_attempts, 1);

  double f;
  double h;
  double h2;
  oracle_soc_logit_z_h2_host(
      z, sol0, negative_branch, D_scaled_mul_x_part,
      D_scaled_squared_part, len, nThread, nBlock, oracle_gpu,
      &f, &h, &h2);
  if (valid_warm && pdcs_soc_logit_converged(
          f, h, z, left, right, sol0, abs_tol, rel_tol)) {
    GRID_SOC_PROFILE_ADD(warm_start_direct_accepts, 1);
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
            f, h, z, left, right, sol0, abs_tol, rel_tol)) {
      GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
      return z;
    }
    double candidate;
    if (!pdcs_soc_logit_candidate(
            z, f, h, h2, left, right, left_f, right_f, use_halley,
            &candidate)) break;
    GRID_SOC_PROFILE_ADD(newton_attempts, 1);
    double candidate_f;
    double candidate_h;
    double candidate_h2;
    oracle_soc_logit_z_h2_host(
        candidate, sol0, negative_branch, D_scaled_mul_x_part,
        D_scaled_squared_part, len, nThread, nBlock, oracle_gpu,
        &candidate_f, &candidate_h, &candidate_h2);
    if (candidate_f > 0.0) {
      right = candidate;
      right_f = candidate_f;
    } else {
      left = candidate;
      left_f = candidate_f;
    }
    if (!isfinite(candidate_f)) break;
    GRID_SOC_PROFILE_ADD(newton_accepts, 1);
    z = candidate;
    f = candidate_f;
    h = candidate_h;
    h2 = candidate_h2;
  }
  if (pdcs_soc_logit_converged(
          f, h, z, left, right, sol0, abs_tol, rel_tol)) {
    GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
    return z;
  }

  bool used_fallback = false;
  int last_updated_side = 0;
  for (int iter = 0; iter < 256 && right - left > rel_tol; ++iter) {
    used_fallback = true;
    GRID_SOC_PROFILE_ADD(bisection_iterations, 1);
    const double denominator = right_f - left_f;
    double candidate = left - left_f * (right - left) / denominator;
    const double guard = fmax(64.0 * 2.220446049250313e-16,
                              1e-6 * (right - left));
    if (!isfinite(candidate) || !isfinite(denominator) ||
        fabs(denominator) <= 1e-300 || candidate <= left + guard ||
        candidate >= right - guard) {
      candidate = 0.5 * (left + right);
      last_updated_side = 0;
    }
    f = oracle_soc_logit_z_f_host(
        candidate, sol0, negative_branch, D_scaled_mul_x_part,
        D_scaled_squared_part, len, nThread, nBlock, oracle_gpu);
    z = candidate;
    if (!isfinite(f) || f == 0.0) break;
    if (f > 0.0) {
      right = candidate;
      right_f = f;
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
    } else {
      left = candidate;
      left_f = f;
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
    }
  }
  if (used_fallback) GRID_SOC_PROFILE_ADD(bisection_events, 1);
  if (f != 0.0) z = 0.5 * (left + right);
  return fmin(fmax(z, -700.0), 700.0);
}
#endif

static double soc_bounded_u_solve_host(
    double sol0, bool increasing, double warm_u,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    long len, int nThread, int nBlock, double* oracle_gpu,
    double endpoint_norm, double abs_tol, double rel_tol) {
  double left = 0.0;
  double right = 1.0;
  const double t_squared = sol0 * sol0;
  const double endpoint_f = endpoint_norm * endpoint_norm - t_squared;
  // Strict endpoint signs are supplied by the failed primal/polar tests.
  double left_f = increasing ? -t_squared : endpoint_f;
  double right_f = increasing ? endpoint_f : -t_squared;
  const bool valid_warm = isfinite(warm_u) && warm_u > left && warm_u < right;
  double u = valid_warm ? warm_u : 0.5;
  if (valid_warm) GRID_SOC_PROFILE_ADD(warm_start_attempts, 1);

  double f;
  double h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
  double h2;
  oracle_soc_bounded_u_h2_host(
      u, sol0, increasing, D_scaled_mul_x_part, D_scaled_squared_part,
      len, nThread, nBlock, oracle_gpu, &f, &h, &h2);
#else
  oracle_soc_bounded_u_h_host(
      u, sol0, increasing, D_scaled_mul_x_part, D_scaled_squared_part,
      len, nThread, nBlock, oracle_gpu, &f, &h);
#endif
  if (valid_warm && pdcs_bounded_soc_projection_converged(
          f, h, u, sol0, abs_tol, rel_tol)) {
    GRID_SOC_PROFILE_ADD(warm_start_direct_accepts, 1);
    return u;
  }
  if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
    right = u;
    right_f = f;
  } else {
    left = u;
    left_f = f;
  }

  bool newton_converged = false;
  for (int iter = 0;
       iter < PDCS_SOC_BOUNDED_NEWTON_STEPS &&
       PDCS_ENABLE_SAFEGUARDED_NEWTON;
       ++iter) {
    if (pdcs_bounded_soc_projection_converged(
            f, h, u, sol0, abs_tol, rel_tol) ||
        pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) {
      newton_converged = true;
      break;
    }
    double candidate;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    if (!pdcs_bounded_soc_candidate(
            u, f, h, h2, left, right, left_f, right_f, &candidate)) break;
#else
    if (!pdcs_bounded_soc_newton_candidate(
            u, f, h, left, right, &candidate)) break;
#endif

    GRID_SOC_PROFILE_ADD(newton_attempts, 1);
#if !PDCS_ENABLE_BOUNDED_SOC_HALLEY
    const double old_abs_f = fabs(f);
#endif
    double candidate_f;
    double candidate_h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    double candidate_h2;
    oracle_soc_bounded_u_h2_host(
        candidate, sol0, increasing, D_scaled_mul_x_part,
        D_scaled_squared_part, len, nThread, nBlock, oracle_gpu,
        &candidate_f, &candidate_h, &candidate_h2);
#else
    oracle_soc_bounded_u_h_host(
        candidate, sol0, increasing, D_scaled_mul_x_part,
        D_scaled_squared_part, len, nThread, nBlock, oracle_gpu,
        &candidate_f, &candidate_h);
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
            candidate_f, candidate_h, candidate, sol0,
            abs_tol, rel_tol)) {
      GRID_SOC_PROFILE_ADD(newton_accepts, 1);
      u = candidate;
      f = candidate_f;
      newton_converged = true;
      break;
    }
    // The candidate is bracket-safe, so retain it even if roundoff prevents
    // a strict residual decrease.  Returning is still controlled only by the
    // residual or bracket width.
#else
    if (fabs(candidate_f) >= old_abs_f) break;
#endif
    GRID_SOC_PROFILE_ADD(newton_accepts, 1);
    u = candidate;
    f = candidate_f;
    h = candidate_h;
#if PDCS_ENABLE_BOUNDED_SOC_HALLEY
    h2 = candidate_h2;
#endif
    if (pdcs_bounded_soc_projection_converged(
            f, h, u, sol0, abs_tol, rel_tol) ||
        pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) {
      newton_converged = true;
      break;
    }
  }
  if (newton_converged ||
      pdcs_bounded_soc_projection_converged(
          f, h, u, sol0, abs_tol, rel_tol) ||
      pdcs_bounded_soc_bracket_converged(left, right, rel_tol)) {
    GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
    return u;
  }

  // Illinois regula falsi keeps the inexpensive F-only oracle in the
  // well-conditioned region. Near an endpoint it uses the fused F+F' oracle
  // so the recovered projection, rather than only the root residual, controls
  // termination. If interpolation becomes unsafe, a strict midpoint is used.
  bool used_fallback = false;
  int last_updated_side = 0;  // -1: left, +1: right.
  for (int iter = 0;
       iter < 128 && !pdcs_bounded_soc_bracket_converged(
           left, right, rel_tol);
       ++iter) {
    used_fallback = true;
    GRID_SOC_PROFILE_ADD(bisection_iterations, 1);
    const double denominator = right_f - left_f;
    double candidate = left - left_f * (right - left) / denominator;
    const double guard = fmax(
        pdcs_bounded_soc_coordinate_resolution(0.5 * (left + right)),
        1e-6 * (right - left));
    if (!isfinite(candidate) || !isfinite(denominator) ||
        fabs(denominator) <= 1e-300 || candidate <= left + guard ||
        candidate >= right - guard) {
      candidate = pdcs_bounded_soc_bisection_midpoint(left, right);
      last_updated_side = 0;
    }
    if (pdcs_bounded_soc_near_endpoint(candidate, rel_tol)) {
      oracle_soc_bounded_u_h_host(
          candidate, sol0, increasing, D_scaled_mul_x_part,
          D_scaled_squared_part, len, nThread, nBlock, oracle_gpu, &f, &h);
    } else {
      f = oracle_soc_bounded_u_f_host(
          candidate, sol0, increasing, D_scaled_mul_x_part,
          D_scaled_squared_part, len, nThread, nBlock, oracle_gpu);
    }
    u = candidate;
    if (!isfinite(f) ||
        pdcs_bounded_soc_projection_converged(
            f, h, u, sol0, abs_tol, rel_tol)) break;

    const bool update_right =
        (increasing && f > 0.0) || (!increasing && f < 0.0);
    if (update_right) {
      right = candidate;
      right_f = f;
      if (last_updated_side == 1) left_f *= 0.5;
      last_updated_side = 1;
    } else {
      left = candidate;
      left_f = f;
      if (last_updated_side == -1) right_f *= 0.5;
      last_updated_side = -1;
    }
  }
  if (used_fallback) GRID_SOC_PROFILE_ADD(bisection_events, 1);
  if (!isfinite(f) || !pdcs_bounded_soc_projection_converged(
          f, h, u, sol0, abs_tol, rel_tol)) {
    u = pdcs_bounded_soc_bisection_midpoint(left, right);
  }
  return fmin(fmax(u, 64.0 * 2.220446049250313e-16),
              1.0 - 64.0 * 2.220446049250313e-16);
}

static bool soc_safeguarded_candidate_host(
    double x, double f, double h, double left, double right,
    bool increasing, double rel_tol, double* candidate) {
  double width = right - left;
  if (!isfinite(f) || !isfinite(h) || fabs(h) <= 1e-18 || width <= 0.0) {
    return false;
  }
#if PDCS_SOC_COORDINATE_MODE == 0
  *candidate = x - f / h;
  double guard = fmax(rel_tol, 1e-8 * width);
  return isfinite(*candidate) && *candidate > left + guard &&
         *candidate < right - guard;
#else
  double shift = pdcs_soc_coordinate_shift(rel_tol);
  bool use_log_coordinate = true;
#if PDCS_SOC_COORDINATE_MODE == 3
  use_log_coordinate = pdcs_soc_newton_needs_log_coordinate(
      x, left, right, increasing, shift);
#endif
  if (!use_log_coordinate) {
    *candidate = x - f / h;
    double guard = fmax(rel_tol, 1e-8 * width);
    return isfinite(*candidate) && *candidate > left + guard &&
           *candidate < right - guard;
  }
  double u = pdcs_soc_root_to_coordinate(x, increasing, shift);
  double u_left = pdcs_soc_root_to_coordinate(left, increasing, shift);
  double u_right = pdcs_soc_root_to_coordinate(right, increasing, shift);
  double derivative_u = h * pdcs_soc_root_coordinate_derivative(
      x, increasing, shift);
  if (!isfinite(derivative_u) || fabs(derivative_u) <= 1e-18) return false;
  double candidate_u = u - f / derivative_u;
  *candidate = pdcs_soc_coordinate_to_root(
      candidate_u, increasing, shift);
  double u_guard = 1e-8 * (u_right - u_left);
  return isfinite(candidate_u) && isfinite(*candidate) &&
         candidate_u > u_left + u_guard &&
         candidate_u < u_right - u_guard &&
         *candidate > left && *candidate < right;
#endif
}

static bool soc_post_bracket_newton_host(
    cublasHandle_t handle, double sol0, double* xi_gpu,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    double* temp_part, long* len_cpu, long* len_gpu, int nThread,
    int nBlock, double* oracle_gpu, bool increasing, double* left,
    double* right, double* xi, double abs_tol, double rel_tol) {
#if !PDCS_ENABLE_SAFEGUARDED_NEWTON || PDCS_SOC_POST_BRACKET_NEWTON_STEPS <= 0
  return false;
#else
  if (!isfinite(*left) || !isfinite(*right) || !(*left < *right)) {
    return false;
  }

  double x = pdcs_soc_bisection_midpoint(
      *left, *right, increasing, rel_tol);
  double f;
  double h;
  oracle_soc_h_host(handle, x, sol0, xi_gpu, D_scaled_mul_x_part,
      D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread,
      nBlock, oracle_gpu, &f, &h);

  for (int iter = 0; iter < PDCS_SOC_POST_BRACKET_NEWTON_STEPS; ++iter) {
    if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
      *right = x;
    } else {
      *left = x;
    }
    double normalized_width = (*right - *left) /
                              (1.0 + *right + *left);
    if (fabs(f) <= abs_tol * abs_tol || normalized_width <= rel_tol) {
      *xi = x;
      return true;
    }

    double candidate;
    if (!soc_safeguarded_candidate_host(
            x, f, h, *left, *right, increasing, rel_tol, &candidate)) {
      break;
    }
    GRID_SOC_PROFILE_ADD(newton_attempts, 1);
    double candidate_f;
    double candidate_h;
    oracle_soc_h_host(handle, candidate, sol0, xi_gpu,
        D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
        len_gpu, nThread, nBlock, oracle_gpu, &candidate_f, &candidate_h);

    // Even a rejected Newton point has a valid sign and can tighten the
    // bracket.  Stop spending derivative work when it does not improve the
    // residual; the caller then continues with guaranteed bisection.
    if (fabs(candidate_f) >= fabs(f)) {
      if ((increasing && candidate_f > 0.0) ||
          (!increasing && candidate_f < 0.0)) {
        *right = candidate;
      } else {
        *left = candidate;
      }
      break;
    }
    GRID_SOC_PROFILE_ADD(newton_accepts, 1);
    double normalized_step = fabs(candidate - x) /
                             (1.0 + fabs(candidate) + fabs(x));
    x = candidate;
    f = candidate_f;
    h = candidate_h;
    if (normalized_step <= rel_tol || fabs(f) <= abs_tol * abs_tol) {
      *xi = x;
      return true;
    }
  }

  // Commit the final accepted point to the bracket.  The old implementation
  // only did this at the start of the next loop iteration, so its last useful
  // Newton step was discarded whenever the step budget was exhausted.
  if ((increasing && f > 0.0) || (!increasing && f < 0.0)) {
    *right = x;
  } else {
    *left = x;
  }
  double normalized_width = (*right - *left) /
                            (1.0 + *right + *left);
  if (fabs(f) <= abs_tol * abs_tol || normalized_width <= rel_tol) {
    *xi = x;
    return true;
  }
  return false;
#endif
}

static bool soc_exponent_expansion_bracket_host(
    cublasHandle_t handle, double sol0, double* xi_gpu,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    double* temp_part, long* len_cpu, long* len_gpu, int nThread,
    int nBlock, double* oracle_gpu, double* left, double* right,
    double* f, double abs_tol) {
#if !PDCS_ENABLE_EXPONENT_EXPANSION
  return false;
#else
  *f = oracle_soc_f_host(handle, *right, sol0, xi_gpu,
      D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
      len_gpu, nThread, nBlock, oracle_gpu);
  if (!isfinite(*f)) return false;
  if (fabs(*f) <= abs_tol) {
    *left = *right;
    return true;
  }
  if (*f >= 0.0) return true;
  double base_q = *right - 0.5;
  if (!(base_q > 0.0) || !isfinite(base_q)) return false;
  int low_exponent = 0;
  int high_exponent = 1;
  bool found = false;
  while (high_exponent <= 1020) {
    double candidate = 0.5 + ldexp(base_q, high_exponent);
    if (!isfinite(candidate)) break;
    double candidate_f = oracle_soc_f_host(handle, candidate, sol0, xi_gpu,
        D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
        len_gpu, nThread, nBlock, oracle_gpu);
    if (!isfinite(candidate_f)) break;
    if (fabs(candidate_f) <= abs_tol) {
      *left = candidate;
      *right = candidate;
      *f = candidate_f;
      return true;
    }
    if (candidate_f >= 0.0) {
      *f = candidate_f;
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
    double candidate_f = oracle_soc_f_host(handle, candidate, sol0, xi_gpu,
        D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
        len_gpu, nThread, nBlock, oracle_gpu);
    if (!isfinite(candidate_f)) return false;
    if (fabs(candidate_f) <= abs_tol) {
      *left = candidate;
      *right = candidate;
      *f = candidate_f;
      return true;
    }
    if (candidate_f >= 0.0) {
      high_exponent = mid_exponent;
      *f = candidate_f;
    }
    else low_exponent = mid_exponent;
  }
  *left = 0.5 + ldexp(base_q, low_exponent);
  *right = 0.5 + ldexp(base_q, high_exponent);
  return true;
#endif
}

__global__ void recover_sol_case01(double* __restrict__ sol, double* __restrict__ t_warm_start, double* __restrict__ D_scaled_squared, long* __restrict__ n) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j == 0){
    sol[0] = sol[0] / (1 - 2 * t_warm_start[0]) * minVal;
  }
  if (j > 0 && j < *n){
    sol[j] = sol[j] / (1 + 2 * t_warm_start[0] * D_scaled_squared[j]) * minVal;
  }
}

__global__ void recover_sol_bounded_u(
    double* __restrict__ sol, double u, bool increasing,
    double* __restrict__ D_scaled_squared, long* __restrict__ n) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j == 0) {
    if (increasing) {
      // t < 0 and u = 1 - (2 lambda)^(-1).
      sol[0] = -sol[0] * (1.0 - u) / u * minVal;
    } else {
      // t > 0 and u = 2 lambda.
      sol[0] = sol[0] / (1.0 - u) * minVal;
    }
  }
  if (j > 0 && j < *n) {
    const double c = D_scaled_squared[j];
    if (increasing) {
      sol[j] = sol[j] * (1.0 - u) / (c + 1.0 - u) * minVal;
    } else {
      sol[j] = sol[j] / (1.0 + c * u) * minVal;
    }
  }
}

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
__global__ void recover_sol_logit_z(
    double* __restrict__ sol, double z, bool negative_branch,
    double* __restrict__ D_scaled_squared, long* __restrict__ n) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  double s;
  double v;
  pdcs_soc_logistic_pair(z, &s, &v);
  if (j == 0) {
    sol[0] = negative_branch ? -sol[0] * v / s * minVal
                             : sol[0] / s * minVal;
  }
  if (j > 0 && j < *n) {
    const double c = D_scaled_squared[j];
    sol[j] = negative_branch ? sol[j] * v / (c + v) * minVal
                             : sol[j] / (1.0 + c * v) * minVal;
  }
}
#endif

__global__ void binary_search_case0(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu,  double* __restrict__ oracleVal_gpu, double *t_warm_start_gpu, bool* __restrict__ d_auxiliary_flag, bool* __restrict__ d_return_flag, double abs_tol, double rel_tol) {
  if (*d_auxiliary_flag){
    *xiRight_gpu = *t_warm_start_gpu;
  }else{
    *xiLeft_gpu = *t_warm_start_gpu;
  }
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void binary_search_case1(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, double *t_warm_start_gpu, bool* __restrict__ d_auxiliary_flag, bool* __restrict__ d_return_flag, double abs_tol, double rel_tol) {
  // For t < 0, f(xi) is increasing on (1/2, +inf): a negative value
  // means the root is to the right of the current midpoint.
  if (*d_auxiliary_flag){
    *xiLeft_gpu = *t_warm_start_gpu;
  }else{
    *xiRight_gpu = *t_warm_start_gpu;
  }
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void average_xi(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ t_warm_start_gpu) {
  *t_warm_start_gpu = (*xiLeft_gpu + *xiRight_gpu) / 2;
}

__global__ void end_while_loop(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, double rel_tol, double abs_tol) {
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void enlarge_xi_right(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu) {
    *xiLeft_gpu = *xiRight_gpu;
    *xiRight_gpu *= 2;
}

__global__ void recover_sol_case3(double* __restrict__ sol_gpu, double* __restrict__ temp_gpu, double* __restrict__ D_scaled_gpu, long* __restrict__ n_gpu, double* __restrict__ t_warm_start_gpu, double* __restrict__ D_scaled_squared_gpu) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j > 0 && j < *n_gpu){
    sol_gpu[j] = sol_gpu[j] / (1 + D_scaled_squared_gpu[j]) * minVal;
    temp_gpu[j] = D_scaled_gpu[j] * sol_gpu[j];
  }
}

extern "C" void soc_proj_diagonal(cublasHandle_t handle,
                                 double* sol_gpu,
                                 long* len_gpu,
                                 long* n_gpu,
                                 long* len_cpu,
                                 long* n_cpu,
                                 double* D_scaled_gpu,
                                 double* D_scaled_squared_gpu,
                                 double* D_scaled_mul_x_gpu,
                                 double* temp_gpu,
                                 double* t_warm_start_gpu,
                                 int nThread,
                                 int nBlock,
                                 bool* d_return_flag,
                                 bool* d_auxiliary_flag,
                                 double abs_tol,
                                 double rel_tol) {
  GRID_SOC_PROFILE_ADD(projection_events, 1);
  // This is deliberately a cuBLAS implementation: Dnrm2 performs the
  // large-vector reductions while small kernels only evaluate/recover the
  // scalar root.  The temporary root bounds must not alias D_scaled[0] or
  // D_scaled_squared[0], which are persistent solver scaling data.
  const int scale_blocks = (*n_cpu + nThread - 1) / nThread;
  scale_vector<<<scale_blocks, nThread>>>(sol_gpu, *n_cpu, minVal_inv);

  double* x_tail = sol_gpu + 1;
  double* d_tail = D_scaled_gpu + 1;
  double* d_squared_tail = D_scaled_squared_gpu + 1;
  double* d_times_x_tail = D_scaled_mul_x_gpu + 1;
  double* work_tail = temp_gpu + 1;
  double* oracle = temp_gpu;
#if !PDCS_ENABLE_GRID_SOC_FASTPATH
  bool host_flag = false;
#endif

#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  thrust::device_ptr<double> x_thrust = thrust::device_pointer_cast(x_tail);
  thrust::device_ptr<double> d_thrust = thrust::device_pointer_cast(d_tail);
  thrust::device_ptr<double> d_times_x_thrust = thrust::device_pointer_cast(d_times_x_tail);
  thrust::device_ptr<double> work_thrust = thrust::device_pointer_cast(work_tail);
#endif

  double sol0;
  pdcs_record_cuda(cudaMemcpy(
      &sol0, sol_gpu, sizeof(double), cudaMemcpyDeviceToHost));

  // Test the polar cone with ||x ./ D|| and feasibility with ||D .* x||.
#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  GRID_SOC_PROFILE_ADD(vector_reductions, 1);
  pdcs_record_cuda(cudaMemset(oracle, 0, 2 * sizeof(double)));
  int warp_count = (nThread + 31) / 32;
  soc_initial_norm_pair_kernel<<<nBlock, nThread,
      2 * warp_count * sizeof(double)>>>(
      x_tail, d_tail, d_times_x_tail, *len_cpu, oracle);
  double initial_sums[2];
  pdcs_record_cuda(cudaMemcpy(
      initial_sums, oracle, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  double polar_norm = sqrt(fmax(initial_sums[0], 0.0));
  double weighted_norm = sqrt(fmax(initial_sums[1], 0.0));
#else
  thrust::transform(x_thrust, x_thrust + *len_cpu, d_thrust, work_thrust,
                    thrust::divides<double>());
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, work_tail, 1, oracle));
#endif
#if PDCS_ENABLE_GRID_SOC_FASTPATH
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double polar_norm;
  pdcs_record_cuda(cudaMemcpy(
      &polar_norm, oracle, sizeof(double), cudaMemcpyDeviceToHost));
#endif
  if (polar_norm <= -sol0 && sol0 <= 0.0) {
    GRID_SOC_PROFILE_ADD(polar_events, 1);
    pdcs_record_cuda(cudaMemset(sol_gpu, 0, *n_cpu * sizeof(double)));
    return;
  }
#else
  pdcs_record_cuda(cudaMemset(d_return_flag, 0, sizeof(bool)));
  soc_cone_dual<<<nBlock, nThread>>>(sol_gpu, n_gpu, oracle, d_return_flag);
  pdcs_record_cuda(cudaMemcpy(
      &host_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost));
  if (host_flag) return;
#endif

#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  thrust::transform(d_thrust, d_thrust + *len_cpu, x_thrust,
                    d_times_x_thrust, thrust::multiplies<double>());
  pdcs_record_cublas(
      cublasDnrm2_v2(handle, *len_cpu, d_times_x_tail, 1, oracle));
#endif
#if PDCS_ENABLE_GRID_SOC_FASTPATH
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double weighted_norm;
  pdcs_record_cuda(cudaMemcpy(
      &weighted_norm, oracle, sizeof(double), cudaMemcpyDeviceToHost));
#endif
  if (weighted_norm <= sol0) {
    GRID_SOC_PROFILE_ADD(interior_events, 1);
    scale_vector<<<scale_blocks, nThread>>>(sol_gpu, *n_cpu, minVal);
    return;
  }
#else
  pdcs_record_cuda(cudaMemset(d_return_flag, 0, sizeof(bool)));
  soc_cone_heuristic<<<1, 1>>>(sol_gpu, oracle, d_return_flag);
  pdcs_record_cuda(cudaMemcpy(
      &host_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost));
  if (host_flag) {
    scale_vector<<<scale_blocks, nThread>>>(sol_gpu, *n_cpu, minVal);
    return;
  }
#endif

  double warm_x;
  pdcs_record_cuda(cudaMemcpy(
      &warm_x, t_warm_start_gpu, sizeof(double), cudaMemcpyDeviceToHost));

#if PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT
  if (PDCS_ENABLE_GRID_SOC_LOGIT_ROOT &&
      *n_cpu >= PDCS_SOC_LOGIT_MIN_DIMENSION &&
      (sol0 >= rel_tol || sol0 <= -rel_tol)) {
    GRID_SOC_PROFILE_ADD(root_events, 1);
    const bool negative_branch = sol0 < 0.0;
    if (negative_branch) {
      GRID_SOC_PROFILE_ADD(negative_root_events, 1);
    } else {
      GRID_SOC_PROFILE_ADD(positive_root_events, 1);
    }
    const double z = soc_logit_z_solve_host(
        sol0, negative_branch, warm_x, d_times_x_tail, d_squared_tail,
        *len_cpu, nThread, nBlock, oracle,
        negative_branch ? polar_norm : weighted_norm, abs_tol, rel_tol);
    pdcs_record_cuda(cudaMemcpy(
        t_warm_start_gpu, &z, sizeof(double), cudaMemcpyHostToDevice));
    recover_sol_logit_z<<<nBlock, nThread>>>(
        sol_gpu, z, negative_branch, D_scaled_squared_gpu, n_gpu);
    return;
  }
#endif
#if PDCS_ENABLE_BOUNDED_SOC_ROOT
  if (sol0 >= rel_tol || sol0 <= -rel_tol) {
    GRID_SOC_PROFILE_ADD(root_events, 1);
    const bool increasing = sol0 < 0.0;
    if (increasing) {
      GRID_SOC_PROFILE_ADD(negative_root_events, 1);
    } else {
      GRID_SOC_PROFILE_ADD(positive_root_events, 1);
    }
    const double u = soc_bounded_u_solve_host(
        sol0, increasing, warm_x, d_times_x_tail, d_squared_tail,
        *len_cpu, nThread, nBlock, oracle,
        increasing ? polar_norm : weighted_norm, abs_tol, rel_tol);
    pdcs_record_cuda(cudaMemcpy(
        t_warm_start_gpu, &u, sizeof(double), cudaMemcpyHostToDevice));
    recover_sol_bounded_u<<<nBlock, nThread>>>(
        sol_gpu, u, increasing, D_scaled_squared_gpu, n_gpu);
    return;
  }
#endif

  if (sol0 >= rel_tol) {
    GRID_SOC_PROFILE_ADD(root_events, 1);
    GRID_SOC_PROFILE_ADD(positive_root_events, 1);
    double left = 0.0;
    double right = 0.5;
    double f = 1.0;
    if (warm_x > left && warm_x < right) {
      GRID_SOC_PROFILE_ADD(warm_start_attempts, 1);
      double h;
      oracle_soc_h_host(handle, warm_x, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle, &f, &h);
      bool warm_direct = fabs(f) <= abs_tol * abs_tol;
      if (warm_direct) {
        GRID_SOC_PROFILE_ADD(warm_start_direct_accepts, 1);
      }
      double x = warm_x;
      int accepted_newton_steps = 0;
      for (int iter = 0;
           iter < PDCS_SOC_NEWTON_STEPS && PDCS_ENABLE_SAFEGUARDED_NEWTON;
           ++iter) {
        if (f < 0.0) right = x; else left = x;
        if (fabs(f) <= abs_tol * abs_tol) {
          left = x;
          right = x;
          break;
        }
        double candidate;
        if (!soc_safeguarded_candidate_host(
                x, f, h, left, right, false, rel_tol, &candidate)) break;
        GRID_SOC_PROFILE_ADD(newton_attempts, 1);
        double old_abs_f = fabs(f);
        x = candidate;
        oracle_soc_h_host(handle, x, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle, &f, &h);
        if (fabs(f) >= old_abs_f) {
          if (f < 0.0) right = x; else left = x;
          break;
        }
        GRID_SOC_PROFILE_ADD(newton_accepts, 1);
        ++accepted_newton_steps;
      }
      if (!warm_direct && accepted_newton_steps > 0 && left == right) {
        GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
      }
    }
    if (left != right) f = 1.0;
    double xi = 0.5 * (left + right);
    bool newton_converged = soc_post_bracket_newton_host(
        handle, sol0, t_warm_start_gpu, d_times_x_tail, d_squared_tail,
        work_tail, len_cpu, len_gpu, nThread, nBlock, oracle, false,
        &left, &right, &xi, abs_tol, rel_tol);
    if (newton_converged) {
      GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
      f = 0.0;
    } else {
      xi = 0.5 * (left + right);
      f = 1.0;
    }
    bool used_bisection = false;
    while ((right - left) / (1.0 + right + left) > rel_tol &&
           fabs(f) > abs_tol) {
      used_bisection = true;
      GRID_SOC_PROFILE_ADD(bisection_iterations, 1);
      xi = pdcs_soc_bisection_midpoint(left, right, false, rel_tol);
      f = oracle_soc_f_host(handle, xi, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      if (f < 0.0) right = xi; else left = xi;
    }
    if (used_bisection) GRID_SOC_PROFILE_ADD(bisection_events, 1);
    pdcs_record_cuda(cudaMemcpy(
        t_warm_start_gpu, &xi, sizeof(double), cudaMemcpyHostToDevice));
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu,
                                              D_scaled_squared_gpu, n_gpu);
  } else if (sol0 <= -rel_tol) {
    GRID_SOC_PROFILE_ADD(root_events, 1);
    GRID_SOC_PROFILE_ADD(negative_root_events, 1);
    double left = 0.5, right = 1.0;
    double f = 1.0;
    if (warm_x > left) {
      GRID_SOC_PROFILE_ADD(warm_start_attempts, 1);
      double h;
      oracle_soc_h_host(handle, warm_x, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle, &f, &h);
      bool warm_direct = fabs(f) <= abs_tol * abs_tol;
      if (warm_direct) {
        GRID_SOC_PROFILE_ADD(warm_start_direct_accepts, 1);
      }
      double x = warm_x;
      int accepted_newton_steps = 0;
      for (int iter = 0;
           iter < PDCS_SOC_NEWTON_STEPS && PDCS_ENABLE_SAFEGUARDED_NEWTON;
           ++iter) {
        if (f > 0.0) right = x; else left = x;
        if (fabs(f) <= abs_tol * abs_tol) {
          left = x;
          right = x;
          break;
        }
        double candidate;
        if (!soc_safeguarded_candidate_host(
                x, f, h, left, right, true, rel_tol, &candidate)) break;
        GRID_SOC_PROFILE_ADD(newton_attempts, 1);
        double old_abs_f = fabs(f);
        x = candidate;
        oracle_soc_h_host(handle, x, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle, &f, &h);
        if (fabs(f) >= old_abs_f) {
          if (f > 0.0) right = x; else left = x;
          break;
        }
        GRID_SOC_PROFILE_ADD(newton_accepts, 1);
        ++accepted_newton_steps;
      }
      if (!warm_direct && accepted_newton_steps > 0 && left == right) {
        GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
      }
    }
    bool exponent_bracketed = soc_exponent_expansion_bracket_host(
        handle, sol0, t_warm_start_gpu, d_times_x_tail, d_squared_tail,
        work_tail, len_cpu, len_gpu, nThread, nBlock, oracle, &left,
        &right, &f, abs_tol);
    if (!exponent_bracketed) {
      f = oracle_soc_f_host(handle, right, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      while (f < 0.0) {
        left = right;
        right *= 2.0;
        f = oracle_soc_f_host(handle, right, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle);
      }
    }
    double xi = 0.5 * (left + right);
    bool newton_converged = soc_post_bracket_newton_host(
        handle, sol0, t_warm_start_gpu, d_times_x_tail, d_squared_tail,
        work_tail, len_cpu, len_gpu, nThread, nBlock, oracle, true,
        &left, &right, &xi, abs_tol, rel_tol);
    if (newton_converged) {
      GRID_SOC_PROFILE_ADD(newton_converged_events, 1);
      f = 0.0;
    } else {
      xi = 0.5 * (left + right);
      f = 1.0;
    }
    bool used_bisection = false;
    while ((right - left) / (1.0 + right + left) > rel_tol &&
           fabs(f) > abs_tol) {
      used_bisection = true;
      GRID_SOC_PROFILE_ADD(bisection_iterations, 1);
      xi = pdcs_soc_bisection_midpoint(left, right, true, rel_tol);
      f = oracle_soc_f_host(handle, xi, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      if (f > 0.0) right = xi; else left = xi;
    }
    if (used_bisection) GRID_SOC_PROFILE_ADD(bisection_events, 1);
    pdcs_record_cuda(cudaMemcpy(
        t_warm_start_gpu, &xi, sizeof(double), cudaMemcpyHostToDevice));
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu,
                                              D_scaled_squared_gpu, n_gpu);
  } else {
    recover_sol_case3<<<nBlock, nThread>>>(sol_gpu, temp_gpu, D_scaled_gpu,
                                            n_gpu, t_warm_start_gpu,
                                            D_scaled_squared_gpu);
    pdcs_record_cublas(
        cublasDnrm2_v2(handle, *len_cpu, temp_gpu + 1, 1, sol_gpu));
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

extern "C" int few_block_proj(cublasHandle_t handle,
                             double* arr, 
                             double* bl, 
                             double* bu, 
                             double* D_scaled,  
                             double* D_scaled_squared,  
                             double* D_scaled_mul_x, 
                             double* temp, 
                             double* t_warm_start, 
                             long* cpu_head_start,  
                             long* ns_gpu, 
                             long* ns_cpu, 
                             int blkNum, 
                             long* cpu_proj_type,  
                             int ThreadPerBlock, 
                             int nBlock,
                             double abs_tol,
                             double rel_tol)
{
  if (blkNum == 0) return 0;
  if (handle == nullptr || arr == nullptr || bl == nullptr || bu == nullptr ||
      D_scaled == nullptr || D_scaled_squared == nullptr ||
      D_scaled_mul_x == nullptr || temp == nullptr ||
      t_warm_start == nullptr || cpu_head_start == nullptr ||
      ns_gpu == nullptr || ns_cpu == nullptr || cpu_proj_type == nullptr ||
      blkNum < 0 || ThreadPerBlock <= 0 || nBlock <= 0 ||
      !isfinite(abs_tol) || abs_tol <= 0.0 ||
      !isfinite(rel_tol) || rel_tol <= 0.0) {
    return PDCS_GRIDWISE_INVALID_ARGUMENT;
  }

  // Reject unknown cone codes before modifying any output.
  for (int i = 0; i < blkNum; ++i) {
    const long projection_type = cpu_proj_type[i];
    if (projection_type < 0 || projection_type > 29) {
      return PDCS_GRIDWISE_INVALID_ARGUMENT;
    }
  }

  pdcs_gridwise_cuda_status = 0;
  pdcs_gridwise_cublas_status = 0;
  // Clear a prior CUDA runtime error so this invocation reports only its own
  // work. All native calls are serialized by the Julia wrapper.
  cudaGetLastError();
  for (int i = 0; i < blkNum; ++i)
  {
    long *n_gpu = ns_gpu+i;
    long n_cpu = ns_cpu[i];
    long head_start = cpu_head_start[i];
    double *sol = arr + head_start;
    if (cpu_proj_type[i] == 0 || cpu_proj_type[i] == 1){
      // dual_free_proj
      ;
    }
    else if (cpu_proj_type[i] == 17 || cpu_proj_type[i] == 19 || cpu_proj_type[i] == 18){
      // box
      double *sub_bl = bl + head_start;
      double *sub_bu = bu + head_start;
      box_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, sub_bl, sub_bu, n_gpu);
    }
    else if (cpu_proj_type[i] == 2){
      // zeros<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
      pdcs_record_cuda(cudaMemset(sol, 0, n_cpu * sizeof(double)));
    }
    else if (cpu_proj_type[i] == 3 || cpu_proj_type[i] == 4){
      // dual_positive
      positive_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
    }
    else if (cpu_proj_type[i] == 5 || cpu_proj_type[i] == 7 || cpu_proj_type[i] == 20 || cpu_proj_type[i] == 21){
      long len_cpu = n_cpu - 1;
      double *sub_temp = temp + head_start;
      // The Julia wrapper guarantees that `temp` does not alias `arr`.
      // Preserve the cone's warm-start value for diagonal projections.
      soc_proj(handle, sol, &n_cpu, n_gpu, &len_cpu, sub_temp,
               ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 6 || cpu_proj_type[i] == 22){
      long len_cpu = n_cpu - 1;
      double *sub_D_scaled = D_scaled + head_start;
      double *sub_D_scaled_squared = D_scaled_squared + head_start;
      double *sub_D_scaled_mul_x = D_scaled_mul_x + head_start;
      double *sub_temp = temp + head_start;
#if PDCS_ENABLE_GRID_SOC_FASTPATH
      bool* d_auxiliary_flag = nullptr;
      bool* d_return_flag = nullptr;
      long* len_gpu = nullptr;
#else
      bool* d_auxiliary_flag;
      bool h_auxiliary_flag = false;
      pdcs_record_cuda(cudaMalloc(&d_auxiliary_flag, sizeof(bool)));
      pdcs_record_cuda(cudaMemcpy(
          d_auxiliary_flag, &h_auxiliary_flag, sizeof(bool),
          cudaMemcpyHostToDevice));
      bool* d_return_flag;
      bool h_return_flag = false;
      pdcs_record_cuda(cudaMalloc(&d_return_flag, sizeof(bool)));
      pdcs_record_cuda(cudaMemcpy(
          d_return_flag, &h_return_flag, sizeof(bool),
          cudaMemcpyHostToDevice));
      long* len_gpu;
      pdcs_record_cuda(cudaMalloc(&len_gpu, sizeof(long)));
      pdcs_record_cuda(cudaMemcpy(
          len_gpu, &len_cpu, sizeof(long), cudaMemcpyHostToDevice));
#endif
      soc_proj_diagonal(handle,
                       sol, 
                       len_gpu,
                       n_gpu, 
                       &len_cpu,
                       &n_cpu,
                       sub_D_scaled, 
                       sub_D_scaled_squared, 
                       sub_D_scaled_mul_x, 
                       sub_temp, 
                       &t_warm_start[i], 
                       ThreadPerBlock, 
                       nBlock, 
                       d_return_flag, 
                       d_auxiliary_flag,
                       abs_tol,
                       rel_tol);
#if !PDCS_ENABLE_GRID_SOC_FASTPATH
      pdcs_record_cuda(cudaFree(len_gpu));
      pdcs_record_cuda(cudaFree(d_return_flag));
      pdcs_record_cuda(cudaFree(d_auxiliary_flag));
#endif
    }
    else if (cpu_proj_type[i] == 8 || cpu_proj_type[i] == 10 || cpu_proj_type[i] == 23 || cpu_proj_type[i] == 24){
      printf("use cublas for rsoc projection is developing!\n");
    //   rsoc_proj(handle, sol, &n, sub_D_scaled_mul_x, sub_temp, ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 9 || cpu_proj_type[i] == 25){
      printf("use cublas for rsoc diagonal projection is developing!\n");
    //   rsoc_proj_diagonal(handle, sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 11 || cpu_proj_type[i] == 16 || cpu_proj_type[i] == 28){
      // dualExponent_proj
      dualExponent_proj_kernel<<<1, 1>>>(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 14 || cpu_proj_type[i] == 13 || cpu_proj_type[i] == 26 ){
      // exponent_proj
      exponent_proj_kernel<<<1, 1>>>(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 12 || cpu_proj_type[i] == 29){
      // dualExponent_proj_diagonal
      double *sub_D_scaled = D_scaled + head_start;
      double *sub_temp = temp + head_start;
      dualExponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_D_scaled, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 15 || cpu_proj_type[i] == 27){
      // exponent_proj_diagonal
      double *sub_D_scaled = D_scaled + head_start;
      double *sub_temp = temp + head_start;
      invert_exp_diagonal<<<1, 3>>>(sub_D_scaled, sub_temp);
      exponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
  }
  pdcs_record_cuda(cudaDeviceSynchronize());
  if (pdcs_gridwise_cublas_status != 0) {
    return PDCS_GRIDWISE_CUBLAS_STATUS_BASE + pdcs_gridwise_cublas_status;
  }
  if (pdcs_gridwise_cuda_status != 0) {
    return PDCS_GRIDWISE_CUDA_STATUS_BASE + pdcs_gridwise_cuda_status;
  }
  return 0;
}
