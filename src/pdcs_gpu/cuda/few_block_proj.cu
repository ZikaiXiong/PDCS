#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>


#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
#define minVal 1e-3
#define minVal_inv 1e+3

#include "exp_proj_kernel.cu"

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

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  // temp for storing the norm of the vector
  cublasDnrm2(handle, *len_cpu, sol + 1, 1, temp);
  soc_proj_scale_kernel<<<1, 1>>>(sol, temp, n_gpu);
  cublasDscal(handle, *n_cpu, temp, sol, 1);


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
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu);
  oracle_soc_f_sqrt_final_case0<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
}

extern "C" void oracle_soc_f_sqrt_case1(cublasHandle_t handle, double *xi, double *x, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len_cpu, long *len_gpu, int nThread, int nBlock,  double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
  oracle_soc_f_sqrt_kernel<<<nBlock, nThread>>>(xi, x, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_gpu, oracleVal_gpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu);
  oracle_soc_f_sqrt_final_case1<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
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
  if (*d_auxiliary_flag){
    *xiRight_gpu = *t_warm_start_gpu;
  }else{
    *xiLeft_gpu = *t_warm_start_gpu;
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
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  double scale_factor_inv = minVal_inv;
  cublasDscal_v2(handle, *n_cpu, &scale_factor_inv, sol_gpu, 1);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  double *x2end = sol_gpu + 1;
  double *D_scaled_part = D_scaled_gpu + 1;
  double *temp_part = temp_gpu + 1;
  double *D_scaled_mul_x_part = D_scaled_mul_x_gpu + 1;
  double *D_scaled_squared_part = D_scaled_squared_gpu + 1;
  bool h_return_flag = false;
  bool h_auxiliary_flag = false;
  double *xiRight_gpu = D_scaled_gpu;
  double *xiLeft_gpu = D_scaled_squared_gpu;
  double *oracleVal_gpu = temp_gpu;

  thrust::device_ptr<double> x2end_thrust = thrust::device_pointer_cast(x2end);
  thrust::device_ptr<double> temp_thrust = thrust::device_pointer_cast(temp_part);
  thrust::device_ptr<double> D_scaled_part_thrust = thrust::device_pointer_cast(D_scaled_part);
  thrust::device_ptr<double> D_scaled_mul_x_part_thrust = thrust::device_pointer_cast(D_scaled_mul_x_part);
  thrust::device_ptr<double> D_scaled_squared_part_thrust = thrust::device_pointer_cast(D_scaled_squared_part);

  thrust::transform(x2end_thrust, x2end_thrust + *len_cpu, D_scaled_part_thrust, temp_thrust, thrust::divides<double>());
  // vvrscl<<<nBlock, nThread>>>(len_gpu, x2end, D_scaled_part, temp_part);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu);

  soc_cone_dual<<<nBlock, nThread>>>(sol_gpu, n_gpu, oracleVal_gpu, d_return_flag);
  cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if (h_return_flag){
    // printf("soc_cone_dual return\n");
    return;
  }
  // // // D_scaled_mul_x_part = D_scaled_part .* x2end
  // vvscal<<<nBlock, nThread>>>(len_gpu, D_scaled_part, x2end, D_scaled_mul_x_part);
  thrust::transform(D_scaled_part_thrust, D_scaled_part_thrust + *len_cpu, x2end_thrust, D_scaled_mul_x_part_thrust, thrust::multiplies<double>());
  cublasDnrm2_v2(handle, *len_cpu, D_scaled_mul_x_part, 1, temp_gpu);
  soc_cone_heuristic<<<1, 1>>>(sol_gpu, temp_gpu, d_return_flag);
  cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if (h_return_flag){
    double scale_factor = minVal;
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasDscal_v2(handle, *n_cpu, &scale_factor, sol_gpu, 1);
    return;
  }
  double sol0;
  cudaMemcpy(&sol0, sol_gpu, sizeof(double), cudaMemcpyDeviceToHost);
  if (sol0 >= rel_tol) {
    check_t_range_case0<<<1, 1>>>(t_warm_start_gpu, xiLeft_gpu, xiRight_gpu, oracleVal_gpu, d_auxiliary_flag);
    cudaMemcpy(&h_auxiliary_flag, d_auxiliary_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_auxiliary_flag){
      oracle_soc_f_sqrt_case0(handle, t_warm_start_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
      if (h_return_flag){
        recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu, D_scaled_squared_gpu, n_gpu);
        return;
      }
      binary_search_case0<<<1, 1>>>(xiLeft_gpu, xiRight_gpu, oracleVal_gpu, t_warm_start_gpu, d_auxiliary_flag, d_return_flag, abs_tol, rel_tol);
    }
    cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    while (!h_return_flag) {
      average_xi<<<1, 1>>>(xiLeft_gpu, xiRight_gpu, t_warm_start_gpu);
      oracle_soc_f_sqrt_case0(handle, t_warm_start_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
      binary_search_case0<<<1, 1>>>(xiLeft_gpu, xiRight_gpu, oracleVal_gpu, t_warm_start_gpu, d_auxiliary_flag, d_return_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu, D_scaled_squared_gpu, n_gpu);
    return;
  }
  else if (sol0 <= -rel_tol) {
    initialize_case1<<<1, 1>>>(xiRight_gpu, xiLeft_gpu, oracleVal_gpu);
    oracle_soc_f_sqrt_case1(handle, xiRight_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, temp_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
    cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_return_flag){
      return;
    }
    cudaMemcpy(&h_auxiliary_flag, d_auxiliary_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    while (h_auxiliary_flag) {
      enlarge_xi_right<<<1, 1>>>(xiLeft_gpu, xiRight_gpu);
      oracle_soc_f_sqrt_case1(handle, xiRight_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, temp_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_auxiliary_flag, d_auxiliary_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    average_xi<<<1, 1>>>(xiLeft_gpu, xiRight_gpu, t_warm_start_gpu);
    oracle_soc_f_sqrt_case1(handle, t_warm_start_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, temp_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
    cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    while (!h_return_flag) {
      average_xi<<<1, 1>>>(xiLeft_gpu, xiRight_gpu, t_warm_start_gpu);
      oracle_soc_f_sqrt_case1(handle, t_warm_start_gpu, sol_gpu, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu, len_gpu, nThread, nBlock, temp_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
      binary_search_case1<<<1, 1>>>(xiRight_gpu, xiLeft_gpu, oracleVal_gpu, t_warm_start_gpu, d_auxiliary_flag, d_return_flag, abs_tol, rel_tol);
      cudaMemcpy(&h_return_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu, D_scaled_squared_gpu, n_gpu);
    return;
  }
  else {
    recover_sol_case3<<<nBlock, nThread>>>(sol_gpu, temp_gpu, D_scaled_gpu, n_gpu, t_warm_start_gpu, D_scaled_squared_gpu);
    cublasDnrm2_v2(handle, *len_cpu, temp_gpu + 1, 1, sol_gpu);
    return;
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

extern "C" void few_block_proj(cublasHandle_t handle,
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
  for (int i = 0; i < blkNum; ++i)
  {
    long *n_gpu = ns_gpu+i;
    long n_cpu = ns_cpu[i];
    double *sol = arr + cpu_head_start[i];
    double *sub_D_scaled = D_scaled + cpu_head_start[i];
    double *sub_D_scaled_squared = D_scaled_squared + cpu_head_start[i];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + cpu_head_start[i];
    double *sub_temp = temp + cpu_head_start[i];
    double *sub_bl = bl + cpu_head_start[i];
    double *sub_bu = bu + cpu_head_start[i];
    if (cpu_proj_type[i] == 0 || cpu_proj_type[i] == 1){
      // dual_free_proj
      ;
    }
    else if (cpu_proj_type[i] == 17 || cpu_proj_type[i] == 19 || cpu_proj_type[i] == 18){
      // box
      box_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, sub_bl, sub_bu, n_gpu);
    }
    else if (cpu_proj_type[i] == 2){
      // zeros<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
      cudaMemset(sol, 0, n_cpu * sizeof(double));
    }
    else if (cpu_proj_type[i] == 3 || cpu_proj_type[i] == 4){
      // dual_positive
      positive_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
    }
    else if (cpu_proj_type[i] == 5 || cpu_proj_type[i] == 7 || cpu_proj_type[i] == 20 || cpu_proj_type[i] == 21){
      long len_cpu = n_cpu - 1;
      double *d_temp;
      cudaMalloc(&d_temp, sizeof(double));
      cudaMemcpy(d_temp, sub_temp, sizeof(double), cudaMemcpyHostToDevice);
      soc_proj(handle, sol, &n_cpu, n_gpu, &len_cpu, d_temp, ThreadPerBlock, nBlock);
      cudaFree(d_temp);
    }
    else if (cpu_proj_type[i] == 6 || cpu_proj_type[i] == 22){
      bool* d_auxiliary_flag;
      bool h_auxiliary_flag = false;
      cudaMalloc(&d_auxiliary_flag, sizeof(bool));
      cudaMemcpy(d_auxiliary_flag, &h_auxiliary_flag, sizeof(bool), cudaMemcpyHostToDevice);
      bool* d_return_flag;
      bool h_return_flag = false;
      cudaMalloc(&d_return_flag, sizeof(bool));
      cudaMemcpy(d_return_flag, &h_return_flag, sizeof(bool), cudaMemcpyHostToDevice);
      long* len_gpu;
      long len_cpu = n_cpu - 1;
      cudaMalloc(&len_gpu, sizeof(long));
      cudaMemcpy(len_gpu, &len_cpu, sizeof(long), cudaMemcpyHostToDevice);
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
      cudaFree(len_gpu);
      cudaFree(d_return_flag);
      cudaFree(d_auxiliary_flag);
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
      dualExponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_D_scaled, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 15 || cpu_proj_type[i] == 27){
      // exponent_proj_diagonal
      exponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_D_scaled, &t_warm_start[i], abs_tol, rel_tol);
    }
  }
}

void check_status(cublasStatus_t status){
  if (status != CUBLAS_STATUS_SUCCESS){
    printf("cuBLAS error with status code: %d\n", status);
  }
}

extern "C" void create_cublas_handle_inner(cublasHandle_t* handle){
  cublasStatus_t status = cublasCreate(handle);
  check_status(status);
}

extern "C" void destroy_cublas_handle_inner(cublasHandle_t handle){
  cublasStatus_t status = cublasDestroy(handle);
  check_status(status);
}
