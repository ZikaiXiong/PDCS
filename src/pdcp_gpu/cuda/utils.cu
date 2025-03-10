





extern "C" __global__ void
primal_update(double* primal_sol, double* primal_sol_lag, double* primal_sol_diff, double* d_c, double tau, long n)
{
    // x.primal_sol.x .= x.primal_sol_lag.x .+ sol.params.tau * (primal_sol_diff.x .- solver.data.d_c)
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < n) {
        primal_sol[global_thread_idx] = primal_sol_lag[global_thread_idx] + tau * (primal_sol_diff[global_thread_idx] - d_c[global_thread_idx]);
    }
}

extern "C" __global__ void
extrapolation_update(double* primal_sol_diff, double* primal_sol, double* primal_sol_lag, long n)
{
    // primal_sol_diff.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < n) {
        primal_sol_diff[global_thread_idx] = 2.0 * primal_sol[global_thread_idx] - primal_sol_lag[global_thread_idx];
    }
}

extern "C" __global__ void
dual_update(double* dual_sol, double* dual_sol_lag, double* dual_sol_diff, double* d_h, double sigma, long n)
{
    // y.dual_sol.y .= y.dual_sol_lag.y .- sol.params.sigma * (dual_sol_diff.y .- solver.data.coeff.d_h)
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < n) {
        dual_sol[global_thread_idx] = dual_sol_lag[global_thread_idx] - sigma * (dual_sol_diff[global_thread_idx] - d_h[global_thread_idx]);
    }
}

extern "C" __global__ void
reflection_update(double* primal_sol, double* primal_sol_lag, double* primal_sol_mean, double* dual_sol, double* dual_sol_lag, double* dual_sol_mean, double extra_coeff, long primal_n, long dual_n, long inner_iter, double eta_cum, double eta)
{
    // sol.x.primal_sol.x .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.x.primal_sol.x .- extra_coeff * sol.x.primal_sol_lag.x) + 1 / (inner_iter + 2) * sol.x.primal_sol.x
    // sol.y.dual_sol.y .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.y.dual_sol.y .- extra_coeff * sol.y.dual_sol_lag.y) + 1 / (inner_iter + 2) * sol.y.dual_sol.y
    // # average update
    // sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
    // sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double double_inner_iter = (double) inner_iter;
    double temp = (double_inner_iter + 1.0) / (double_inner_iter + 2.0);
    if (global_thread_idx < primal_n) {
        primal_sol[global_thread_idx] = temp * ((1.0 + extra_coeff) * primal_sol[global_thread_idx] - extra_coeff * primal_sol_lag[global_thread_idx]) + (1.0 / (double_inner_iter + 2.0)) * primal_sol[global_thread_idx];
        primal_sol_mean[global_thread_idx] = (primal_sol_mean[global_thread_idx] * eta_cum + primal_sol[global_thread_idx] * eta) / (eta_cum + eta);
    }
    if (global_thread_idx < dual_n) {
        dual_sol[global_thread_idx] = temp * ((1.0 + extra_coeff) * dual_sol[global_thread_idx] - extra_coeff * dual_sol_lag[global_thread_idx]) + (1.0 / (double_inner_iter + 2.0)) * dual_sol[global_thread_idx];
        dual_sol_mean[global_thread_idx] = (dual_sol_mean[global_thread_idx] * eta_cum + dual_sol[global_thread_idx] * eta) / (eta_cum + eta);
    }
}

extern "C" __global__ void
calculate_diff(double* dual_sol, double* dual_sol_lag, double* dual_sol_diff, long dual_n, double* primal_sol, double* primal_sol_lag, double* primal_sol_diff, long primal_n)
{
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < dual_n) {
        dual_sol_diff[global_thread_idx] = dual_sol[global_thread_idx] - dual_sol_lag[global_thread_idx];
    }
    if (global_thread_idx < primal_n) {
        primal_sol_diff[global_thread_idx] = primal_sol[global_thread_idx] - primal_sol_lag[global_thread_idx];
    }
}

extern "C" __global__ void 
axpyz(double* z, double alpha, double* y, double* x, long n)
{
    // z .= alpha * y .+ z
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < n) {
        z[global_thread_idx] = alpha * y[global_thread_idx] + x[global_thread_idx];
    }
}

extern "C" __global__ void
average_seq( double* primal_sol_mean, double* primal_sol, long primal_n, double* dual_sol_mean, double* dual_sol,  long dual_n, long inner_iter)
{
    // sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * inner_iter .+ sol.x.primal_sol.x) / (inner_iter + 1)
    // sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * inner_iter .+ sol.y.dual_sol.y) / (inner_iter + 1)
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double double_inner_iter = (double) inner_iter;
    if (global_thread_idx < primal_n) {
        primal_sol_mean[global_thread_idx] = (primal_sol_mean[global_thread_idx] * double_inner_iter + primal_sol[global_thread_idx]) / (double_inner_iter + 1.0);
    }
    if (global_thread_idx < dual_n) {
        dual_sol_mean[global_thread_idx] = (dual_sol_mean[global_thread_idx] * double_inner_iter + dual_sol[global_thread_idx]) / (double_inner_iter + 1.0);
    }
}

extern "C" __global__ void 
get_row_index(int* row_ptr, long m, int *row_idx){
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx >= row_ptr[m] - 1) {
        return;
    }
    // Binary search to find the corresponding row
    long left = 0;
    long right = m - 1;
    while (left <= right) {
        long mid = left + (right - left) / 2;
        if (global_thread_idx >= row_ptr[mid] - 1 && global_thread_idx < row_ptr[mid + 1] - 1) {
            row_idx[global_thread_idx] = mid + 1; // for julia index
            break;
        } else if (global_thread_idx < row_ptr[mid] - 1) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
}

extern "C" __global__ void
rescale_coo(double* d_G, int* row_idx, int* col_idx, double* row_scaling, double* col_scaling, long nnz)
{
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx >= nnz) {
        return;
    }
    long col = col_idx[global_thread_idx] - 1;
    long row = row_idx[global_thread_idx] - 1;
    double value = d_G[global_thread_idx];

    double row_scale = row_scaling[row];
    double col_scale = col_scaling[col];
    d_G[global_thread_idx] = value / (row_scale * col_scale);
}

extern "C" __global__ void
rescale_csr(double* d_G, int* row_ptr, int* col_idx, double* row_scaling, double* col_scaling, long m, long n)
{
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (global_thread_idx == 0) {
    //     printf("Debug: Thread %d: RowPtr = %d, ColIdx = %d, RowScaling = %f, ColScaling = %f, m = %d, n = %d\n",
    //            global_thread_idx, row_ptr[m], col_idx[global_thread_idx], row_scaling[m], col_scaling[global_thread_idx], m, n);
    //     for (long i = 0; i < m; i++) {
    //         printf("Debug: RowPtr = %d\n", row_ptr[i]);
    //     }
    //     printf("Debug: nnz = %d\n", row_ptr[m]);
    //     for (long i = 0; i < row_ptr[m]; i++) {
    //         printf("Debug: ColIdx = %d\n", col_idx[i]);
    //     }
    // }
    
    if (global_thread_idx >= row_ptr[m] - 1) {
        return;
    }

    // 找到对应行
    // long row = 0;
    // while (row < m && global_thread_idx >= row_ptr[row + 1] - 1) {
    //     row++;
    // }
    // Binary search to find the corresponding row
    long left = 0;
    long right = m - 1;
    long row = 0;
    
    while (left <= right) {
        long mid = left + (right - left) / 2;
        if (global_thread_idx >= row_ptr[mid] - 1 && global_thread_idx < row_ptr[mid + 1] - 1) {
            row = mid;
            break;
        } else if (global_thread_idx < row_ptr[mid] - 1) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }


    // 获取列索引和非零值
    long col = col_idx[global_thread_idx] - 1;
    double value = d_G[global_thread_idx];

    double row_scale = row_scaling[row];
    double col_scale = col_scaling[col];
    // printf("Debug csr: global_thread_idx = %ld, row = %ld, col = %ld\n", global_thread_idx, row, col);



    // 缩放计算
    d_G[global_thread_idx] = value / (row_scale * col_scale);
}


extern "C" __global__ void
replace_inf_with_zero(double* d_bl, double* d_bu, long n)
{
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < n) {
        d_bl[global_thread_idx] = (d_bl[global_thread_idx] == -INFINITY) ? 0.0 : d_bl[global_thread_idx];
        d_bu[global_thread_idx] = (d_bu[global_thread_idx] == INFINITY) ? 0.0 : d_bu[global_thread_idx];
    }
}

extern "C" __global__ void max_abs_row_elementwise_kernel(
    const double* values, const int* row_idx, const long nnz, double* result)
{
    long global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_idx < nnz){
        int row = row_idx[global_thread_idx] - 1;
        double value = fabs(values[global_thread_idx]);
        atomicMax((unsigned long long*)&result[row],
                 __double_as_longlong(value));
    }
}


extern "C" __global__ void max_abs_row_kernel(
    const double* values, const int* rowptr, long nrows, double* result) 
{
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    long warpIdx = globalThreadIdx / warpSize; // 全局 warp 索引
    long laneIdx = globalThreadIdx % warpSize; // 当前线程在warp中的lane索引
    long rowIdx = warpIdx; // 每个warp对应一个行

    if (rowIdx >= nrows) return; // 防止越界

    // 获取该行的起始和结束索引
    long row_start = rowptr[rowIdx] - 1;
    long row_end = rowptr[rowIdx + 1] - 1;

    double max_val = 0.0;

    // 使用warp中的线程并行处理该行
    for (long i = row_start + laneIdx; i < row_end; i += warpSize) {
        max_val = fmax(max_val, fabs(values[i]));
    }

    // 使用warp-level reduction共享 max_val
    for (long offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmax(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }

    // Warp的第一个线程（laneIdx == 0）写回结果
    if (laneIdx == 0) {
        result[rowIdx] = max_val;
    }
}


extern "C" __global__ void max_abs_col_elementwise_kernel(
    const double* values, const int* colind, const long nnz, double* result) 
{
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < nnz){
        int col = colind[globalThreadIdx] - 1;
        double value = fabs(values[globalThreadIdx]);
        atomicMax((unsigned long long*)&result[col],
                 __double_as_longlong(value));
    }
}


extern "C" __global__ void max_abs_col_kernel(
    const double* values, const int* colind, const int* rowptr,
    long nrows, long ncols, double* result) 
{
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    long warpIdx = globalThreadIdx / warpSize; // 全局 warp 索引
    long laneIdx = globalThreadIdx % warpSize; // 当前线程在 warp 中的 lane 索引
    long colIdx = warpIdx; // 每个 warp 对应一个列

    if (colIdx >= ncols) return; // 防止越界

    double max_val = 0.0;
    for (long row = laneIdx; row < nrows; row += warpSize) {
        long left = rowptr[row] - 1;
        long right = rowptr[row + 1] - 2;  // 修改右边界
        long mid = left + (right - left) / 2;  // 更安全的中点计算   
        while (left <= right) {  // 修改循环条件
            mid = left + (right - left) / 2;  // 更安全的中点计算
            if (colind[mid] - 1 == colIdx) {
                max_val = fmax(max_val, fabs(values[mid]));
                break;  // 找到后可以直接退出
            } else if (colind[mid] - 1 > colIdx) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
    for (long offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmax(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneIdx == 0) {
        result[colIdx] = max_val;
    }
}





extern "C" __global__ void alpha_norm_row_kernel(
    const double* values, const int* rowptr, long nrows, double* result, double alpha) 
{
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    long warpIdx = globalThreadIdx / warpSize; // 全局 warp 索引
    long laneIdx = globalThreadIdx % warpSize; // 当前线程在warp中的lane索引
    long rowIdx = warpIdx; // 每个warp对应一个行

    if (rowIdx >= nrows) return; // 防止越界

    // 获取该行的起始和结束索引
    long row_start = rowptr[rowIdx] - 1;
    long row_end = rowptr[rowIdx + 1] - 1;

    double alpha_norm_val = 0.0;

    // 使用warp中的线程并行处理该行
    for (long i = row_start + laneIdx; i < row_end; i += warpSize) {
        // alpha_norm_val += powf(fabsf(values[i]), alpha);
        alpha_norm_val += fabs(values[i]);
    }

    for (long offset = warpSize / 2; offset > 0; offset /= 2) {
        alpha_norm_val += __shfl_down_sync(0xFFFFFFFF, alpha_norm_val, offset);
    }

    // Warp的第一个线程（laneIdx == 0）计算最终结果并写回
    if (laneIdx == 0) {
        // result[rowIdx] = pow(alpha_norm_val, 1.0 / alpha);
        result[rowIdx] = alpha_norm_val;
    }
}


extern "C" __global__ void alpha_norm_col_kernel(
    const double* values, const int* colind, const int* rowptr,
    long nrows, long ncols, double* result, double alpha) 
{
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    long warpIdx = globalThreadIdx / warpSize; // 全局 warp 索引
    long laneIdx = globalThreadIdx % warpSize; // 当前线程在 warp 中的 lane 索引
    long colIdx = warpIdx; // 每个 warp 对应一个列

    if (colIdx >= ncols) return; // 防止越界

    double alpha_norm_val = 0.0;

    // 每个线程遍历行范围，找到属于当前列的元素
    for (long row = laneIdx; row < nrows; row += warpSize) {
        long left = rowptr[row] - 1;
        long right = rowptr[row + 1] - 2;  // 修改右边界
        long mid = left + (right - left) / 2;  // 更安全的中点计算   
        while (left <= right) {  // 修改循环条件
            mid = left + (right - left) / 2;  // 更安全的中点计算
            if (colind[mid] - 1 == colIdx) {
                // alpha_norm_val += pow(fabs(values[mid]), alpha);
                alpha_norm_val += fabs(values[mid]);
                break;  // 找到后可以直接退出
            } else if (colind[mid] - 1 > colIdx) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }

    // 使用 warp-level reduction 共享 max_val
    for (long offset = warpSize / 2; offset > 0; offset /= 2) {
        alpha_norm_val += __shfl_down_sync(0xFFFFFFFF, alpha_norm_val, offset);
    }

    // Warp 的第一个线程（laneIdx == 0）写回结果
    if (laneIdx == 0) {
        // result[colIdx] = pow(alpha_norm_val, 1.0 / alpha);
        result[colIdx] = alpha_norm_val;
    }
}

extern "C" __global__ void alpha_norm_col_elementwise_kernel(
        const double* values, const int* colind, const long nnz, double* result, double alpha, int ncols) 
{
    // Note: this kernel is not used to calculate ^{1/alpha} norm 
    long globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First pass: accumulate alpha powers
    if (globalThreadIdx < nnz) {
        int col = colind[globalThreadIdx] - 1;
        // double value = pow(fabs(values[globalThreadIdx]), alpha);
        double value = fabs(values[globalThreadIdx]);
        atomicAdd(&result[col], value);
    }
}