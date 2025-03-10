massive_block_proj_path = joinpath(MODULE_DIR, "cuda/massive_block_proj.ptx")
massive_func_name = "massive_block_proj"

massive_mod = CuModule(read(massive_block_proj_path))
massive_kernel = CuFunction(massive_mod, massive_func_name)


function massive_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    nBlock = Int64(ceil((blkNum + ThreadPerBlock- 1) / ThreadPerBlock))
    nBlock = min(nBlock, 10240)
    CUDA.@sync begin
        CUDA.cudacall(
        massive_kernel,
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

moderate_block_proj_path = joinpath(MODULE_DIR, "cuda/moderate_block_proj.ptx")
moderate_func_name = "moderate_block_proj"

moderate_mod = CuModule(read(moderate_block_proj_path))
moderate_kernel = CuFunction(moderate_mod, moderate_func_name)

function moderate_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    nBlock = blkNum + 1
    CUDA.@sync begin
        CUDA.cudacall(
        moderate_kernel,
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

sufficient_block_proj_path = joinpath(MODULE_DIR, "cuda/sufficient_block_proj.ptx")
sufficient_func_name = "sufficient_block_proj"

sufficient_mod = CuModule(read(sufficient_block_proj_path))
sufficient_kernel = CuFunction(sufficient_mod, sufficient_func_name)

function sufficient_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    nBlock = Int64(ceil((blkNum + 1) * 32 / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(
        sufficient_kernel,
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

lib = Libdl.dlopen(joinpath(MODULE_DIR, "cuda/libfew_block_proj.so"))
cublasCreate = Libdl.dlsym(lib, Symbol("create_cublas_handle_inner"))
cublasDestroy = Libdl.dlsym(lib, Symbol("destroy_cublas_handle_inner"))

mutable struct CUBLASHandle
    handle::Ptr{Nothing}
    # 无参数构造函数
    CUBLASHandle() = new(C_NULL)
    # 带参数构造函数
    CUBLASHandle(handle::Ptr{Nothing}) = new(handle)
end


function create_cublas_handle()
    handle_ptr = Ref{Ptr{Nothing}}(C_NULL)  # 初始化为空指针
    @ccall $cublasCreate(handle_ptr::Ref{Ptr{Nothing}})::Cvoid  # 调用动态库
    handle = CUBLASHandle(handle_ptr[])  # 创建 Julia 的 CUBLASHandle 结构体
    @info ("cuBLAS handle created: $(handle.handle)")
    return handle
end

function destroy_cublas_handle(handle::CUBLASHandle)
    @ccall $cublasDestroy(handle.handle::Ptr{Nothing})::Cvoid  # 销毁句柄
end

few_block_proj_kernel = Libdl.dlsym(lib, Symbol("few_block_proj"))



function few_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, cpu_head_start::Vector{Int64}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, cpu_ns::Vector{Int64}, blkNum::Int64, cpu_proj_type::Vector{Int64}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    nBlock = Int64(ceil((maximum(cpu_ns) + ThreadPerBlock + 1) / ThreadPerBlock))
    nThread = Int64(ThreadPerBlock)
    
    @ccall $few_block_proj_kernel(handle.handle::Ptr{Nothing},
                             vec::CuPtr{Cdouble}, 
                             bl::CuPtr{Cdouble}, 
                             bu::CuPtr{Cdouble}, 
                             D_scaled::CuPtr{Cdouble}, 
                             D_scaled_squared::CuPtr{Cdouble}, 
                             D_scaled_mul_x::CuPtr{Cdouble}, 
                             temp::CuPtr{Cdouble}, 
                             t_warm_start::CuPtr{Cdouble}, 
                             cpu_head_start::Ptr{Clong},
                             gpu_ns::CuPtr{Clong}, 
                             cpu_ns::Ptr{Clong}, 
                             blkNum::Cint, 
                             cpu_proj_type::Ptr{Clong}, 
                             nThread::Cint, 
                             nBlock::Cint,
                             abs_tol::Cdouble,
                             rel_tol::Cdouble)::Cvoid# sync
    CUDA.synchronize()
end



utils_path = joinpath(MODULE_DIR, "cuda/utils.ptx")
reflection_update_func_name = "reflection_update"
primal_update_func_name = "primal_update"
dual_update_func_name = "dual_update"
extrapolation_update_func_name = "extrapolation_update"
calculate_diff_func_name = "calculate_diff"
axpyz_func_name = "axpyz"
average_seq_func_name = "average_seq"
rescale_csr_func_name = "rescale_csr"

utils_mod = CuModule(read(utils_path))
reflection_update_kernel = CuFunction(utils_mod, reflection_update_func_name)
primal_update_kernel = CuFunction(utils_mod, primal_update_func_name)
dual_update_kernel = CuFunction(utils_mod, dual_update_func_name)
extrapolation_update_kernel = CuFunction(utils_mod, extrapolation_update_func_name)
calculate_diff_kernel = CuFunction(utils_mod, calculate_diff_func_name)
axpyz_kernel = CuFunction(utils_mod, axpyz_func_name)
average_seq_kernel = CuFunction(utils_mod, average_seq_func_name)

function reflection_update(primal_sol::T, primal_sol_lag::T, primal_sol_mean::T, dual_sol::T, dual_sol_lag::T, dual_sol_mean::T, extra_coeff::Float64, primal_n::Int64, dual_n::Int64, inner_iter::Int64, eta_cum::Float64, eta::Float64) where T<:CuArray
    nBlock = Int64(ceil((max(primal_n, dual_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(reflection_update_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64, Int64, Int64, Float64, Float64), 
        primal_sol, primal_sol_lag, primal_sol_mean, dual_sol, dual_sol_lag, dual_sol_mean, extra_coeff, primal_n, dual_n, inner_iter, eta_cum, eta;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function primal_update(primal_sol::T, primal_sol_lag::T, primal_sol_diff::T, d_c::T, tau::Float64, n::Int64) where T<:CuArray
    nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(primal_update_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function dual_update(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, d_h::T, sigma::Float64, n::Int64) where T<:CuArray
    nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(dual_update_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function extrapolation_update(primal_sol_diff::T, primal_sol::T, primal_sol_lag::T, n::Int64) where T<:CuArray
    nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(extrapolation_update_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        primal_sol_diff, primal_sol, primal_sol_lag, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function calculate_diff(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, dual_n::Int64, primal_sol::T, primal_sol_lag::T, primal_sol_diff::T,  primal_n::Int64) where T<:CuArray
    nBlock = Int64(ceil((max(dual_n, primal_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(calculate_diff_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function axpyz(z::T, alpha::Float64, y::T, x::T, n::Int64) where T<:CuArray
    nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    # z .= alpha * y .+ x
    CUDA.@sync begin
        CUDA.cudacall(axpyz_kernel, 
        (CuPtr{Float64}, Float64, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        z, alpha, y, x, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

function average_seq(; primal_sol_mean::T, primal_sol::T, primal_n::Int64, dual_sol_mean::T, dual_sol::T, dual_n::Int64, inner_iter::Int64) where T<:CuArray
    nBlock = Int64(ceil((max(primal_n, dual_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(average_seq_kernel, 
        (CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64), 
        primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

rescale_csr_kernel = CuFunction(utils_mod, rescale_csr_func_name)

function rescale_csr(d_G::CUDA.CUSPARSE.CuSparseMatrixCSR, row_scaling::CuArray, col_scaling::CuArray, m::Int64, n::Int64)
    nBlock = Int64(ceil((length(d_G.nzVal) + ThreadPerBlock - 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(rescale_csr_kernel, 
        (CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64),
         d_G.nzVal, d_G.rowPtr, d_G.colVal, row_scaling, col_scaling, m, n; 
         blocks = nBlock, threads = ThreadPerBlock)
    end
end

replace_inf_with_zero_func_name = "replace_inf_with_zero"
replace_inf_with_zero_kernel = CuFunction(utils_mod, replace_inf_with_zero_func_name)

function replace_inf_with_zero(bl::T, bu::T, n::Int64) where T<:CuArray
    nBlock = Int64(ceil((n + ThreadPerBlock + 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(replace_inf_with_zero_kernel, (CuPtr{Float64}, CuPtr{Float64}, Int64), bl, bu, n; blocks = nBlock, threads = ThreadPerBlock)
    end
end
max_abs_row_func_name = "max_abs_row_kernel"
max_abs_row_kernel = CuFunction(utils_mod, max_abs_row_func_name)

function max_abs_row(d_G, result)
    # Use appropriate methods to extract CuSparseMatrixCSR data
    rowptr = d_G.rowPtr    # Access row pointers directly
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    nrows = Int64(nrows)
    result .= 1.0
    nBlock = Int64(ceil((nrows + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(max_abs_row_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, rowptr, nrows, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

max_abs_col_func_name = "max_abs_col_kernel"
max_abs_col_kernel = CuFunction(utils_mod, max_abs_col_func_name)

function max_abs_col(d_G, result)
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)
    nrows = Int64(nrows)
    ncols = Int64(ncols)
    result .= 1.0
    nBlock = Int64(ceil((ncols + ThreadPerBlock + 1) * 32 / ThreadPerBlock))

    CUDA.@sync begin
        CUDA.cudacall(max_abs_col_kernel, 
        (CuPtr{Float64}, CuPtr{Int32}, CuPtr{Int32}, Int64, Int64, CuPtr{Float64}), 
        d_G.nzVal, d_G.colVal, d_G.rowPtr, nrows, ncols, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


alpha_norm_row_func_name = "alpha_norm_row_kernel"
alpha_norm_row_kernel = CuFunction(utils_mod, alpha_norm_row_func_name)

function alpha_norm_row(d_G, alpha, result)
    rowptr = d_G.rowPtr    # Access row pointers directly
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    nBlock = Int64(ceil((nrows + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(alpha_norm_row_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, Int, CuPtr{Float64}, Float64), 
        values, rowptr, nrows, result, alpha;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

alpha_norm_col_func_name = "alpha_norm_col_kernel"
alpha_norm_col_kernel = CuFunction(utils_mod, alpha_norm_col_func_name)

function alpha_norm_col(d_G, alpha, result)
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)
    nBlock = Int64(ceil((ncols + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(alpha_norm_col_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, CuPtr{Int}, Int, Int, CuPtr{Float64}, Float64), 
        d_G.nzVal, d_G.colVal, d_G.rowPtr, nrows, ncols, result, alpha;
        blocks = nBlock, threads = ThreadPerBlock)
    end
    # delete this since alpha = 1.0
    # result .= result .^ (1.0 / alpha)
end

get_row_index_func_name = "get_row_index"
get_row_index_kernel = CuFunction(utils_mod, get_row_index_func_name)

function get_row_index(d_G, row_idx)
    nnz = length(d_G.nzVal)
    nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)

    CUDA.@sync begin
        CUDA.cudacall(get_row_index_kernel, 
        (CuPtr{Int}, Int64, CuPtr{Int}), 
        d_G.rowPtr, nrows, row_idx;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

rescale_coo_func_name = "rescale_coo"
rescale_coo_kernel = CuFunction(utils_mod, rescale_coo_func_name)

function rescale_coo(d_G::CUDA.CUSPARSE.CuSparseMatrixCSR, row_scaling::CuArray, col_scaling::CuArray, m::Int64, n::Int64, row_idx::CuArray)
    nnz = length(d_G.nzVal)
    nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(rescale_coo_kernel, 
        (CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64),
         d_G.nzVal, row_idx, d_G.colVal, row_scaling, col_scaling, nnz; 
         blocks = nBlock, threads = ThreadPerBlock)
    end
end


max_abs_row_elementwise_func_name = "max_abs_row_elementwise_kernel"
max_abs_row_elementwise_kernel = CuFunction(utils_mod, max_abs_row_elementwise_func_name)

function max_abs_row_elementwise(d_G, row_idx, result)
    # Use appropriate methods to extract CuSparseMatrixCSR data
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    nrows = Int64(nrows)
    result .= 0.0
    nnz = length(d_G.nzVal)
    nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(max_abs_row_elementwise_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, row_idx, nnz, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


max_abs_col_elementwise_func_name = "max_abs_col_elementwise_kernel"
max_abs_col_elementwise_kernel = CuFunction(utils_mod, max_abs_col_elementwise_func_name)

function max_abs_col_elementwise(d_G, result)
    values = d_G.nzVal
    col_idx = d_G.colVal
    ncols = size(d_G, 2)
    ncols = Int64(ncols)
    result .= 0.0
    nnz = length(d_G.nzVal)
    nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(max_abs_col_elementwise_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, col_idx, nnz, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


alpha_norm_col_elementwise_func_name = "alpha_norm_col_elementwise_kernel"
alpha_norm_col_elementwise_kernel = CuFunction(utils_mod, alpha_norm_col_elementwise_func_name)

function alpha_norm_col_elementwise(d_G, alpha, result)
    nnz = length(d_G.nzVal)
    ncols = size(d_G, 2)
    nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(alpha_norm_col_elementwise_kernel, 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}, Float64, Int64), 
        d_G.nzVal, d_G.colVal, nnz, result, alpha, ncols;
        blocks = nBlock, threads = ThreadPerBlock)
    end
    # delete this since alpha = 1.0
    # result .= result .^ (1.0 / alpha)
end
