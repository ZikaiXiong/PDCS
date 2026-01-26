
const _massive_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _massive_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _massive_block_proj_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _massive_block_proj_path = joinpath(MODULE_DIR, "cuda/massive_block_proj.ptx")   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _massive_block_proj_name = "massive_block_proj"

function get_massive_block_proj_kernel()::CuFunction
    k = _massive_block_proj_kernel[]
    k !== nothing && return k

    lock(_massive_block_proj_lock)
    try
        # double-check after locking
        k = _massive_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_massive_block_proj_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _massive_block_proj_name)

        _massive_block_proj_mod[]    = mod
        _massive_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_massive_block_proj_lock)
    end
end


function massive_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # nBlock = Int64(ceil((blkNum + ThreadPerBlock- 1) / ThreadPerBlock))
    # nBlock = min(nBlock, 10240)
    nBlock = cld(blkNum + ThreadPerBlock, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
        get_massive_block_proj_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

const _moderate_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _moderate_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _moderate_block_proj_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _moderate_block_proj_path = joinpath(MODULE_DIR, "cuda/moderate_block_proj.ptx")   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _moderate_block_proj_name = "moderate_block_proj"

function get_moderate_block_proj_kernel()::CuFunction
    k = _moderate_block_proj_kernel[]
    k !== nothing && return k

    lock(_moderate_block_proj_lock)
    try
        # double-check after locking
        k = _moderate_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_moderate_block_proj_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _moderate_block_proj_name)

        _moderate_block_proj_mod[]    = mod
        _moderate_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_moderate_block_proj_lock)
    end
end

function moderate_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    nBlock = blkNum + 1
    CUDA.@sync begin
        CUDA.cudacall(
        get_moderate_block_proj_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end


const _sufficient_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _sufficient_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _sufficient_block_proj_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _sufficient_block_proj_path = joinpath(MODULE_DIR, "cuda/sufficient_block_proj.ptx")   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _sufficient_block_proj_name = "sufficient_block_proj"

function get_sufficient_block_proj_kernel()::CuFunction
    k = _sufficient_block_proj_kernel[]
    k !== nothing && return k

    lock(_sufficient_block_proj_lock)
    try
        # double-check after locking
        k = _sufficient_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_sufficient_block_proj_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _sufficient_block_proj_name)

        _sufficient_block_proj_mod[]    = mod
        _sufficient_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_sufficient_block_proj_lock)
    end
end

function sufficient_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # nBlock = Int64(ceil((blkNum + 1) * 32 / ThreadPerBlock))
    nBlock = cld((blkNum + 1) * 32, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
        get_sufficient_block_proj_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

lib = Libdl.dlopen(joinpath(MODULE_DIR, "cuda/libfew_block_proj.so"))
cublasCreate = Libdl.dlsym(lib, Symbol("create_cublas_handle_inner"))
cublasDestroy = Libdl.dlsym(lib, Symbol("destroy_cublas_handle_inner"))

const libcublas_path = CUDA.CUBLAS.libcublas  # the artifact path you printed
const cublasStatus_t = Cint
const CUBLAS_STATUS_SUCCESS = 0
const cublasHandle_t = Ptr{Cvoid}

mutable struct CUBLASHandle
    handle::cublasHandle_t
end

function create_cublas_handle()
    # Ensure CUDA context exists
    CUDA.zeros(Float32, 1)

    h = Ref{cublasHandle_t}(C_NULL)

    status = ccall((:cublasCreate_v2, libcublas_path),
                   cublasStatus_t, (Ref{cublasHandle_t},), h)

    status == CUBLAS_STATUS_SUCCESS || error("cublasCreate_v2 failed: status=$status")
    h[] != C_NULL || error("cublasCreate_v2 returned NULL handle")

    return CUBLASHandle(h[])
end

function destroy_cublas_handle(ch::CUBLASHandle)
    ch.handle == C_NULL && return nothing

    status = ccall((:cublasDestroy_v2, libcublas_path),
                   cublasStatus_t, (cublasHandle_t,), ch.handle)

    status == CUBLAS_STATUS_SUCCESS || error("cublasDestroy_v2 failed: status=$status")
    ch.handle = C_NULL
    return nothing
end


function few_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, cpu_head_start::Vector{Int64}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, cpu_ns::Vector{Int64}, blkNum::Int64, cpu_proj_type::Vector{Int64}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # nBlock = Int64(ceil((maximum(cpu_ns) + ThreadPerBlock + 1) / ThreadPerBlock))
    nThread = Int64(ThreadPerBlock)
    nBlock = cld(maximum(cpu_ns) + ThreadPerBlock + 1, ThreadPerBlock)
    fptr = few_block_proj_ptr[]
    fptr != C_NULL || error("few_block_proj not initialized. Did __init__() run?")
    @ccall $fptr(handle.handle::Ptr{Nothing},
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


# reflection_update_func_name = "reflection_update"
const _reflection_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _reflection_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _reflection_update_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _reflection_update_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _reflection_update_name = "reflection_update"

function get_reflection_update_kernel()::CuFunction
    k = _reflection_update_kernel[]
    k !== nothing && return k

    lock(_reflection_update_lock)
    try
        # double-check after locking
        k = _reflection_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_reflection_update_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _reflection_update_name)

        _reflection_update_mod[]    = mod
        _reflection_update_kernel[] = fun
        return fun
    finally
        unlock(_reflection_update_lock)
    end
end

function reflection_update(primal_sol::T, primal_sol_lag::T, primal_sol_mean::T, dual_sol::T, dual_sol_lag::T, dual_sol_mean::T, extra_coeff::Float64, primal_n::Int64, dual_n::Int64, inner_iter::Int64, eta_cum::Float64, eta::Float64) where T<:CuArray
    # nBlock = Int64(ceil((max(primal_n, dual_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(max(primal_n, dual_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_reflection_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64, Int64, Int64, Float64, Float64), 
        primal_sol, primal_sol_lag, primal_sol_mean, dual_sol, dual_sol_lag, dual_sol_mean, extra_coeff, primal_n, dual_n, inner_iter, eta_cum, eta;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# primal_update_func_name = "primal_update"
const _primal_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _primal_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _primal_update_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _primal_update_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _primal_update_name = "primal_update"

function get_primal_update_kernel()::CuFunction
    k = _primal_update_kernel[]
    k !== nothing && return k

    lock(_primal_update_lock)
    try
        # double-check after locking
        k = _primal_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_primal_update_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _primal_update_name)

        _primal_update_mod[]    = mod
        _primal_update_kernel[] = fun
        return fun
    finally
        unlock(_primal_update_lock)
    end
end

function primal_update(primal_sol::T, primal_sol_lag::T, primal_sol_diff::T, d_c::T, tau::Float64, n::Int64) where T<:CuArray
    # nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_primal_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# dual_update_func_name = "dual_update"
const _dual_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _dual_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _dual_update_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _dual_update_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _dual_update_name = "dual_update"

function get_dual_update_kernel()::CuFunction
    k = _dual_update_kernel[]
    k !== nothing && return k

    lock(_dual_update_lock)
    try
        # double-check after locking
        k = _dual_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_dual_update_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _dual_update_name)

        _dual_update_mod[]    = mod
        _dual_update_kernel[] = fun
        return fun
    finally
        unlock(_dual_update_lock)
    end
end


function dual_update(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, d_h::T, sigma::Float64, n::Int64) where T<:CuArray
    # nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_dual_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# extrapolation_update_func_name = "extrapolation_update"
const _extrapolation_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _extrapolation_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _extrapolation_update_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _extrapolation_update_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _extrapolation_update_name = "extrapolation_update"

function get_extrapolation_update_kernel()::CuFunction
    k = _extrapolation_update_kernel[]
    k !== nothing && return k

    lock(_extrapolation_update_lock)
    try
        # double-check after locking
        k = _extrapolation_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_extrapolation_update_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _extrapolation_update_name)

        _extrapolation_update_mod[]    = mod
        _extrapolation_update_kernel[] = fun
        return fun
    finally
        unlock(_extrapolation_update_lock)
    end
end

function extrapolation_update(primal_sol_diff::T, primal_sol::T, primal_sol_lag::T, n::Int64) where T<:CuArray
    # nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_extrapolation_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        primal_sol_diff, primal_sol, primal_sol_lag, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# calculate_diff_func_name = "calculate_diff"
const _calculate_diff_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _calculate_diff_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _calculate_diff_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _calculate_diff_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _calculate_diff_name = "calculate_diff"

function get_calculate_diff_kernel()::CuFunction
    k = _calculate_diff_kernel[]
    k !== nothing && return k

    lock(_calculate_diff_lock)
    try
        # double-check after locking
        k = _calculate_diff_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_calculate_diff_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _calculate_diff_name)

        _calculate_diff_mod[]    = mod
        _calculate_diff_kernel[] = fun
        return fun
    finally
        unlock(_calculate_diff_lock)
    end
end


function calculate_diff(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, dual_n::Int64, primal_sol::T, primal_sol_lag::T, primal_sol_diff::T,  primal_n::Int64) where T<:CuArray
    # nBlock = Int64(ceil((max(dual_n, primal_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(max(dual_n, primal_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_calculate_diff_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# axpyz_func_name = "axpyz"
const _axpyz_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _axpyz_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _axpyz_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _axpyz_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _axpyz_name = "axpyz"

function get_axpyz_kernel()::CuFunction
    k = _axpyz_kernel[]
    k !== nothing && return k

    lock(_axpyz_lock)
    try
        # double-check after locking
        k = _axpyz_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_axpyz_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _axpyz_name)

        _axpyz_mod[]    = mod
        _axpyz_kernel[] = fun
        return fun
    finally
        unlock(_axpyz_lock)
    end
end


function axpyz(z::T, alpha::Float64, y::T, x::T, n::Int64) where T<:CuArray
    # nBlock = Int64(ceil((n + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    # z .= alpha * y .+ x
    CUDA.@sync begin
        CUDA.cudacall(get_axpyz_kernel(), 
        (CuPtr{Float64}, Float64, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        z, alpha, y, x, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# average_seq_func_name = "average_seq"
const _average_seq_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _average_seq_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _average_seq_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _average_seq_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _average_seq_name = "average_seq"

function get_average_seq_kernel()::CuFunction
    k = _average_seq_kernel[]
    k !== nothing && return k

    lock(_average_seq_lock)
    try
        # double-check after locking
        k = _average_seq_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_average_seq_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _average_seq_name)

        _average_seq_mod[]    = mod
        _average_seq_kernel[] = fun
        return fun
    finally
        unlock(_average_seq_lock)
    end
end


function average_seq(; primal_sol_mean::T, primal_sol::T, primal_n::Int64, dual_sol_mean::T, dual_sol::T, dual_n::Int64, inner_iter::Int64) where T<:CuArray
    # nBlock = Int64(ceil((max(primal_n, dual_n) + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(max(primal_n, dual_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_average_seq_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64), 
        primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

const _rescale_csr_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_csr_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_csr_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _rescale_csr_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _rescale_csr_name = "rescale_csr"

function get_rescale_csr_kernel()::CuFunction
    k = _rescale_csr_kernel[]
    k !== nothing && return k

    lock(_rescale_csr_lock)
    try
        # double-check after locking
        k = _rescale_csr_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_rescale_csr_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _rescale_csr_name)

        _rescale_csr_mod[]    = mod
        _rescale_csr_kernel[] = fun
        return fun
    finally
        unlock(_rescale_csr_lock)
    end
end

function rescale_csr(d_G::CUDA.CUSPARSE.CuSparseMatrixCSR, row_scaling::CuArray, col_scaling::CuArray, m::Int64, n::Int64)
    # nBlock = Int64(ceil((length(d_G.nzVal) + ThreadPerBlock - 1) / ThreadPerBlock))
    nBlock = cld(length(d_G.nzVal) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_rescale_csr_kernel(), 
        (CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64),
         d_G.nzVal, d_G.rowPtr, d_G.colVal, row_scaling, col_scaling, m, n; 
         blocks = nBlock, threads = ThreadPerBlock)
    end
end

const _replace_inf_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _replace_inf_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _replace_inf_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _replace_inf_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _replace_inf_name = "replace_inf_with_zero"

function get_replace_inf_kernel()::CuFunction
    k = _replace_inf_kernel[]
    k !== nothing && return k

    lock(_replace_inf_lock)
    try
        # double-check after locking
        k = _replace_inf_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_replace_inf_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _replace_inf_name)

        _replace_inf_mod[]    = mod
        _replace_inf_kernel[] = fun
        return fun
    finally
        unlock(_replace_inf_lock)
    end
end


function replace_inf_with_zero(bl::CuArray{Float64,1}, bu::CuArray{Float64,1}, n::Int)
    threads = ThreadPerBlock
    blocks  = cld(n, threads)

    k = get_replace_inf_kernel()

    CUDA.@sync CUDA.cudacall(
        k,
        (CuPtr{Cdouble}, CuPtr{Cdouble}, Clong),
        bl, bu, Clong(n);
        threads=threads, blocks=blocks
    )
    return nothing
end



const _max_abs_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _max_abs_row_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _max_abs_row_name = "max_abs_row_kernel"

function get_max_abs_row_kernel()::CuFunction
    k = _max_abs_row_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_lock)
    try
        # double-check after locking
        k = _max_abs_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_max_abs_row_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _max_abs_row_name)

        _max_abs_row_mod[]    = mod
        _max_abs_row_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_lock)
    end
end

function max_abs_row(d_G, result)
    # Use appropriate methods to extract CuSparseMatrixCSR data
    rowptr = d_G.rowPtr    # Access row pointers directly
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    nrows = Int64(nrows)
    result .= 1.0
    nBlock = Int64(ceil((nrows + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    CUDA.@sync begin
        CUDA.cudacall(get_max_abs_row_kernel(), 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, rowptr, nrows, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


const _max_abs_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _max_abs_col_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _max_abs_col_name = "max_abs_col_kernel"

function get_max_abs_col_kernel()::CuFunction
    k = _max_abs_col_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_lock)
    try
        # double-check after locking
        k = _max_abs_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_max_abs_col_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _max_abs_col_name)

        _max_abs_col_mod[]    = mod
        _max_abs_col_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_lock)
    end
end

function max_abs_col(d_G, result)
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)
    nrows = Int64(nrows)
    ncols = Int64(ncols)
    result .= 1.0
    #  nBlock = Int64(ceil((ncols + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    nBlock = cld((ncols + ThreadPerBlock + 1) * 32, ThreadPerBlock)

    CUDA.@sync begin
        CUDA.cudacall(get_max_abs_col_kernel(), 
        (CuPtr{Float64}, CuPtr{Int32}, CuPtr{Int32}, Int64, Int64, CuPtr{Float64}), 
        d_G.nzVal, d_G.colVal, d_G.rowPtr, nrows, ncols, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end



const _alpha_norm_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_row_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _alpha_norm_row_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _alpha_norm_row_name = "alpha_norm_row_kernel"

function get_alpha_norm_row_kernel()::CuFunction
    k = _alpha_norm_row_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_row_lock)
    try
        # double-check after locking
        k = _alpha_norm_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_alpha_norm_row_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _alpha_norm_row_name)

        _alpha_norm_row_mod[]    = mod
        _alpha_norm_row_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_row_lock)
    end
end

function alpha_norm_row(d_G, alpha, result)
    rowptr = d_G.rowPtr    # Access row pointers directly
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    # nBlock = Int64(ceil((nrows + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    nBlock = cld((nrows + ThreadPerBlock + 1) * 32, ThreadPerBlock)
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(get_alpha_norm_row_kernel(), 
        (CuPtr{Float64}, CuPtr{Int}, Int, CuPtr{Float64}, Float64), 
        values, rowptr, nrows, result, alpha;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end



const _alpha_norm_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _alpha_norm_col_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _alpha_norm_col_name = "alpha_norm_col_kernel"

function get_alpha_norm_col_kernel()::CuFunction
    k = _alpha_norm_col_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_lock)
    try
        # double-check after locking
        k = _alpha_norm_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_alpha_norm_col_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _alpha_norm_col_name)

        _alpha_norm_col_mod[]    = mod
        _alpha_norm_col_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_lock)
    end
end

function alpha_norm_col(d_G, alpha, result)
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)
    # nBlock = Int64(ceil((ncols + ThreadPerBlock + 1) * 32 / ThreadPerBlock))
    nBlock = cld((ncols + ThreadPerBlock + 1) * 32, ThreadPerBlock)
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(get_alpha_norm_col_kernel(), 
        (CuPtr{Float64}, CuPtr{Int}, CuPtr{Int}, Int, Int, CuPtr{Float64}, Float64), 
        d_G.nzVal, d_G.colVal, d_G.rowPtr, nrows, ncols, result, alpha;
        blocks = nBlock, threads = ThreadPerBlock)
    end
    # delete this since alpha = 1.0
    # result .= result .^ (1.0 / alpha)
end



const _get_row_index_path = utils_path
const _get_row_index_name = "get_row_index"
const _get_row_index_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _get_row_index_lock   = SpinLock()
const _get_row_index_mod    = Ref{Union{Nothing,CuModule}}(nothing)

function get_row_index_kernel()::CuFunction
    k = _get_row_index_kernel[]
    k !== nothing && return k

    lock(_get_row_index_lock)
    try
        # double-check after locking
        k = _get_row_index_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_get_row_index_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _get_row_index_name)

        _get_row_index_mod[]    = mod
        _get_row_index_kernel[] = fun
        return fun
    finally
        unlock(_get_row_index_lock)
    end
end


function get_row_index(d_G, row_idx)
    nnz = length(d_G.nzVal)
    # nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nBlock = cld(nnz + ThreadPerBlock + 1, ThreadPerBlock)
    nrows = size(d_G, 1)
    ncols = size(d_G, 2)

    k = get_row_index_kernel()

    CUDA.@sync begin
        CUDA.cudacall(k, 
        (CuPtr{Int}, Int64, CuPtr{Int}), 
        d_G.rowPtr, nrows, row_idx;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end



const _rescale_coo_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_coo_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_coo_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _rescale_coo_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _rescale_coo_name = "rescale_coo"

function get_rescale_coo_kernel()::CuFunction
    k = _rescale_coo_kernel[]
    k !== nothing && return k

    lock(_rescale_coo_lock)
    try
        # double-check after locking
        k = _rescale_coo_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_rescale_coo_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _rescale_coo_name)

        _rescale_coo_mod[]    = mod
        _rescale_coo_kernel[] = fun
        return fun
    finally
        unlock(_rescale_coo_lock)
    end
end

function rescale_coo(d_G::CUDA.CUSPARSE.CuSparseMatrixCSR, row_scaling::CuArray, col_scaling::CuArray, m::Int64, n::Int64, row_idx::CuArray)
    nnz = length(d_G.nzVal)
    # nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nBlock = cld(nnz + ThreadPerBlock + 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_rescale_coo_kernel(), 
        (CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64),
         d_G.nzVal, row_idx, d_G.colVal, row_scaling, col_scaling, nnz; 
         blocks = nBlock, threads = ThreadPerBlock)
    end
end




const _max_abs_row_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_elementwise_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _max_abs_row_elementwise_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _max_abs_row_elementwise_name = "max_abs_row_elementwise_kernel"

function get_max_abs_row_elementwise_kernel()::CuFunction
    k = _max_abs_row_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_elementwise_lock)
    try
        # double-check after locking
        k = _max_abs_row_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_max_abs_row_elementwise_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _max_abs_row_elementwise_name)

        _max_abs_row_elementwise_mod[]    = mod
        _max_abs_row_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_elementwise_lock)
    end
end

function max_abs_row_elementwise(d_G, row_idx, result)
    # Use appropriate methods to extract CuSparseMatrixCSR data
    values = d_G.nzVal     # Access non-zero values directly
    nrows = size(d_G, 1)   # Number of rows
    nrows = Int64(nrows)
    result .= 0.0
    nnz = length(d_G.nzVal)
    # nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nBlock = cld(nnz + ThreadPerBlock + 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_max_abs_row_elementwise_kernel(), 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, row_idx, nnz, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


const _max_abs_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_elementwise_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _max_abs_col_elementwise_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _max_abs_col_elementwise_name = "max_abs_col_elementwise_kernel"

function get_max_abs_col_elementwise_kernel()::CuFunction
    k = _max_abs_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_elementwise_lock)
    try
        # double-check after locking
        k = _max_abs_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_max_abs_col_elementwise_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _max_abs_col_elementwise_name)

        _max_abs_col_elementwise_mod[]    = mod
        _max_abs_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_elementwise_lock)
    end
end



function max_abs_col_elementwise(d_G, result)
    values = d_G.nzVal
    col_idx = d_G.colVal
    ncols = size(d_G, 2)
    ncols = Int64(ncols)
    result .= 0.0
    nnz = length(d_G.nzVal)
    # nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nBlock = cld(nnz + ThreadPerBlock + 1, ThreadPerBlock)
    k = get_max_abs_col_elementwise_kernel()
    CUDA.@sync begin
        CUDA.cudacall(k, 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}), 
        values, col_idx, nnz, result;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


const _alpha_norm_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_elementwise_lock   = SpinLock()

# set this to your PTX/CUBIN path once
const _alpha_norm_col_elementwise_path = utils_path   # e.g. joinpath(@__DIR__, "kernels", "replace_inf.ptx")
const _alpha_norm_col_elementwise_name = "alpha_norm_col_elementwise_kernel"

function get_alpha_norm_col_elementwise_kernel()::CuFunction
    k = _alpha_norm_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_elementwise_lock)
    try
        # double-check after locking
        k = _alpha_norm_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)   # ensure context exists

        # Load once
        bytes = read(_alpha_norm_col_elementwise_path)     # Vector{UInt8}
        mod   = CuModule(bytes)            # if this is PTX text you can also do CuModule(String(bytes))
        fun   = CuFunction(mod, _alpha_norm_col_elementwise_name)

        _alpha_norm_col_elementwise_mod[]    = mod
        _alpha_norm_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_elementwise_lock)
    end
end
function alpha_norm_col_elementwise(d_G, alpha, result)
    nnz = length(d_G.nzVal)
    ncols = size(d_G, 2)
    # nBlock = Int64(ceil((nnz + ThreadPerBlock + 1) / ThreadPerBlock))
    nBlock = cld(nnz + ThreadPerBlock + 1, ThreadPerBlock)
    result .= 0.0
    CUDA.@sync begin
        CUDA.cudacall(get_alpha_norm_col_elementwise_kernel(), 
        (CuPtr{Float64}, CuPtr{Int}, Int64, CuPtr{Float64}, Float64, Int64), 
        d_G.nzVal, d_G.colVal, nnz, result, alpha, ncols;
        blocks = nBlock, threads = ThreadPerBlock)
    end
    # delete this since alpha = 1.0
    # result .= result .^ (1.0 / alpha)
end
