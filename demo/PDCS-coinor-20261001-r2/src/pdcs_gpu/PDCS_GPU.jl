__precompile__()
module PDCS_GPU

using Random, SparseArrays, LinearAlgebra
using Printf
using Match
using DataStructures
using Base.Threads
using JuMP
using Polynomials
using Statistics
using CUDA, PythonCall
using CUDA.CUSPARSE
using Libdl
using Logging
using Dates
import Base: unsafe_convert
using Base.Threads: SpinLock
using SnoopPrecompile

# Logging.with_logger(Logging.NullLogger()) do
#     CUDA.allowscalar(true)
# end

const rpdhg_float = Float64
const rpdhg_int = Int32
const positive_zero = 1e-20
const negative_zero = -1e-20
const proj_rel_tol = 1e-12
const proj_abs_tol = 1e-16
const ThreadPerBlock = 256

# Public-entry defaults shared by direct solves, solver structs, preprocessing,
# and the MOI wrapper. Keeping these in one place prevents the entry points
# from silently drifting apart.
const DEFAULT_SCALAR_CONE_RESCALING = false
const DEFAULT_USE_ADAPTIVE_DIAGONAL_SCALAR_RESCALING = true
const DEFAULT_USE_ADAPTIVE_STEP = false
const DEFAULT_USE_ADAPTIVE_STEP_SIZE_WEIGHT = true

const MODULE_DIR = @__DIR__

# cuBLAS reproducibility is process-wide because its workspace policy is read
# from the environment and handles are cached. Configure it before the first
# PDCS-owned handle is created. Users can opt out before importing PDCS.
const _cublas_reproducible_enabled = Ref(true)

@inline function _pdcs_env_enabled(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    return raw ∉ ("0", "false", "no", "off")
end

include("./csc_to_csr.jl")
include("./plain_multi_logger.jl")


## standard formulation of the optimization problem ##

# def var solver and methods

# main algorithm

const _kernlib_ref = Ref{Ptr{Cvoid}}(C_NULL)
const few_block_proj_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _GRIDWISE_NATIVE_ABI_VERSION = 2
const _gridwise_native_abi_version = Ref{Union{Nothing,Int}}(nothing)

# Grid-wise projection is the only PDCS path that crosses a Julia/CUDA shared
# library boundary.  Keep its state explicit so a stale or incompatible
# artifact cannot turn a package import into an opaque CUDA failure.
const _gridwise_native_enabled = Ref(false)
const _gridwise_artifact_dir = Ref("")
const _gridwise_native_failure = Ref{Union{Nothing,String}}(nothing)
const _gridwise_runtime_state = Ref{Symbol}(:uninitialized)
const _gridwise_fallback_warned = Ref(false)

@inline function _gridwise_mode()
    # Scientific runs must exercise the requested grid-wise implementation.
    # Compatibility fallback remains available, but only by explicit opt-in.
    mode = lowercase(strip(get(ENV, "PDCS_GRIDWISE_MODE", "native")))
    mode in ("auto", "native", "block") && return mode
    @warn "Invalid PDCS_GRIDWISE_MODE; using strict native mode" mode
    return "native"
end

@inline function _gridwise_artifact_directory()
    configured = strip(get(ENV, "PDCS_CUDA_PROJECTION_ARTIFACT_DIR", ""))
    root = isempty(configured) ? joinpath(@__DIR__, "cuda") : configured
    return normpath(abspath(root))
end

@inline function _gridwise_required_artifacts()
    return (
        "libfew_block_proj.so",
        "moderate_block_proj.ptx",
        "sufficient_block_proj.ptx",
        "massive_block_proj.ptx",
        "utils.ptx",
    )
end

function _disable_gridwise_native!(reason::AbstractString)
    _gridwise_native_enabled[] = false
    _gridwise_runtime_state[] = :disabled
    _gridwise_native_failure[] = String(reason)
    _gridwise_native_abi_version[] = nothing
    @warn "Grid-wise native projection is unavailable" reason mode = _gridwise_mode()
    return nothing
end


function __init__()
    _gridwise_native_enabled[] = false
    _gridwise_native_abi_version[] = nothing
    _gridwise_native_failure[] = nothing
    _gridwise_runtime_state[] = :uninitialized
    _gridwise_fallback_warned[] = false
    _heterogeneous_projection_enabled[] =
        lowercase(get(ENV, "PDCS_ENABLE_HETEROGENEOUS_PROJECTION", "1")) ∉
        ("0", "false", "no", "off")
    _cublas_reproducible_enabled[] =
        _pdcs_env_enabled("PDCS_CUBLAS_REPRODUCIBLE", true)
    if _cublas_reproducible_enabled[]
        workspace_config = get(ENV, "CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        workspace_config in (":16:8", ":4096:8") || throw(ArgumentError(
            "reproducible cuBLAS requires CUBLAS_WORKSPACE_CONFIG=:16:8 " *
            "or :4096:8; got $(repr(workspace_config))",
        ))
        ENV["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
        # Apply the same prescribed-precision policy to CUDA.jl-managed
        # cuBLAS handles. The separately owned grid-wise handle is configured
        # explicitly when it is created in gpu_kernel.jl.
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    end
    artifact_dir = _gridwise_artifact_directory()
    _gridwise_artifact_dir[] = artifact_dir
    mode = _gridwise_mode()
    if mode == "block"
        _disable_gridwise_native!("PDCS_GRIDWISE_MODE=block")
        return
    end
    CUDA.functional() || begin
        _gridwise_runtime_state[] = :unavailable
        _gridwise_native_failure[] = "CUDA is not functional"
        return
    end

    # Open the projection library, but do not assume that a library with the
    # right filename has the current ABI.  In particular, old builds created
    # the cuBLAS handle through CUDA.jl and do not export the native ownership
    # helpers required by the current implementation.
    libpath = joinpath(artifact_dir, "libfew_block_proj.so")
    if !isfile(libpath)
        _disable_gridwise_native!("missing native projection library: $libpath")
        mode == "native" && error(_gridwise_native_failure[])
        return
    end
    try
        _kernlib_ref[] = Libdl.dlopen(libpath)
        few_block_proj_ptr[] = Libdl.dlsym(_kernlib_ref[], :few_block_proj)
        few_block_proj_ptr[] != C_NULL ||
            error("symbol few_block_proj is NULL in $libpath")
        for symbol in (
            :create_cublas_handle_inner,
            :configure_cublas_handle_inner,
            :cublas_handle_configuration_inner,
            :destroy_cublas_handle_inner,
            :pdcs_gridwise_abi_version,
        )
            Libdl.dlsym(_kernlib_ref[], symbol)
        end
        abi_ptr = Libdl.dlsym(_kernlib_ref[], :pdcs_gridwise_abi_version)
        abi_version = Int(ccall(abi_ptr, Cint, ()))
        abi_version == _GRIDWISE_NATIVE_ABI_VERSION || error(
            "native grid-wise ABI version $abi_version does not match " *
            "required version $(_GRIDWISE_NATIVE_ABI_VERSION)",
        )
        _gridwise_native_abi_version[] = abi_version
        _gridwise_native_enabled[] = true
        _gridwise_runtime_state[] = :uninitialized
        register_gridWise_cublas_cleanup!()
    catch err
        _kernlib_ref[] = C_NULL
        few_block_proj_ptr[] = C_NULL
        _disable_gridwise_native!(
            "incompatible native projection library $libpath: " *
            sprint(showerror, err),
        )
        mode == "native" && rethrow()
    end
end

"""Return non-invasive grid-wise artifact and runtime diagnostics."""
function gridWise_runtime_status()
    artifact_dir = isempty(_gridwise_artifact_dir[]) ?
        _gridwise_artifact_directory() : _gridwise_artifact_dir[]
    required = _gridwise_required_artifacts()
    missing = String[
        joinpath(artifact_dir, name) for name in required
        if !isfile(joinpath(artifact_dir, name))
    ]
    configuration = nothing
    if _gridwise_runtime_state[] === :passed &&
       isdefined(@__MODULE__, :_gridWise_cublas_handle)
        handle = _gridWise_cublas_handle[]
        if handle !== nothing && handle.handle != C_NULL
            try
                configuration = gridWise_cublas_configuration()
            catch
                configuration = nothing
            end
        end
    end
    return (
        mode = _gridwise_mode(),
        artifact_dir = artifact_dir,
        missing_artifacts = missing,
        native_enabled = _gridwise_native_enabled[],
        state = _gridwise_runtime_state[],
        failure = _gridwise_native_failure[],
        abi_version = _gridwise_native_abi_version[],
        required_abi_version = _GRIDWISE_NATIVE_ABI_VERSION,
        configuration = configuration,
    )
end

"""Run the one-time grid-wise cuBLAS/alias self-test and return its status."""
function check_gridWise_runtime!()
    isdefined(@__MODULE__, :_ensure_gridwise_runtime!) ||
        error("PDCS_GPU projection code has not finished loading")
    lock(_gridwise_call_lock)
    try
        _ensure_gridwise_runtime!()
        return gridWise_runtime_status()
    finally
        unlock(_gridwise_call_lock)
    end
end



## general formulation of the optimization problem ##
include("./gpu_kernel.jl")
include("./def_struct.jl")
include("./exp_proj.jl")
include("./soc_rsoc_proj.jl")
include("./projection_strategy.jl")
include("./def_rpdhg_gen.jl")
include("./preprocess.jl")
include("./postprocess.jl")

# # # main algorithm
include("./termination.jl")
include("rpdhg_alg_gpu_gen_scaling.jl")
include("./rpdhg_alg_gpu_gen.jl")



include("./utils.jl")
include("./MOI_wrapper/MOI_wrapper.jl")
include("../MOI_bulk_cache.jl")
include("./cvxpy_wrapper/py2jl.jl")
include("./cvxpy_wrapper/data_updating.jl")

include("./precompile.jl")
redirect_stdout(devnull) do; 
    SnoopPrecompile.@precompile_all_calls begin
        if CUDA.has_cuda() &&
           get(ENV, "PDCS_SKIP_GPU_PRECOMPILE", "0") != "1" &&
           _precompile_projection_artifacts_ready()
            @info "============precompile PDCS_GPU============"
            try
                __precompile_gpu()
                __precompile_gpu_clean_pointer()
                @info "============precompile PDCS_GPU done============"
            catch err
                # GPU contexts and native handles are machine-local.  A
                # package precompile must remain usable when the build host
                # has a different driver, device, or CUDA.jl artifact.
                @warn "GPU precompile smoke test skipped; runtime self-test will run after loading PDCS" exception = (err, catch_backtrace())
            end
        else
            @info "============ PDCS_GPU GPU precompile skipped ============"
        end
    end
end

export rpdhg_gpu_solve, conic_cache_from_data, model_from_conic_data
export enable_projection_work_profile!, disable_projection_work_profile!
export reset_projection_work_profile!, projection_work_profile_summary
export gridWise_runtime_status, check_gridWise_runtime!


end
