"""
GPU Kernel Wrapper Functions for PDCS_GPU

This file provides Julia wrappers for CUDA kernels used in the PDCS GPU solver.
It implements lazy loading of PTX (CUDA kernel) files with thread-safe initialization.

Main components:
1. Block projection kernels (massive, moderate, sufficient) - for projecting onto constraint sets
2. cuBLAS handle management - for using cuBLAS library functions
3. Few block projection - uses a shared library (.so) for complex projections
4. Utility kernels - various helper functions for the RPDHG algorithm
5. Matrix scaling and norm computation kernels - for preconditioning and scaling
6. Elementwise operations - for sparse matrix operations

All kernels use a lazy loading pattern with thread-safe initialization to ensure
kernels are loaded only once, even in multi-threaded environments.
"""

# ============================================================================
# Section 1: Block Projection Kernels
# ============================================================================
# These kernels perform projections onto constraint sets for different block sizes.
# The kernels are loaded from PTX files and use lazy initialization.

# ----------------------------------------------------------------------------
# Optional projection-work profiler (rebuttal experiment 1)
# ----------------------------------------------------------------------------

"""Host mirror of `cuda/root_profile.h::RootProfileRecord` (88-byte ABI)."""
struct ProjectionProfileRecord
    branch_code::Int32
    interval_expansion_iterations::Int32
    newton_attempts::Int32
    newton_accepts::Int32
    bisection_iterations::Int32
    oracle_evaluations::Int32
    gradient_evaluations::Int32
    warm_start_attempted::Int32
    warm_start_accepted::Int32
    max_iter_reached::Int32
    termination_reason::Int32
    output_finite::Int32
    vector_vector_reductions::Int32
    final_residual::Float64
    final_bracket_left::Float64
    final_bracket_right::Float64
    normalized_bracket_width::Float64
end

@assert isbitstype(ProjectionProfileRecord)
@assert sizeof(ProjectionProfileRecord) == 88

mutable struct ProjectionWorkCount
    projection_events::Int64
    vector_vector_reductions::Int64
    oracle_evaluations::Int64
    gradient_evaluations::Int64
    interval_expansion_iterations::Int64
    bisection_iterations::Int64
    bisection_events::Int64
    newton_attempts::Int64
    newton_accepts::Int64
    newton_attempt_events::Int64
    newton_accept_events::Int64
    warm_start_attempts::Int64
    warm_start_accepts::Int64
    max_iter_reached::Int64
    nonfinite_outputs::Int64
end

mutable struct ProjectionProfileKernelState
    module_handle::CuModule
    projection_kernel::CuFunction
    initialize_kernel::CuFunction
    records::CuArray{ProjectionProfileRecord,1,CUDA.DeviceMemory}
end

const _projection_work_profile_enabled = Ref(false)
const _projection_work_profile_scope = Ref{Symbol}(:all)
const _projection_work_profile_iteration_active = Ref(false)
const _projection_work_profile_iteration_index = Ref(0)
const _projection_work_profile_warmup_iterations = Ref(0)
const _projection_work_profile_sample_iterations = Ref(typemax(Int))
const _projection_work_profile_counts =
    Dict{Tuple{Symbol,Int64,Bool},ProjectionWorkCount}()
const _projection_work_profile_lock = ReentrantLock()
const _projection_profile_states = Dict{Symbol,ProjectionProfileKernelState}()

@inline _projection_artifact_path(filename) = joinpath(
    get(ENV, "PDCS_CUDA_PROJECTION_ARTIFACT_DIR", joinpath(MODULE_DIR, "cuda")),
    filename,
)

const _projection_profile_paths = Dict(
    :threadWise => _projection_artifact_path("massive_block_proj_profile.ptx"),
    :blockWise => _projection_artifact_path("moderate_block_proj_profile.ptx"),
    :warpWise => _projection_artifact_path("sufficient_block_proj_profile.ptx"),
)
const _projection_profile_kernel_names = Dict(
    :threadWise => "massive_block_proj",
    :blockWise => "moderate_block_proj",
    :warpWise => "sufficient_block_proj",
)

@inline function _projection_profile_cone(code::Int64)
    code in (5, 6, 7, 20, 21, 22) && return :soc
    code in (8, 9, 10, 23, 24, 25) && return :rsoc
    code in (13, 14, 15, 26, 27) && return :exp
    code in (11, 12, 16, 28, 29) && return :dual_exp
    return nothing
end

@inline _projection_profile_is_diagonal(code::Int64) =
    code in (6, 9, 12, 15, 22, 25, 27, 29)

@inline _projection_work_profile_should_record() =
    _projection_work_profile_enabled[] &&
    (_projection_work_profile_scope[] === :all ||
     _projection_work_profile_iteration_active[])

@inline function _begin_projection_work_iteration!()
    if _projection_work_profile_enabled[] &&
       _projection_work_profile_scope[] === :pdhg_iterations
        index = (_projection_work_profile_iteration_index[] += 1)
        warmup = _projection_work_profile_warmup_iterations[]
        samples = _projection_work_profile_sample_iterations[]
        _projection_work_profile_iteration_active[] =
            index > warmup &&
            (samples == typemax(Int) || index - warmup <= samples)
    end
    return
end

@inline function _end_projection_work_iteration!()
    if _projection_work_profile_scope[] === :pdhg_iterations
        _projection_work_profile_iteration_active[] = false
    end
    return
end

function _get_projection_profile_state(strategy::Symbol)
    haskey(_projection_profile_paths, strategy) ||
        throw(ArgumentError("unsupported projection profiling strategy: $strategy"))
    lock(_projection_work_profile_lock)
    try
        haskey(_projection_profile_states, strategy) &&
            return _projection_profile_states[strategy]

        CUDA.functional() || error("CUDA is not functional")
        profile_path = joinpath(
            get(
                ENV,
                "PDCS_CUDA_PROJECTION_ARTIFACT_DIR",
                joinpath(MODULE_DIR, "cuda"),
            ),
            basename(_projection_profile_paths[strategy]),
        )
        isfile(profile_path) || error(
            "missing $profile_path; run `make rebuild-profile` in src/pdcs_gpu/cuda",
        )
        profile_sources = (
            joinpath(MODULE_DIR, "cuda/root_profile.h"),
            joinpath(MODULE_DIR, "cuda/exp_proj.cu"),
            joinpath(MODULE_DIR, "cuda/massive_block_proj.cu"),
            joinpath(MODULE_DIR, "cuda/moderate_block_proj.cu"),
            joinpath(MODULE_DIR, "cuda/sufficient_block_proj.cu"),
        )
        mtime(profile_path) >= maximum(mtime, profile_sources) || error(
            "stale projection profile PTX; run `make rebuild-profile` in src/pdcs_gpu/cuda",
        )

        module_handle = CuModule(read(profile_path))
        state = ProjectionProfileKernelState(
            module_handle,
            CuFunction(module_handle, _projection_profile_kernel_names[strategy]),
            CuFunction(module_handle, "pdcs_root_profile_initialize"),
            CuArray{ProjectionProfileRecord}(undef, 0),
        )
        _projection_profile_states[strategy] = state
        return state
    finally
        unlock(_projection_work_profile_lock)
    end
end

function _ensure_projection_profile_capacity!(
    state::ProjectionProfileKernelState,
    record_count::Int,
)
    if length(state.records) < record_count
        state.records = CuArray{ProjectionProfileRecord}(undef, record_count)
    end
    return state
end

function _accumulate_projection_work!(
    records::Vector{ProjectionProfileRecord},
    cone_sizes::Vector{Int64},
    projection_codes::Vector{Int64},
    block_count::Int64,
)
    count = min(Int(block_count), length(records), length(cone_sizes), length(projection_codes))
    lock(_projection_work_profile_lock)
    try
        for i in 1:count
            code = projection_codes[i]
            cone = _projection_profile_cone(code)
            cone === nothing && continue
            key = (cone, cone_sizes[i], _projection_profile_is_diagonal(code))
            work = get!(_projection_work_profile_counts, key) do
                ProjectionWorkCount(
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                )
            end
            record = records[i]
            work.projection_events += 1
            work.vector_vector_reductions += record.vector_vector_reductions
            work.oracle_evaluations += record.oracle_evaluations
            work.gradient_evaluations += record.gradient_evaluations
            work.interval_expansion_iterations +=
                record.interval_expansion_iterations
            work.bisection_iterations += record.bisection_iterations
            work.bisection_events += record.bisection_iterations > 0
            work.newton_attempts += record.newton_attempts
            work.newton_accepts += record.newton_accepts
            work.newton_attempt_events += record.newton_attempts > 0
            work.newton_accept_events += record.newton_accepts > 0
            work.warm_start_attempts += record.warm_start_attempted
            work.warm_start_accepts += record.warm_start_accepted
            work.max_iter_reached += record.max_iter_reached
            work.nonfinite_outputs += record.output_finite == 0
        end
    finally
        unlock(_projection_work_profile_lock)
    end
    return
end

function _profile_block_projection!(
    strategy::Symbol,
    vec,
    bl,
    bu,
    D_scaled,
    D_scaled_squared,
    D_scaled_mul_x,
    temp,
    t_warm_start,
    gpu_head_start,
    gpu_ns,
    blkNum::Int64,
    proj_type,
    abs_tol::Float64,
    rel_tol::Float64;
    blocks::Int,
    threads::Int,
    record_count::Int,
)
    state = _ensure_projection_profile_capacity!(
        _get_projection_profile_state(strategy),
        record_count,
    )
    CUDA.@sync begin
        CUDA.cudacall(
            state.initialize_kernel,
            (CuPtr{ProjectionProfileRecord}, Int64),
            state.records,
            Int64(record_count);
            blocks = cld(record_count, ThreadPerBlock),
            threads = ThreadPerBlock,
        )
        CUDA.cudacall(
            state.projection_kernel,
            (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
             CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
             CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64}, Float64, Float64),
            vec,
            bl,
            bu,
            D_scaled,
            D_scaled_squared,
            D_scaled_mul_x,
            temp,
            t_warm_start,
            gpu_head_start,
            gpu_ns,
            Cint(blkNum),
            proj_type,
            abs_tol,
            rel_tol;
            blocks = blocks,
            threads = threads,
        )
    end
    _accumulate_projection_work!(
        Array(@view state.records[1:record_count]),
        Array(gpu_ns),
        Array(proj_type),
        blkNum,
    )
    return
end

"""
    enable_projection_work_profile!(;
        scope=:all, warmup_iterations=0, sample_iterations=typemax(Int))

Enable the diagnostic projection kernels and reset their host counters. The
counter records one full-cone elementwise product followed by a scalar
reduction. This mode adds synchronization and device-to-host copies and must
not be used for runtime measurements.

With `scope=:pdhg_iterations`, only the primal/dual projection pair in each
PDHG step is recorded, matching the `M*T` denominator in R3.5. `scope=:all`
also includes termination, restart, and infeasibility-check projections.
"""
function enable_projection_work_profile!(;
    scope::Symbol = :all,
    warmup_iterations::Integer = 0,
    sample_iterations::Integer = typemax(Int),
)
    scope in (:all, :pdhg_iterations) || throw(ArgumentError(
        "projection profile scope must be :all or :pdhg_iterations",
    ))
    warmup_iterations >= 0 || throw(ArgumentError(
        "projection profile warmup_iterations must be nonnegative",
    ))
    sample_iterations > 0 || throw(ArgumentError(
        "projection profile sample_iterations must be positive",
    ))
    reset_projection_work_profile!()
    _projection_work_profile_scope[] = scope
    _projection_work_profile_iteration_active[] = false
    _projection_work_profile_iteration_index[] = 0
    _projection_work_profile_warmup_iterations[] = Int(warmup_iterations)
    _projection_work_profile_sample_iterations[] = Int(sample_iterations)
    _projection_work_profile_enabled[] = true
    return
end

"""Disable diagnostic kernels without discarding the accumulated summary."""
function disable_projection_work_profile!()
    _projection_work_profile_enabled[] = false
    _projection_work_profile_iteration_active[] = false
    return projection_work_profile_summary()
end

"""Clear the accumulated projection-work counters."""
function reset_projection_work_profile!()
    lock(_projection_work_profile_lock)
    try
        empty!(_projection_work_profile_counts)
    finally
        unlock(_projection_work_profile_lock)
    end
    return
end

"""Return total and per-cone projection-work counts collected so far."""
function projection_work_profile_summary()
    lock(_projection_work_profile_lock)
    try
        by_cone = [
            (
                cone_type = key[1],
                cone_dimension = key[2],
                diagonal = key[3],
                projection_events = count.projection_events,
                vector_vector_reductions = count.vector_vector_reductions,
                average_vector_vector_reductions = count.projection_events == 0 ?
                    0.0 : count.vector_vector_reductions / count.projection_events,
                oracle_evaluations = count.oracle_evaluations,
                average_function_evaluations =
                    count.projection_events == 0 ? 0.0 :
                    count.oracle_evaluations / count.projection_events,
                gradient_evaluations = count.gradient_evaluations,
                average_gradient_evaluations =
                    count.projection_events == 0 ? 0.0 :
                    count.gradient_evaluations / count.projection_events,
                interval_expansion_iterations =
                    count.interval_expansion_iterations,
                bisection_iterations = count.bisection_iterations,
                bisection_events = count.bisection_events,
                bisection_event_fraction = count.projection_events == 0 ?
                    0.0 : count.bisection_events / count.projection_events,
                newton_attempts = count.newton_attempts,
                newton_accepts = count.newton_accepts,
                newton_attempt_events = count.newton_attempt_events,
                newton_accept_events = count.newton_accept_events,
                newton_attempt_event_fraction = count.projection_events == 0 ?
                    0.0 : count.newton_attempt_events / count.projection_events,
                newton_accept_event_fraction = count.projection_events == 0 ?
                    0.0 : count.newton_accept_events / count.projection_events,
                warm_start_attempts = count.warm_start_attempts,
                warm_start_accepts = count.warm_start_accepts,
                max_iter_reached = count.max_iter_reached,
                nonfinite_outputs = count.nonfinite_outputs,
            ) for (key, count) in sort!(
                collect(_projection_work_profile_counts);
                by = pair -> (
                    string(pair.first[1]),
                    pair.first[2],
                    pair.first[3],
                ),
            )
        ]
        total_events = mapreduce(
            row -> row.projection_events,
            +,
            by_cone;
            init = 0,
        )
        total_reductions = mapreduce(
            row -> row.vector_vector_reductions,
            +,
            by_cone;
            init = 0,
        )
        total_functions = mapreduce(
            row -> row.oracle_evaluations,
            +,
            by_cone;
            init = 0,
        )
        total_gradients = mapreduce(
            row -> row.gradient_evaluations,
            +,
            by_cone;
            init = 0,
        )
        return (
            enabled = _projection_work_profile_enabled[],
            scope = _projection_work_profile_scope[],
            warmup_iterations = _projection_work_profile_warmup_iterations[],
            sample_iterations = _projection_work_profile_sample_iterations[],
            projection_events = total_events,
            vector_vector_reductions = total_reductions,
            average_vector_vector_reductions = total_events == 0 ?
                0.0 : total_reductions / total_events,
            function_evaluations = total_functions,
            average_function_evaluations = total_events == 0 ?
                0.0 : total_functions / total_events,
            gradient_evaluations = total_gradients,
            average_gradient_evaluations = total_events == 0 ?
                0.0 : total_gradients / total_events,
            by_cone = by_cone,
        )
    finally
        unlock(_projection_work_profile_lock)
    end
end

# ----------------------------------------------------------------------------
# Massive Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for large blocks. Loads the PTX file and provides thread-safe access.
# The kernel projects vectors onto constraint sets for blocks with many elements.

# Storage for the loaded CUDA module (PTX file)
const _massive_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
# Storage for the CUDA function (kernel entry point)
const _massive_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _massive_soc_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _massive_block_proj_indexed_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _simple_block_proj_indexed_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
# Spin lock for thread-safe lazy initialization
const _massive_block_proj_lock   = SpinLock()

# Path to the PTX file containing the compiled CUDA kernel
# Name of the kernel function within the PTX file
const _massive_block_proj_name = "massive_block_proj"
const _massive_soc_block_proj_name = "massive_soc_block_proj"
const _massive_block_proj_indexed_name = "massive_block_proj_indexed"
const _simple_block_proj_indexed_name = "simple_block_proj_indexed"

struct ProjectionLayoutValidation
    gpu_ns::Any
    proj_type::Any
    block_count::Int64
    vector_length::Int
    device::Any
end

const _projection_layout_validations = IdDict{Any,ProjectionLayoutValidation}()
const _projection_layout_validation_lock = ReentrantLock()

"""Copy and validate immutable cone metadata once per device layout."""
function _validate_device_projection_layout_once(
    gpu_head_start::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    gpu_ns::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    proj_type::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    blkNum::Int64, vector_length::Int,
)
    blkNum == 0 && return nothing
    lock(_projection_layout_validation_lock)
    try
        cached = get(_projection_layout_validations, gpu_head_start, nothing)
        active_device = CUDA.device()
        if cached !== nothing && cached.gpu_ns === gpu_ns &&
           cached.proj_type === proj_type && cached.block_count == blkNum &&
           cached.vector_length == vector_length &&
           cached.device == active_device
            return nothing
        end
        heads = Array(@view gpu_head_start[1:blkNum])
        sizes = Array(@view gpu_ns[1:blkNum])
        types = Array(@view proj_type[1:blkNum])
        previous_stop = 0
        for index in 1:blkNum
            head = heads[index]
            dimension = sizes[index]
            code = types[index]
            head >= previous_stop || throw(ArgumentError(
                "cone $index starts at $head and overlaps or precedes the prior cone",
            ))
            dimension > 0 || throw(ArgumentError(
                "cone $index must have positive size; got $dimension",
            ))
            head <= vector_length - dimension || throw(DimensionMismatch(
                "cone $index range [$head, $(head + dimension)) exceeds " *
                "vector length $vector_length",
            ))
            0 <= code <= 29 || throw(ArgumentError(
                "cone $index has unknown projection type $code",
            ))
            if _is_soc_projection_code(code)
                dimension >= 2 || throw(DimensionMismatch(
                    "SOC cone $index must have at least two entries",
                ))
            elseif _is_exp_projection_code(code)
                dimension == 3 || throw(DimensionMismatch(
                    "exponential cone $index must have exactly three entries; " *
                    "got $dimension",
                ))
            end
            previous_stop = head + dimension
        end
        _projection_layout_validations[gpu_head_start] =
            ProjectionLayoutValidation(
                gpu_ns, proj_type, blkNum, vector_length, active_device,
            )
        return nothing
    finally
        unlock(_projection_layout_validation_lock)
    end
end

"""Validate device-buffer ABI invariants shared by PTX projection wrappers."""
function _validate_device_projection_arguments(
    vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T,
    D_scaled_mul_x::T, temp::T, t_warm_start::T,
    gpu_head_start::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    gpu_ns::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}, blkNum::Int64,
    proj_type::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    abs_tol::Float64, rel_tol::Float64,
) where {T<:CuArray}
    0 <= blkNum <= typemax(Cint) || throw(ArgumentError(
        "projection block count must be in 0:$(typemax(Cint)); got $blkNum",
    ))
    for (name, values) in (
        ("gpu_head_start", gpu_head_start), ("gpu_ns", gpu_ns),
        ("proj_type", proj_type),
    )
        length(values) >= blkNum || throw(DimensionMismatch(
            "$name has length $(length(values)), but blkNum=$blkNum",
        ))
    end
    length(t_warm_start) >= blkNum || throw(DimensionMismatch(
        "t_warm_start has length $(length(t_warm_start)), but blkNum=$blkNum",
    ))
    isfinite(abs_tol) && abs_tol > 0 || throw(ArgumentError(
        "projection abs_tol must be finite and positive; got $abs_tol",
    ))
    isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError(
        "projection rel_tol must be finite and positive; got $rel_tol",
    ))
    active_device = CUDA.device()
    required_length = length(vec)
    for (name, values) in (
        ("vec", vec), ("bl", bl), ("bu", bu),
        ("D_scaled", D_scaled),
        ("D_scaled_squared", D_scaled_squared),
        ("D_scaled_mul_x", D_scaled_mul_x), ("temp", temp),
    )
        ndims(values) == 1 || throw(DimensionMismatch(
            "$name must be a one-dimensional CuArray",
        ))
        eltype(values) === Float64 || throw(ArgumentError(
            "$name must use Float64 storage for the PTX ABI; got " *
            "$(eltype(values))",
        ))
        length(values) >= required_length || throw(DimensionMismatch(
            "$name has length $(length(values)); expected at least $required_length",
        ))
        CUDA.device(values) == active_device || throw(ArgumentError(
            "$name is not allocated on the active CUDA device",
        ))
    end
    for (name, values) in (
        ("gpu_head_start", gpu_head_start), ("gpu_ns", gpu_ns),
        ("proj_type", proj_type),
    )
        CUDA.device(values) == active_device || throw(ArgumentError(
            "$name is not allocated on the active CUDA device",
        ))
    end
    _validate_device_projection_layout_once(
        gpu_head_start, gpu_ns, proj_type, blkNum, required_length,
    )
    return nothing
end

struct HeterogeneousProjectionPlan
    masked_projection_types::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    thread_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    thread_cone_count::Int64
    serial_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    serial_cone_count::Int64
    compact_soc_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    compact_soc_cone_count::Int64
    compact_warp_soc_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    compact_warp_soc_cone_count::Int64
    simple_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    simple_cone_count::Int64
    native_cone_indices::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}
    native_cone_count::Int64
    fully_compacted::Bool
    soc_only::Bool
end

const _heterogeneous_projection_enabled = Ref(true)
# The endpoint-safe compact diagonal-SOC path passes the full joint_FC_12
# regression.  Keep a mutable switch for benchmark A/Bs against the native
# cooperative reduction without disabling production compaction.
const _diagonal_soc_compaction_enabled = Ref(true)
struct CachedHeterogeneousProjectionPlan
    gpu_ns::Any
    blk_num::Int64
    plan::HeterogeneousProjectionPlan
end

const _heterogeneous_projection_plans =
    IdDict{Any,CachedHeterogeneousProjectionPlan}()
const _heterogeneous_projection_plan_lock = ReentrantLock()
const _serial_compaction_minimum = 768
const _primal_exp_compaction_minimum = 512
const _thread_soc_dimension_limit = 4
const _thread_soc_compaction_minimum = 256
const _warp_soc_dimension_limit = 32
const _warp_soc_compaction_minimum = 64

@inline _is_exp_projection_code(code::Int64) =
    code in (11, 12, 13, 14, 15, 16, 26, 27, 28, 29)

@inline _is_primal_exp_projection_code(code::Int64) =
    code in (13, 14, 15, 26, 27)

@inline _is_soc_projection_code(code::Int64) =
    code in (5, 6, 7, 20, 21, 22)

@inline _is_diagonal_soc_projection_code(code::Int64) =
    code in (6, 22)

@inline _is_rsoc_projection_code(code::Int64) =
    code in (8, 9, 10, 23, 24, 25)

@inline _is_simple_projection_code(code::Int64) =
    code in (0, 1, 2, 3, 4, 17, 18, 19)

function _get_heterogeneous_projection_plan(
    gpu_ns::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
    blkNum::Int64,
    proj_type::CUDA.CuArray{Int64,1,CUDA.DeviceMemory},
)
    lock(_heterogeneous_projection_plan_lock)
    try
        cached = get(_heterogeneous_projection_plans, proj_type, nothing)
        if cached !== nothing && cached.gpu_ns === gpu_ns &&
           cached.blk_num == blkNum
            return cached.plan
        end

        0 <= blkNum <= typemax(Cint) || throw(ArgumentError(
            "projection block count must fit Cint; got $blkNum",
        ))
        length(gpu_ns) >= blkNum || throw(DimensionMismatch(
            "gpu_ns has length $(length(gpu_ns)), but blkNum=$blkNum",
        ))
        length(proj_type) >= blkNum || throw(DimensionMismatch(
            "proj_type has length $(length(proj_type)), but blkNum=$blkNum",
        ))
        cpu_ns = Array(@view gpu_ns[1:blkNum])
        cpu_types = Array(@view proj_type[1:blkNum])
        for index in 1:blkNum
            dimension = cpu_ns[index]
            code = cpu_types[index]
            dimension > 0 || throw(ArgumentError(
                "cone $index must have positive size; got $dimension",
            ))
            0 <= code <= 29 || throw(ArgumentError(
                "cone $index has unknown projection type $code",
            ))
            if _is_soc_projection_code(code)
                dimension >= 2 || throw(DimensionMismatch(
                    "SOC cone $index must have at least two entries",
                ))
            elseif _is_exp_projection_code(code)
                dimension == 3 || throw(DimensionMismatch(
                    "exponential cone $index must have exactly three entries; " *
                    "got $dimension",
                ))
            end
        end
        soc_only = all(
            code -> _is_simple_projection_code(code) ||
                    _is_soc_projection_code(code) ||
                    _is_rsoc_projection_code(code),
            cpu_types,
        )
        cone_count = Int(blkNum)
        serial_indices = Int64[]
        tiny_soc_indices = Int64[]
        warp_soc_indices = Int64[]
        simple_indices = Int64[]
        native_indices = Int64[]
        for julia_idx in 1:cone_count
            code = cpu_types[julia_idx]
            dimension = cpu_ns[julia_idx]
            if code in (0, 1)
                # Free cones are identity projections and need no kernel work.
                continue
            elseif _is_exp_projection_code(code) ||
                   (_is_simple_projection_code(code) && dimension <= 32)
                push!(serial_indices, Int64(julia_idx - 1))
            elseif _is_soc_projection_code(code) &&
                   (!_is_diagonal_soc_projection_code(code) ||
                    _diagonal_soc_compaction_enabled[]) &&
                   dimension <= _thread_soc_dimension_limit
                push!(tiny_soc_indices, Int64(julia_idx - 1))
            elseif _is_soc_projection_code(code) &&
                   (!_is_diagonal_soc_projection_code(code) ||
                    _diagonal_soc_compaction_enabled[]) &&
                   dimension <= _warp_soc_dimension_limit
                push!(warp_soc_indices, Int64(julia_idx - 1))
            elseif _is_simple_projection_code(code)
                push!(simple_indices, Int64(julia_idx - 1))
            else
                push!(native_indices, Int64(julia_idx - 1))
            end
        end

        # Same-GPU dispatch sweeps on H100 show that packing a handful of EXP
        # cones loses to the native mapping; the serial path amortizes its
        # launch/occupancy cost at roughly 512 cones.  Standard SOCs of
        # dimension <= 4 use one thread once 256 are available. Dimensions
        # 5--32 use one warp from 64 cones onward. Diagonally rescaled SOCs
        # may use the same compact mappings now that all four implementations
        # apply the endpoint-safe forward-error stopping rule.
        primal_exp_count = count(
            index -> _is_primal_exp_projection_code(cpu_types[index + 1]),
            serial_indices,
        )
        serial_is_profitable =
            primal_exp_count >= _primal_exp_compaction_minimum ||
            length(serial_indices) >= _serial_compaction_minimum
        has_compaction_candidate =
            serial_is_profitable ||
            length(tiny_soc_indices) >= _thread_soc_compaction_minimum ||
            length(warp_soc_indices) >= _warp_soc_compaction_minimum
        compact_soc_indices =
            length(tiny_soc_indices) >= _thread_soc_compaction_minimum ?
            tiny_soc_indices : Int64[]
        compact_warp_soc_indices =
            length(warp_soc_indices) >= _warp_soc_compaction_minimum ?
            warp_soc_indices : Int64[]
        original_work_blocks = length(serial_indices) +
            length(tiny_soc_indices) + length(warp_soc_indices) +
            length(simple_indices) + length(native_indices)
        compact_work_blocks = length(native_indices) +
            length(simple_indices) +
            (length(tiny_soc_indices) < _thread_soc_compaction_minimum ?
                length(tiny_soc_indices) :
                cld(length(tiny_soc_indices), ThreadPerBlock)) +
            (length(warp_soc_indices) < _warp_soc_compaction_minimum ?
                length(warp_soc_indices) :
                cld(length(warp_soc_indices) * 32, ThreadPerBlock)) +
            (serial_is_profitable ?
                cld(length(serial_indices), ThreadPerBlock) :
                length(serial_indices))
        fully_compacted = has_compaction_candidate &&
            5 * compact_work_blocks <= 4 * original_work_blocks
        thread_indices = Int64[]
        if !fully_compacted
            empty!(serial_indices)
            empty!(compact_soc_indices)
            empty!(compact_warp_soc_indices)
            empty!(simple_indices)
            empty!(native_indices)
            masked = proj_type
        else
            serial_is_profitable || append!(native_indices, serial_indices)
            serial_is_profitable || empty!(serial_indices)
            length(tiny_soc_indices) < _thread_soc_compaction_minimum &&
                append!(native_indices, tiny_soc_indices)
            length(warp_soc_indices) < _warp_soc_compaction_minimum &&
                append!(native_indices, warp_soc_indices)
            thread_indices = vcat(serial_indices, compact_soc_indices)
            masked_cpu_types = copy(cpu_types)
            for cone_idx in Iterators.flatten(
                (thread_indices, compact_warp_soc_indices),
            )
                masked_cpu_types[cone_idx + 1] = 0
            end
            masked = CuArray(masked_cpu_types)
            fully_compacted = true
        end
        plan = HeterogeneousProjectionPlan(
            masked,
            CuArray(thread_indices),
            Int64(length(thread_indices)),
            CuArray(serial_indices),
            Int64(length(serial_indices)),
            CuArray(compact_soc_indices),
            Int64(length(compact_soc_indices)),
            CuArray(compact_warp_soc_indices),
            Int64(length(compact_warp_soc_indices)),
            CuArray(simple_indices),
            Int64(length(simple_indices)),
            CuArray(native_indices),
            Int64(length(native_indices)),
            fully_compacted,
            soc_only,
        )
        _heterogeneous_projection_plans[proj_type] =
            CachedHeterogeneousProjectionPlan(gpu_ns, blkNum, plan)
        return plan
    finally
        unlock(_heterogeneous_projection_plan_lock)
    end
end

function get_simple_block_proj_indexed_kernel()::CuFunction
    k = _simple_block_proj_indexed_kernel[]
    k !== nothing && return k
    get_massive_block_proj_kernel()
    lock(_massive_block_proj_lock)
    try
        k = _simple_block_proj_indexed_kernel[]
        k !== nothing && return k
        mod = _massive_block_proj_mod[]
        mod !== nothing || error("massive projection CUDA module is not loaded")
        k = CuFunction(mod, _simple_block_proj_indexed_name)
        _simple_block_proj_indexed_kernel[] = k
        return k
    finally
        unlock(_massive_block_proj_lock)
    end
end

"""
    get_massive_block_proj_kernel() -> CuFunction

Lazily loads and returns the massive block projection CUDA kernel.

Uses double-checked locking pattern for thread-safe initialization:
1. First check without lock (fast path)
2. If not loaded, acquire lock
3. Double-check after acquiring lock (another thread might have loaded it)
4. If still not loaded, load the PTX file and cache the kernel

Returns the CuFunction that can be used with CUDA.cudacall().
"""
function get_massive_block_proj_kernel()::CuFunction
    # Fast path: check if already loaded (no lock needed for read)
    k = _massive_block_proj_kernel[]
    k !== nothing && return k

    # Slow path: acquire lock and load kernel
    lock(_massive_block_proj_lock)
    try
        # Double-check after locking (another thread might have loaded it while we waited)
        k = _massive_block_proj_kernel[]
        k !== nothing && return k

        # Verify CUDA is available
        CUDA.functional() || error("CUDA is not functional")
        # Ensure CUDA context exists (creates context if needed)
        CUDA.zeros(Float32, 1)

        # Load the PTX file from disk
        bytes = read(_projection_artifact_path("massive_block_proj.ptx"))
        # Create CUDA module from PTX bytes
        mod   = CuModule(bytes)
        # Get the kernel function from the module
        fun   = CuFunction(mod, _massive_block_proj_name)

        # Cache the module and function for future use
        _massive_block_proj_mod[]    = mod
        _massive_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_massive_block_proj_lock)
    end
end

function get_massive_block_proj_indexed_kernel()::CuFunction
    k = _massive_block_proj_indexed_kernel[]
    k !== nothing && return k
    get_massive_block_proj_kernel()
    lock(_massive_block_proj_lock)
    try
        k = _massive_block_proj_indexed_kernel[]
        k !== nothing && return k
        mod = _massive_block_proj_mod[]
        mod !== nothing || error("massive projection CUDA module is not loaded")
        k = CuFunction(mod, _massive_block_proj_indexed_name)
        _massive_block_proj_indexed_kernel[] = k
        return k
    finally
        unlock(_massive_block_proj_lock)
    end
end

function get_massive_soc_block_proj_kernel()::CuFunction
    k = _massive_soc_block_proj_kernel[]
    k !== nothing && return k
    get_massive_block_proj_kernel()
    lock(_massive_block_proj_lock)
    try
        k = _massive_soc_block_proj_kernel[]
        k !== nothing && return k
        mod = _massive_block_proj_mod[]
        mod !== nothing || error("massive projection CUDA module is not loaded")
        k = CuFunction(mod, _massive_soc_block_proj_name)
        _massive_soc_block_proj_kernel[] = k
        return k
    finally
        unlock(_massive_block_proj_lock)
    end
end

function _launch_indexed_thread_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
    t_warm_start, gpu_head_start, gpu_ns, proj_type,
    abs_tol::Float64, rel_tol::Float64,
)
    plan.thread_cone_count == 0 && return
    CUDA.cudacall(
        get_massive_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Int64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64},
         Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns, plan.thread_cone_indices,
        Cint(plan.thread_cone_count), proj_type, abs_tol, rel_tol;
        blocks = cld(plan.thread_cone_count, ThreadPerBlock),
        threads = ThreadPerBlock,
    )
    return
end

function _launch_indexed_compact_soc_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
    t_warm_start, gpu_head_start, gpu_ns, proj_type,
    abs_tol::Float64, rel_tol::Float64,
)
    plan.compact_soc_cone_count == 0 && return
    CUDA.cudacall(
        get_massive_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Int64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64},
         Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns,
        plan.compact_soc_cone_indices, Cint(plan.compact_soc_cone_count),
        proj_type, abs_tol, rel_tol;
        blocks = cld(plan.compact_soc_cone_count, ThreadPerBlock),
        threads = ThreadPerBlock,
    )
    return
end

function _launch_indexed_simple_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, gpu_head_start, gpu_ns, proj_type,
)
    plan.simple_cone_count == 0 && return
    CUDA.cudacall(
        get_simple_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64},
         CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64}),
        vec, bl, bu, gpu_head_start, gpu_ns, plan.simple_cone_indices,
        Cint(plan.simple_cone_count), proj_type;
        blocks = plan.simple_cone_count,
        threads = ThreadPerBlock,
    )
    return
end


"""
    massive_block_proj(vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol)

Projects vectors onto constraint sets for large blocks (massive blocks).

Arguments:
- vec: Vector to project (modified in-place)
- bl: Lower bounds for the projection
- bu: Upper bounds for the projection
- D_scaled: Scaled diagonal matrix D
- D_scaled_squared: D scaled and squared
- D_scaled_mul_x: D scaled multiplied by x
- temp: Temporary storage array
- t_warm_start: Warm start values for t
- gpu_head_start: GPU array of block head start indices
- gpu_ns: GPU array of block sizes
- blkNum: Number of blocks
- proj_type: Type of projection for each block
- abs_tol: Absolute tolerance for projection
- rel_tol: Relative tolerance for projection

The kernel computes the projection onto constraint sets defined by the bounds
and scaling matrices, using the specified projection types for each block.
"""
function massive_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    _validate_device_projection_arguments(
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
        abs_tol, rel_tol,
    )
    blkNum == 0 && return nothing
    # Calculate number of thread blocks needed
    # cld(x, y) = ceil(x/y) = smallest integer >= x/y
    nBlock = max(1, cld(blkNum, ThreadPerBlock))
    if _projection_work_profile_should_record()
        return _profile_block_projection!(
            :threadWise,
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
            t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
            abs_tol, rel_tol;
            blocks = nBlock,
            threads = ThreadPerBlock,
            record_count = nBlock * ThreadPerBlock,
        )
    end
    
    plan = _heterogeneous_projection_enabled[] ?
        _get_heterogeneous_projection_plan(gpu_ns, blkNum, proj_type) : nothing
    # Keep the SOC-only entry point available for controlled microbenchmarks,
    # but do not select it in the solver.  Although it uses fewer registers,
    # the qssp180 stress run showed a large end-to-end regression relative to
    # the general thread-wise kernel.  The general kernel is therefore the
    # correctness/performance default until the specialization has passed the
    # full projection matrix and PDCS hard-case gate.
    projection_kernel = get_massive_block_proj_kernel()

    # Launch kernel and wait for completion
    CUDA.@sync begin
        CUDA.cudacall(
        projection_kernel,
        # Kernel function signature: all pointers to device memory
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64}, Float64, Float64),
        # Kernel arguments
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, Cint(blkNum), proj_type, abs_tol, rel_tol;
        # Launch configuration: nBlock blocks, ThreadPerBlock threads per block
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

"""Thread-wise projection: one GPU thread processes one cone block."""
threadWise_block_proj(args...) = massive_block_proj(args...)

# ----------------------------------------------------------------------------
# Moderate Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for medium-sized blocks. Similar structure to massive_block_proj but
# optimized for different block size ranges.

const _moderate_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _moderate_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _moderate_block_proj_indexed_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _moderate_block_proj_lock   = SpinLock()

const _moderate_block_proj_name = "moderate_block_proj"
const _moderate_block_proj_indexed_name = "moderate_block_proj_indexed"

"""
    get_moderate_block_proj_kernel() -> CuFunction

Lazily loads and returns the moderate block projection CUDA kernel.
Uses the same thread-safe lazy loading pattern as massive_block_proj.
"""
function get_moderate_block_proj_kernel()::CuFunction
    k = _moderate_block_proj_kernel[]
    k !== nothing && return k

    lock(_moderate_block_proj_lock)
    try
        k = _moderate_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_projection_artifact_path("moderate_block_proj.ptx"))
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _moderate_block_proj_name)

        _moderate_block_proj_mod[]    = mod
        _moderate_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_moderate_block_proj_lock)
    end
end

function get_moderate_block_proj_indexed_kernel()::CuFunction
    k = _moderate_block_proj_indexed_kernel[]
    k !== nothing && return k
    get_moderate_block_proj_kernel()
    lock(_moderate_block_proj_lock)
    try
        k = _moderate_block_proj_indexed_kernel[]
        k !== nothing && return k
        mod = _moderate_block_proj_mod[]
        mod !== nothing || error("moderate projection CUDA module is not loaded")
        k = CuFunction(mod, _moderate_block_proj_indexed_name)
        _moderate_block_proj_indexed_kernel[] = k
        return k
    finally
        unlock(_moderate_block_proj_lock)
    end
end

function _launch_indexed_block_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
    t_warm_start, gpu_head_start, gpu_ns, proj_type,
    abs_tol::Float64, rel_tol::Float64,
)
    serial_blocks = cld(plan.serial_cone_count, ThreadPerBlock)
    total_blocks = plan.native_cone_count + plan.simple_cone_count +
                   serial_blocks
    total_blocks == 0 && return
    CUDA.cudacall(
        get_moderate_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Int64}, CuPtr{Int64},
         CuPtr{Int64}, Clong, CuPtr{Int64}, Clong,
         CuPtr{Int64}, Clong, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns,
        plan.native_cone_indices, Clong(plan.native_cone_count),
        plan.simple_cone_indices, Clong(plan.simple_cone_count),
        plan.serial_cone_indices, Clong(plan.serial_cone_count),
        proj_type, abs_tol, rel_tol;
        blocks = total_blocks,
        threads = ThreadPerBlock,
    )
    return
end

"""
    moderate_block_proj(...)

Projects vectors onto constraint sets for medium-sized blocks.
Similar to massive_block_proj but uses a different block configuration:
nBlock = blkNum + 1 (one block per constraint block plus one extra).
"""
function moderate_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    _validate_device_projection_arguments(
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
        abs_tol, rel_tol,
    )
    blkNum == 0 && return nothing
    # For moderate blocks, use one block per constraint block plus one
    nBlock = blkNum + 1
    if _projection_work_profile_should_record()
        return _profile_block_projection!(
            :blockWise,
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
            t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
            abs_tol, rel_tol;
            blocks = nBlock,
            threads = ThreadPerBlock,
            record_count = nBlock,
        )
    end
    plan = _heterogeneous_projection_enabled[] ?
        _get_heterogeneous_projection_plan(gpu_ns, blkNum, proj_type) : nothing
    launch_projection_types = plan === nothing ? proj_type :
        plan.masked_projection_types
    CUDA.@sync begin
        if plan === nothing || !plan.fully_compacted
            CUDA.cudacall(
            get_moderate_block_proj_kernel(),
            (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64}, Float64, Float64),
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, Cint(blkNum), launch_projection_types, abs_tol, rel_tol;
            blocks = nBlock, threads = ThreadPerBlock
            )
        else
            _launch_indexed_block_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
            _launch_indexed_compact_soc_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
            _launch_indexed_compact_warp_soc_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
        end
    end
end

"""Block-wise projection: one GPU thread block processes one cone block."""
blockWise_block_proj(args...) = moderate_block_proj(args...)


# ----------------------------------------------------------------------------
# Sufficient Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for blocks with sufficient parallelism. Optimized for cases where
# there are enough elements to fully utilize GPU threads.

const _sufficient_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _sufficient_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _sufficient_block_proj_indexed_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _sufficient_block_proj_lock   = SpinLock()

const _sufficient_block_proj_name = "sufficient_block_proj"
const _sufficient_block_proj_indexed_name = "sufficient_block_proj_indexed"

"""
    get_sufficient_block_proj_kernel() -> CuFunction

Lazily loads and returns the sufficient block projection CUDA kernel.
Uses the same thread-safe lazy loading pattern.
"""
function get_sufficient_block_proj_kernel()::CuFunction
    k = _sufficient_block_proj_kernel[]
    k !== nothing && return k

    lock(_sufficient_block_proj_lock)
    try
        k = _sufficient_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_projection_artifact_path("sufficient_block_proj.ptx"))
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _sufficient_block_proj_name)

        _sufficient_block_proj_mod[]    = mod
        _sufficient_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_sufficient_block_proj_lock)
    end
end

function get_sufficient_block_proj_indexed_kernel()::CuFunction
    k = _sufficient_block_proj_indexed_kernel[]
    k !== nothing && return k
    get_sufficient_block_proj_kernel()
    lock(_sufficient_block_proj_lock)
    try
        k = _sufficient_block_proj_indexed_kernel[]
        k !== nothing && return k
        mod = _sufficient_block_proj_mod[]
        mod !== nothing || error("sufficient projection CUDA module is not loaded")
        k = CuFunction(mod, _sufficient_block_proj_indexed_name)
        _sufficient_block_proj_indexed_kernel[] = k
        return k
    finally
        unlock(_sufficient_block_proj_lock)
    end
end

function _launch_indexed_warp_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
    t_warm_start, gpu_head_start, gpu_ns, proj_type,
    abs_tol::Float64, rel_tol::Float64,
)
    plan.native_cone_count == 0 && return
    CUDA.cudacall(
        get_sufficient_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Int64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64},
         Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns, plan.native_cone_indices,
        Cint(plan.native_cone_count), proj_type, abs_tol, rel_tol;
        blocks = cld(plan.native_cone_count * 32, ThreadPerBlock),
        threads = ThreadPerBlock,
    )
    return
end

function _launch_indexed_compact_warp_soc_projection!(
    plan::HeterogeneousProjectionPlan,
    vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
    t_warm_start, gpu_head_start, gpu_ns, proj_type,
    abs_tol::Float64, rel_tol::Float64,
)
    plan.compact_warp_soc_cone_count == 0 && return
    CUDA.cudacall(
        get_sufficient_block_proj_indexed_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64},
         CuPtr{Int64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64},
         Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns,
        plan.compact_warp_soc_cone_indices,
        Cint(plan.compact_warp_soc_cone_count), proj_type, abs_tol, rel_tol;
        blocks = cld(plan.compact_warp_soc_cone_count * 32, ThreadPerBlock),
        threads = ThreadPerBlock,
    )
    return
end

"""
    sufficient_block_proj(...)

Projects vectors onto constraint sets for blocks with sufficient parallelism.
Uses a different block configuration: nBlock = ceil((blkNum + 1) * 32 / ThreadPerBlock),
which provides more blocks for better GPU utilization.
"""
function sufficient_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    _validate_device_projection_arguments(
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
        t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
        abs_tol, rel_tol,
    )
    blkNum == 0 && return nothing
    # Calculate blocks: (blkNum + 1) * 32 elements distributed across thread blocks
    nBlock = cld((blkNum + 1) * 32, ThreadPerBlock)
    if _projection_work_profile_should_record()
        return _profile_block_projection!(
            :warpWise,
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
            t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type,
            abs_tol, rel_tol;
            blocks = nBlock,
            threads = ThreadPerBlock,
            record_count = nBlock * cld(ThreadPerBlock, 32),
        )
    end
    plan = _heterogeneous_projection_enabled[] ?
        _get_heterogeneous_projection_plan(gpu_ns, blkNum, proj_type) : nothing
    launch_projection_types = plan === nothing ? proj_type :
        plan.masked_projection_types
    CUDA.@sync begin
        if plan === nothing || !plan.fully_compacted
            CUDA.cudacall(
            get_sufficient_block_proj_kernel(),
            (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Cint, CuPtr{Int64}, Float64, Float64),
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, Cint(blkNum), launch_projection_types, abs_tol, rel_tol;
            blocks = nBlock, threads = ThreadPerBlock
            )
        else
            _launch_indexed_simple_projection!(
                plan, vec, bl, bu, gpu_head_start, gpu_ns, proj_type,
            )
            _launch_indexed_warp_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
        end
        if plan !== nothing
            _launch_indexed_thread_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
            _launch_indexed_compact_warp_soc_projection!(
                plan, vec, bl, bu, D_scaled, D_scaled_squared,
                D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns,
                proj_type, abs_tol, rel_tol,
            )
        end
    end
end

"""Warp-wise projection: one GPU warp processes one cone block."""
warpWise_block_proj(args...) = sufficient_block_proj(args...)

# ============================================================================
# Section 2: cuBLAS Handle Management
# ============================================================================
# cuBLAS is NVIDIA's CUDA Basic Linear Algebra Subroutines library.
# We need to create and manage cuBLAS handles for performing linear algebra
# operations on the GPU (e.g., matrix-vector products, norms).

# cuBLAS type definitions
const cublasStatus_t = Cint                    # cuBLAS status code type
const CUBLAS_STATUS_SUCCESS = 0                # Success status code
const cublasHandle_t = Ptr{Cvoid}              # cuBLAS handle type (opaque pointer)

"""
    CUBLASHandle

Wrapper struct for cuBLAS handle.
A cuBLAS handle is required for all cuBLAS operations and manages the
library's internal state and resources.
"""
mutable struct CUBLASHandle
    handle::cublasHandle_t  # Opaque pointer to cuBLAS context
    workspace::CuArray      # Non-aliasing scratch space for grid-wise kernels
    device::Any             # CUDA device on which the handle was created
end

# The grid-wise kernel is the only projection strategy that needs a cuBLAS
# handle. Create it lazily so importing PDCS does not create a CUDA context,
# and keep ownership in this module rather than in individual solver calls.
const _gridWise_cublas_handle = Ref{Union{Nothing,CUBLASHandle}}(nothing)
const _gridWise_cublas_handle_lock = SpinLock()
const _gridWise_cublas_atexit_registered = Ref(false)
const _gridwise_runtime_lock = SpinLock()
const _gridwise_call_lock = ReentrantLock()
const _gridwise_fallback_workspace = Ref{Any}(nothing)
const _gridwise_fallback_metadata_cache = Ref{Any}(nothing)
const _gridwise_fallback_lock = SpinLock()

"""
    create_cublas_handle() -> CUBLASHandle

Creates a new cuBLAS handle for performing linear algebra operations on GPU.

The handle must be created before using any cuBLAS functions and should be
destroyed when no longer needed using destroy_cublas_handle().

Returns a CUBLASHandle wrapper containing the opaque cuBLAS handle pointer.
"""
function create_cublas_handle()
    # Ensure CUDA context exists (required before creating cuBLAS handle)
    CUDA.zeros(Float32, 1)

    # Create and configure the handle through the same shared object that
    # consumes it. CUDA.jl and the native projection library can resolve to
    # different cuBLAS installations, whose opaque handles are not
    # interchangeable.
    h = Ref{cublasHandle_t}(C_NULL)
    create_ptr = Libdl.dlsym(_kernlib_ref[], :create_cublas_handle_inner)
    ccall(create_ptr, Cvoid, (Ref{cublasHandle_t},), h)
    h[] != C_NULL || error("native cublasCreate_v2 returned NULL handle")
    configure_ptr =
        Libdl.dlsym(_kernlib_ref[], :configure_cublas_handle_inner)
    status = ccall(
        configure_ptr,
        Cint,
        (cublasHandle_t, Cint),
        h[],
        _cublas_reproducible_enabled[] ? 1 : 0,
    )
    if status != CUBLAS_STATUS_SUCCESS
        destroy_ptr =
            Libdl.dlsym(_kernlib_ref[], :destroy_cublas_handle_inner)
        ccall(destroy_ptr, Cvoid, (cublasHandle_t,), h[])
        error("native cuBLAS handle configuration failed with status $status")
    end
    return CUBLASHandle(h[], CUDA.zeros(Float64, 0), CUDA.device())
end

"""Return the effective reproducibility settings of the grid-wise handle."""
function gridWise_cublas_configuration()
    wrapper = get_gridWise_cublas_handle()
    atomics = Ref{Cint}()
    math_mode = Ref{Cint}()
    configuration_ptr =
        Libdl.dlsym(_kernlib_ref[], :cublas_handle_configuration_inner)
    status = ccall(
        configuration_ptr,
        Cint,
        (cublasHandle_t, Ref{Cint}, Ref{Cint}),
        wrapper.handle,
        atomics,
        math_mode,
    )
    status == CUBLAS_STATUS_SUCCESS ||
        error("native cuBLAS configuration query failed with status $status")
    return (
        reproducible = _cublas_reproducible_enabled[],
        workspace_config = get(ENV, "CUBLAS_WORKSPACE_CONFIG", ""),
        atomics_mode = atomics[],
        math_mode = math_mode[],
    )
end

"""
    get_gridWise_cublas_handle() -> CUBLASHandle

Return the process-local cuBLAS handle used by `gridWise_block_proj`, creating
it only on first use. It is subsequently reused by every projection in every
iteration of the solve.
"""
function get_gridWise_cublas_handle()
    current = _gridWise_cublas_handle[]
    if current !== nothing && current.handle != C_NULL
        current.device == CUDA.device() || error(
            "grid-wise cuBLAS handle belongs to CUDA device " *
            "$(current.device), but the active device is $(CUDA.device()); " *
            "release the handle before switching devices",
        )
        return current
    end

    lock(_gridWise_cublas_handle_lock)
    try
        current = _gridWise_cublas_handle[]
        if current === nothing || current.handle == C_NULL
            current = create_cublas_handle()
            _gridWise_cublas_handle[] = current
        end
        return current
    finally
        unlock(_gridWise_cublas_handle_lock)
    end
end

"""
    destroy_cublas_handle(ch::CUBLASHandle)

Destroys a cuBLAS handle, freeing associated resources.

Should be called when the handle is no longer needed. Safe to call multiple
times (idempotent if handle is already C_NULL).
"""
function destroy_cublas_handle(ch::CUBLASHandle)
    # Early return if handle is already null
    ch.handle == C_NULL && return nothing

    # Destroy through the same native library that created and consumed it.
    destroy_ptr = Libdl.dlsym(_kernlib_ref[], :destroy_cublas_handle_inner)
    ccall(destroy_ptr, Cvoid, (cublasHandle_t,), ch.handle)
    # Mark handle as destroyed
    ch.handle = C_NULL
    return nothing
end

"""
    release_gridWise_cublas_handle!()

Release the cached grid-wise cuBLAS handle. Normal solver calls should not use
this between projections: the handle is intentionally reused for the whole
optimization. It is available for explicit teardown after the final solve.
"""
function release_gridWise_cublas_handle!()
    lock(_gridwise_call_lock)
    try
        lock(_gridWise_cublas_handle_lock)
        try
            current = _gridWise_cublas_handle[]
            current === nothing || destroy_cublas_handle(current)
            _gridWise_cublas_handle[] = nothing
        finally
            unlock(_gridWise_cublas_handle_lock)
        end
    finally
        unlock(_gridwise_call_lock)
    end
    return nothing
end

"""Register one Julia-side cleanup callback for the cached grid-wise handle."""
function register_gridWise_cublas_cleanup!()
    _gridWise_cublas_atexit_registered[] && return nothing
    _gridWise_cublas_atexit_registered[] = true
    atexit(release_gridWise_cublas_handle!)
    return nothing
end

"""Emit one explicit warning when a grid-wise call switches to block-wise."""
function _warn_gridwise_fallback_once!(reason = _gridwise_native_failure[])
    _gridwise_fallback_warned[] && return nothing
    lock(_gridwise_runtime_lock)
    try
        _gridwise_fallback_warned[] && return nothing
        @warn(
            "Using block-wise projection fallback for grid-wise request",
            mode = _gridwise_mode(),
            state = _gridwise_runtime_state[],
            reason = reason,
        )
        _gridwise_fallback_warned[] = true
    finally
        unlock(_gridwise_runtime_lock)
    end
    return nothing
end

"""Return a persistent non-aliasing workspace on the active CUDA device."""
function _gridwise_workspace!(n::Integer, ::Type{T}; handle = nothing) where {T}
    if handle !== nothing
        handle.device == CUDA.device() || error(
            "grid-wise workspace requested on a different CUDA device",
        )
        workspace = handle.workspace
        if !(workspace isa CuArray) || eltype(workspace) !== T ||
           length(workspace) < n || CUDA.device(workspace) != CUDA.device()
            handle.workspace = CUDA.zeros(T, n)
        end
        return handle.workspace
    end
    workspace = _gridwise_fallback_workspace[]
    if !(workspace isa CuArray) || eltype(workspace) !== T ||
       length(workspace) < n || CUDA.device(workspace) != CUDA.device()
        workspace = CUDA.zeros(T, n)
        _gridwise_fallback_workspace[] = workspace
    end
    return workspace
end

"""Upload CPU cone metadata once and refresh it when the caller mutates it."""
function _gridwise_fallback_metadata(
    cpu_head_start::Vector{Int64}, cpu_proj_type::Vector{Int64},
)
    lock(_gridwise_fallback_lock)
    try
        cached = _gridwise_fallback_metadata_cache[]
        active_device = CUDA.device()
        if cached === nothing || cached[1] !== cpu_head_start ||
           cached[2] !== cpu_proj_type || cached[5] != active_device
            gpu_head_start = CuArray(cpu_head_start)
            gpu_proj_type = CuArray(cpu_proj_type)
            _gridwise_fallback_metadata_cache[] =
                (cpu_head_start, cpu_proj_type, gpu_head_start, gpu_proj_type,
                 active_device)
        else
            gpu_head_start, gpu_proj_type = cached[3], cached[4]
            copyto!(gpu_head_start, cpu_head_start)
            copyto!(gpu_proj_type, cpu_proj_type)
        end
        cached = _gridwise_fallback_metadata_cache[]
        return cached[3], cached[4]
    finally
        unlock(_gridwise_fallback_lock)
    end
end

const _GRIDWISE_SOC_TYPES = (5, 6, 7, 20, 21, 22)
const _GRIDWISE_EXP_TYPES = (11, 12, 13, 14, 15, 16, 26, 27, 28, 29)

"""Validate the host layout and device buffers before crossing the C ABI."""
function _validate_gridwise_arguments(
    vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T,
    D_scaled_mul_x::T, temp::T, t_warm_start::T,
    cpu_head_start::Vector{Int64},
    gpu_ns::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}, cpu_ns::Vector{Int64},
    blkNum::Int64, cpu_proj_type::Vector{Int64},
    abs_tol::Float64, rel_tol::Float64,
) where {T<:CuArray}
    0 < blkNum <= typemax(Cint) || throw(ArgumentError(
        "grid-wise block count must be in 1:$(typemax(Cint)); got $blkNum",
    ))
    for (name, values) in (
        ("cpu_head_start", cpu_head_start),
        ("gpu_ns", gpu_ns),
        ("cpu_ns", cpu_ns),
        ("cpu_proj_type", cpu_proj_type),
    )
        length(values) >= blkNum || throw(DimensionMismatch(
            "$name has length $(length(values)), but blkNum=$blkNum",
        ))
    end
    length(t_warm_start) >= blkNum || throw(DimensionMismatch(
        "t_warm_start has length $(length(t_warm_start)), but blkNum=$blkNum",
    ))
    isfinite(abs_tol) && abs_tol > 0 || throw(ArgumentError(
        "grid-wise abs_tol must be finite and positive; got $abs_tol",
    ))
    isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError(
        "grid-wise rel_tol must be finite and positive; got $rel_tol",
    ))

    active_device = CUDA.device()
    required_length = length(vec)
    for (name, values) in (
        ("vec", vec), ("bl", bl), ("bu", bu),
        ("D_scaled", D_scaled),
        ("D_scaled_squared", D_scaled_squared),
        ("D_scaled_mul_x", D_scaled_mul_x), ("temp", temp),
    )
        ndims(values) == 1 || throw(DimensionMismatch(
            "$name must be a one-dimensional CuArray",
        ))
        eltype(values) === Float64 || throw(ArgumentError(
            "$name must use Float64 storage for the native C ABI; got " *
            "$(eltype(values))",
        ))
        CUDA.device(values) == active_device || throw(ArgumentError(
            "$name is allocated on $(CUDA.device(values)), but the active " *
            "CUDA device is $active_device",
        ))
    end
    CUDA.device(gpu_ns) == active_device || throw(ArgumentError(
        "gpu_ns is not allocated on the active CUDA device",
    ))

    required_bl = 1
    required_bu = 1
    required_D_scaled = 1
    required_D_scaled_squared = 1
    required_D_scaled_mul_x = 1
    required_temp = 1
    previous_stop = 0
    for index in 1:blkNum
        head = cpu_head_start[index]
        size = cpu_ns[index]
        head >= previous_stop || throw(ArgumentError(
            "cone $index starts at $head and overlaps or precedes the prior cone",
        ))
        size > 0 || throw(ArgumentError(
            "cone $index must have positive size; got $size",
        ))
        head <= required_length - size || throw(DimensionMismatch(
            "cone $index range [$head, $(head + size)) exceeds vector length " *
            "$required_length",
        ))
        projection_type = cpu_proj_type[index]
        0 <= projection_type <= 29 || throw(ArgumentError(
            "cone $index has unknown projection type $projection_type",
        ))
        stop = head + size
        if projection_type in (17, 18, 19)
            required_bl = max(required_bl, stop)
            required_bu = max(required_bu, stop)
        elseif projection_type in (5, 7, 20, 21)
            required_temp = max(required_temp, stop)
        elseif projection_type in (6, 22)
            required_D_scaled = max(required_D_scaled, stop)
            required_D_scaled_squared = max(required_D_scaled_squared, stop)
            required_D_scaled_mul_x = max(required_D_scaled_mul_x, stop)
            required_temp = max(required_temp, stop)
        elseif projection_type in (12, 15, 27, 29)
            required_D_scaled = max(required_D_scaled, stop)
            required_temp = max(required_temp, stop)
        end
        if projection_type in _GRIDWISE_SOC_TYPES
            size >= 2 || throw(DimensionMismatch(
                "SOC cone $index must have at least two entries",
            ))
        elseif projection_type in _GRIDWISE_EXP_TYPES
            size == 3 || throw(DimensionMismatch(
                "exponential cone $index must have exactly three entries; got $size",
            ))
        end
        previous_stop = head + size
    end
    for (name, values, required) in (
        ("vec", vec, required_length),
        ("bl", bl, required_bl),
        ("bu", bu, required_bu),
        ("D_scaled", D_scaled, required_D_scaled),
        ("D_scaled_squared", D_scaled_squared, required_D_scaled_squared),
        ("D_scaled_mul_x", D_scaled_mul_x, required_D_scaled_mul_x),
        ("temp", temp, required_temp),
    )
        length(values) >= required || throw(DimensionMismatch(
            "$name has length $(length(values)); expected at least $required " *
            "for the active grid-wise projection types",
        ))
    end
    return nothing
end

"""Invoke the native entry point after all arguments have been validated."""
function _call_native_few_block_proj!(
    fptr, handle::CUBLASHandle, vec::T, bl::T, bu::T,
    D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T,
    t_warm_start::T, cpu_head_start::Vector{Int64},
    gpu_ns::CUDA.CuArray{Int64,1,CUDA.DeviceMemory}, cpu_ns::Vector{Int64},
    blkNum::Int64, cpu_proj_type::Vector{Int64},
    abs_tol::Float64, rel_tol::Float64,
) where {T<:CuArray}
    nThread = Int64(ThreadPerBlock)
    nBlock = cld(maximum(cpu_ns) + ThreadPerBlock + 1, ThreadPerBlock)
    status = @ccall $fptr(
        handle.handle::Ptr{Nothing},
        vec::CuPtr{Cdouble}, bl::CuPtr{Cdouble}, bu::CuPtr{Cdouble},
        D_scaled::CuPtr{Cdouble}, D_scaled_squared::CuPtr{Cdouble},
        D_scaled_mul_x::CuPtr{Cdouble}, temp::CuPtr{Cdouble},
        t_warm_start::CuPtr{Cdouble}, cpu_head_start::Ptr{Clong},
        gpu_ns::CuPtr{Clong}, cpu_ns::Ptr{Clong}, blkNum::Cint,
        cpu_proj_type::Ptr{Clong}, nThread::Cint, nBlock::Cint,
        abs_tol::Cdouble, rel_tol::Cdouble,
    )::Cint
    if status != 0
        reason = if status >= 2000
            "native grid-wise cuBLAS failure (status $(status - 2000))"
        elseif status >= 1000
            "native grid-wise CUDA failure (status $(status - 1000))"
        else
            "native grid-wise projection failure (status $status)"
        end
        _gridwise_native_enabled[] = false
        _gridwise_runtime_state[] = :failed
        _gridwise_native_failure[] = reason
        error(reason)
    end
    return nothing
end

"""Check the native SOC reduction, scaling, and alias-safe workspace once."""
function _gridwise_native_selftest!(handle::CUBLASHandle)
    n = 3
    bl = CUDA.fill(-Inf, n)
    bu = CUDA.fill(Inf, n)
    d = CUDA.ones(Float64, n)
    d2 = d .* d
    dmx = CUDA.zeros(Float64, n)
    starts = Int64[0]
    sizes = Int64[n]
    gpu_sizes = CuArray(sizes)
    types = Int64[20] # ordinary SOC

    cases = (
        (name = "interior", input = [2.0, 1.0, 0.0], expected = [2.0, 1.0, 0.0]),
        (name = "exterior", input = [0.0, 2.0, 0.0], expected = [1.0, 1.0, 0.0]),
        (name = "polar", input = [-2.0, 1.0, 0.0], expected = zeros(3)),
    )
    for case in cases
        vec = CuArray(case.input)
        warm = CUDA.ones(Float64, 1)
        temp = _gridwise_workspace!(n, Float64; handle = handle)
        for repetition in 1:3
            _call_native_few_block_proj!(
                few_block_proj_ptr[], handle, vec, bl, bu, d, d2, dmx,
                temp, warm, starts, gpu_sizes, sizes, Int64(1), types,
                1e-12, 1e-12,
            )
            result = Array(vec)
            err = maximum(abs.(result .- case.expected))
            isfinite(err) && err <= 1e-10 || error(
                "grid-wise SOC self-test failed (case=$(case.name), " *
                "repetition=$repetition, error=$err, " *
                "result=$(repr(result)), expected=$(repr(case.expected)))",
            )
        end
    end
    return nothing
end

"""Run the native self-test once; return false when block-wise fallback is safer."""
function _ensure_gridwise_runtime!()
    if !_gridwise_native_enabled[]
        _gridwise_mode() == "native" && error(
            "grid-wise native projection is unavailable: " *
            string(_gridwise_native_failure[]),
        )
        return false
    end
    state = _gridwise_runtime_state[]
    state === :passed && return true
    state === :failed && return false

    lock(_gridwise_runtime_lock)
    try
        state = _gridwise_runtime_state[]
        state === :passed && return true
        state === :failed && return false
        try
            handle = get_gridWise_cublas_handle()
            # This query is deliberately made through the same shared object
            # that creates the handle. It catches stale ABIs before a solver
            # iteration can silently consume an invalid opaque pointer.
            gridWise_cublas_configuration()
            if _pdcs_env_enabled("PDCS_GRIDWISE_SELFTEST", true)
                _gridwise_native_selftest!(handle)
            end
            _gridwise_runtime_state[] = :passed
            return true
        catch err
            reason = sprint(showerror, err, catch_backtrace())
            _gridwise_native_enabled[] = false
            _gridwise_runtime_state[] = :failed
            _gridwise_native_failure[] = reason
            strict = _gridwise_mode() == "native" ||
                     _pdcs_env_enabled("PDCS_GRIDWISE_STRICT", false)
            strict && rethrow()
            @warn "Grid-wise runtime self-test failed; using block-wise fallback" reason
            return false
        end
    finally
        unlock(_gridwise_runtime_lock)
    end
end


# ============================================================================
# Section 3: Few Block Projection (Shared Library Function)
# ============================================================================
# This function calls a C function from the shared library libfew_block_proj.so.
# Unlike the PTX kernels, this uses a function pointer loaded from a .so file.
# The function performs projections for cases with few blocks.

"""
    few_block_proj(vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, cpu_head_start, gpu_ns, cpu_ns, blkNum, cpu_proj_type, abs_tol, rel_tol)

Projects vectors onto constraint sets for cases with few blocks.

This function calls a C function from the shared library libfew_block_proj.so.
The function pointer must be initialized before calling this function.

Arguments:
- vec: Vector to project (GPU array, modified in-place)
- bl, bu: Lower and upper bounds (GPU arrays)
- D_scaled, D_scaled_squared, D_scaled_mul_x: Scaled diagonal matrices (GPU arrays)
- temp: Temporary storage (GPU array)
- t_warm_start: Warm start values (GPU array)
- cpu_head_start: Block head start indices (CPU array)
- gpu_ns: Block sizes (GPU array)
- cpu_ns: Block sizes (CPU array, used for block configuration)
- blkNum: Number of blocks
- cpu_proj_type: Projection type for each block (CPU array)
- abs_tol, rel_tol: Tolerances for projection

The native function pointer is initialized by the module's `__init__()` method.
The cuBLAS handle is initialized lazily on the first grid-wise projection.
"""
function few_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, cpu_head_start::Vector{Int64}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, cpu_ns::Vector{Int64}, blkNum::Int64, cpu_proj_type::Vector{Int64}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    blkNum == 0 && return nothing
    lock(_gridwise_call_lock)
    try
        _validate_gridwise_arguments(
            vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp,
            t_warm_start, cpu_head_start, gpu_ns, cpu_ns, blkNum,
            cpu_proj_type, abs_tol, rel_tol,
        )
        aliases_workspace = pointer(vec) == pointer(temp)
        if _projection_work_profile_should_record()
            # The grid-wise implementation lives in a shared library rather
            # than profile PTX. In diagnostic mode only, execute the same
            # formulas through the instrumented block-wise kernel.
            profile_temp = aliases_workspace ?
                _gridwise_workspace!(length(vec), eltype(vec)) : temp
            return moderate_block_proj(
                vec,
                bl,
                bu,
                D_scaled,
                D_scaled_squared,
                D_scaled_mul_x,
                profile_temp,
                t_warm_start,
                CuArray(cpu_head_start),
                gpu_ns,
                blkNum,
                CuArray(cpu_proj_type),
                abs_tol,
                rel_tol,
            )
        end

        # A stale .so, a missing artifact, or a cuBLAS runtime mismatch must
        # never reach the opaque native call. Native mode fails explicitly;
        # auto/block mode is an opt-in compatibility policy.
        native_ready = _ensure_gridwise_runtime!()
        fallback_reason = _gridwise_native_failure[]
        if !native_ready
            _warn_gridwise_fallback_once!(fallback_reason)
            gpu_head_start, gpu_proj_type = _gridwise_fallback_metadata(
                cpu_head_start, cpu_proj_type,
            )
            fallback_temp = aliases_workspace ?
                _gridwise_workspace!(length(vec), eltype(vec)) : temp
            return moderate_block_proj(
                vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x,
                fallback_temp, t_warm_start, gpu_head_start, gpu_ns, blkNum,
                gpu_proj_type, abs_tol, rel_tol,
            )
        end

        fptr = few_block_proj_ptr[]
        fptr != C_NULL || error(
            "grid-wise native projection is enabled but few_block_proj is unavailable",
        )
        gridWise_handle = get_gridWise_cublas_handle()
        projection_temp = aliases_workspace ?
            _gridwise_workspace!(
                length(vec), eltype(vec); handle = gridWise_handle,
            ) : temp
        # CUDA.jl may queue work on a task-local stream. The native library
        # uses its own handle/stream, so complete producer work before crossing
        # the ABI. The C entry point synchronizes again and returns any runtime
        # failure as a checked status code.
        CUDA.synchronize()
        _call_native_few_block_proj!(
            fptr, gridWise_handle, vec, bl, bu, D_scaled, D_scaled_squared,
            D_scaled_mul_x, projection_temp, t_warm_start, cpu_head_start,
            gpu_ns, cpu_ns, blkNum, cpu_proj_type, abs_tol, rel_tol,
        )
        return nothing
    finally
        unlock(_gridwise_call_lock)
    end
end

"""Grid-wise projection: the GPU grid cooperates on a small number of cones."""
gridWise_block_proj(args...) = few_block_proj(args...)



# ============================================================================
# Section 4: Utility Kernels (from utils.ptx)
# ============================================================================
# These kernels implement various utility functions for the RPDHG algorithm,
# including primal/dual updates, reflection operations, averaging, and
# matrix operations. All kernels are loaded from a single PTX file: utils.ptx

# Path to the shared PTX file containing all utility kernels
utils_path = joinpath(
    get(
        ENV,
        "PDCS_CUDA_PROJECTION_ARTIFACT_DIR",
        joinpath(MODULE_DIR, "cuda"),
    ),
    "utils.ptx",
)

# CUDA.jl's CPU CSC convenience upload and the historical utility PTX both
# assume 32-bit sparse indices. These kernels are compiled by CUDA.jl for the
# actual array types, so matrices with more than typemax(Int32) stored entries
# keep Int64 row pointers and column indices throughout preprocessing.
@inline function _sparse_thread_index(::Type{Ti}) where {Ti<:Integer}
    return _sparse_unsigned_thread_index(
        Ti,
        CUDA.blockIdx().x,
        CUDA.blockDim().x,
        CUDA.threadIdx().x,
    )
end

function _rescale_csr_sparse_kernel!(
    values,
    rowptr,
    colval,
    row_scaling,
    col_scaling,
    nrows,
)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        for position in first_position:last_position
            column = @inbounds colval[position]
            @inbounds values[position] /= row_scaling[row] * col_scaling[column]
        end
    end
    return
end

function _max_abs_row_sparse_kernel!(values, rowptr, result, nrows)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        maximum_value = 0.0
        for position in first_position:last_position
            maximum_value = max(maximum_value, abs(@inbounds values[position]))
        end
        @inbounds result[row] = maximum_value
    end
    return
end


function _alpha_norm_row_sparse_kernel!(values, rowptr, alpha, result, nrows)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        total = 0.0
        for position in first_position:last_position
            value = abs(@inbounds values[position])
            total += alpha == 1.0 ? value : value^alpha
        end
        @inbounds result[row] = total
    end
    return
end

function _fill_row_sparse_kernel!(rowptr, row_indices, nrows)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        for position in first_position:last_position
            @inbounds row_indices[position] = row
        end
    end
    return
end

function _rescale_coo_sparse_kernel!(
    values,
    row_indices,
    col_indices,
    row_scaling,
    col_scaling,
    num_entries,
)
    Ti = eltype(row_indices)
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
        row = @inbounds row_indices[position]
        column = @inbounds col_indices[position]
        @inbounds values[position] /= row_scaling[row] * col_scaling[column]
    end
    return
end

function _max_abs_indexed_sparse_kernel!(values, indices, result, num_entries)
    Ti = eltype(indices)
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
        output_index = @inbounds indices[position]
        value = abs(@inbounds values[position])
        CUDA.@atomic result[output_index] = max(result[output_index], value)
    end
    return
end

function _alpha_norm_indexed_sparse_kernel!(
    values,
    indices,
    alpha,
    result,
    num_entries,
)
    Ti = eltype(indices)
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
        output_index = @inbounds indices[position]
        value = abs(@inbounds values[position])
        contribution = alpha == 1.0 ? value : value^alpha
        CUDA.@atomic result[output_index] += contribution
    end
    return
end

# ----------------------------------------------------------------------------
# Reflection Update Kernel
# ----------------------------------------------------------------------------
# Updates primal and dual solutions using reflection/extrapolation techniques
# in the aggressive-update path.

const _reflection_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _reflection_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _reflection_update_lock   = SpinLock()

const _reflection_update_path = utils_path
const _reflection_update_name = "reflection_update"

"""
    get_reflection_update_kernel() -> CuFunction

Lazily loads and returns the reflection update CUDA kernel.
Uses thread-safe lazy loading pattern.
"""
function get_reflection_update_kernel()::CuFunction
    k = _reflection_update_kernel[]
    k !== nothing && return k

    lock(_reflection_update_lock)
    try
        k = _reflection_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_reflection_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _reflection_update_name)

        _reflection_update_mod[]    = mod
        _reflection_update_kernel[] = fun
        return fun
    finally
        unlock(_reflection_update_lock)
    end
end

"""
    reflection_update(
        primal_sol,
        primal_sol_lag,
        primal_sol_mean,
        primal_halpern_candidate,
        primal_restart_anchor,
        dual_sol,
        dual_sol_lag,
        dual_sol_mean,
        dual_halpern_candidate,
        dual_restart_anchor,
        extra_coeff,
        primal_n,
        dual_n,
        inner_iter,
        eta_cum,
        eta,
        use_reflection,
        use_halpern,
        use_inline_halpern,
    )

Updates the reflected main trajectory, its running mean, and an auxiliary
Halpern restart candidate.

This kernel implements the reflection step in the aggressive-update path.
The reflection uses extrapolation
coefficients (eta, eta_cum) to combine current and lagged solutions.

Arguments:
- primal_sol: Current primal solution (GPU array, modified in-place)
- primal_sol_lag: Lagged primal solution (GPU array)
- primal_sol_mean: Running average of primal solution (GPU array, modified in-place)
- primal_halpern_candidate: Auxiliary primal Halpern restart candidate
- primal_restart_anchor: Primal restart anchor from Algorithm 1
- dual_sol: Current dual solution (GPU array, modified in-place)
- dual_sol_lag: Lagged dual solution (GPU array)
- dual_sol_mean: Running average of dual solution (GPU array, modified in-place)
- dual_halpern_candidate: Auxiliary dual Halpern restart candidate
- dual_restart_anchor: Dual restart anchor from Algorithm 1
- extra_coeff: Extra extrapolation coefficient
- primal_n: Dimension of primal variable
- dual_n: Dimension of dual variable
- inner_iter: Current inner iteration number
- eta_cum: Cumulative extrapolation coefficient
- eta: Current extrapolation coefficient
- use_weighted_average: Use step-size weights for the running mean; when false,
  use a uniform arithmetic mean over the inner iterations
- use_reflection: Whether to apply the reflection/extrapolation term
- use_halpern: Whether to form the auxiliary restart-only Halpern candidate.
- use_inline_halpern: Whether to apply the legacy inline update to the main
  trajectory. This reproduces the archived formula that mixes the reflected
  point with the unreflected PDHG point from the same iteration.
"""
function reflection_update(
    primal_sol::T,
    primal_sol_lag::T,
    primal_sol_mean::T,
    primal_halpern_candidate::T,
    primal_restart_anchor::T,
    dual_sol::T,
    dual_sol_lag::T,
    dual_sol_mean::T,
    dual_halpern_candidate::T,
    dual_restart_anchor::T,
    extra_coeff::Float64,
    primal_n::Int64,
    dual_n::Int64,
    inner_iter::Int64,
    eta_cum::Float64,
    eta::Float64,
    use_weighted_average::Bool,
    use_reflection::Bool,
    use_halpern::Bool,
    use_inline_halpern::Bool,
) where T<:CuArray
    # Calculate blocks based on maximum dimension (primal or dual)
    nBlock = cld(max(primal_n, dual_n), ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
            get_reflection_update_kernel(),
            (
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                Float64,
                Int64,
                Int64,
                Int64,
                Float64,
                Float64,
                Int32,
                Int32,
                Int32,
                Int32,
            ),
            primal_sol,
            primal_sol_lag,
            primal_sol_mean,
            primal_halpern_candidate,
            primal_restart_anchor,
            dual_sol,
            dual_sol_lag,
            dual_sol_mean,
            dual_halpern_candidate,
            dual_restart_anchor,
            extra_coeff,
            primal_n,
            dual_n,
            inner_iter,
            eta_cum,
            eta,
            Int32(use_weighted_average),
            Int32(use_reflection),
            Int32(use_halpern),
            Int32(use_inline_halpern);
            blocks = nBlock,
            threads = ThreadPerBlock,
        )
    end
end


# ----------------------------------------------------------------------------
# Primal Update Kernel
# ----------------------------------------------------------------------------
# Updates the primal variable in the RPDHG algorithm.
# Implements: x^{k+1} = x^k - tau * (c + G^T * y^k + d_c)

const _primal_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _primal_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _primal_update_lock   = SpinLock()

const _primal_update_path = utils_path
const _primal_update_name = "primal_update"

"""
    get_primal_update_kernel() -> CuFunction

Lazily loads and returns the primal update CUDA kernel.
"""
function get_primal_update_kernel()::CuFunction
    k = _primal_update_kernel[]
    k !== nothing && return k

    lock(_primal_update_lock)
    try
        k = _primal_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_primal_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _primal_update_name)

        _primal_update_mod[]    = mod
        _primal_update_kernel[] = fun
        return fun
    finally
        unlock(_primal_update_lock)
    end
end

"""
    primal_update(primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n)

Updates the primal variable in the RPDHG algorithm.

Performs the primal update step: x^{k+1} = x^k - tau * (c + G^T * y^k + d_c)
where tau is the primal step size.

Arguments:
- primal_sol: Current primal solution (GPU array, modified in-place)
- primal_sol_lag: Previous primal solution (GPU array, modified in-place)
- primal_sol_diff: Difference vector (GPU array, modified in-place)
- d_c: Scaled gradient term (GPU array)
- tau: Primal step size parameter
- n: Dimension of primal variable
"""
function primal_update(primal_sol::T, primal_sol_lag::T, primal_sol_diff::T, d_c::T, tau::Float64, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_primal_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# ----------------------------------------------------------------------------
# Dual Update Kernel
# ----------------------------------------------------------------------------
# Updates the dual variable in the RPDHG algorithm.
# Implements: y^{k+1} = y^k + sigma * (G * x^{k+1} - h + d_h)

const _dual_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _dual_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _dual_update_lock   = SpinLock()

const _dual_update_path = utils_path
const _dual_update_name = "dual_update"

"""
    get_dual_update_kernel() -> CuFunction

Lazily loads and returns the dual update CUDA kernel.
"""
function get_dual_update_kernel()::CuFunction
    k = _dual_update_kernel[]
    k !== nothing && return k

    lock(_dual_update_lock)
    try
        k = _dual_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_dual_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _dual_update_name)

        _dual_update_mod[]    = mod
        _dual_update_kernel[] = fun
        return fun
    finally
        unlock(_dual_update_lock)
    end
end

"""
    dual_update(dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n)

Updates the dual variable in the RPDHG algorithm.

Performs the dual update step: y^{k+1} = y^k + sigma * (G * x^{k+1} - h + d_h)
where sigma is the dual step size.

Arguments:
- dual_sol: Current dual solution (GPU array, modified in-place)
- dual_sol_lag: Previous dual solution (GPU array, modified in-place)
- dual_sol_diff: Difference vector (GPU array, modified in-place)
- d_h: Scaled constraint violation term (GPU array)
- sigma: Dual step size parameter
- n: Dimension of dual variable
"""
function dual_update(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, d_h::T, sigma::Float64, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_dual_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# Extrapolation Update Kernel
# ----------------------------------------------------------------------------
# Computes the extrapolation/difference between current and lagged primal solutions.
# Used by aggressive updates: x_diff = x - x_lag

const _extrapolation_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _extrapolation_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _extrapolation_update_lock   = SpinLock()

const _extrapolation_update_path = utils_path
const _extrapolation_update_name = "extrapolation_update"

"""
    get_extrapolation_update_kernel() -> CuFunction

Lazily loads and returns the extrapolation update CUDA kernel.
"""
function get_extrapolation_update_kernel()::CuFunction
    k = _extrapolation_update_kernel[]
    k !== nothing && return k

    lock(_extrapolation_update_lock)
    try
        k = _extrapolation_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_extrapolation_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _extrapolation_update_name)

        _extrapolation_update_mod[]    = mod
        _extrapolation_update_kernel[] = fun
        return fun
    finally
        unlock(_extrapolation_update_lock)
    end
end

"""
    extrapolation_update(primal_sol_diff, primal_sol, primal_sol_lag, n)

Computes the difference between current and lagged primal solutions.

Calculates: primal_sol_diff = primal_sol - primal_sol_lag
This difference is used by the aggressive-update path.

Arguments:
- primal_sol_diff: Output difference vector (GPU array, modified in-place)
- primal_sol: Current primal solution (GPU array)
- primal_sol_lag: Lagged primal solution (GPU array)
- n: Dimension of primal variable
"""
function extrapolation_update(primal_sol_diff::T, primal_sol::T, primal_sol_lag::T, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_extrapolation_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        primal_sol_diff, primal_sol, primal_sol_lag, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# ----------------------------------------------------------------------------
# Calculate Difference Kernel
# ----------------------------------------------------------------------------
# Computes differences between current and lagged solutions for both
# primal and dual variables. Used for tracking convergence and restart logic.

const _calculate_diff_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _calculate_diff_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _calculate_diff_lock   = SpinLock()

const _calculate_diff_path = utils_path
const _calculate_diff_name = "calculate_diff"

"""
    get_calculate_diff_kernel() -> CuFunction

Lazily loads and returns the calculate difference CUDA kernel.
"""
function get_calculate_diff_kernel()::CuFunction
    k = _calculate_diff_kernel[]
    k !== nothing && return k

    lock(_calculate_diff_lock)
    try
        k = _calculate_diff_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_calculate_diff_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _calculate_diff_name)

        _calculate_diff_mod[]    = mod
        _calculate_diff_kernel[] = fun
        return fun
    finally
        unlock(_calculate_diff_lock)
    end
end

"""
    calculate_diff(dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n)

Computes differences between current and lagged solutions for both primal and dual variables.

Calculates:
- dual_sol_diff = dual_sol - dual_sol_lag
- primal_sol_diff = primal_sol - primal_sol_lag

These differences are used for convergence monitoring and adaptive restart strategies.

Arguments:
- dual_sol: Current dual solution (GPU array)
- dual_sol_lag: Lagged dual solution (GPU array)
- dual_sol_diff: Dual difference vector (GPU array, modified in-place)
- dual_n: Dimension of dual variable
- primal_sol: Current primal solution (GPU array)
- primal_sol_lag: Lagged primal solution (GPU array)
- primal_sol_diff: Primal difference vector (GPU array, modified in-place)
- primal_n: Dimension of primal variable
"""
function calculate_diff(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, dual_n::Int64, primal_sol::T, primal_sol_lag::T, primal_sol_diff::T,  primal_n::Int64) where T<:CuArray
    # Use maximum dimension to determine number of blocks
    nBlock = cld(max(dual_n, primal_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_calculate_diff_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# AXPYZ Kernel (BLAS-like operation)
# ----------------------------------------------------------------------------
# Performs the operation: z = alpha * y + x
# This is a common linear algebra operation used throughout the algorithm.

const _axpyz_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _axpyz_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _axpyz_lock   = SpinLock()

const _axpyz_path = utils_path
const _axpyz_name = "axpyz"

"""
    get_axpyz_kernel() -> CuFunction

Lazily loads and returns the axpyz CUDA kernel.
"""
function get_axpyz_kernel()::CuFunction
    k = _axpyz_kernel[]
    k !== nothing && return k

    lock(_axpyz_lock)
    try
        k = _axpyz_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_axpyz_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _axpyz_name)

        _axpyz_mod[]    = mod
        _axpyz_kernel[] = fun
        return fun
    finally
        unlock(_axpyz_lock)
    end
end

"""
    axpyz(z, alpha, y, x, n)

Performs the BLAS-like operation: z = alpha * y + x

This is equivalent to: z[i] = alpha * y[i] + x[i] for all i.

Arguments:
- z: Output vector (GPU array, modified in-place)
- alpha: Scalar coefficient
- y: First input vector (GPU array)
- x: Second input vector (GPU array)
- n: Length of vectors
"""
function axpyz(z::T, alpha::Float64, y::T, x::T, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_axpyz_kernel(), 
        (CuPtr{Float64}, Float64, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        z, alpha, y, x, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# Average Sequence Kernel
# ----------------------------------------------------------------------------
# Computes running averages of primal and dual solutions.
# Used for averaging methods in RPDHG: mean = (k * mean + new_value) / (k + 1)

const _average_seq_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _average_seq_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _average_seq_lock   = SpinLock()

const _average_seq_path = utils_path
const _average_seq_name = "average_seq"

"""
    get_average_seq_kernel() -> CuFunction

Lazily loads and returns the average sequence CUDA kernel.
"""
function get_average_seq_kernel()::CuFunction
    k = _average_seq_kernel[]
    k !== nothing && return k

    lock(_average_seq_lock)
    try
        k = _average_seq_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_average_seq_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _average_seq_name)

        _average_seq_mod[]    = mod
        _average_seq_kernel[] = fun
        return fun
    finally
        unlock(_average_seq_lock)
    end
end

"""
    average_seq(; primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter)

Updates running averages of primal and dual solutions.

Implements exponential moving average:
- primal_sol_mean = (inner_iter * primal_sol_mean + primal_sol) / (inner_iter + 1)
- dual_sol_mean = (inner_iter * dual_sol_mean + dual_sol) / (inner_iter + 1)

Arguments:
- primal_sol_mean: Running average of primal solution (GPU array, modified in-place)
- primal_sol: Current primal solution (GPU array)
- primal_n: Dimension of primal variable
- dual_sol_mean: Running average of dual solution (GPU array, modified in-place)
- dual_sol: Current dual solution (GPU array)
- dual_n: Dimension of dual variable
- inner_iter: Current inner iteration number (used as weight)
"""
function average_seq(; primal_sol_mean::T, primal_sol::T, primal_n::Int64, dual_sol_mean::T, dual_sol::T, dual_n::Int64, inner_iter::Int64) where T<:CuArray
    nBlock = cld(max(primal_n, dual_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_average_seq_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64), 
        primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ============================================================================
# Section 5: Matrix Scaling and Norm Computation Kernels
# ============================================================================
# These kernels are used for preconditioning and scaling operations on
# sparse matrices stored in CSR (Compressed Sparse Row) format.

# ----------------------------------------------------------------------------
# Rescale CSR Matrix Kernel
# ----------------------------------------------------------------------------
# Scales a CSR matrix by row and column scaling factors.
# Performs: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]

const _rescale_csr_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_csr_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_csr_lock   = SpinLock()

const _rescale_csr_path = utils_path
const _rescale_csr_name = "rescale_csr"

"""
    get_rescale_csr_kernel() -> CuFunction

Lazily loads and returns the rescale CSR matrix CUDA kernel.
"""
function get_rescale_csr_kernel()::CuFunction
    k = _rescale_csr_kernel[]
    k !== nothing && return k

    lock(_rescale_csr_lock)
    try
        k = _rescale_csr_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_rescale_csr_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _rescale_csr_name)

        _rescale_csr_mod[]    = mod
        _rescale_csr_kernel[] = fun
        return fun
    finally
        unlock(_rescale_csr_lock)
    end
end

"""
    rescale_csr(d_G, row_scaling, col_scaling, m, n)

Scales a CSR sparse matrix by row and column scaling factors.

Performs element-wise scaling: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]
This is used for matrix preconditioning to improve numerical conditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR, modified in-place)
- row_scaling: Row scaling factors (GPU array)
- col_scaling: Column scaling factors (GPU array)
- m: Number of rows
- n: Number of columns
"""
function rescale_csr(
    d_G::CUDA.CUSPARSE.CuSparseMatrixCSR,
    row_scaling::CuArray,
    col_scaling::CuArray,
    m::Int64,
    n::Int64,
)
    m == 0 && return
    typed_m = eltype(d_G.rowPtr)(m)
    nblocks = cld(m, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _rescale_csr_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            d_G.colVal,
            row_scaling,
            col_scaling,
            typed_m,
        )
    end
    return
end

# ----------------------------------------------------------------------------
# Replace Infinity with Zero Kernel
# ----------------------------------------------------------------------------
# Replaces infinite values in bound arrays with zero.
# Used to handle unbounded variables (Inf bounds) in the optimization problem.

const _replace_inf_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _replace_inf_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _replace_inf_lock   = SpinLock()

const _replace_inf_path = utils_path
const _replace_inf_name = "replace_inf_with_zero"

"""
    get_replace_inf_kernel() -> CuFunction

Lazily loads and returns the replace infinity kernel.
"""
function get_replace_inf_kernel()::CuFunction
    k = _replace_inf_kernel[]
    k !== nothing && return k

    lock(_replace_inf_lock)
    try
        k = _replace_inf_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_replace_inf_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _replace_inf_name)

        _replace_inf_mod[]    = mod
        _replace_inf_kernel[] = fun
        return fun
    finally
        unlock(_replace_inf_lock)
    end
end

"""
    replace_inf_with_zero(bl, bu, n)

Replaces infinite values in bound arrays with zero.

For unbounded variables, bounds are set to Inf. This kernel replaces
Inf values with 0 to avoid numerical issues in GPU computations.

Arguments:
- bl: Lower bounds (GPU array, modified in-place)
- bu: Upper bounds (GPU array, modified in-place)
- n: Length of bound arrays
"""
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



# ----------------------------------------------------------------------------
# Max Absolute Row Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each row of a CSR sparse matrix.
# Used for row scaling in preconditioning: result[i] = max_j |G[i,j]|

const _max_abs_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_lock   = SpinLock()

const _max_abs_row_path = utils_path
const _max_abs_row_name = "max_abs_row_kernel"

"""
    get_max_abs_row_kernel() -> CuFunction

Lazily loads and returns the max absolute row CUDA kernel.
"""
function get_max_abs_row_kernel()::CuFunction
    k = _max_abs_row_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_lock)
    try
        k = _max_abs_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_row_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_row_name)

        _max_abs_row_mod[]    = mod
        _max_abs_row_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_lock)
    end
end

"""
    max_abs_row(d_G, result)

Computes the maximum absolute value in each row of a CSR sparse matrix.

For each row i, computes: result[i] = max_j |G[i,j]|
This is used for row scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of row maxima (GPU array, modified in-place)
          The function resets it to 0.0 before computing the maxima
"""
function max_abs_row(d_G, result)
    result .= 0.0
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_row_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            result,
            typed_nrows,
        )
    end
    return
end


# ----------------------------------------------------------------------------
# Max Absolute Column Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each column of a CSR sparse matrix.
# Used for column scaling in preconditioning: result[j] = max_i |G[i,j]|

const _max_abs_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_lock   = SpinLock()

const _max_abs_col_path = utils_path
const _max_abs_col_name = "max_abs_col_kernel"

"""
    get_max_abs_col_kernel() -> CuFunction

Lazily loads and returns the max absolute column CUDA kernel.
"""
function get_max_abs_col_kernel()::CuFunction
    k = _max_abs_col_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_lock)
    try
        k = _max_abs_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_col_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_col_name)

        _max_abs_col_mod[]    = mod
        _max_abs_col_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_lock)
    end
end

"""
    max_abs_col(d_G, result)

Computes the maximum absolute value in each column of a CSR sparse matrix.

For each column j, computes: result[j] = max_i |G[i,j]|
This is used for column scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of column maxima (GPU array, modified in-place)
          The function resets it to 0.0 before computing the maxima
"""
function max_abs_col(d_G, result)
    return max_abs_col_elementwise(d_G, result)
end



# ----------------------------------------------------------------------------
# Alpha Norm Row Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm (L_alpha norm) for each row of a CSR sparse matrix.
# For alpha=1, this is the L1 norm (sum of absolute values).
# Used for row scaling in preconditioning: result[i] = (sum_j |G[i,j]|^alpha)^(1/alpha)

const _alpha_norm_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_row_lock   = SpinLock()

const _alpha_norm_row_path = utils_path
const _alpha_norm_row_name = "alpha_norm_row_kernel"

"""
    get_alpha_norm_row_kernel() -> CuFunction

Lazily loads and returns the alpha norm row CUDA kernel.
"""
function get_alpha_norm_row_kernel()::CuFunction
    k = _alpha_norm_row_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_row_lock)
    try
        k = _alpha_norm_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_row_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_row_name)

        _alpha_norm_row_mod[]    = mod
        _alpha_norm_row_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_row_lock)
    end
end

"""
    alpha_norm_row(d_G, alpha, result)

Computes the alpha-norm for each row of a CSR sparse matrix.

For each row i, computes: result[i] = (sum_j |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This is used for row scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of row norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function alpha_norm_row(d_G, alpha, result)
    result .= 0.0
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _alpha_norm_row_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            Float64(alpha),
            result,
            typed_nrows,
        )
    end
    return
end



# ----------------------------------------------------------------------------
# Alpha Norm Column Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm (L_alpha norm) for each column of a CSR sparse matrix.
# For alpha=1, this is the L1 norm (sum of absolute values).
# Used for column scaling in preconditioning: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)

const _alpha_norm_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_lock   = SpinLock()

const _alpha_norm_col_path = utils_path
const _alpha_norm_col_name = "alpha_norm_col_kernel"

"""
    get_alpha_norm_col_kernel() -> CuFunction

Lazily loads and returns the alpha norm column CUDA kernel.
"""
function get_alpha_norm_col_kernel()::CuFunction
    k = _alpha_norm_col_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_lock)
    try
        k = _alpha_norm_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_col_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_col_name)

        _alpha_norm_col_mod[]    = mod
        _alpha_norm_col_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_lock)
    end
end

"""
    alpha_norm_col(d_G, alpha, result)

Computes the alpha-norm for each column of a CSR sparse matrix.

For each column j, computes: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This is used for column scaling in matrix preconditioning.

Note: When alpha=1.0, the final power operation (^(1/alpha)) is skipped
since it's just the identity operation.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of column norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function alpha_norm_col(d_G, alpha, result)
    return alpha_norm_col_elementwise(d_G, alpha, result)
end



# ----------------------------------------------------------------------------
# Get Row Index Kernel
# ----------------------------------------------------------------------------
# Computes the row index for each non-zero element in a CSR sparse matrix.
# This is useful for elementwise operations that need to know which row
# each non-zero element belongs to.

const _get_row_index_path = utils_path
const _get_row_index_name = "get_row_index"
const _get_row_index_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _get_row_index_lock   = SpinLock()
const _get_row_index_mod    = Ref{Union{Nothing,CuModule}}(nothing)

"""
    get_row_index_kernel() -> CuFunction

Lazily loads and returns the get row index CUDA kernel.
"""
function get_row_index_kernel()::CuFunction
    k = _get_row_index_kernel[]
    k !== nothing && return k

    lock(_get_row_index_lock)
    try
        k = _get_row_index_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_get_row_index_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _get_row_index_name)

        _get_row_index_mod[]    = mod
        _get_row_index_kernel[] = fun
        return fun
    finally
        unlock(_get_row_index_lock)
    end
end

"""
    get_row_index(d_G, row_idx)

Computes the row index for each non-zero element in a CSR sparse matrix.

For each non-zero element at position k in the CSR format, computes which row
it belongs to. This is useful for elementwise operations that need row information.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- row_idx: Output array of row indices (GPU array, modified in-place)
           Length should equal the number of non-zero elements (nnz)
"""
function get_row_index(d_G, row_idx)
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _fill_row_sparse_kernel!(
            d_G.rowPtr,
            row_idx,
            typed_nrows,
        )
    end
    return
end



# ----------------------------------------------------------------------------
# Rescale COO (Coordinate) Format Kernel
# ----------------------------------------------------------------------------
# Scales a CSR matrix by row and column scaling factors, using row indices
# computed from the CSR format. Similar to rescale_csr but uses precomputed
# row indices for better performance in elementwise operations.
# Performs: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]

const _rescale_coo_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_coo_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_coo_lock   = SpinLock()

const _rescale_coo_path = utils_path
const _rescale_coo_name = "rescale_coo"

"""
    get_rescale_coo_kernel() -> CuFunction

Lazily loads and returns the rescale COO CUDA kernel.
"""
function get_rescale_coo_kernel()::CuFunction
    k = _rescale_coo_kernel[]
    k !== nothing && return k

    lock(_rescale_coo_lock)
    try
        k = _rescale_coo_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_rescale_coo_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _rescale_coo_name)

        _rescale_coo_mod[]    = mod
        _rescale_coo_kernel[] = fun
        return fun
    finally
        unlock(_rescale_coo_lock)
    end
end

"""
    rescale_coo(d_G, row_scaling, col_scaling, m, n, row_idx)

Scales a CSR sparse matrix by row and column scaling factors using precomputed row indices.

Performs element-wise scaling: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]
This version uses precomputed row indices (from get_row_index) for better performance
when performing multiple scaling operations.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR, modified in-place)
- row_scaling: Row scaling factors (GPU array)
- col_scaling: Column scaling factors (GPU array)
- m: Number of rows
- n: Number of columns
- row_idx: Precomputed row indices for each non-zero element (GPU array)
           Should be computed using get_row_index() before calling this function
"""
function rescale_coo(
    d_G::CUDA.CUSPARSE.CuSparseMatrixCSR,
    row_scaling::CuArray,
    col_scaling::CuArray,
    m::Int64,
    n::Int64,
    row_idx::CuArray,
)
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(row_idx)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _rescale_coo_sparse_kernel!(
            d_G.nzVal,
            row_idx,
            d_G.colVal,
            row_scaling,
            col_scaling,
            typed_entries,
        )
    end
    return
end




# ============================================================================
# Section 6: Elementwise Operations
# ============================================================================
# These kernels perform elementwise operations on sparse matrices using
# precomputed row indices. They are optimized for cases where row indices
# are already known, avoiding repeated computation.

# ----------------------------------------------------------------------------
# Max Absolute Row Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each row using elementwise operations
# with precomputed row indices. More efficient than max_abs_row when row
# indices are already available.

const _max_abs_row_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_elementwise_lock   = SpinLock()

const _max_abs_row_elementwise_path = utils_path
const _max_abs_row_elementwise_name = "max_abs_row_elementwise_kernel"

"""
    get_max_abs_row_elementwise_kernel() -> CuFunction

Lazily loads and returns the max absolute row elementwise CUDA kernel.
"""
function get_max_abs_row_elementwise_kernel()::CuFunction
    k = _max_abs_row_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_elementwise_lock)
    try
        k = _max_abs_row_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_row_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_row_elementwise_name)

        _max_abs_row_elementwise_mod[]    = mod
        _max_abs_row_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_elementwise_lock)
    end
end

"""
    max_abs_row_elementwise(d_G, row_idx, result)

Computes the maximum absolute value in each row using elementwise operations.

For each row i, computes: result[i] = max_j |G[i,j]|
This version uses precomputed row indices for better performance when
performing multiple row-wise operations.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- row_idx: Precomputed row indices for each non-zero element (GPU array)
           Should be computed using get_row_index() before calling
- result: Output vector of row maxima (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function max_abs_row_elementwise(d_G, row_idx, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(row_idx)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_indexed_sparse_kernel!(
            d_G.nzVal,
            row_idx,
            result,
            typed_entries,
        )
    end
    return
end


# ----------------------------------------------------------------------------
# Max Absolute Column Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each column using elementwise operations.
# More efficient than max_abs_col when processing elements directly.

const _max_abs_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_elementwise_lock   = SpinLock()

const _max_abs_col_elementwise_path = utils_path
const _max_abs_col_elementwise_name = "max_abs_col_elementwise_kernel"

"""
    get_max_abs_col_elementwise_kernel() -> CuFunction

Lazily loads and returns the max absolute column elementwise CUDA kernel.
"""
function get_max_abs_col_elementwise_kernel()::CuFunction
    k = _max_abs_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_elementwise_lock)
    try
        k = _max_abs_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_col_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_col_elementwise_name)

        _max_abs_col_elementwise_mod[]    = mod
        _max_abs_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_elementwise_lock)
    end
end

"""
    max_abs_col_elementwise(d_G, result)

Computes the maximum absolute value in each column using elementwise operations.

For each column j, computes: result[j] = max_i |G[i,j]|
This version processes elements directly from the CSR format, which can be
more efficient than the row-based approach for certain matrix structures.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of column maxima (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function max_abs_col_elementwise(d_G, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(d_G.colVal)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_indexed_sparse_kernel!(
            d_G.nzVal,
            d_G.colVal,
            result,
            typed_entries,
        )
    end
    return
end


# ----------------------------------------------------------------------------
# Alpha Norm Column Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm for each column using elementwise operations.
# More efficient than alpha_norm_col when processing elements directly.

const _alpha_norm_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_elementwise_lock   = SpinLock()

const _alpha_norm_col_elementwise_path = utils_path
const _alpha_norm_col_elementwise_name = "alpha_norm_col_elementwise_kernel"

"""
    get_alpha_norm_col_elementwise_kernel() -> CuFunction

Lazily loads and returns the alpha norm column elementwise CUDA kernel.
"""
function get_alpha_norm_col_elementwise_kernel()::CuFunction
    k = _alpha_norm_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_elementwise_lock)
    try
        k = _alpha_norm_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_col_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_col_elementwise_name)

        _alpha_norm_col_elementwise_mod[]    = mod
        _alpha_norm_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_elementwise_lock)
    end
end

"""
    alpha_norm_col_elementwise(d_G, alpha, result)

Computes the alpha-norm for each column using elementwise operations.

For each column j, computes: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This version processes elements directly from the CSR format.

Note: When alpha=1.0, the final power operation (^(1/alpha)) is skipped
since it's just the identity operation.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of column norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function alpha_norm_col_elementwise(d_G, alpha, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(d_G.colVal)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _alpha_norm_indexed_sparse_kernel!(
            d_G.nzVal,
            d_G.colVal,
            Float64(alpha),
            result,
            typed_entries,
        )
    end
    return
end
