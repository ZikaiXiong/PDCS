import MathOptInterface as MOI
"""
    struct ScaledPSDCone <: MOI.AbstractVectorSet
        side_dimension::Int
    end

Similar to `MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}` but it the
vectorization is the lower triangular part column-wise (or the upper triangular
part row-wise).
"""
struct ScaledPSDCone <: MOI.AbstractVectorSet
    side_dimension::Int
end

function MOI.Utilities.set_with_dimension(::Type{ScaledPSDCone}, dim)
    return ScaledPSDCone(div(-1 + isqrt(1 + 8 * dim), 2))
end

Base.copy(x::ScaledPSDCone) = ScaledPSDCone(x.side_dimension)

MOI.side_dimension(x::ScaledPSDCone) = x.side_dimension

function MOI.dimension(x::ScaledPSDCone)
    return div(x.side_dimension * (x.side_dimension + 1), 2)
end

struct ScaledPSDConeBridge{T,F} <: MOI.Bridges.Constraint.SetMapBridge{
    T,
    ScaledPSDCone,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
    F,
    F,
}
    constraint::MOI.ConstraintIndex{F,ScaledPSDCone}
end

function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{ScaledPSDConeBridge{T}},
    ::Type{F},
    ::Type{MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}},
) where {T,F<:MOI.AbstractVectorFunction}
    return ScaledPSDConeBridge{T,F}
end

function MOI.Bridges.map_set(
    ::Type{<:ScaledPSDConeBridge},
    set::MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
)
    return ScaledPSDCone(MOI.side_dimension(set))
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:ScaledPSDConeBridge},
    set::ScaledPSDCone,
)
    return MOI.Scaled(MOI.PositiveSemidefiniteConeTriangle(set.side_dimension))
end

function _upper_to_lower_triangular_permutation(dim::Int)
    side_dimension = MOI.Utilities.side_dimension_for_vectorized_dimension(dim)
    permutation = zeros(Int, dim)
    i = 0
    for row in 1:side_dimension
        start = div(row * (row + 1), 2)
        for col in row:side_dimension
            i += 1
            permutation[i] = start
            start += col
        end
    end
    return sortperm(permutation), permutation
end

function _transform_function(func, moi_to_pdhgclp::Bool)
    scalars = MOI.Utilities.eachscalar(func)
    d = length(scalars)
    upper_to_lower, lower_to_upper = _upper_to_lower_triangular_permutation(d)
    if moi_to_pdhgclp
        return scalars[lower_to_upper]
    else
        return scalars[upper_to_lower]
    end
end

# Map ConstraintFunction from MOI -> PDHGCLP
function MOI.Bridges.map_function(::Type{<:ScaledPSDConeBridge}, f)
    return _transform_function(f, true)
end

# Used to map the ConstraintPrimal from PDHGCLP -> MOI
function MOI.Bridges.inverse_map_function(::Type{<:ScaledPSDConeBridge}, f)
    return _transform_function(f, false)
end

# Used to map the ConstraintDual from PDHGCLP -> MOI
function MOI.Bridges.adjoint_map_function(::Type{<:ScaledPSDConeBridge}, f)
    return _transform_function(f, false)
end

# Used to set ConstraintDualStart
function MOI.Bridges.inverse_adjoint_map_function(
    ::Type{<:ScaledPSDConeBridge},
    f,
)
    return _transform_function(f, true)
end

MOI.Utilities.@product_of_sets(
    _Cones,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.DualExponentialCone,
    # MOI.PowerCone{T},
    # MOI.DualPowerCone{T},
)

struct _SetConstants{T}
    b::Vector{T}
    power_coefficients::Dict{Int,T}
    _SetConstants{T}() where {T} = new{T}(T[], Dict{Int,T}())
end

function Base.empty!(x::_SetConstants)
    empty!(x.b)
    empty!(x.power_coefficients)
    return x
end

Base.resize!(x::_SetConstants, n) = resize!(x.b, n)

function MOI.Utilities.load_constants(x::_SetConstants, offset, f)
    MOI.Utilities.load_constants(x.b, offset, f)
    return
end

function MOI.Utilities.load_constants(
    x::_SetConstants{T},
    offset,
    set::Union{MOI.PowerCone{T},MOI.DualPowerCone{T}},
) where {T}
    x.power_coefficients[offset+1] = set.exponent
    return
end

function MOI.Utilities.function_constants(x::_SetConstants, rows)
    return MOI.Utilities.function_constants(x.b, rows)
end

function MOI.Utilities.set_from_constants(x::_SetConstants, S, rows)
    return MOI.Utilities.set_from_constants(x.b, S, rows)
end

function MOI.Utilities.set_from_constants(
    x::_SetConstants{T},
    ::Type{S},
    rows,
) where {T,S<:Union{MOI.PowerCone{T},MOI.DualPowerCone{T}}}
    @assert length(rows) == 3
    return S(x.power_coefficients[first(rows)])
end

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            T,
            MOI.Utilities.OneBasedIndexing,
        },
        _SetConstants{Float64},
        _Cones{Float64},
    },
}

function _to_sparse(
    ::Type{T},
    A::MOI.Utilities.MutableSparseMatrixCSC{
        Float64,
        T,
        MOI.Utilities.OneBasedIndexing,
    },
) where {T}
    return -A.nzval, A.rowval, A.colptr
end

mutable struct MOISolution
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    exit_code::Int
    exit_status::Symbol
    objective_value::Float64
    dual_objective_value::Float64
    solve_time_sec::Float64
    iterations::Int
    objective_constant::Float64
    function MOISolution(;primal = Float64[], dual = Float64[], slack = Float64[], exit_code = 0, exit_status = :unknown, objective_value = NaN, dual_objective_value = NaN, solve_time_sec = NaN, iterations = 0, objective_constant = 0.0)
        new(primal, dual, slack, exit_code, exit_status, objective_value, dual_objective_value, solve_time_sec, iterations, objective_constant)
    end
end

"""
    Optimizer()

Create a new PDHG-CLP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    cones::Union{Nothing,_Cones{Float64}}
    sol::MOISolution
    silent::Bool
    options::Dict{Symbol,Any}

    Optimizer() = new(nothing, MOISolution(), false, Dict{Symbol,Any}())
end

function MOI.get(::Optimizer, ::MOI.Bridges.ListOfNonstandardBridges)
    return [ScaledPSDConeBridge{Float64}]
end

MOI.get(::Optimizer, ::MOI.SolverName) = "PDHG-CLP"

MOI.is_empty(optimizer::Optimizer) = optimizer.cones === nothing

function MOI.empty!(optimizer::Optimizer)
    optimizer.cones = nothing
    optimizer.sol = MOISolution()
    return
end

###
### MOI.RawOptimizerAttribute
###

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return optimizer.options[Symbol(param.name)] = value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[Symbol(param.name)]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.TimeLimitSec
###

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, time_limit::Real)
    optimizer.options[:time_limit_secs] = convert(Float64, time_limit)
    return
end

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(optimizer.options, :time_limit_secs)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec)
    value = get(optimizer.options, :time_limit_secs, nothing)
    return value::Union{Float64,Nothing}
end

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
        # MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
    },
)
    return true
end

###
### MOI.AbstractVariableAttribute
###

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

###
### MOI.AbstractConstraintAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ConstraintPrimalStart,MOI.ConstraintDualStart},
    ::Type{<:MOI.ConstraintIndex},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{
        <:Union{
            MOI.Zeros,
            MOI.Nonnegatives,
            MOI.SecondOrderCone,
            MOI.RotatedSecondOrderCone,
            MOI.ExponentialCone,
            MOI.DualExponentialCone,
            # MOI.PowerCone{Float64},
            # MOI.DualPowerCone{Float64},
        },
    },
)
    return true
end

# customized
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{
        <:Union{
            MOI.EqualTo{Float64},
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.Interval{Float64},
        }
    }
)
    return true
end

# # customized
# function MOI.supports_constraint(
#     ::Optimizer,
#     ::Type{MOI.VectorOfVariables},
#     ::Type{
#         <:Union{
#             MOI.SecondOrderCone,
#             MOI.RotatedSecondOrderCone,
#             # may be not supported for two exponential cones
#             MOI.ExponentialCone,
#             MOI.DualExponentialCone,
#         }
#     }
# )
#     return true
# end


function _map_sets(f, ::Type{T}, sets, ::Type{S}) where {T,S}
    F = MOI.VectorAffineFunction{Float64}
    cis = MOI.get(sets, MOI.ListOfConstraintIndices{F,S}())
    return T[f(MOI.get(sets, MOI.ConstraintSet(), ci)) for ci in cis]
end

function MOI.optimize!(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache{T}},
) where {T}
    # The real stuff starts here.
    MOI.empty!(dest)
    index_map = MOI.Utilities.identity_index_map(src)
    Ab = src.model.constraints
    A = Ab.coefficients

    for (F, S) in keys(src.constraints) # src.constraints is very important
        throw(MOI.UnsupportedConstraint{F,S}())
        # if F == MOI.VectorOfVariables
        #     if S == MOI.SecondOrderCone
        #         println("SecondOrderCone")
        #     elseif S == MOI.RotatedSecondOrderCone
        #         println("RotatedSecondOrderCone")
        #     elseif S == MOI.ExponentialCone
        #         println("ExponentialCone")
        #     elseif S == MOI.DualExponentialCone
        #         println("DualExponentialCone")
        #     else
        #         throw(MOI.UnsupportedConstraint{F,S}())
        #     end
        # else
        #     throw(MOI.UnsupportedConstraint{F,S}())
        # end
    end
    # key: src.model.variables is also very important
    model_attributes = MOI.get(src, MOI.ListOfModelAttributesSet())
    max_sense = false
    obj_attr = nothing
    for attr in model_attributes
        if attr == MOI.ObjectiveSense()
            max_sense = MOI.get(src, attr) == MOI.MAX_SENSE
        elseif attr == MOI.Name()
            continue  # This can be skipped without consequence
        elseif attr isa MOI.ObjectiveFunction
            obj_attr = attr
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    objective_constant, c = 0.0, zeros(A.n)
    if obj_attr == MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
        obj = MOI.get(src, obj_attr)
        objective_constant = MOI.constant(obj)
        for term in obj.terms
            c[term.variable.value] += (max_sense ? -1 : 1) * term.coefficient
        end
    elseif obj_attr !== nothing
        throw(MOI.UnsupportedAttribute(obj_attr))
    end

    # `model.primal` contains the result of the previous optimization.
    # It is used as a warm start if its length is the same, e.g.
    # probably because no variable and/or constraint has been added.
    if A.n != length(dest.sol.primal)
        resize!(dest.sol.primal, A.n)
        fill!(dest.sol.primal, 0.0)
    end
    if A.m != length(dest.sol.dual)
        resize!(dest.sol.dual, A.m)
        fill!(dest.sol.dual, 0.0)
    end
    if A.m != length(dest.sol.slack)
        resize!(dest.sol.slack, A.m)
        fill!(dest.sol.slack, 0.0)
    end
    # Set starting values and throw error for other variable attributes
    has_warm_start = false
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName()
            # Skip
        elseif attr == MOI.VariablePrimalStart()
            has_warm_start = true
            for (i, x) in enumerate(vis_src)
                dest.sol.primal[i] = something(MOI.get(src, attr, x), 0.0)
            end
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    # Set starting values and throw error for other constraint attributes
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if attr == MOI.ConstraintName()
                # Skip
            elseif attr == MOI.ConstraintPrimalStart()
                has_warm_start = true
                for ci in cis_src
                    start = MOI.get(src, attr, ci)
                    if start !== nothing
                        rows = MOI.Utilities.rows(Ab, ci)
                        dest.sol.slack[rows] .= start
                    end
                end
            elseif attr == MOI.ConstraintDualStart()
                has_warm_start = true
                for ci in cis_src
                    start = MOI.get(src, attr, ci)
                    if start !== nothing
                        rows = MOI.Utilities.rows(Ab, ci)
                        dest.sol.dual[rows] .= start
                    end
                end
            else
                throw(MOI.UnsupportedAttribute(attr))
            end
        end
    end
    options = copy(dest.options)
    if dest.silent
        options[:verbose] = 0
    end
    dest.cones = deepcopy(Ab.sets)
    mGzero = sum(_map_sets(MOI.dimension, T, Ab, MOI.Zeros))
    mGnonnegative = sum(_map_sets(MOI.dimension, T, Ab, MOI.Nonnegatives))
    G = SparseMatrixCSC{Float64, Int64}(A.m, A.n, Int64.(A.colptr), Int64.(A.rowval), A.nzval)
    bl = fill(-Inf, A.n)
    bl .= src.model.variables.lower
    bu = fill(Inf, A.n)
    bu .= src.model.variables.upper
    if isempty(G)
        error("The constraint matrix G is empty, PDHG solver cannot be applied.")
    end
    if !haskey(options, :verbose)
        options[:verbose] = 0
    end
    if !haskey(options, :logfile)
        options[:logfile] = nothing
        if options[:verbose] > 0
            println("logfile is not set, using default value: nothing.")
        end
    end
    if !haskey(options, :time_limit_secs)
        options[:time_limit_secs] = 1000.0
        if options[:verbose] > 0
            println("time_limit_secs is not set, using default value: 1000.0.")
        end
    end
    if !haskey(options, :use_scaling)
        options[:use_scaling] = true
        if options[:verbose] > 0
            println("use_scaling is not set, using default value: true.")
        end
    end
    if !haskey(options, :rescaling_method)
        options[:rescaling_method] = :ruiz_pock_chambolle
        if options[:verbose] > 0
            println("rescaling_method is not set, using default value: :ruiz_pock_chambolle. Note that if :use_scaling is false, this option is ignored.")
        end
    end
    if !haskey(options, :use_adaptive_restart)
        options[:use_adaptive_restart] = true
        if options[:verbose] > 0
            println("use_adaptive_restart is not set, using default value: true.")
        end
    end
    if !haskey(options, :use_adaptive_step_size_weight)
        options[:use_adaptive_step_size_weight] = true
        if options[:verbose] > 0
            println("use_adaptive_step_size_weight is not set, using default value: true.")
        end
    end
    if !haskey(options, :use_resolving)
        options[:use_resolving] = true
        if options[:verbose] > 0
            println("use_resolving is not set, using default value: true.")
        end
    end
    if !haskey(options, :print_freq)
        options[:print_freq] = 2000
        if options[:verbose] > 0
            println("print_freq is not set, using default value: 5000.")
        end
    end
    if !haskey(options, :use_accelerated)
        options[:use_accelerated] = false
        if options[:verbose] > 0
            println("use_accelerated is not set, using default value: true.")
        end
    end
    if !haskey(options, :use_aggressive)
        options[:use_aggressive] = true
        if options[:verbose] > 0
            println("use_aggressive is not set, using default value: true.")
        end
    end
    if !haskey(options, :kkt_restart_freq)
        options[:kkt_restart_freq] = 2000
        if options[:verbose] > 0
            println("kkt_restart_freq is not set, using default value: 500.")
        end
    end
    if !haskey(options, :duality_gap_restart_freq)
        options[:duality_gap_restart_freq] = 2000
        if options[:verbose] > 0
            println("duality_gap_restart_freq is not set, using default value: 2000.")
        end
    end
    if !haskey(options, :use_kkt_restart)
        options[:use_kkt_restart] = false
        if options[:verbose] > 0
            println("use_kkt_restart is not set, using default value: true.")
        end
    end
    if !haskey(options, :use_duality_gap_restart)
        options[:use_duality_gap_restart] = true
        if options[:verbose] > 0
            println("use_duality_gap_restart is not set, using default value: true.")
        end
    end
    if !haskey(options, :rel_tol)
        options[:rel_tol] = 1e-6
        if options[:verbose] > 0
            println("rel_tol is not set, using default value: 1e-6.")
        end
    end
    if !haskey(options, :abs_tol)
        options[:abs_tol] = 1e-6
        if options[:verbose] > 0
            println("abs_tol is not set, using default value: 1e-6.")
        end
    end
    if options[:use_scaling]
        sol_res = PDCP_GPU.rpdhg_gpu_solve(
            n = A.n,
            m = A.m,
            nb = A.n,
            c_cpu = c,
            G_cpu = G,
            h_cpu = -Ab.constants.b,
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            socG = _map_sets(MOI.dimension, T, Ab, MOI.SecondOrderCone),
            rsocG = _map_sets(MOI.dimension, T, Ab, MOI.RotatedSecondOrderCone),
            expG = length(_map_sets(MOI.dimension, T, Ab, MOI.ExponentialCone)),
            dual_expG = length(_map_sets(MOI.dimension, T, Ab, MOI.DualExponentialCone)),
            bl_cpu = bl,
            bu_cpu = bu,
            soc_x = Vector{Integer}([]),
            rsoc_x = Vector{Integer}([]),
            exp_x = 0,
            dual_exp_x = 0,
            print_freq = options[:print_freq],
            use_preconditioner = true,
            rescaling_method = options[:rescaling_method],
            method = :average,
            time_limit = options[:time_limit_secs],
            use_adaptive_restart = options[:use_adaptive_restart],
            use_adaptive_step_size_weight = options[:use_adaptive_step_size_weight],
            use_resolving = options[:use_resolving],
            use_accelerated = options[:use_accelerated],
            use_aggressive = options[:use_aggressive],
            verbose = options[:verbose],
            rel_tol = options[:rel_tol],
            abs_tol = options[:abs_tol],
            logfile_name = options[:logfile],
            kkt_restart_freq = options[:kkt_restart_freq],
            duality_gap_restart_freq = options[:duality_gap_restart_freq],
            use_kkt_restart = options[:use_kkt_restart],
            use_duality_gap_restart = options[:use_duality_gap_restart],
        )
        dest.sol = MOISolution(
            primal = sol_res.x.recovered_primal.primal_sol.x,
            dual = sol_res.y.recovered_dual.dual_sol.y,
            slack = sol_res.y.slack.primal_sol.x,
            exit_code = sol_res.info.exit_code,
            exit_status = sol_res.info.exit_status,
            objective_value = (max_sense ? -1 : 1) * sol_res.info.pObj,
            dual_objective_value = (max_sense ? -1 : 1) * sol_res.info.dObj,
            solve_time_sec = sol_res.info.time,
            iterations = sol_res.info.iter,
            objective_constant = objective_constant,
        )
    else
        sol_res = PDCP_GPU.rpdhg_gpu_solve(
            n = A.n,
            m = A.m,
            nb = A.n,
            c_cpu = c,
            G_cpu = G,
            h_cpu = -Ab.constants.b,
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            socG = _map_sets(MOI.dimension, T, Ab, MOI.SecondOrderCone),
            rsocG = _map_sets(MOI.dimension, T, Ab, MOI.RotatedSecondOrderCone),
            expG = length(_map_sets(MOI.dimension, T, Ab, MOI.ExponentialCone)),
            dual_expG = length(_map_sets(MOI.dimension, T, Ab, MOI.DualExponentialCone)),
            bl_cpu = bl,
            bu_cpu = bu,
            soc_x = Vector{Integer}([]),
            rsoc_x = Vector{Integer}([]),
            exp_x = 0,
            dual_exp_x = 0,
            print_freq = options[:print_freq],
            use_preconditioner = false,
            method = :average,
            time_limit = options[:time_limit_secs],
            use_adaptive_restart = options[:use_adaptive_restart],
            use_adaptive_step_size_weight = options[:use_adaptive_step_size_weight],
            use_resolving = options[:use_resolving],
            use_accelerated = options[:use_accelerated],
            use_aggressive = options[:use_aggressive],
            verbose = options[:verbose],
            rel_tol = options[:rel_tol],
            abs_tol = options[:abs_tol],
            logfile_name = options[:logfile],
            kkt_restart_freq = options[:kkt_restart_freq],
            duality_gap_restart_freq = options[:duality_gap_restart_freq],
            use_kkt_restart = options[:use_kkt_restart],
            use_duality_gap_restart = options[:use_duality_gap_restart],
        )
        dest.sol = MOISolution(
            primal = sol_res.x.primal_sol.x,
            dual = sol_res.y.dual_sol.y,
            slack = sol_res.y.slack.primal_sol.x,
            exit_code = sol_res.info.exit_code,
            exit_status = sol_res.info.exit_status,
            objective_value = (max_sense ? -1 : 1) * sol_res.info.pObj,
            dual_objective_value = (max_sense ? -1 : 1) * sol_res.info.dObj,
            solve_time_sec = sol_res.info.time,
            iterations = sol_res.info.iter,
            objective_constant = objective_constant,
        )
    end
    return index_map, false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = MOI.Utilities.UniversalFallback(OptimizerCache{Integer}())
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.sol.solve_time_sec
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.sol.raw_status
end

"""
    PDHGIterations()

The number of PDHG iterations completed during the solve.
"""
struct PDHGIterations <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::PDHGIterations) = true

function MOI.get(optimizer::Optimizer, ::PDHGIterations)
    return optimizer.sol.iterations
end

"""
exit_status:
    :optimal 0
    :max_iter 1
    :primal_infeasible_low_acc 2
    :primal_infeasible_high_acc 3
    :dual_infeasible_low_acc 4
    :dual_infeasible_high_acc 5
    :time_limit 6   
    :continue 7
    :numerical_error 8
"""
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    s = optimizer.sol.exit_code
    @assert 0 <= s <= 8
    if s == 0
        return MOI.OPTIMAL
    elseif s == 1
        return MOI.ITERATION_LIMIT
    elseif s == 2
        return MOI.INFEASIBLE
    elseif s == 3
        return MOI.DUAL_INFEASIBLE
    elseif s == 4
        return MOI.INFEASIBLE
    elseif s == 5
        return MOI.DUAL_INFEASIBLE
    elseif s == 6
        return MOI.TIME_LIMIT
    elseif s == 7
        return MOI.ITERATION_LIMIT
    elseif s == 8
        return MOI.NUMERICAL_ERROR
    else
        throw(MOI.UnsupportedAttribute(MOI.TerminationStatus()))
        if occursin("reached time_limit_secs", optimizer.sol.raw_status)
            return MOI.TIME_LIMIT
        elseif occursin("reached max_iters", optimizer.sol.raw_status)
            return MOI.ITERATION_LIMIT
        else
            return MOI.ALMOST_OPTIMAL
        end
    # elseif s == -5
    #     return MOI.INTERRUPTED
    # elseif s == -4
    #     return MOI.INVALID_MODEL
    # elseif s == -3
    #     return MOI.SLOW_PROGRESS
    # elseif s == -2
    #     return MOI.INFEASIBLE
    # elseif s == -1
    #     return MOI.DUAL_INFEASIBLE
    # elseif s == 1
    #     return MOI.OPTIMAL
    # else
    #     @assert s == 0
    #     return MOI.OPTIMIZE_NOT_CALLED
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.objective_value
    if !MOI.Utilities.is_ray(MOI.get(optimizer, MOI.PrimalStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.dual_objective_value
    if !MOI.Utilities.is_ray(MOI.get(optimizer, MOI.DualStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif optimizer.sol.exit_code in (0)
        return MOI.FEASIBLE_POINT
    elseif optimizer.sol.exit_code in (2, 3, 4, 5, 6)
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif optimizer.sol.exit_code == (1, 7)
        return MOI.NO_SOLUTION
    end
    return MOI.INFEASIBLE_POINT
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.primal[vi.value]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.slack[MOI.Utilities.rows(optimizer.cones, ci)]
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif optimizer.sol.exit_code in (0)
        return MOI.FEASIBLE_POINT
    elseif optimizer.sol.exit_code in (2, 3, 4, 5, 6)
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif optimizer.sol.exit_code == (1, 7)
        return MOI.NO_SOLUTION
    end
    return MOI.INFEASIBLE_POINT
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.dual[MOI.Utilities.rows(optimizer.cones, ci)]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1
