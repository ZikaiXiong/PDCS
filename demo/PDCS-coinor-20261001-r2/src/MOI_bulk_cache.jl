const _CONIC_DATA_PROPERTIES = (
    :objective_sense,
    :objective_constant,
    :objective_coefficients,
    :variable_lower,
    :variable_upper,
    :num_rows,
    :num_variables,
    :colptr,
    :rowval,
    :nzval,
    :affine_constants,
    :cone_blocks,
    :layout,
    :timings,
)

function _require_conic_properties(data)
    missing = Symbol[
        name for name in _CONIC_DATA_PROPERTIES if !hasproperty(data, name)
    ]
    isempty(missing) || throw(ArgumentError(
        "conic data is missing required properties: " * join(string.(missing), ", "),
    ))
    return nothing
end

function _checked_conic_dimension(value, name::Symbol)
    value isa Integer || throw(ArgumentError("$name must be an integer"))
    0 <= value <= typemax(Int) || throw(ArgumentError("$name is outside the Int range"))
    return Int(value)
end

function _float_vector(value, name::Symbol)
    value isa AbstractVector || throw(ArgumentError("$name must be a vector"))
    return value isa Vector{Float64} ? value : Float64.(value)
end

function _index_vectors(colptr_value, rowval_value)
    colptr_value isa AbstractVector{<:Integer} || throw(ArgumentError(
        "colptr must be an integer vector",
    ))
    rowval_value isa AbstractVector{<:Integer} || throw(ArgumentError(
        "rowval must be an integer vector",
    ))
    if colptr_value isa Vector &&
       rowval_value isa Vector &&
       eltype(colptr_value) == eltype(rowval_value)
        return colptr_value, rowval_value
    end
    return Int.(colptr_value), Int.(rowval_value)
end

function _cone_category(set_type)
    set_type === MOI.Zeros && return 1
    set_type === MOI.Nonnegatives && return 2
    set_type === MOI.SecondOrderCone && return 3
    set_type === MOI.ExponentialCone && return 5
    set_type === MOI.DualExponentialCone && return 6
    throw(ArgumentError("unsupported canonical cone type $set_type"))
end

function _validate_cone_blocks(blocks, num_rows::Int)
    blocks isa AbstractVector || throw(ArgumentError("cone_blocks must be a vector"))
    expected_row = 1
    previous_category = 0
    for (index, block) in pairs(blocks)
        for property in (:set_type, :dimension, :first_row)
            hasproperty(block, property) || throw(ArgumentError(
                "cone block $index is missing property $property",
            ))
        end
        category = _cone_category(getproperty(block, :set_type))
        category >= previous_category || throw(ArgumentError(
            "cone blocks are not in canonical category order",
        ))
        dimension = _checked_conic_dimension(getproperty(block, :dimension), :dimension)
        dimension > 0 || throw(ArgumentError("cone block dimensions must be positive"))
        first_row = _checked_conic_dimension(getproperty(block, :first_row), :first_row)
        first_row == expected_row || throw(ArgumentError(
            "cone block $index starts at row $first_row, expected $expected_row",
        ))
        if category in (5, 6)
            dimension == 3 || throw(ArgumentError(
                "exponential cone blocks must have dimension 3",
            ))
        elseif category == 3
            dimension >= 2 || throw(ArgumentError(
                "second-order cone blocks must have dimension at least 2",
            ))
        end
        expected_row = Base.checked_add(expected_row, dimension)
        previous_category = category
    end
    expected_row - 1 == num_rows || throw(DimensionMismatch(
        "cone blocks contain $(expected_row - 1) rows, expected $num_rows",
    ))
    return blocks
end

function _validated_conic_data(data)
    _require_conic_properties(data)
    num_rows = _checked_conic_dimension(data.num_rows, :num_rows)
    num_variables = _checked_conic_dimension(data.num_variables, :num_variables)
    objective_sense = data.objective_sense
    objective_sense in (MOI.MIN_SENSE, MOI.MAX_SENSE, MOI.FEASIBILITY_SENSE) ||
        throw(ArgumentError("unsupported objective sense $objective_sense"))
    objective_constant = Float64(data.objective_constant)
    isfinite(objective_constant) || throw(ArgumentError("objective constant must be finite"))
    objective = _float_vector(data.objective_coefficients, :objective_coefficients)
    lower = _float_vector(data.variable_lower, :variable_lower)
    upper = _float_vector(data.variable_upper, :variable_upper)
    constants = _float_vector(data.affine_constants, :affine_constants)
    nzval = _float_vector(data.nzval, :nzval)
    colptr, rowval = _index_vectors(data.colptr, data.rowval)

    length(objective) == num_variables || throw(DimensionMismatch(
        "objective has $(length(objective)) entries, expected $num_variables",
    ))
    length(lower) == num_variables || throw(DimensionMismatch(
        "lower bounds have $(length(lower)) entries, expected $num_variables",
    ))
    length(upper) == num_variables || throw(DimensionMismatch(
        "upper bounds have $(length(upper)) entries, expected $num_variables",
    ))
    length(constants) == num_rows || throw(DimensionMismatch(
        "affine constants have $(length(constants)) entries, expected $num_rows",
    ))
    length(colptr) == num_variables + 1 || throw(DimensionMismatch(
        "colptr has $(length(colptr)) entries, expected $(num_variables + 1)",
    ))
    length(rowval) == length(nzval) || throw(DimensionMismatch(
        "rowval and nzval lengths differ",
    ))
    all(isfinite, objective) || throw(ArgumentError("objective coefficients must be finite"))
    all(isfinite, constants) || throw(ArgumentError("affine constants must be finite"))
    all(isfinite, nzval) || throw(ArgumentError("matrix coefficients must be finite"))
    all(!iszero, nzval) || throw(ArgumentError("matrix contains explicit zero coefficients"))

    for variable in 1:num_variables
        lo = lower[variable]
        hi = upper[variable]
        isnan(lo) && throw(ArgumentError("lower bound $variable is NaN"))
        isnan(hi) && throw(ArgumentError("upper bound $variable is NaN"))
        lo != Inf || throw(ArgumentError("lower bound $variable is +Inf"))
        hi != -Inf || throw(ArgumentError("upper bound $variable is -Inf"))
        lo <= hi || throw(ArgumentError("lower bound exceeds upper bound at variable $variable"))
    end

    first(colptr) == 1 || throw(ArgumentError("colptr must use one-based indexing"))
    last(colptr) == length(rowval) + 1 || throw(DimensionMismatch(
        "last colptr entry must equal nnz + 1",
    ))
    for column in 1:num_variables
        first_index = Int(colptr[column])
        after = Int(colptr[column + 1])
        first_index <= after || throw(ArgumentError("colptr is not monotone"))
        previous_row = 0
        for position in first_index:(after - 1)
            row = Int(rowval[position])
            1 <= row <= num_rows || throw(ArgumentError(
                "row index $row in column $column is outside 1:$num_rows",
            ))
            row > previous_row || throw(ArgumentError(
                "row indices in column $column are not strictly increasing",
            ))
            previous_row = row
        end
    end
    blocks = _validate_cone_blocks(data.cone_blocks, num_rows)
    return (;
        objective_sense,
        objective_constant,
        objective,
        lower,
        upper,
        num_rows,
        num_variables,
        colptr,
        rowval,
        nzval,
        constants,
        blocks,
    )
end

function _variables_container(lower::Vector{Float64}, upper::Vector{Float64})
    masks = Vector{UInt16}(undef, length(lower))
    for variable in eachindex(lower)
        lo = lower[variable]
        hi = upper[variable]
        masks[variable] = if lo == -Inf && hi == Inf
            0x0000
        elseif lo == hi
            0x0001
        elseif hi == Inf
            0x0002
        elseif lo == -Inf
            0x0004
        else
            0x0008
        end
    end
    return MOI.Utilities.VariablesContainer{Float64}(masks, lower, upper)
end

function _sets_from_blocks(blocks)
    sets = _Cones{Float64}()
    for block in blocks
        set_type = getproperty(block, :set_type)
        category = MOI.Utilities.set_index(sets, set_type)
        category === nothing && throw(ArgumentError("unsupported cone type $set_type"))
        MOI.Utilities.add_set(
            sets,
            category,
            Int(getproperty(block, :dimension)),
        )
    end
    MOI.Utilities.final_touch(sets)
    return sets
end

function _objective_container(validated)
    objective = MOI.Utilities.ObjectiveContainer{Float64}()
    MOI.set(objective, MOI.ObjectiveSense(), validated.objective_sense)
    terms = MOI.ScalarAffineTerm{Float64}[
        MOI.ScalarAffineTerm(value, MOI.VariableIndex(column))
        for (column, value) in pairs(validated.objective) if !iszero(value)
    ]
    MOI.set(
        objective,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(terms, validated.objective_constant),
    )
    return objective
end

function _coefficient_matrix(validated)
    Ti = eltype(validated.colptr)
    coefficients = MOI.Utilities.MutableSparseMatrixCSC{
        Float64,
        Ti,
        MOI.Utilities.OneBasedIndexing,
    }()
    coefficients.m = validated.num_rows
    coefficients.n = validated.num_variables
    coefficients.colptr = validated.colptr
    coefficients.rowval = validated.rowval
    coefficients.nzval = validated.nzval
    coefficients.nz_added = zeros(Ti, validated.num_variables)
    return coefficients
end

"""
    conic_cache_from_data(data)

Validate a property-compatible conic data object and build the finalized PDCS
`OptimizerCache` without incremental constraint insertion.
"""
function conic_cache_from_data(data)
    validated = _validated_conic_data(data)
    variables = _variables_container(validated.lower, validated.upper)
    objective = _objective_container(validated)
    coefficients = _coefficient_matrix(validated)
    constants = _SetConstants{Float64}(validated.constants)
    sets = _sets_from_blocks(validated.blocks)
    constraints = MOI.Utilities.MatrixOfConstraints{Float64}(
        coefficients,
        constants,
        sets,
    )
    constraints.final_touch = true
    cache_model = MOI.Utilities.GenericModel{Float64}(
        objective,
        variables,
        constraints,
    )
    return MOI.Utilities.UniversalFallback(cache_model)
end

"""
    model_from_conic_data(data; optimizer=Optimizer())

Return a JuMP model whose finalized PDCS cache borrows compatible arrays from
`data`. The first `optimize!` dispatches directly to PDCS's two-argument method.
"""
function model_from_conic_data(data; optimizer=Optimizer())
    optimizer isa MOI.AbstractOptimizer || throw(ArgumentError(
        "optimizer must be an MOI.AbstractOptimizer instance",
    ))
    MOI.is_empty(optimizer) || throw(ArgumentError("optimizer must be empty"))
    started = time_ns()
    cache = conic_cache_from_data(data)
    Ti = eltype(cache.model.constraints.coefficients.colptr)
    empty_cache = MOI.Utilities.UniversalFallback(OptimizerCache{Ti}())
    backend = MOI.Utilities.CachingOptimizer(empty_cache, optimizer)
    model = JuMP.direct_model(backend)
    backend.model_cache = cache
    model.ext[:PDCS_conic_data] = data
    model.ext[:PDCS_optimizer_cache] = cache
    model.ext[:PDCS_cache_build_seconds] = Float64(time_ns() - started) / 1.0e9
    model.ext[:JumpRW_CBF_layout] = getproperty(data, :layout)
    return model
end
