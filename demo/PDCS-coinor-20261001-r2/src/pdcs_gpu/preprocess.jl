# function l2_norm_row_kernel_julia(values, rowptr, nrows, result)
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if idx <= nrows
#         sum_squares = 0.0f0
#         for i in rowptr[idx]:(rowptr[idx + 1] - 1)
#             sum_squares += values[i]^2
#         end
#         result[idx] = sqrt(sum_squares)
#     end
#     return
# end

# function l2_norm_row_julia(d_G)
#     rowptr = d_G.rowPtr
#     values = d_G.nzVal
#     nrows = size(d_G, 1)
#     result = CUDA.zeros(Float64, nrows)

#     @cuda threads=128 blocks=ceil(Int, nrows / 128) l2_norm_row_kernel(values, rowptr, nrows, result)

#     return result
# end

# function l2_norm_col_kernel_julia(values, colind, ncols, result)
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if idx <= ncols
#         sum_squares = 0.0f0
#         for i = 1:length(values)
#             if colind[i] == idx
#                 sum_squares += values[i]^2
#             end
#         end
#         result[idx] = sqrt(sum_squares)
#     end
#     return
# end

# function l2_norm_col_julia(d_G)
#     colind = d_G.colVal
#     values = d_G.nzVal
#     ncols = size(d_G, 2)
#     result = CUDA.zeros(Float64, ncols)

#     @cuda threads=128 blocks=ceil(Int, ncols / 128) l2_norm_col_kernel_julia(values, colind, ncols, result)

#     return result
# end

# function max_abs_row_kernel_julia(values, rowptr, nrows, result)
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if idx <= nrows
#         max_val = 0.0
#         for i in rowptr[idx]:(rowptr[idx + 1] - 1)
#             max_val = max(max_val, abs(values[i]))
#         end
#         result[idx] = max_val
#     end
#     return 
# end

# function max_abs_row_julia(d_G)
#     # Use appropriate methods to extract CuSparseMatrixCSR data
#     rowptr = d_G.rowPtr    # Access row pointers directly
#     values = d_G.nzVal     # Access non-zero values directly
#     nrows = size(d_G, 1)   # Number of rows
#     result = CUDA.zeros(Float64, nrows)

#     @cuda threads=128 blocks=ceil(Int, nrows / 128) max_abs_row_kernel_julia(values, rowptr, nrows, result)

#     return result
# end

# function max_abs_col_kernel_julia(values, colind, ncols, result)
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if idx <= ncols
#         max_val = 0.0f0
#         # Iterate through all non-zero elements
#         for i = 1:length(values)
#             # If this element belongs to current column
#             if colind[i] == idx
#                 max_val = max(max_val, abs(values[i]))
#             end
#         end
#         result[idx] = max_val
#     end
#     return
# end

# function max_abs_col_julia(d_G)
#     # Extract CSR matrix data
#     colind = d_G.colVal   # Column indices
#     values = d_G.nzVal    # Non-zero values
#     ncols = size(d_G, 2)  # Number of columns
#     result = CUDA.zeros(Float64, ncols)

#     @cuda threads=128 blocks=ceil(Int, ncols / 128) max_abs_col_kernel_julia(values, colind, ncols, result)

#     return result
# end

# """
# Returns the l2 norm of each row or column of a matrix. The method rescales
# the sum-of-squares computation by the largest absolute value if nonzero in order
# to avoid overflow.

# # Arguments
# - `matrix::SparseMatrixCSC{Float64, Int64}`: a sparse matrix.
# - `dimension::Int64`: the dimension we want to compute the norm over. Must be
#   1 or 2.
#   - 1 means we compute the norm of each row.
#   - 2 means we compute the norm of each column.

# # Returns
# An array with the l2 norm of a matrix over the given dimension.
# """
# function l2_norm_julia(matrix::SparseMatrixCSC{Float64,Int64}, dimension::Int64)
#     if dimension == 1
#         scaled_matrix = matrix * Diagonal(1 ./ scale_factor)
#         return l2_norm_col(scaled_matrix)
#     end

#     if dimension == 2
#         scaled_matrix = Diagonal(1 ./ scale_factor) * matrix
#         return l2_norm_row(scaled_matrix)
#     end
# end


# """
# Rescales a quadratic programming problem by dividing each row and column of the
# constraint matrix by the sqrt its respective L2 norm, adjusting the other
# problem data accordingly.

# # Arguments
# - `problem::QuadraticProgrammingProblem`: The input quadratic programming
#   problem. This is modified to store the transformed problem.

# # Returns

# A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
# the original problem is recovered by
# `unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
# """
# function l2_norm_rescaling_julia!(;
#     problem::probData,
#     Dr_product::CuArray{Float64},
#     DGl_product::CuArray{Float64}
#     )
#     error("Not implemented, since we need to calculate the column l2 norm of the matrix G")
#     num_constraints, num_variables = size(problem.constraint_matrix)

#     norm_of_rows_G = l2_norm(problem.coeff.d_G, 2)

#     # not the norm_of_columns_Q and norm_of_columns_A separately
#     norm_of_columns_G = l2_norm(problem.coeff.d_G, 1)

#     norm_of_rows_G[iszero.(norm_of_rows_G)] .= 1.0
#     norm_of_columns_G[iszero.(norm_of_columns_G)] .= 1.0


#     column_rescale_factor_G = sqrt.(norm_of_columns_G)
#     row_rescale_factor_G = sqrt.(norm_of_rows_G)
#     scale_data!(
#         data = problem,
#         Dr = column_rescale_factor_G,
#         DGl = row_rescale_factor_G,
#         Dr_product = Dr_product,
#         DGl_product = DGl_product,
#         sol = sol, dual_sol = dual_sol)
# end


function scale_preconditioner!(;
    Dr_product::CuArray{Float64},
    Dl_product::CuArray{Float64},
    Dr_product_inv_normalized::CuArray{Float64},
    Dr_product_normalized::CuArray{Float64},
    Dl_product_inv_normalized::CuArray{Float64},
    Dr_product_inv_normalized_squared::CuArray{Float64},
    Dr_product_normalized_squared::CuArray{Float64},
    Dl_product_inv_normalized_squared::CuArray{Float64},
    primal_sol::solVecPrimal,
    dual_sol::solVecDual,
    primalConstScale::Vector{Bool},
    dualConstScale::Vector{Bool}
)
    primalConstScale .= false
    dualConstScale .= false
    blkCountPrimal = 1
    blkCountDual = 1
    if primal_sol.box_index > 0
        blkCountPrimal += 1
    end
    if length(dual_sol.mGzeroIndices) > 0
        blkCountDual += 1
    end
    if length(dual_sol.mGnonnegativeIndices) > 0
        blkCountDual += 1
    end
    if length(primal_sol.soc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.soc_cone_indices_start, primal_sol.soc_cone_indices_end)
            CUDA.@allowscalar Dr_product_inv_normalized[start_idx:end_idx] .= Dr_product[start_idx] ./ Dr_product[start_idx:end_idx]
            CUDA.@allowscalar Dr_product_normalized[start_idx:end_idx] .= 1 ./ Dr_product_inv_normalized[start_idx:end_idx]
            # check Dr_product_inv_normalized[start_idx:end_idx] == 1.0
            if all(x -> isapprox(x, 1.0, atol = 1e-20), Dr_product_inv_normalized[start_idx:end_idx])
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end # if length(primal_sol.soc_cone_indices_start) > 0
    if length(primal_sol.rsoc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.rsoc_cone_indices_start, primal_sol.rsoc_cone_indices_end)
            CUDA.@allowscalar Dr_product_inv_normalized[start_idx:end_idx] .= sqrt(Dr_product[start_idx] * Dr_product[start_idx + 1]) ./ Dr_product[start_idx:end_idx]
            CUDA.@allowscalar Dr_product_normalized[start_idx:end_idx] .= 1 ./ Dr_product_inv_normalized[start_idx:end_idx]
            # check Dr_product_inv_normalized[start_idx:end_idx] == sqrt(Dr_product[start_idx] * Dr_product[start_idx + 1])
            if all(x -> isapprox(x, 1.0, atol = 1e-20), Dr_product_inv_normalized[start_idx + 2:end_idx])
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end # if length(primal_sol.rsoc_cone_indices_start) > 0

    if length(primal_sol.exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.exp_cone_indices_start, primal_sol.exp_cone_indices_end)
            CUDA.@allowscalar Dr_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dr_product[start_idx:end_idx]
            reference_scale = CUDA.@allowscalar Dr_product[start_idx]
            if all(
                value -> isapprox(
                    value,
                    reference_scale;
                    atol = 1e-20,
                    rtol = 0.0,
                ),
                view(Dr_product, start_idx:end_idx),
            )
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end

    if length(primal_sol.dual_exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.dual_exp_cone_indices_start, primal_sol.dual_exp_cone_indices_end)
            CUDA.@allowscalar Dr_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dr_product[start_idx:end_idx]
            reference_scale = CUDA.@allowscalar Dr_product[start_idx]
            if all(
                value -> isapprox(
                    value,
                    reference_scale;
                    atol = 1e-20,
                    rtol = 0.0,
                ),
                view(Dr_product, start_idx:end_idx),
            )
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end

    if length(dual_sol.soc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.soc_cone_indices_start, dual_sol.soc_cone_indices_end)
            CUDA.@allowscalar Dl_product_inv_normalized[start_idx:end_idx] .= Dl_product[start_idx] ./ Dl_product[start_idx:end_idx]
            # check Dl_product_inv_normalized[start_idx:end_idx] == 1.0
            if all(x -> isapprox(x, 1.0, atol = 1e-20), Dl_product_inv_normalized[start_idx:end_idx])
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.soc_cone_indices_start) > 0

    if length(dual_sol.rsoc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.rsoc_cone_indices_start, dual_sol.rsoc_cone_indices_end)
            CUDA.@allowscalar Dl_product_inv_normalized[start_idx:end_idx] .= sqrt(Dl_product[start_idx] * Dl_product[start_idx + 1]) ./ Dl_product[start_idx:end_idx]
            # check Dl_product_inv_normalized[start_idx:end_idx] == sqrt(Dl_product[start_idx] * Dl_product[start_idx + 1])
            if all(x -> isapprox(x, 1.0, atol = 1e-20), Dl_product_inv_normalized[start_idx + 2:end_idx])
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.rsoc_cone_indices_start) > 0

    if length(dual_sol.exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.exp_cone_indices_start, dual_sol.exp_cone_indices_end)
            CUDA.@allowscalar Dl_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dl_product[start_idx:end_idx]
            reference_scale = CUDA.@allowscalar Dl_product[start_idx]
            if all(
                value -> isapprox(
                    value,
                    reference_scale;
                    atol = 1e-20,
                    rtol = 0.0,
                ),
                view(Dl_product, start_idx:end_idx),
            )
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.exp_cone_indices_start) > 0

    if length(dual_sol.dual_exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.dual_exp_cone_indices_start, dual_sol.dual_exp_cone_indices_end)
            CUDA.@allowscalar Dl_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dl_product[start_idx:end_idx]
            reference_scale = CUDA.@allowscalar Dl_product[start_idx]
            if all(
                value -> isapprox(
                    value,
                    reference_scale;
                    atol = 1e-20,
                    rtol = 0.0,
                ),
                view(Dl_product, start_idx:end_idx),
            )
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.dual_exp_cone_indices_start) > 0

    Dr_product_inv_normalized_squared .= Dr_product_inv_normalized .^ 2
    Dr_product_normalized_squared .= Dr_product_normalized .^ 2
    Dl_product_inv_normalized_squared .= Dl_product_inv_normalized .^ 2
end


"""
Rescales `problem` in place, then `problem` is modified such that:

    c = Dr^-1 c 
    Q = D_{Ql}^-1 Q Dr^-1
    bl = Dr bl
    bu = Dr bu
    A = D_{Al}^-1 A Dr^-1
    h = D_{Ql}^-1 h
    b = D_{Al}^-1 b

    x = Dr x
    yQ = D_{Ql}^-1 yQ
    yA = D_{Al}^-1 yA

    Dr_product = Dr_product * Dr
    DQl_product = DQl_product * D_{Ql}
    DAl_product = DAl_product * D_{Al}

The scaling vectors must be positive.
"""
function scale_data!(;
    data::probData,
    Dr::CuArray{Float64},
    DGl::CuArray{Float64},
    Dr_product::CuArray{Float64},
    DGl_product::CuArray{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual,
    row_idx::CuArray
)
    data.d_c ./= Dr
    sol.primal_sol.x .*= Dr
    sol.primal_sol_lag.x .*= Dr
    sol.primal_sol_mean.x .*= Dr
    if data.nb > 0
        # do not need to change sol.bl additionally, since it is the same as data.bl
        Dr_part = @view Dr[1:data.nb]
        data.d_bl .*= Dr_part
        data.d_bu .*= Dr_part
    end
    Dr_product .*= Dr
    if data.m > 0
        dual_sol.dual_sol.y .*= DGl
        dual_sol.dual_sol_lag.y .*= DGl
        dual_sol.dual_sol_mean.y .*= DGl
    end

    if isa(data.coeff, coeffUnion)
        if length(DGl) != 0
            # rescale_csr(data.coeff.d_G, DGl, Dr, data.m, data.n)
            rescale_coo(data.coeff.d_G, DGl, Dr, data.m, data.n, row_idx)
            # data.coeff.d_G .= (Diagonal(1 ./ DGl) * data.coeff.d_G) * Diagonal(1 ./ Dr)
            data.coeff.d_h ./= DGl
            DGl_product .*= DGl
        end
    else
        error("Unknown type of data.coeff")
    end
    return
end


"""Collect the inclusive ranges of all non-polyhedral cone blocks."""
function structured_cone_ranges(sol::Union{solVecPrimal,solVecDual})
    starts = Int64[]
    ends = Int64[]
    for cone_name in (:soc_cone, :rsoc_cone, :exp_cone, :dual_exp_cone)
        start_field = Symbol(string(cone_name), "_indices_start")
        end_field = Symbol(string(cone_name), "_indices_end")
        append!(starts, Int64.(getproperty(sol, start_field)))
        append!(ends, Int64.(getproperty(sol, end_field)))
    end
    return starts, ends
end


function scalarize_cone_rescaling_kernel!(
    scaling,
    cone_starts,
    cone_ends,
)
    cone_index = Int64(CUDA.blockIdx().x)
    lane = Int64(CUDA.threadIdx().x)
    lanes = Int64(CUDA.blockDim().x)
    start_index = cone_starts[cone_index]
    end_index = cone_ends[cone_index]

    local_max = 0.0
    element_index = start_index + lane - 1
    while element_index <= end_index
        local_max = max(local_max, scaling[element_index])
        element_index += lanes
    end

    shared_max = CUDA.@cuStaticSharedMem(Float64, ThreadPerBlock)
    shared_max[lane] = local_max
    CUDA.sync_threads()

    offset = lanes >> 1
    while offset > 0
        if lane <= offset
            shared_max[lane] = max(shared_max[lane], shared_max[lane + offset])
        end
        CUDA.sync_threads()
        offset >>= 1
    end

    block_max = shared_max[1]
    element_index = start_index + lane - 1
    while element_index <= end_index
        scaling[element_index] = block_max
        element_index += lanes
    end
    return
end


function scalarize_masked_cone_rescaling_kernel!(
    scaling,
    cone_starts,
    cone_ends,
    scalar_cone_mask,
)
    cone_index = Int64(CUDA.blockIdx().x)
    @inbounds scalar_cone_mask[cone_index] || return
    lane = Int64(CUDA.threadIdx().x)
    lanes = Int64(CUDA.blockDim().x)
    start_index = @inbounds cone_starts[cone_index]
    end_index = @inbounds cone_ends[cone_index]

    local_max = 0.0
    element_index = start_index + lane - 1
    while element_index <= end_index
        local_max = max(local_max, @inbounds scaling[element_index])
        element_index += lanes
    end

    shared_max = CUDA.@cuStaticSharedMem(Float64, ThreadPerBlock)
    shared_max[lane] = local_max
    CUDA.sync_threads()

    offset = lanes >> 1
    while offset > 0
        if lane <= offset
            shared_max[lane] = max(shared_max[lane], shared_max[lane + offset])
        end
        CUDA.sync_threads()
        offset >>= 1
    end

    block_max = shared_max[1]
    element_index = start_index + lane - 1
    while element_index <= end_index
        @inbounds scaling[element_index] = block_max
        element_index += lanes
    end
    return
end


function mark_non_simple_matrix_coordinates_kernel!(
    row_is_simple,
    column_is_simple,
    values,
    row_indices,
    column_indices,
    num_entries,
)
    entry_index =
        (Int64(CUDA.blockIdx().x) - 1) * Int64(CUDA.blockDim().x) +
        Int64(CUDA.threadIdx().x)
    if entry_index <= num_entries
        value = @inbounds values[entry_index]
        if !(value == 0.0 || value == 1.0 || value == -1.0)
            row = @inbounds row_indices[entry_index]
            column = @inbounds column_indices[entry_index]
            # Concurrent stores are idempotent: every writer stores `false`.
            @inbounds row_is_simple[row] = false
            @inbounds column_is_simple[column] = false
        end
    end
    return
end


function simple_cone_mask_kernel!(
    scalar_cone_mask,
    coordinate_is_simple,
    cone_starts,
    cone_ends,
)
    cone_index = Int64(CUDA.blockIdx().x)
    lane = Int64(CUDA.threadIdx().x)
    lanes = Int64(CUDA.blockDim().x)
    start_index = @inbounds cone_starts[cone_index]
    end_index = @inbounds cone_ends[cone_index]

    local_simple = Int32(1)
    element_index = start_index + lane - 1
    while element_index <= end_index
        if !(@inbounds coordinate_is_simple[element_index])
            local_simple = Int32(0)
        end
        element_index += lanes
    end

    shared_simple = CUDA.@cuStaticSharedMem(Int32, ThreadPerBlock)
    shared_simple[lane] = local_simple
    CUDA.sync_threads()

    offset = lanes >> 1
    while offset > 0
        if lane <= offset
            shared_simple[lane] = min(
                shared_simple[lane],
                shared_simple[lane + offset],
            )
        end
        CUDA.sync_threads()
        offset >>= 1
    end
    if lane == 1
        @inbounds scalar_cone_mask[cone_index] = shared_simple[1] == Int32(1)
    end
    return
end


"""Return per-cone masks for original matrix blocks containing only 0 and ±1."""
function adaptive_scalar_cone_masks(
    matrix::CUDA.CUSPARSE.CuSparseMatrixCSR,
    row_indices::CuArray,
    primal_cone_starts::CuArray{Int64},
    primal_cone_ends::CuArray{Int64},
    dual_cone_starts::CuArray{Int64},
    dual_cone_ends::CuArray{Int64},
)
    row_is_simple = CUDA.ones(Bool, size(matrix, 1))
    column_is_simple = CUDA.ones(Bool, size(matrix, 2))
    num_entries = length(matrix.nzVal)
    if num_entries > 0
        CUDA.@sync begin
            CUDA.@cuda threads = ThreadPerBlock blocks = cld(num_entries, ThreadPerBlock) mark_non_simple_matrix_coordinates_kernel!(
                row_is_simple,
                column_is_simple,
                matrix.nzVal,
                row_indices,
                matrix.colVal,
                Int64(num_entries),
            )
        end
    end

    primal_mask = CUDA.ones(Bool, length(primal_cone_starts))
    dual_mask = CUDA.ones(Bool, length(dual_cone_starts))
    if !isempty(primal_mask)
        CUDA.@sync begin
            CUDA.@cuda threads = ThreadPerBlock blocks = length(primal_mask) simple_cone_mask_kernel!(
                primal_mask,
                column_is_simple,
                primal_cone_starts,
                primal_cone_ends,
            )
        end
    end
    if !isempty(dual_mask)
        CUDA.@sync begin
            CUDA.@cuda threads = ThreadPerBlock blocks = length(dual_mask) simple_cone_mask_kernel!(
                dual_mask,
                row_is_simple,
                dual_cone_starts,
                dual_cone_ends,
            )
        end
    end
    CUDA.unsafe_free!(row_is_simple)
    CUDA.unsafe_free!(column_is_simple)
    return primal_mask, dual_mask
end


"""
    scalarize_cone_rescaling!(scaling, cone_starts, cone_ends)

Replace every candidate rescaling factor in each structured cone block by the
maximum factor in that block. One CUDA block handles one cone, so this remains
usable for both many three-dimensional exponential cones and high-dimensional
SOC blocks without host round trips.
"""
function scalarize_cone_rescaling!(
    scaling::CuArray{Float64},
    cone_starts::CuArray{Int64},
    cone_ends::CuArray{Int64},
)
    cone_count = length(cone_starts)
    cone_count == length(cone_ends) ||
        throw(DimensionMismatch("cone start/end arrays must have equal length"))
    cone_count == 0 && return scaling
    CUDA.@sync begin
        CUDA.@cuda threads = ThreadPerBlock blocks = cone_count scalarize_cone_rescaling_kernel!(
            scaling,
            cone_starts,
            cone_ends,
        )
    end
    return scaling
end


"""Scalarize only the cone blocks selected by `scalar_cone_mask`."""
function scalarize_cone_rescaling!(
    scaling::CuArray{Float64},
    cone_starts::CuArray{Int64},
    cone_ends::CuArray{Int64},
    scalar_cone_mask::CuArray{Bool},
)
    cone_count = length(cone_starts)
    cone_count == length(cone_ends) == length(scalar_cone_mask) ||
        throw(DimensionMismatch("cone ranges and mask must have equal length"))
    cone_count == 0 && return scaling
    CUDA.@sync begin
        CUDA.@cuda threads = ThreadPerBlock blocks = cone_count scalarize_masked_cone_rescaling_kernel!(
            scaling,
            cone_starts,
            cone_ends,
            scalar_cone_mask,
        )
    end
    return scaling
end


"""Preprocesses the original problem, and returns a ScaledQpProblem struct.
Applies L_inf Ruiz rescaling for `l_inf_ruiz_iterations` iterations. If
`l2_norm_rescaling` is true, applies L2 norm rescaling. `problem` is not
modified.
"""
function rescale_problem!(;
  l_inf_ruiz_iterations::Int,
#   l2_norm_rescal::Bool,
  pock_chambolle_alpha::Union{Float64,Nothing},
  data::probData,
  Dr_product::CuArray{Float64},
  Dl_product::CuArray{Float64},
  sol::solVecPrimal,
  dual_sol::solVecDual,
  variable_rescaling::CuArray{Float64},
  constraint_rescaling_G::CuArray{Float64},
  scalar_cone_rescaling::Bool = DEFAULT_SCALAR_CONE_RESCALING,
  use_adaptive_diagonal_scalar_rescaling::Bool =
      DEFAULT_USE_ADAPTIVE_DIAGONAL_SCALAR_RESCALING
)
    row_idx = similar(data.coeff.d_G.colVal, nnz(data.coeff.d_G))
    get_row_index(data.coeff.d_G, row_idx)
    primal_cone_starts = nothing
    primal_cone_ends = nothing
    dual_cone_starts = nothing
    dual_cone_ends = nothing
    primal_scalar_cone_mask = nothing
    dual_scalar_cone_mask = nothing
    if scalar_cone_rescaling || use_adaptive_diagonal_scalar_rescaling
        primal_cone_starts_cpu, primal_cone_ends_cpu = structured_cone_ranges(sol)
        dual_cone_starts_cpu, dual_cone_ends_cpu = structured_cone_ranges(dual_sol)
        primal_cone_starts = CuArray(primal_cone_starts_cpu)
        primal_cone_ends = CuArray(primal_cone_ends_cpu)
        dual_cone_starts = CuArray(dual_cone_starts_cpu)
        dual_cone_ends = CuArray(dual_cone_ends_cpu)
    end
    if use_adaptive_diagonal_scalar_rescaling && !scalar_cone_rescaling
        primal_scalar_cone_mask, dual_scalar_cone_mask =
            adaptive_scalar_cone_masks(
                data.coeff.d_G,
                row_idx,
                primal_cone_starts,
                primal_cone_ends,
                dual_cone_starts,
                dual_cone_ends,
            )
    end
    variable_rescaling .= 1.0
    constraint_rescaling_G .= 1.0
    if l_inf_ruiz_iterations > 0
        ruiz_rescaling!(
            problem = data,
            num_iterations = l_inf_ruiz_iterations,
            p = Inf,
            Dr_product = Dr_product,
            DGl_product = Dl_product,
            sol = sol, dual_sol = dual_sol, row_idx = row_idx,
            variable_rescaling = variable_rescaling,
            constraint_rescaling_G = constraint_rescaling_G,
            scalar_cone_rescaling = scalar_cone_rescaling,
            primal_scalar_cone_mask = primal_scalar_cone_mask,
            dual_scalar_cone_mask = dual_scalar_cone_mask,
            primal_cone_starts = primal_cone_starts,
            primal_cone_ends = primal_cone_ends,
            dual_cone_starts = dual_cone_starts,
            dual_cone_ends = dual_cone_ends
        )
    end

    # @info("complete ruiz_rescaling!")

    if !isnothing(pock_chambolle_alpha)
        pock_chambolle_rescaling!(
            problem = data,
            alpha = pock_chambolle_alpha,
            Dr_product = Dr_product,
            DGl_product = Dl_product,
            sol = sol, dual_sol = dual_sol, row_idx = row_idx,
            variable_rescaling = variable_rescaling,
            constraint_rescaling_G = constraint_rescaling_G,
            scalar_cone_rescaling = scalar_cone_rescaling,
            primal_scalar_cone_mask = primal_scalar_cone_mask,
            dual_scalar_cone_mask = dual_scalar_cone_mask,
            primal_cone_starts = primal_cone_starts,
            primal_cone_ends = primal_cone_ends,
            dual_cone_starts = dual_cone_starts,
            dual_cone_ends = dual_cone_ends
        )
    end
    # @info("complete pock_chambolle_rescaling!")
    CUDA.unsafe_free!(row_idx)
    primal_cone_starts === nothing || CUDA.unsafe_free!(primal_cone_starts)
    primal_cone_ends === nothing || CUDA.unsafe_free!(primal_cone_ends)
    dual_cone_starts === nothing || CUDA.unsafe_free!(dual_cone_starts)
    dual_cone_ends === nothing || CUDA.unsafe_free!(dual_cone_ends)
    primal_scalar_cone_mask === nothing || CUDA.unsafe_free!(primal_scalar_cone_mask)
    dual_scalar_cone_mask === nothing || CUDA.unsafe_free!(dual_scalar_cone_mask)
end



"""
Applies the rescaling proposed by Pock and Cambolle (2011),
"Diagonal preconditioning for first order primal-dual algorithms
in convex optimization"
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.381.6056&rep=rep1&type=pdf

Although presented as a form of diagonal preconditioning, it can be
equivalently implemented by rescaling the problem data.

Each column of the constraint matrix is divided by
sqrt(sum_{elements e in the column} |e|^(2 - alpha))
and each row of the constraint matrix is divided by
sqrt(sum_{elements e in the row} |e|^alpha)

Lemma 2 in Pock and Chambolle demonstrates that this rescaling causes the
operator norm of the rescaled constraint matrix to be less than or equal to
one, which is a desireable property for PDHG.

# Arguments
- `problem::QuadraticProgrammingProblem`: the quadratic programming problem.
  This is modified to store the transformed problem.
- `alpha::Float64`: the exponent parameter. Must be in the interval [0, 2].

# Returns

A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function pock_chambolle_rescaling!(;
    problem::probData,
    alpha::Float64,
    Dr_product::CuArray{Float64},
    DGl_product::CuArray{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual,
    row_idx::CuArray,
    variable_rescaling::CuArray{Float64},
    constraint_rescaling_G::CuArray{Float64},
    scalar_cone_rescaling::Bool = false,
    primal_scalar_cone_mask::Union{Nothing,CuArray{Bool}} = nothing,
    dual_scalar_cone_mask::Union{Nothing,CuArray{Bool}} = nothing,
    primal_cone_starts::Union{Nothing,CuArray{Int64}} = nothing,
    primal_cone_ends::Union{Nothing,CuArray{Int64}} = nothing,
    dual_cone_starts::Union{Nothing,CuArray{Int64}} = nothing,
    dual_cone_ends::Union{Nothing,CuArray{Int64}} = nothing
)
    @assert 0 <= alpha <= 2
    variable_rescaling .= 1.0
    constraint_rescaling_G .= 1.0
    num_constraints = problem.m

    # variable_rescaling_new = vec(
    #     sqrt.(
    #     mapreduce(
    #         t -> abs(t)^(2 - alpha),
    #         +,
    #         problem.coeff.d_G,
    #         dims = 1,
    #         init = 0.0,
    #     ),
    #     ),
    # )
    # alpha_norm_col(problem.coeff.d_G, 2 - alpha, variable_rescaling)
    # variable_rescaling .= sqrt.(variable_rescaling)
    # println("variable_rescaling[1:10]: ", variable_rescaling[1:10])
    # println("variable_rescaling_new[1:10]: ", variable_rescaling_new[1:10])
    # error("debug")
    alpha_norm_col_elementwise(problem.coeff.d_G, 2 - alpha, variable_rescaling)
    variable_rescaling .= sqrt.(variable_rescaling)
    alpha_norm_row(problem.coeff.d_G, alpha, constraint_rescaling_G)
    constraint_rescaling_G .= sqrt.(constraint_rescaling_G)
    # constraint_res = vec(
    #     sqrt.(
    #     mapreduce(t -> abs(t)^alpha, +, problem.coeff.d_G, dims = 2, init = 0.0),
    #     ),
    # )
    # println("constraint_res[1:10]: ", constraint_res[1:10])
    # println("constraint_rescaling_G[1:10]: ", constraint_rescaling_G[1:10])
    constraint_rescaling_G[iszero.(constraint_rescaling_G)] .= 1.0
    variable_rescaling[iszero.(variable_rescaling)] .= 1.0
    if scalar_cone_rescaling
        any(isnothing, (
            primal_cone_starts,
            primal_cone_ends,
            dual_cone_starts,
            dual_cone_ends,
        )) && error("scalar cone rescaling requires cone range arrays")
        scalarize_cone_rescaling!(
            variable_rescaling,
            primal_cone_starts,
            primal_cone_ends,
        )
        scalarize_cone_rescaling!(
            constraint_rescaling_G,
            dual_cone_starts,
            dual_cone_ends,
        )
    elseif primal_scalar_cone_mask !== nothing
        scalarize_cone_rescaling!(
            variable_rescaling,
            primal_cone_starts,
            primal_cone_ends,
            primal_scalar_cone_mask,
        )
        scalarize_cone_rescaling!(
            constraint_rescaling_G,
            dual_cone_starts,
            dual_cone_ends,
            dual_scalar_cone_mask,
        )
    end
    ## debugging
    # variable_rescaling .= 1.0
    # constraint_rescaling_G .= 1.0
    scale_data!(
        data = problem,
        Dr = variable_rescaling,
        DGl = constraint_rescaling_G,
        Dr_product = Dr_product, 
        DGl_product = DGl_product,
        sol = sol, dual_sol = dual_sol, row_idx = row_idx
    )
end


"""
Uses a modified Ruiz rescaling algorithm to rescale the matrix M=[Q,A';A,0]
where Q is objective_matrix and A is constraint_matrix, and returns the
cumulative scaling vectors. More details of Ruiz rescaling algorithm can be
found at: http://www.numerical.rl.ac.uk/reports/drRAL2001034.pdf.

In the p=Inf case, both matrices approach having all row and column LInf norms
of M equal to 1 as the number of iterations goes to infinity. This convergence
is fast (linear).

In the p=2 case, the goal is all row L2 norms of [Q,A'] equal to 1 and all row
L2 norms of A equal to sqrt(num_variables/(num_constraints+num_variables))
for QP, and all row L2 norms of A equal to
sqrt(num_variables/num_constraints) for LP. Having a different
goal for the row and col norms is required since the sum of squares of the
entries of the A matrix is the same when the sum is grouped by rows or grouped
by columns. In particular, for the LP case, all L2 norms of A must be
sqrt(num_variables/num_constraints) when all row L2 norm of [Q,A'] equal to 1.

The Ruiz rescaling paper (link above) only analyzes convergence in the p < Inf
case when the matrix is square, and it does not preserve the symmetricity of
the matrix, and that is why we need to modify it for p=2 case.

TODO: figure out when this converges.

# Arguments
- `problem::QuadraticProgrammingProblem`: the quadratic programming problem.
  This is modified to store the transformed problem.
- `num_iterations::Int64` the number of iterations to run Ruiz rescaling
  algorithm. Must be positive.
- `p::Float64`: which norm to use. Must be 2 or Inf.

# Returns

A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function ruiz_rescaling!(;
    problem::probData,
    num_iterations::Int64,
    p::Float64 = Inf,
    Dr_product::CuArray{Float64},
    DGl_product::CuArray{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual,
    row_idx::CuArray,
    variable_rescaling::CuArray{Float64},
    constraint_rescaling_G::CuArray{Float64},
    scalar_cone_rescaling::Bool = false,
    primal_scalar_cone_mask::Union{Nothing,CuArray{Bool}} = nothing,
    dual_scalar_cone_mask::Union{Nothing,CuArray{Bool}} = nothing,
    primal_cone_starts::Union{Nothing,CuArray{Int64}} = nothing,
    primal_cone_ends::Union{Nothing,CuArray{Int64}} = nothing,
    dual_cone_starts::Union{Nothing,CuArray{Int64}} = nothing,
    dual_cone_ends::Union{Nothing,CuArray{Int64}} = nothing
)
    num_constraints = problem.m
    for i in 1:num_iterations
        variable_rescaling .= 1.0
        constraint_rescaling_G .= 1.0
        if p == Inf
            # variable_rescaling = 
            #     sqrt.(
            #         max_abs_col(problem.coeff.d_G)
            #     )
            # variable_rescaling_julia = max_abs_col_julia(problem.coeff.d_G)
            # variable_rescaling = sqrt.(max_abs_col(problem.coeff.d_G))
            max_abs_col_elementwise(problem.coeff.d_G, variable_rescaling)
            # max_abs_col(problem.coeff.d_G, variable_rescaling)
            variable_rescaling .= sqrt.(variable_rescaling)
            # if (norm(variable_rescaling_julia - variable_rescaling, Inf) > 1e-2)
            #     println(findmax(abs.(variable_rescaling_julia - variable_rescaling)))
            # end
            # @assert variable_rescaling_julia = variable_rescaling
        # else
        #     @assert p == 2
        #     variable_rescaling = 
        #         sqrt.(
        #             l2_norm(problem.coeff.d_G, 1)
        #         )
        end
        variable_rescaling[iszero.(variable_rescaling)] .= 1.0

        if num_constraints == 0
            error("No constraints to rescale.")
        else
            if p == Inf
                # constraint_rescaling_G =
                #     sqrt.(
                #         max_abs_row(problem.coeff.d_G)
                #     )
                # constraint_rescaling_G = sqrt.(max_abs_row(problem.coeff.d_G))
                # max_abs_row_elementwise(problem.coeff.d_G, row_idx, constraint_rescaling_G)
                max_abs_row(problem.coeff.d_G, constraint_rescaling_G)
                constraint_rescaling_G .= sqrt.(constraint_rescaling_G)
            # else
            #     @assert p == 2
            #     target_row_norm = sqrt(problem.n / (problem.m))
            #     norm_of_rows_G = l2_norm_row(problem.coeff.d_G)
            #     norm_of_G = sqrt.(norm_of_rows_G)
            #     constraint_rescaling_G = norm_of_G / target_row_norm
            end
            constraint_rescaling_G[iszero.(constraint_rescaling_G)] .= 1.0
        end
        if scalar_cone_rescaling
            any(isnothing, (
                primal_cone_starts,
                primal_cone_ends,
                dual_cone_starts,
                dual_cone_ends,
            )) && error("scalar cone rescaling requires cone range arrays")
            scalarize_cone_rescaling!(
                variable_rescaling,
                primal_cone_starts,
                primal_cone_ends,
            )
            scalarize_cone_rescaling!(
                constraint_rescaling_G,
                dual_cone_starts,
                dual_cone_ends,
            )
        elseif primal_scalar_cone_mask !== nothing
            scalarize_cone_rescaling!(
                variable_rescaling,
                primal_cone_starts,
                primal_cone_ends,
                primal_scalar_cone_mask,
            )
            scalarize_cone_rescaling!(
                constraint_rescaling_G,
                dual_cone_starts,
                dual_cone_ends,
                dual_scalar_cone_mask,
            )
        end
        scale_data!(
            data = problem,
            Dr = variable_rescaling,
            DGl = constraint_rescaling_G,
            Dr_product = Dr_product, 
            DGl_product = DGl_product,
            sol = sol, dual_sol = dual_sol,
            row_idx = row_idx
        )
    end # for i in 1:num_iterations
end

function scale_solution!(;
    Dr_product::CuArray{Float64},
    Dl_product::CuArray{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    sol.primal_sol.x .*= Dr_product
    sol.primal_sol_lag.x .*= Dr_product
    sol.primal_sol_mean.x .*= Dr_product

    dual_sol.dual_sol.y .*= Dl_product
    dual_sol.dual_sol_lag.y .*= Dl_product
    dual_sol.dual_sol_mean.y .*= Dl_product
end # recover_solution
