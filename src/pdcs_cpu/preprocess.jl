

"""
Returns the l2 norm of each row or column of a matrix. The method rescales
the sum-of-squares computation by the largest absolute value if nonzero in order
to avoid overflow.

# Arguments
- `matrix::SparseMatrixCSC{Float64, Int64}`: a sparse matrix.
- `dimension::Int64`: the dimension we want to compute the norm over. Must be
  1 or 2.
  - 1 means we compute the norm of each row.
  - 2 means we compute the norm of each column.

# Returns
An array with the l2 norm of a matrix over the given dimension.
"""
function l2_norm(matrix::SparseMatrixCSC{Float64,Int64}, dimension::Int64)
    scale_factor = vec(maximum(abs, matrix, dims = dimension))
    scale_factor[iszero.(scale_factor)] .= 1.0
    if dimension == 1
        scaled_matrix = matrix * Diagonal(1 ./ scale_factor)
        return scale_factor .*
            vec(sqrt.(sum(t -> t^2, scaled_matrix, dims = dimension)))
    end

    if dimension == 2
        scaled_matrix = Diagonal(1 ./ scale_factor) * matrix
        return scale_factor .*
            vec(sqrt.(sum(t -> t^2, scaled_matrix, dims = dimension)))
    end
end


"""
Rescales a quadratic programming problem by dividing each row and column of the
constraint matrix by the sqrt its respective L2 norm, adjusting the other
problem data accordingly.

# Arguments
- `problem::QuadraticProgrammingProblem`: The input quadratic programming
  problem. This is modified to store the transformed problem.

# Returns

A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function l2_norm_rescaling!(;
    problem::probData,
    Dr_product::Vector{Float64},
    DGl_product::Vector{Float64}
    )
    error("Not implemented, since we need to calculate the column l2 norm of the matrix G")
    num_constraints, num_variables = size(problem.constraint_matrix)

    norm_of_rows_G = vec(l2_norm(problem.coeff.G, 2))

    # not the norm_of_columns_Q and norm_of_columns_A separately
    norm_of_columns_G = vec(l2_norm(problem.coeff.G, 1))

    norm_of_rows_G[iszero.(norm_of_rows_G)] .= 1.0
    norm_of_columns_G[iszero.(norm_of_columns_G)] .= 1.0


    column_rescale_factor_G = sqrt.(norm_of_columns_G)
    row_rescale_factor_G = sqrt.(norm_of_rows_G)
    scale_data!(
        data = problem,
        Dr = column_rescale_factor_G,
        DGl = row_rescale_factor_G,
        Dr_product = Dr_product,
        DGl_product = DGl_product,
        sol = sol, dual_sol = dual_sol)
end


function scale_preconditioner!(;
    Dr_product::Vector{Float64},
    Dl_product::Vector{Float64},
    Dr_product_inv_normalized::Vector{Float64},
    Dr_product_normalized::Vector{Float64},
    Dl_product_inv_normalized::Vector{Float64},
    Dr_product_inv_normalized_squared::Vector{Float64},
    Dr_product_normalized_squared::Vector{Float64},
    Dl_product_inv_normalized_squared::Vector{Float64},
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
            Dr_product_inv_normalized[start_idx:end_idx] .= Dr_product[start_idx] ./ Dr_product[start_idx:end_idx]
            Dr_product_normalized[start_idx:end_idx] .= 1 ./ Dr_product_inv_normalized[start_idx:end_idx]
            # check Dr_product_inv_normalized[start_idx:end_idx] == 1.0
            if all(x -> isapprox(x, 1.0, atol = 1e-15), Dr_product_inv_normalized[start_idx:end_idx])
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end # if length(primal_sol.soc_cone_indices_start) > 0
    if length(primal_sol.rsoc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.rsoc_cone_indices_start, primal_sol.rsoc_cone_indices_end)
            Dr_product_inv_normalized[start_idx:end_idx] .= sqrt(Dr_product[start_idx] * Dr_product[start_idx + 1]) ./ Dr_product[start_idx:end_idx]
            Dr_product_normalized[start_idx:end_idx] .= 1 ./ Dr_product_inv_normalized[start_idx:end_idx]
            # check Dr_product_inv_normalized[start_idx:end_idx] == sqrt(Dr_product[start_idx] * Dr_product[start_idx + 1])
            if all(x -> isapprox(x, 1.0, atol = 1e-15), Dr_product_inv_normalized[start_idx + 2:end_idx])
                primalConstScale[blkCountPrimal] = true
            end
            blkCountPrimal += 1
        end
    end # if length(primal_sol.rsoc_cone_indices_start) > 0

    if length(primal_sol.exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.exp_cone_indices_start, primal_sol.exp_cone_indices_end)
            Dr_product_inv_normalized[start_idx:end_idx] = 1.0 ./ Dr_product[start_idx:end_idx]
            blkCountPrimal += 1
        end
    end

    if length(primal_sol.dual_exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(primal_sol.dual_exp_cone_indices_start, primal_sol.dual_exp_cone_indices_end)
            Dr_product_inv_normalized[start_idx:end_idx] = 1.0 ./ Dr_product[start_idx:end_idx]
            blkCountPrimal += 1
        end
    end

    if length(dual_sol.soc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.soc_cone_indices_start, dual_sol.soc_cone_indices_end)
            Dl_product_inv_normalized[start_idx:end_idx] .= Dl_product[start_idx] ./ Dl_product[start_idx:end_idx]
            # check Dl_product_inv_normalized[start_idx:end_idx] == 1.0
            if all(x -> isapprox(x, 1.0, atol = 1e-15), Dl_product_inv_normalized[start_idx:end_idx])
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.soc_cone_indices_start) > 0

    if length(dual_sol.rsoc_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.rsoc_cone_indices_start, dual_sol.rsoc_cone_indices_end)
            Dl_product_inv_normalized[start_idx:end_idx] .= sqrt(Dl_product[start_idx] * Dl_product[start_idx + 1]) ./ Dl_product[start_idx:end_idx]
            # check Dl_product_inv_normalized[start_idx:end_idx] == sqrt(Dl_product[start_idx] * Dl_product[start_idx + 1])
            if all(x -> isapprox(x, 1.0, atol = 1e-15), Dl_product_inv_normalized[start_idx + 2:end_idx])
                dualConstScale[blkCountDual] = true
            end
            blkCountDual += 1
        end
    end # if length(dual_sol.rsoc_cone_indices_start) > 0

    if length(dual_sol.exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.exp_cone_indices_start, dual_sol.exp_cone_indices_end)
            Dl_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dl_product[start_idx:end_idx]
            blkCountDual += 1
        end
    end # if length(dual_sol.exp_cone_indices_start) > 0

    if length(dual_sol.dual_exp_cone_indices_start) > 0
        for (start_idx, end_idx) in zip(dual_sol.dual_exp_cone_indices_start, dual_sol.dual_exp_cone_indices_end)
            Dl_product_inv_normalized[start_idx:end_idx] .= 1.0 ./ Dl_product[start_idx:end_idx]
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
    Dr::Vector{Float64},
    DGl::Vector{Float64},
    Dr_product::Vector{Float64},
    DGl_product::AbstractVector{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual,
)

    @assert all(t -> t > 0, Dr)
    if length(DGl) != 0
        @assert all(t -> t > 0, DGl)
    end
    data.c ./= Dr
    sol.primal_sol.x .*= Dr
    sol.primal_sol_lag.x .*= Dr
    sol.primal_sol_mean.x .*= Dr
    if data.nb > 0
        # do not need to change sol.bl additionally, since it is the same as data.bl
        data.bl .*= Dr[1:data.nb]
        data.bu .*= Dr[1:data.nb]
    end
    Dr_product .*= Dr
    if data.m > 0
        dual_sol.dual_sol.y .*= DGl
        dual_sol.dual_sol_lag.y .*= DGl
        dual_sol.dual_sol_mean.y .*= DGl
    end

    if isa(data.coeff, coeffUnion)
        if length(DGl) != 0
            data.coeff.G .= (Diagonal(1 ./ DGl) * data.coeff.G) * Diagonal(1 ./ Dr)
            data.coeff.h ./= DGl
            DGl_product .*= DGl
        end
    else
        error("Unknown type of data.coeff")
    end
    return
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
  Dr_product::Vector{Float64},
  Dl_product::Vector{Float64},
  sol::solVecPrimal,
  dual_sol::solVecDual,
)

    if l_inf_ruiz_iterations > 0
        ruiz_rescaling!(
            problem = data,
            num_iterations = l_inf_ruiz_iterations,
            p = Inf,
            Dr_product = Dr_product,
            DGl_product = Dl_product,
            sol = sol, dual_sol = dual_sol
        )
    end

    ## if l2_norm_rescal
    ##     l2_norm_rescaling!(data)
    ## end

    if !isnothing(pock_chambolle_alpha)
        pock_chambolle_rescaling!(
            problem = data,
            alpha = pock_chambolle_alpha,
            Dr_product = Dr_product,
            DGl_product = Dl_product,
            sol = sol, dual_sol = dual_sol
        )
    end
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
    Dr_product::Vector{Float64},
    DGl_product::AbstractVector{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    @assert 0 <= alpha <= 2

    num_constraints = problem.m

    variable_rescaling = vec(
        sqrt.(
        mapreduce(
            t -> abs(t)^(2 - alpha),
            +,
            problem.coeff.G,
            dims = 1,
            init = 0.0,
        ),
        ),
    )
    constraint_rescaling_G = vec(
        sqrt.(
            mapreduce(t -> abs(t)^alpha, +, problem.coeff.G, dims = 2, init = 0.0),
        ),
    )
    constraint_rescaling_G[iszero.(constraint_rescaling_G)] .= 1.0
    variable_rescaling[iszero.(variable_rescaling)] .= 1.0
    ## debugging
    # variable_rescaling .= 1.0
    # constraint_rescaling_G .= 1.0
    scale_data!(
        data = problem,
        Dr = variable_rescaling,
        DGl = constraint_rescaling_G,
        Dr_product = Dr_product, 
        DGl_product = DGl_product,
        sol = sol, dual_sol = dual_sol)
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
    Dr_product::Vector{Float64},
    DGl_product::AbstractVector{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    num_constraints = problem.m

    for i in 1:num_iterations
        if p == Inf
            variable_rescaling = vec(
                sqrt.(
                    maximum(abs, problem.coeff.G, dims = 1),
                ),
            )
        else
            @assert p == 2
            variable_rescaling = vec(
                sqrt.(
                    l2_norm(problem.coeff.G, 1)
                ),
            )
        end
        variable_rescaling[iszero.(variable_rescaling)] .= 1.0

        if num_constraints == 0
            error("No constraints to rescale.")
        else
            if p == Inf
                constraint_rescaling_G =
                vec(sqrt.(maximum(abs, problem.coeff.G, dims = 2)))
            else
                @assert p == 2
                target_row_norm = sqrt(problem.n / (problem.m))
                norm_of_rows_G = vec(l2_norm(problem.coeff.G, 2))
                norm_of_G = vec(sqrt.(norm_of_rows_G))
                constraint_rescaling_G = vec(sqrt.(norm_of_G / target_row_norm))
            end
            constraint_rescaling_G[iszero.(constraint_rescaling_G)] .= 1.0
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
            sol = sol, dual_sol = dual_sol)
    end # for i in 1:num_iterations
end

function scale_solution!(;
    Dr_product::Vector{Float64},
    Dl_product::Vector{Float64},
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


function block_diagonal_preconditioner!(;
    problem::probData,
    diag_precond::Diagonal_preconditioner,
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    # debugging
    diag_precond.Dr_product.x .= 1.0
    diag_precond.Dl_product.y .= 1.0

    Dl = diag_precond.Dl
    Dr = diag_precond.Dr
    Dr_product = diag_precond.Dr_product
    Dl_product = diag_precond.Dl_product
    primalBlkBase = 1
    dualBlkBase = 1
    if length(Dr_product.xbox) > 0
        primalBlkBase += 1
    end

    if Dl_product.mGzero > 0
        dualBlkBase += 1
    end

    if Dl_product.mGnonnegative > 0
        dualBlkBase += 1
    end

    # let box constraint, equality constraint, and nonnegative constraint scale not change 
    Dr.x .= Dr_product.x
    Dl.y .= Dl_product.y

    for i in primalBlkBase:Dr.blkLen
        Dr.x_slice[i] .= max(max.(Dr_product.x[i]), 1e-2)
        # Dr.x_slice[i] .= 1.0
    end

    for i in dualBlkBase:Dl.blkLen
        Dl.y_slice[i] .= max(max.(Dl_product.y[i]), 1e-2)
        # Dl.y_slice[i] .= 1.0
    end

    Dr_product.x .= 1.0
    Dl_product.y .= 1.0

    scale_data!(
        data = problem,
        Dr = Dr.x,
        DGl = Dl.y,
        Dr_product = diag_precond.Dr_product.x, 
        DGl_product = diag_precond.Dl_product.y,
        sol = sol, dual_sol = dual_sol
    )

    scale_preconditioner!(
        Dr_product = diag_precond.Dr_product.x,
        Dl_product = diag_precond.Dl_product.y,
        Dr_product_inv_normalized = diag_precond.Dr_product_inv_normalized.x,
        Dr_product_normalized = diag_precond.Dr_product_normalized.x,
        Dl_product_inv_normalized = diag_precond.Dl_product_inv_normalized.y,
        Dr_product_inv_normalized_squared = diag_precond.Dr_product_inv_normalized_squared.x,
        Dr_product_normalized_squared = diag_precond.Dr_product_normalized_squared.x,
        Dl_product_inv_normalized_squared = diag_precond.Dl_product_inv_normalized_squared.y,
        primal_sol = sol,
        dual_sol = dual_sol,
        primalConstScale = diag_precond.primalConstScale,
        dualConstScale = diag_precond.dualConstScale
    )

    scale_solution!(
        Dr_product = diag_precond.Dr_product.x,
        Dl_product = diag_precond.Dl_product.y,
        sol = sol, dual_sol = dual_sol
    )


end