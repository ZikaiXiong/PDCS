
mutable struct primalVector
    x::Vector{rpdhg_float}
    xbox::AbstractVector{rpdhg_float}
    x_slice::Vector{AbstractVector{rpdhg_float}}
    t_warm_start::Vector{rpdhg_float}
    x_slice_part::Vector{AbstractVector{rpdhg_float}}
    x_slice_proj!::Vector{Function}
    x_slice_proj_diagonal!::Vector{Function}
    x_slice_proj_slack!::Vector{Function}
    x_slice_func_symbol::Vector{Symbol}
    blkLen::Integer
    function primalVector(; x, box_index, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end)
        blkLen = length(soc_cone_indices_start) + length(rsoc_cone_indices_start) + length(exp_cone_indices_start) + length(dual_exp_cone_indices_start)
        if box_index > 0
            blkLen += 1
        end
        if blkLen > 0
            x_slice = Vector{AbstractVector{rpdhg_float}}(undef, blkLen)
            x_slice_part = Vector{AbstractVector{rpdhg_float}}(undef, blkLen)
            x_slice_proj! = Vector{Function}(undef, blkLen)
            x_slice_proj_diagonal! = Vector{Function}(undef, blkLen)
            x_slice_proj_slack! = Vector{Function}(undef, blkLen)
            x_slice_func_symbol = Vector{Symbol}(undef, blkLen)
            t_warm_start = Vector{rpdhg_float}(undef, blkLen)
            t_warm_start .= 1.0
        else
            x_slice = Vector{AbstractVector{rpdhg_float}}([])
            x_slice_proj! = Vector{Function}([])
            x_slice_part = Vector{AbstractVector{rpdhg_float}}([])
            x_slice_proj_diagonal! = Vector{Function}([])
            x_slice_proj_slack! = Vector{Function}([])
            x_slice_func_symbol = Vector{Symbol}([])
            t_warm_start = Vector{rpdhg_float}([])
        end
        baseIndex = 1
        if box_index > 0
            xbox = @view x[1:box_index]
            x_slice[baseIndex] = xbox
            x_slice_proj![baseIndex] = x -> println("proj_box! not implemented");
            x_slice_proj_diagonal![baseIndex] = x -> println("proj_box_diagonal! not implemented"); 
            x_slice_part[baseIndex] = @view x[1:box_index]
            x_slice_func_symbol[baseIndex] = :proj_box!
            x_slice_proj_slack![baseIndex] = x -> println("proj_box_slack! not implemented");
            baseIndex += 1
        else
            xbox = Vector{rpdhg_float}([])
        end
        if length(soc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(soc_cone_indices_start, soc_cone_indices_end)
                x_slice[baseIndex] = @view x[start_idx:end_idx]
                x_slice_proj![baseIndex] = x -> println("proj_soc_cone! not implemented");
                x_slice_proj_diagonal![baseIndex] = x -> println("proj_soc_cone_diagonal! not implemented");
                x_slice_part[baseIndex] = @view x[start_idx+1:end_idx]
                x_slice_func_symbol[baseIndex] = :proj_soc_cone!
                x_slice_proj_slack![baseIndex] = x -> println("proj_soc_cone_slack! not implemented");
                baseIndex += 1
            end
        end
        if length(rsoc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(rsoc_cone_indices_start, rsoc_cone_indices_end)
                x_slice[baseIndex] = @view x[start_idx:end_idx]
                x_slice_proj![baseIndex] = x -> println("proj_rsoc_cone! not implemented");
                x_slice_proj_diagonal![baseIndex] = x -> println("proj_rsoc_cone_diagonal! not implemented");
                x_slice_part[baseIndex] = @view x[start_idx+2:end_idx]
                x_slice_func_symbol[baseIndex] = :proj_rsoc_cone!
                x_slice_proj_slack![baseIndex] = x -> println("proj_rsoc_cone_slack! not implemented");
                baseIndex += 1
            end
        end
        if length(exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(exp_cone_indices_start, exp_cone_indices_end)
                x_slice[baseIndex] = @view x[start_idx:end_idx]
                x_slice_proj![baseIndex] = x -> println("proj_exp_cone! not implemented");
                x_slice_proj_diagonal![baseIndex] = x -> println("proj_exp_cone_diagonal! not implemented");
                x_slice_part[baseIndex] = @view x[start_idx:end_idx]
                x_slice_func_symbol[baseIndex] = :proj_exp_cone!
                x_slice_proj_slack![baseIndex] = x -> println("proj_exp_cone_slack! not implemented");
                baseIndex += 1
            end
        end
        if length(dual_exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(dual_exp_cone_indices_start, dual_exp_cone_indices_end)
                x_slice[baseIndex] = @view x[start_idx:end_idx]
                x_slice_proj![baseIndex] = x -> println("proj_dual_exp_cone! not implemented");
                x_slice_proj_diagonal![baseIndex] = x -> println("proj_dual_exp_cone_diagonal! not implemented");
                x_slice_part[baseIndex] = @view x[start_idx:end_idx]
                x_slice_func_symbol[baseIndex] = :proj_dual_exp_cone!
                x_slice_proj_slack![baseIndex] = x -> println("proj_dual_exp_cone_slack! not implemented");
                baseIndex += 1
            end
        end
        new(x, xbox, x_slice, t_warm_start, x_slice_part, x_slice_proj!, x_slice_proj_diagonal!, x_slice_proj_slack!, x_slice_func_symbol, blkLen)
    end
end

mutable struct dualVector
    y::Vector{rpdhg_float}
    y_slice::Vector{AbstractVector{rpdhg_float}}
    y_slice_part::Vector{AbstractVector{rpdhg_float}}
    t_warm_start::Vector{rpdhg_float}
    y_slice_proj!::Vector{Function}
    y_slice_proj_diagonal!::Vector{Function}
    y_slice_con_proj!::Vector{Function}
    y_slice_func_symbol::Vector{Symbol}
    blkLen::Integer
    mGzero::Integer
    mGnonnegative::Integer
    function dualVector(; y, m, mGzero, mGnonnegative, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end)
        blkLen = length(soc_cone_indices_start) + length(rsoc_cone_indices_start) + length(exp_cone_indices_start) + length(dual_exp_cone_indices_start)
        if mGzero > 0
            blkLen += 1
        end
        if mGnonnegative > 0
            blkLen += 1
        end
        if blkLen > 0
            y_slice = Vector{AbstractVector{rpdhg_float}}(undef, blkLen)
            y_slice_part = Vector{AbstractVector{rpdhg_float}}(undef, blkLen)
            y_slice_proj! = Vector{Function}(undef, blkLen)
            y_slice_proj_diagonal! = Vector{Function}(undef, blkLen)
            y_slice_con_proj! = Vector{Function}(undef, blkLen)
            y_slice_func_symbol = Vector{Symbol}(undef, blkLen)
            t_warm_start = Vector{rpdhg_float}(undef, blkLen)
            t_warm_start .= 1.0
        else
            y_slice = Vector{AbstractVector{rpdhg_float}}([])
            y_slice_proj! = Vector{Function}([])
            y_slice_part = Vector{AbstractVector{rpdhg_float}}([])
            y_slice_proj_diagonal! = Vector{Function}([])
            y_slice_con_proj! = Vector{Function}([])
            y_slice_func_symbol = Vector{Symbol}([])
            t_warm_start = Vector{rpdhg_float}([])
        end
        baseIndex = 1
        if mGzero > 0
            y_free = @view y[1:mGzero];
            y_slice[baseIndex] = @view y[1:mGzero];
            y_slice_proj![baseIndex] = x -> println("dual_zero_proj! not implemented");
            y_slice_proj_diagonal![baseIndex] = x -> println("dual_zero_proj_diagonal! not implemented");
            y_slice_con_proj![baseIndex] = x -> println("dual_free_proj_con! not implemented");
            y_slice_func_symbol[baseIndex] = :dual_free_proj!
            y_slice_part[baseIndex] = @view y[1:mGzero]
            baseIndex += 1
        end
        if mGnonnegative > 0
            y_pos = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice[baseIndex] = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice_proj![baseIndex] = x -> println("dual_positive_proj! not implemented");
            y_slice_proj_diagonal![baseIndex] = x -> println("dual_positive_proj_diagonal! not implemented");
            y_slice_part[baseIndex] = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice_func_symbol[baseIndex] = :dual_positive_proj!
            y_slice_con_proj![baseIndex] = x -> println("dual_positive_proj_con! not implemented");
            baseIndex += 1
        end
        if length(soc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(soc_cone_indices_start, soc_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_soc_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_soc_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_soc_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx+1:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_soc_proj!
                baseIndex += 1
            end
        end
        if length(rsoc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(rsoc_cone_indices_start, rsoc_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_rsoc_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_rsoc_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_rsoc_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx+2:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_rsoc_proj!
                baseIndex += 1
            end
        end
        if length(exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(exp_cone_indices_start, exp_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_exp_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_exp_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_exp_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_exp_proj!
                baseIndex += 1
            end
        end
        if length(dual_exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(dual_exp_cone_indices_start, dual_exp_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_dual_exp_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_dual_exp_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_dual_exp_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_DUALEXP_proj!
                baseIndex += 1
            end
        end

        new(y, y_slice, y_slice_part, t_warm_start, y_slice_proj!, y_slice_proj_diagonal!, y_slice_con_proj!, y_slice_func_symbol, blkLen, mGzero, mGnonnegative)
    end
end

mutable struct timesInfo
    interior::Integer
    boundary::Integer
    zero::Integer
    status::Symbol
    function timesInfo()
        new(0, 0, 0, :unknown)
    end
end

""" Diagonal_preconditioner for the PDHGCLP solver
    DQl: the diagonal of the matrix Ql
    DAl: the diagonal of the matrix Al
    Dr: the diagonal of the matrix r
    DQl_product: the product of the diagonal of the matrix Ql and the vector
    DAl_product: the product of the diagonal of the matrix Al and the vector
    Dr_product: the product of the diagonal of the matrix r and the vector
    Dr_product_inv_normalized: the product of the inverse of the diagonal of the matrix r and the vector
    DQl_product_normalized: the product of the inverse of the diagonal of the matrix Ql and the vector

    new_c = c ./ Dr_product
    new_Q = diag(DQl^{-1}) * Q * diag(Dr^{-1})
    new_A = diag(DAl^{-1}) * A * diag(Dr^{-1})
    new_h = diag(DQl^{-1}) * h
    new_b = diag(DAl^{-1}) * b
    new_bl = Dr[1:data.nb] * l
    new_bu = Dr[1:data.nb] * u
"""

mutable struct Diagonal_preconditioner
    Dl::dualVector
    Dr::primalVector

    Dl_temp::dualVector
    Dr_temp::primalVector

    m::Integer
    n::Integer

    Dl_product::dualVector
    Dr_product::primalVector

    Dr_product_inv_normalized::primalVector # primal variable projection Dr[1]./Dr -- primal variable projection
    Dr_product_normalized::primalVector # primal variable projection Dr./Dr[1]  -- slack variable projection
    Dl_product_inv_normalized::dualVector # dual variable projection Dl[1]./Dr -- dual variable projection

    Dr_product_inv_normalized_squared::primalVector
    Dr_product_normalized_squared::primalVector
    Dl_product_inv_normalized_squared::dualVector

    primalConstScale::Vector{Bool}
    dualConstScale::Vector{Bool}

    primalProjInfo::Vector{timesInfo}
    dualProjInfo::Vector{timesInfo}
    slackProjInfo::Vector{timesInfo}

    function Diagonal_preconditioner(; Dl, Dr, m, n, Dl_product = deepcopy(Dl), Dr_product = deepcopy(Dr), Dl_temp = deepcopy(Dl), Dr_temp = deepcopy(Dr),
        Dr_product_inv_normalized = deepcopy(Dr), Dr_product_normalized = deepcopy(Dr), Dl_product_inv_normalized = deepcopy(Dl),
        Dr_product_inv_normalized_squared = deepcopy(Dr), Dr_product_normalized_squared = deepcopy(Dr), Dl_product_inv_normalized_squared = deepcopy(Dl))

        Dl_product.y .= 1.0
        Dr_product.x .= 1.0
        Dr_product_inv_normalized.x .= 1.0
        Dr_product_normalized.x .= 1.0
        Dl_product_inv_normalized.y .= 1.0
        Dr_product_inv_normalized_squared.x .= 1.0
        Dr_product_normalized_squared.x .= 1.0
        Dl_product_inv_normalized_squared.y .= 1.0
        primalConstScale = Vector{Bool}(undef, Dr.blkLen)
        dualConstScale = Vector{Bool}(undef, Dl.blkLen)
        primalConstScale .= false
        dualConstScale .= false
        primalProjInfo = Vector{timesInfo}(undef, Dr.blkLen)
        dualProjInfo = Vector{timesInfo}(undef, Dl.blkLen)
        slackProjInfo = Vector{timesInfo}(undef, Dr.blkLen)
        for i in 1:Dr.blkLen
            primalProjInfo[i] = timesInfo()
            slackProjInfo[i] = timesInfo()
        end
        for i in 1:Dl.blkLen
            dualProjInfo[i] = timesInfo()
        end
        new(Dl, Dr, Dl_temp, Dr_temp, m, n, Dl_product, Dr_product,
        Dr_product_inv_normalized, Dr_product_normalized, Dl_product_inv_normalized, Dr_product_inv_normalized_squared,
        Dr_product_normalized_squared, Dl_product_inv_normalized_squared, primalConstScale, dualConstScale, primalProjInfo, dualProjInfo, slackProjInfo)
    end
end




mutable struct solVecPrimalRecovered
    primal_sol::primalVector
    primal_sol_lag::primalVector
    primal_sol_mean::primalVector
end




"""
solVecPrimal is a struct that stores the primal solution of the optimization problem.
    - primal_sol: the primal solution vector
    - primal_sol_lag: the previous primal solution vector
    - primal_sol_mean: the mean of the previous primal solution vector
    - box_index: the index of the box cone ([1:box_index])
    - bl: the lower bounds of the box cone
    - bu: the upper bounds of the box cone
    - soc_cone_indices_start: the start indices of the SOC cones
    - soc_cone_indices_end: the end indices of the SOC cones
    - rsoc_cone_indices_start: the start indices of the rotated SOC cones
    - rsoc_cone_indices_end: the end indices of the rotated SOC cones
    -- primal_sol[1:box_index] is the box cone
    -- primal_sol[soc_cone_indices_start[i]:soc_cone_indices_end[i]] is the i-th SOC cone
    -- primal_sol[rsoc_cone_indices_start[i]:rsoc_cone_indices_end[i]] is the i-th rotated SOC cone
"""
mutable struct solVecPrimal
    primal_sol::primalVector
    primal_sol_lag::primalVector
    primal_sol_mean::primalVector
    box_index::Integer
    bl::Vector{rpdhg_float}
    bu::Vector{rpdhg_float}
    soc_cone_indices_start::Vector{<:Integer}
    soc_cone_indices_end::Vector{<:Integer}
    rsoc_cone_indices_start::Vector{<:Integer}
    rsoc_cone_indices_end::Vector{<:Integer}
    exp_cone_indices_start::Vector{<:Integer}
    exp_cone_indices_end::Vector{<:Integer}
    dual_exp_cone_indices_start::Vector{<:Integer}
    dual_exp_cone_indices_end::Vector{<:Integer}
    proj!::Function
    proj_diagonal!::Function
    lambd_l::Vector{rpdhg_float}
    lambd_u::Vector{rpdhg_float}
    slack_proj!::Function
    recovered_primal::Union{solVecPrimalRecovered,Nothing}
    function solVecPrimal(; primal_sol, primal_sol_lag, primal_sol_mean, box_index,
        bl, bu, soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        proj!, proj_diagonal!,
        slack_proj!, recovered_primal)
        lambd_l, lambd_u = gen_lambd(bl, bu)
        new(primal_sol, primal_sol_lag, primal_sol_mean, box_index,
        bl, bu, soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        proj!, proj_diagonal!, lambd_l, lambd_u, slack_proj!, recovered_primal)
    end
end

mutable struct solVecDualRecovered
    dual_sol::dualVector
    dual_sol_mean::dualVector
    dual_sol_lag::dualVector
end

"""
solVecDual is a struct that stores the dual solution of the optimization problem.
    - dual_sol: the dual solution vector
    - dual_sol_mean: the mean of the previous dual solution vector
    - len: the length of the dual solution vector
    - m1: the number of rows of matrix Q1, >=0, positive orthant
    - soc_cone_indices_start: the start indices of the SOC cones
    - soc_cone_indices_end: the end indices of the SOC cones
    - rsoc_cone_indices_start: the start indices of the rotated SOC cones
    - rsoc_cone_indices_end: the end indices of the rotated SOC cones
    - slack: the primal solution vector, slack variables
"""
mutable struct solVecDual
    dual_sol::dualVector
    dual_sol_lag::dualVector
    dual_sol_mean::dualVector
    dual_sol_temp::dualVector
    mGzeroIndices::Vector{<:Integer}
    mGnonnegativeIndices::Vector{<:Integer}
    soc_cone_indices_start::Vector{<:Integer}
    soc_cone_indices_end::Vector{<:Integer}
    rsoc_cone_indices_start::Vector{<:Integer}
    rsoc_cone_indices_end::Vector{<:Integer}
    exp_cone_indices_start::Vector{<:Integer}
    exp_cone_indices_end::Vector{<:Integer}
    dual_exp_cone_indices_start::Vector{<:Integer}
    dual_exp_cone_indices_end::Vector{<:Integer}
    slack::solVecPrimal
    proj!::Function
    con_proj!::Function # constraint projection
    proj_diagonal!::Function
    recovered_dual::Union{solVecDualRecovered,Nothing}
    function solVecDual(; dual_sol, dual_sol_lag, dual_sol_mean, dual_sol_temp = deepcopy(dual_sol), mGzero, mGnonnegative,
        soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        slack, proj!, con_proj!, proj_diagonal!, recovered_dual)
        if mGzero > 0
            mGzeroIndices = Vector{Int}([1,mGzero])
        else
            mGzeroIndices = Vector{Int}([])
        end
        if mGnonnegative > 0
            mGnonnegativeIndices = Vector{Int}([mGzero + 1, mGzero + mGnonnegative])
        else
            mGnonnegativeIndices = Vector{Int}([])
        end
        new(dual_sol, dual_sol_lag, dual_sol_mean, dual_sol_temp, mGzeroIndices, mGnonnegativeIndices,
        soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        slack, proj!, con_proj!, proj_diagonal!, recovered_dual)
    end
end


mutable struct PDHGCLPConvergeInfo
    primal_objective::rpdhg_float
    dual_objective::rpdhg_float
    abs_gap::rpdhg_float
    rel_gap::rpdhg_float
    l_inf_rel_primal_res::rpdhg_float
    l_inf_rel_dual_res::rpdhg_float
    l_2_rel_primal_res::rpdhg_float
    l_2_rel_dual_res::rpdhg_float
    l_inf_abs_primal_res::rpdhg_float
    l_inf_abs_dual_res::rpdhg_float
    l_2_abs_primal_res::rpdhg_float
    l_2_abs_dual_res::rpdhg_float
    status::Symbol
    function PDHGCLPConvergeInfo(; primal_objective = 1e+30, dual_objective= 1e+30, abs_gap= 1e+30, rel_gap= 1e+30,
        l_inf_rel_primal_res= 1e+30, l_inf_rel_dual_res= 1e+30, l_2_rel_primal_res= 1e+30, l_2_rel_dual_res= 1e+30,
        l_inf_abs_primal_res= 1e+30, l_inf_abs_dual_res= 1e+30, l_2_abs_primal_res= 1e+30, l_2_abs_dual_res= 1e+30,
        status= :continue)
        new(primal_objective, dual_objective, abs_gap, rel_gap,
        l_inf_rel_primal_res, l_inf_rel_dual_res, l_2_rel_primal_res, l_2_rel_dual_res,
        l_inf_abs_primal_res, l_inf_abs_dual_res, l_2_abs_primal_res, l_2_abs_dual_res,
        status)
    end
end



"""
Information measuring how close a point is to establishing primal or dual
infeasibility (i.e. has no solution); see also TerminationCriteria.
"""
mutable struct PDHGCLPInfeaInfo
    max_primal_ray_infeasibility::rpdhg_float
    primal_ray_objective::rpdhg_float
    max_dual_ray_infeasibility::rpdhg_float
    dual_ray_objective::rpdhg_float
    trend_len::Integer
    primalObj_trend::CircularBuffer{rpdhg_float}
    dualObj_trend::CircularBuffer{rpdhg_float}
    status::Symbol
    function PDHGCLPInfeaInfo(; max_primal_ray_infeasibility = 1e+30,
         primal_ray_objective = 1e+30, max_dual_ray_infeasibility = 1e+30,
         dual_ray_objective = 1e+30, trend_len = 20, primalObj_trend = CircularBuffer{rpdhg_float}(trend_len),
         dualObj_trend = CircularBuffer{rpdhg_float}(trend_len), status = :continue)
        new(max_primal_ray_infeasibility, primal_ray_objective,
         max_dual_ray_infeasibility, dual_ray_objective,
         trend_len, primalObj_trend, dualObj_trend, status)
    end
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
"""

mutable struct PDHGCLPInfo
    # results
    iter::Integer
    iter_stepsize::Integer
    convergeInfo::Vector{PDHGCLPConvergeInfo} # multi sequence to check convergence
    infeaInfo::Vector{PDHGCLPInfeaInfo} # multi sequence to check infeasibility
    time::Float64
    start_time::Float64
    restart_used::Integer
    restart_trigger_mean::Integer
    restart_trigger_ergodic::Integer
    exit_status::Symbol
    pObj::rpdhg_float
    dObj::rpdhg_float
    exit_code::Int
    normalized_duality_gap::Vector{rpdhg_float}
    normalized_duality_gap_restart_threshold::rpdhg_float
    normalized_duality_gap_r::rpdhg_float
    kkt_error::Vector{rpdhg_float}
    kkt_error_restart_threshold::rpdhg_float
    restart_duality_gap_flag::Bool
    binarySearch_t0::rpdhg_float
    omega::rpdhg_float
    function PDHGCLPInfo(; iter, convergeInfo, infeaInfo, time, start_time, restart_used = 0, restart_trigger_mean = 0, restart_trigger_ergodic = 0, exit_status = :continue, pObj = 1e+30, dObj = 1e+30, exit_code = 7, normalized_duality_gap = Vector{rpdhg_float}(undef, 2), normalized_duality_gap_restart_threshold = 0, kkt_error = Vector{rpdhg_float}(undef, 2), kkt_error_restart_threshold = 0)
        normalized_duality_gap[1] = 1e+30
        normalized_duality_gap[2] = 1e+30
        kkt_error[1] = 1e+30
        kkt_error[2] = 1e+30    
        iter_stepsize = 0
        normalized_duality_gap_r = 1e+30
        restart_duality_gap_flag = true
        binarySearch_t0 = 1.0
        omega = 1.0
        new(iter, iter_stepsize, convergeInfo, infeaInfo, time, start_time, restart_used, restart_trigger_mean, restart_trigger_ergodic, exit_status, pObj, dObj, exit_code, normalized_duality_gap, normalized_duality_gap_restart_threshold, normalized_duality_gap_r, kkt_error, kkt_error_restart_threshold, restart_duality_gap_flag, binarySearch_t0, omega)
    end
end




mutable struct PDHGCLPParameters
    # parameters
    max_outer_iter::Integer
    max_inner_iter::Integer
    rel_tol::rpdhg_float
    abs_tol::rpdhg_float
    eps_primal_infeasible_low_acc::rpdhg_float
    eps_dual_infeasible_low_acc::rpdhg_float
    eps_primal_infeasible_high_acc::rpdhg_float
    eps_dual_infeasible_high_acc::rpdhg_float
    sigma::rpdhg_float
    tau::rpdhg_float
    theta::rpdhg_float
    restart_check_freq::Integer
    check_terminate_freq::Integer
    verbose::Integer
    print_freq::Integer
    time_limit::rpdhg_float
    beta_suff::rpdhg_float
    beta_necessary::rpdhg_float
    beta_suff_kkt::rpdhg_float
    beta_necessary_kkt::rpdhg_float
    beta_artificial::rpdhg_float
    function PDHGCLPParameters(;
         max_outer_iter, max_inner_iter, rel_tol, abs_tol,
         eps_primal_infeasible_low_acc, eps_dual_infeasible_low_acc,
         eps_primal_infeasible_high_acc, eps_dual_infeasible_high_acc,
         sigma, tau, theta,
         restart_check_freq, check_terminate_freq, verbose, print_freq, time_limit)
         beta_suff = 0.2
         beta_necessary = 0.9
         beta_suff_kkt = 0.2
         beta_necessary_kkt = 0.9
         beta_artificial = 0.707
        new(max_outer_iter, max_inner_iter, rel_tol, abs_tol,
        eps_primal_infeasible_low_acc, eps_dual_infeasible_low_acc,
        eps_primal_infeasible_high_acc, eps_dual_infeasible_high_acc,
        sigma, tau, theta,
        restart_check_freq, check_terminate_freq, verbose, print_freq, time_limit,
        beta_suff, beta_necessary, beta_suff_kkt, beta_necessary_kkt, beta_artificial)
    end
end

mutable struct Solution
    x::solVecPrimal
    y::solVecDual
    x_best::primalVector
    y_best::dualVector
    primal_res_best::rpdhg_float
    dual_res_best::rpdhg_float
    params::PDHGCLPParameters
    info::PDHGCLPInfo
    function Solution(; x, y, params, info)
        x_best = deepcopy(x.primal_sol)
        y_best = deepcopy(y.dual_sol)
        primal_res_best = 1e+30
        dual_res_best = 1e+30
        new(x, y, x_best, y_best, primal_res_best, dual_res_best, params, info)
    end
end

"""
coeffSplit and coeffUnion are two methods to store data coefficient
    G = [Q, A]
    d = [h, b]

    coeffSplit stores the data coefficient in a split form 
    if the the sparsity of the data coefficient is different
    coeffUnion stores the data coefficient in a union form
    when the sparsity of the data coefficient is in the same level
"""
mutable struct coeffSplitQA{
        hType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
        bType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
    }
    Q::AbstractMatrix{rpdhg_float}
    A::AbstractMatrix{rpdhg_float}
    h::hType
    b::bType
    m::Integer # the row number of [Q1t, Q2t, At]^T
    n::Integer # the column number of Q1 or Q2 or A
    function coeffSplitQA(;
         Q::AbstractMatrix{rpdhg_float},
         A::AbstractMatrix{rpdhg_float}, h::hType, b::bType,
         m::Integer, n::Integer) where {
            hType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
            bType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
        }
        error("Not used, discard it, old function");
        new{hType, bType}(Q, A, h, b, m, n)
    end
end

mutable struct coeffUnion{
    hType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
}    
    G::AbstractMatrix{rpdhg_float}
    h::hType
    m::Integer
    n::Integer
    function coeffUnion(; G::AbstractMatrix{rpdhg_float},
         h::hType, m::Integer, n::Integer) where{
            hType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
         }
        new{hType}(G, h, m, n)
    end
end

mutable struct rpdhgRawData{
    cType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
    coeffType<:Union{coeffSplitQA, coeffUnion},
    coeffTransType<:Union{coeffSplitQA, coeffUnion}
}
    m::Integer
    n::Integer
    nb::Integer
    c::cType
    coeff::coeffType
    coeffTrans::coeffTransType
    bl::Vector{rpdhg_float}
    bu::Vector{rpdhg_float}
    bl_finite::Vector{rpdhg_float} # avoid 0.0 * -Inf
    bu_finite::Vector{rpdhg_float} # avoid 0.0 * Inf
    hNrm1::rpdhg_float
    cNrm1::rpdhg_float
    hNrmInf::rpdhg_float
    cNrmInf::rpdhg_float
    function rpdhgRawData(; m::Integer, n::Integer, nb::Integer,
        c::cType, coeff::coeffType, coeffTrans::coeffTransType,
        bl::Vector{rpdhg_float}, bu::Vector{rpdhg_float},
        hNrm1::rpdhg_float, cNrm1::rpdhg_float, 
        hNrmInf::rpdhg_float, cNrmInf::rpdhg_float) where{
        cType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
        coeffType<:Union{coeffSplitQA, coeffUnion},
        coeffTransType<:Union{coeffSplitQA, coeffUnion},
        }
        bl_finite = deepcopy(bl)
        bu_finite = deepcopy(bu)
        if length(bl) > 0
            bl_finite = replace(bl_finite, -Inf=>0.0)
            bu_finite = replace(bu_finite, Inf=>0.0)
        end
        new{cType, coeffType, coeffTransType}(m, n, nb, c, coeff, coeffTrans, bl, bu, bl_finite, bu_finite, hNrm1, cNrm1, hNrmInf, cNrmInf)
    end
end


"""
probData is a struct that stores the data for the optimization problem.
    - m: the number of rows of the matrix A
    - n: the number of columns of the matrix A
    - c: the vector c
    - coeff: the data coefficient
    - coeffTrans: the transpose of the data coefficient, no vector
    - GlambdaMax: the maximum eigenvalue of the matrix G
    - hNrm1: the 1-norm of the vector d
    - cNrm1: the 1-norm of the vector c
"""

mutable struct probData{
    cType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
    coeffType<:Union{coeffUnion},
    coeffTransType<:Union{coeffUnion}
}
    m::Integer
    n::Integer
    nb::Integer
    c::cType
    coeff::coeffType
    coeffTrans::coeffTransType
    GlambdaMax::rpdhg_float
    GlambdaMax_flag::Integer
    bl::Vector{rpdhg_float}
    bu::Vector{rpdhg_float}
    bl_finite::Vector{rpdhg_float} # avoid 0.0 * -Inf
    bu_finite::Vector{rpdhg_float} # avoid 0.0 * Inf
    hNrm1::rpdhg_float
    hNrm2::rpdhg_float
    cNrm1::rpdhg_float
    cNrm2::rpdhg_float
    hNrmInf::rpdhg_float
    cNrmInf::rpdhg_float
    diagonal_scale::Diagonal_preconditioner
    raw_data::Union{rpdhgRawData,Nothing}
    function probData(; m::Integer, n::Integer, nb::Integer,
         c::cType, coeff::coeffType, coeffTrans::coeffTransType,
         GlambdaMax::rpdhg_float, GlambdaMax_flag::Integer, bl::Vector{rpdhg_float}, bu::Vector{rpdhg_float},
         hNrm1::rpdhg_float, hNrm2::rpdhg_float, cNrm1::rpdhg_float, cNrm2::rpdhg_float,
         hNrmInf::rpdhg_float, cNrmInf::rpdhg_float, diagonal_scale::Diagonal_preconditioner, raw_data::Union{rpdhgRawData,Nothing}) where{
            cType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
            coeffType<:Union{coeffUnion},
            coeffTransType<:Union{coeffUnion},
         }
        bl_finite = deepcopy(bl)
        bu_finite = deepcopy(bu)
        if length(bl) > 0
            bl_finite = replace(bl_finite, -Inf=>0.0)
            bu_finite = replace(bu_finite, Inf=>0.0)
        end
        new{cType, coeffType, coeffTransType}(m, n, nb, c, coeff, coeffTrans, GlambdaMax, GlambdaMax_flag, bl, bu, bl_finite, bu_finite, hNrm1, hNrm2, cNrm1, cNrm2, hNrmInf, cNrmInf, diagonal_scale, raw_data)
    end
end

mutable struct rpdhgSolver
    data::probData
    sol::Solution
    primalMV!::Function
    adjointMV!::Function
    AtAMV!::Function
    addCoeffd!::Function
    dotCoeffd::Function
    function rpdhgSolver(; data, sol, primalMV!, adjointMV!, AtAMV!, addCoeffd!, dotCoeffd)
        new(data, sol, primalMV!, adjointMV!, AtAMV!, addCoeffd!, dotCoeffd)
    end
end

