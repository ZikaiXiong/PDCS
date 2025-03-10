"""
rpdhg_alg_clp_gen.jl
"""

# function recover_soc_cone_rotated_exp_cone_indices(; zero_indices::Integer, boxs_indices::Integer, q::Vector{<:Integer}, rq::Vector{<:Integer}, exp_q::Integer, dual_exp_q::Integer)
#     soc_cone_indices_start = []
#     soc_cone_indices_end = []
#     rsoc_cone_indices_start = []
#     rsoc_cone_indices_end = []
#     exp_cone_indices_start = []
#     exp_cone_indices_end = []
#     dual_exp_cone_indices_start = []
#     dual_exp_cone_indices_end = []
#     start_idx = zero_indices + boxs_indices + 1
#     if length(q) > 0
#         for i in eachindex(q)
#             push!(soc_cone_indices_start, start_idx)
#             start_idx += q[i]
#             if q[i] < 2
#                 throw(ArgumentError("The size of the SOC must be greater than 1"))
#             end
#             push!(soc_cone_indices_end, start_idx - 1)
#         end
#     end
#     if length(rq) > 0
#         for i in eachindex(rq)
#             push!(rsoc_cone_indices_start, start_idx)
#             start_idx += rq[i]
#             if rq[i] < 3
#                 throw(ArgumentError("The size of the rotated SOC must be greater than 2"))
#             end
#             push!(rsoc_cone_indices_end, start_idx - 1)
#         end
#     end
#     if exp_q > 0
#         for i in 1:exp_q
#             push!(exp_cone_indices_start, start_idx)
#             start_idx += 3
#             push!(exp_cone_indices_end, start_idx - 1)
#         end
#     end
#     if dual_exp_q > 0
#         for i in 1:dual_exp_q
#             push!(dual_exp_cone_indices_start, start_idx)
#             start_idx += 3
#             push!(dual_exp_cone_indices_end, start_idx - 1)
#         end
#     end
#     soc_cone_indices_start = Vector{Int}(soc_cone_indices_start)
#     soc_cone_indices_end = Vector{Int}(soc_cone_indices_end)
#     rsoc_cone_indices_start = Vector{Int}(rsoc_cone_indices_start)
#     rsoc_cone_indices_end = Vector{Int}(rsoc_cone_indices_end)
#     exp_cone_indices_start = Vector{Int}(exp_cone_indices_start)
#     exp_cone_indices_end = Vector{Int}(exp_cone_indices_end)
#     dual_exp_cone_indices_start = Vector{Int}(dual_exp_cone_indices_start)
#     dual_exp_cone_indices_end = Vector{Int}(dual_exp_cone_indices_end)  
#     return soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end
# end

function recover_soc_cone_rotated_exp_cone_indices(; zero_indices::Integer, boxs_indices::Integer, q::Vector{<:Integer}, rq::Vector{<:Integer}, exp_q::Integer, dual_exp_q::Integer)

    # 预分配结果数组的大小
    num_soc_cones = length(q)
    num_rsoc_cones = length(rq)
    num_exp_cones = exp_q
    num_dual_exp_cones = dual_exp_q

    soc_cone_indices_start = Vector{Int}(undef, num_soc_cones)
    soc_cone_indices_end = Vector{Int}(undef, num_soc_cones)
    rsoc_cone_indices_start = Vector{Int}(undef, num_rsoc_cones)
    rsoc_cone_indices_end = Vector{Int}(undef, num_rsoc_cones)
    exp_cone_indices_start = Vector{Int}(undef, num_exp_cones)
    exp_cone_indices_end = Vector{Int}(undef, num_exp_cones)
    dual_exp_cone_indices_start = Vector{Int}(undef, num_dual_exp_cones)
    dual_exp_cone_indices_end = Vector{Int}(undef, num_dual_exp_cones)

    # 初始化起始索引
    start_idx = zero_indices + boxs_indices + 1

    # 处理 SOC
    for i in 1:num_soc_cones
        soc_cone_indices_start[i] = start_idx
        if q[i] < 2
            throw(ArgumentError("The size of the SOC must be greater than 1"))
        end
        start_idx += q[i]
        soc_cone_indices_end[i] = start_idx - 1
    end

    # 处理 Rotated SOC
    for i in 1:num_rsoc_cones
        rsoc_cone_indices_start[i] = start_idx
        if rq[i] < 3
            throw(ArgumentError("The size of the rotated SOC must be greater than 2"))
        end
        start_idx += rq[i]
        rsoc_cone_indices_end[i] = start_idx - 1
    end

    # 处理 Exponential Cones
    for i in 1:num_exp_cones
        exp_cone_indices_start[i] = start_idx
        start_idx += 3
        exp_cone_indices_end[i] = start_idx - 1
    end

    # 处理 Dual Exponential Cones
    for i in 1:num_dual_exp_cones
        dual_exp_cone_indices_start[i] = start_idx
        start_idx += 3
        dual_exp_cone_indices_end[i] = start_idx - 1
    end

    return soc_cone_indices_start, soc_cone_indices_end, 
           rsoc_cone_indices_start, rsoc_cone_indices_end,
           exp_cone_indices_start, exp_cone_indices_end,
           dual_exp_cone_indices_start, dual_exp_cone_indices_end
end


function printInfo(;infoAll::PDHGCLPInfo)
    info = infoAll.convergeInfo[1]
    @info (@sprintf("iter:%d rel_p_res(inf):%.3e rel_d_res(inf):%.3e primal_obj:%.3e dual_obj:%.3e rel_pd_gap:%.3e time:%.2f restart_used:%d",
    infoAll.iter, info.l_inf_rel_primal_res, info.l_inf_rel_dual_res, info.primal_objective, info.dual_objective, info.rel_gap, infoAll.time, infoAll.restart_used))
end

function printInfolevel2(;infoAll::PDHGCLPInfo, diff_nrm_x_recovered::rpdhg_float, diff_nrm_y_recovered::rpdhg_float, nrm2_x_recovered::rpdhg_float, nrm2_y_recovered::rpdhg_float, nrmInf_x_recovered::rpdhg_float, nrmInf_y_recovered::rpdhg_float, params::PDHGCLPParameters)
    info = infoAll.convergeInfo[1]
    @info (@sprintf("iter:%d rel_p_res(inf):%.3e rel_d_res(inf):%.3e primal_obj:%.5e dual_obj:%.5e rel_pd_gap:%.3e time:%.2f restart_used:%d, restart_trigger_mean:%d, restart_trigger_ergodic:%d, diff_nrm_x_recovered:%.3e diff_nrm_y_recovered:%.3e nrm2_x_recovered:%.3e nrm2_y_recovered:%.3e, nrmInf_x_recovered:%.3e nrmInf_y_recovered:%.3e dual_gap_ergodic:%.3e, dual_gap_mean:%.3e, dual_gap_restart_threshold:%.3e, dual_gap_r:%.3e, kkt_error_ergodic:%.3e, kkt_error_mean:%.3e, kkt_error_restart_threshold:%.3e, tau:%.3e, sigma:%.3e",
    infoAll.iter, info.l_inf_rel_primal_res, info.l_inf_rel_dual_res, info.primal_objective, info.dual_objective, info.rel_gap, infoAll.time, infoAll.restart_used, infoAll.restart_trigger_mean, infoAll.restart_trigger_ergodic, diff_nrm_x_recovered, diff_nrm_y_recovered, nrm2_x_recovered, nrm2_y_recovered, nrmInf_x_recovered, nrmInf_y_recovered, infoAll.normalized_duality_gap[1], infoAll.normalized_duality_gap[2], infoAll.normalized_duality_gap_restart_threshold, infoAll.normalized_duality_gap_r, infoAll.kkt_error[1], infoAll.kkt_error[2], infoAll.kkt_error_restart_threshold, params.tau, params.sigma))
end

function Mnorm(; solver::rpdhgSolver, x::CuArray, y::CuArray,
     tau::rpdhg_float, sigma::rpdhg_float, AxTemp::dualVector)
    solver.primalMV!(solver.data.coeff, x, AxTemp);
    yAx = CUDA.dot(y, AxTemp.y)
    return CUDA.dot(x, x) / tau - 2 * yAx + CUDA.dot(y, y) / sigma
end

function approximate_cal!(; solver::rpdhgSolver, h1::primalVector, h2::dualVector,
    slack::solVecPrimal, dual_sol_temp::solVecDual, t::rpdhg_float, 
    primal::primalVector, dual::dualVector, 
    tau::rpdhg_float, sigma::rpdhg_float)
    # modify `slack.primal_sol` and `dual_sol_temp.dual_sol`
    slack.primal_sol.x .= primal.x .+ (t * tau / 2) * h1.x
    slack.proj!(slack.primal_sol, solver.sol.x)

    dual_sol_temp.dual_sol.y .= dual.y .+ (t * sigma / 2) * h2.y
    dual_sol_temp.proj!(dual_sol_temp.dual_sol)
end


"""
calculate the normalized duality gap
   rho(r, z)
"""
function binary_search_duality_gap(; solver::rpdhgSolver, 
    r::rpdhg_float, primal::primalVector, 
    dual::dualVector, slack::solVecPrimal,
    dual_sol_temp::solVecDual, tau::rpdhg_float, sigma::rpdhg_float, t0::Ref{rpdhg_float},
    maxIter::Integer = 1000, tol::rpdhg_float = 1e-6)

    t = t0[]
    tRight = t
    tLeft = 0.0
    h1 = slack.primal_sol_mean # no copy
    solver.adjointMV!(solver.data.coeffTrans, dual, h1.x)
    h1.x .-= solver.data.c
    h2 = dual_sol_temp.dual_sol_mean # no copy
    solver.primalMV!(solver.data.coeff, primal.x, h2)
    h2.y .*= -1.0 
    solver.addCoeffd!(solver.data.coeff, h2, 1.0);
    for k = 1: maxIter
        approximate_cal!(solver = solver, h1 = h1, h2 = h2, 
            slack = slack, dual_sol_temp = dual_sol_temp, 
            t = t, primal = primal, dual = dual, 
            tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
        dual_sol_temp.dual_sol_temp.y = dual_sol_temp.dual_sol.y - dual.y
        if Mnorm(solver = solver, x = slack.primal_sol_lag.x,
             y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
             AxTemp = dual_sol_temp.dual_sol_lag) > r
            if k == 1
                break
            else
                tRight = t, tLeft = t / 2, break
            end
        end
        t *= 2
    end
    # binary function search_element(arr::Vector{T}, target::T) where T
    tMid = (tRight + tLeft) / 2
    while tRight - tLeft > tol
        tMid = (tRight + tLeft) / 2
        approximate_cal!(solver = solver, h1 = h1, h2 = h2,
                        slack = slack, dual_sol_temp = dual_sol_temp,
                        t = tMid, primal = primal, dual = dual,
                        tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
        dual_sol_temp.dual_sol_temp.y = dual_sol_temp.dual_sol.y - dual.y
        if Mnorm(solver = solver, x = slack.primal_sol_lag.x,
            y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
             AxTemp = dual_sol_temp.dual_sol_lag) < r
            tLeft = tMid
        else
            tRight = tMid
        end
    end
    t0[] = tMid
    slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
    dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y - dual.y
    val1 = h1.x' * slack.primal_sol_lag.x
    val2 = h2.y' * dual_sol_temp.dual_sol_temp.y
    val = (val1 + val2) / r
    return val
end

function converge_info_calculation(; solver::rpdhgSolver, primal_sol::primalVector, dual_sol::dualVector, slack::solVecPrimal, dual_sol_temp::solVecDual, converge_info::PDHGCLPConvergeInfo)
    pObj = dot(primal_sol.x, solver.data.d_c)
    solver.adjointMV!(solver.data.coeffTrans, dual_sol, slack.primal_sol.x)
    AtyInf = CUDA.norm(slack.primal_sol.x, Inf)
    AtyNrm1 = CUDA.norm(slack.primal_sol.x, 1)
    slack.primal_sol.x .= solver.data.c - slack.primal_sol.x;
    slack.primal_sol_lag.x .= max.(0, slack.primal_sol.x)
    slack.primal_sol_mean.x .= min.(0, slack.primal_sol.x)

    dObj = solver.dotCoeffd(solver.data.coeff, dual_sol)
    dObj += (solver.data.bl_finite' * slack.primal_sol_lag.xbox + solver.data.bu_finite' * slack.primal_sol_mean.xbox)
    abs_gap = abs(pObj - dObj)
    rel_gap = abs_gap / (1 + abs(pObj) + abs(dObj));

    # dual_sol_temp.dual_sol_mean = Gx
    solver.primalMV!(solver.data.coeff, primal_sol.x, dual_sol_temp.dual_sol_mean);
    AxInf = CUDA.norm(dual_sol_temp.dual_sol_mean.y, Inf)
    AxNrm1 = CUDA.norm(dual_sol_temp.dual_sol_mean.y, 1)
    solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
    dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
    solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)

    dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol_mean.y - dual_sol_temp.dual_sol_lag.y
    l_2_abs_primal_res = CUDA.norm(dual_sol_temp.dual_sol_temp.y);
    l_2_rel_primal_res = l_2_abs_primal_res / (1 + max(solver.data.hNrm1, AxNrm1));
    l_inf_abs_primal_res = CUDA.maximum(abs.(dual_sol_temp.dual_sol_temp.y));
    l_inf_rel_primal_res = l_inf_abs_primal_res / (1 + max(solver.data.hNrmInf, AxInf));
    
    slack.primal_sol_lag.x .= slack.primal_sol.x
    solver.sol.x.slack_proj!(slack.primal_sol, slack)
    slack.primal_sol_mean.x .= slack.primal_sol.x - slack.primal_sol_lag.x
    l_2_abs_dual_res = norm(slack.primal_sol_mean.x);
    l_2_rel_dual_res = l_2_abs_dual_res / (1 + max(solver.data.cNrm1, AxNrm1));
    l_inf_abs_dual_res = CUDA.maximum(abs.(slack.primal_sol_mean.x));
    l_inf_rel_dual_res = l_inf_abs_dual_res / (1 + max(solver.data.cNrmInf, AtyInf));
    
    converge_info.rel_gap = rel_gap;
    converge_info.abs_gap = abs_gap;
    converge_info.l_2_abs_primal_res = l_2_abs_primal_res;
    converge_info.l_2_rel_primal_res = l_2_rel_primal_res;
    converge_info.l_inf_abs_primal_res = l_inf_abs_primal_res;
    converge_info.l_inf_rel_primal_res = l_inf_rel_primal_res;
    converge_info.l_2_abs_dual_res = l_2_abs_dual_res;
    converge_info.l_2_rel_dual_res = l_2_rel_dual_res;
    converge_info.l_inf_abs_dual_res = l_inf_abs_dual_res;
    converge_info.l_inf_rel_dual_res = l_inf_rel_dual_res;
    converge_info.primal_objective = pObj;
    converge_info.dual_objective = dObj;
    return;
end

function infeasibility_info_calculation(; solver::rpdhgSolver,
    pObj::Any, dObj::Any, l_inf_primal_ray_infeasibility::Any, l_inf_dual_ray_infeasibility::Any,
    primal_ray_unscaled::primalVector, dual_ray_unscaled::dualVector,
    slack::solVecPrimal, dual_sol_temp::solVecDual, infeasibility_info::PDHGCLPInfeaInfo,
    pObj_seq::rpdhg_float, dObj_seq::rpdhg_float)

    primal_ray = slack.primal_sol
    dual_ray = dual_sol_temp.dual_sol
    primal_ray.x .= primal_ray_unscaled.x
    dual_ray.y .= dual_ray_unscaled.y
    # primal_ray ./= norm(primal_ray, 2)
    # dual_ray ./= norm(dual_ray, 2)
    if pObj === nothing
        pObj = CUDA.dot(primal_ray.x, solver.data.d_c)
    end
    if l_inf_primal_ray_infeasibility === nothing && pObj < 0.0
        # dual_sol_temp.dual_sol_mean = Gx
        solver.primalMV!(solver.data.coeff, primal_ray.x, dual_sol_temp.dual_sol_mean);
        solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
        dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
        solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol_mean.y - dual_sol_temp.dual_sol_lag.y
        l_inf_primal_ray_infeasibility = CUDA.maximum(CUDA.abs.(dual_sol_temp.dual_sol_temp.y));
    end

    if dObj === nothing && l_inf_dual_ray_infeasibility === nothing
        dObj = solver.dotCoeffd(solver.data.coeff, dual_ray)
        solver.adjointMV!(solver.data.coeffTrans, dual_ray, slack.primal_sol.x)
        slack.primal_sol.x .= solver.data.c - slack.primal_sol.x;
        slack.primal_sol_lag.x .= max.(0.0, slack.primal_sol.x)
        slack.primal_sol_mean.x .= min.(0.0, slack.primal_sol.x)
        dObj += (solver.sol.x.bl' * slack.primal_sol_lag.xbox + solver.sol.x.bu' * slack.primal_sol_mean.xbox)
        if dObj > 0
            slack.primal_sol_lag.x .= slack.primal_sol.x
            solver.sol.x.slack_proj!(slack.primal_sol, slack)
            slack.primal_sol_mean.x .= slack.primal_sol.x - slack.primal_sol_lag.x
            l_inf_dual_ray_infeasibility = CUDA.maximum(CUDA.abs.(slack.primal_sol_mean.x));
        end
    end
    infeasibility_info.primal_ray_objective = pObj
    infeasibility_info.dual_ray_objective = dObj
    if l_inf_primal_ray_infeasibility === nothing
        infeasibility_info.max_primal_ray_infeasibility = 1e+10
    else
        infeasibility_info.max_primal_ray_infeasibility = l_inf_primal_ray_infeasibility
    end
    if l_inf_dual_ray_infeasibility === nothing
        infeasibility_info.max_dual_ray_infeasibility = 1e+10
    else
        infeasibility_info.max_dual_ray_infeasibility = l_inf_dual_ray_infeasibility
    end
    push!(infeasibility_info.primalObj_trend, pObj_seq)
    push!(infeasibility_info.dualObj_trend, dObj_seq)
    return
end

function pdhg_one_iter!(; solver::rpdhgSolver, x::solVecPrimal, y::solVecDual, tau::rpdhg_float, sigma::rpdhg_float, slack::solVecPrimal, dual_sol_temp::solVecDual)
    x.primal_sol_lag.x .= x.primal_sol.x
    solver.adjointMV!(solver.data.coeffTrans, y.dual_sol, slack.primal_sol_lag.x)
    x.primal_sol.x .-= tau * (solver.data.c .- slack.primal_sol_lag.x)
    x.proj!(x.primal_sol, x)
    x.primal_sol_lag.x .= x.primal_sol.x * 2 - x.primal_sol_lag.x
    solver.primalMV!(solver.data.coeff, x.primal_sol_lag.x, dual_sol_temp.dual_sol_lag)
    # dual_sol_temp -= d
    solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_lag, -1.0)
    y.dual_sol.y .-= sigma * dual_sol_temp.dual_sol_lag.y
    y.proj!(y.dual_sol)
end



function infoSummary(;info::PDHGCLPInfo)
    @info "---------------------------------------------------"
    @info "------------- Solver Info Summary -----------------"
    @info "---------------------------------------------------"
    @info @sprintf("iter_num:                     %d", info.iter)
    @info @sprintf("exit_status:                  %s", info.exit_status)
    if info.exit_status == :optimal || info.exit_status == :max_iter || info.exit_status == :time_limit
        for converge_info in info.convergeInfo
            if converge_info.status == :optimal || converge_info.status == :max_iter || info.exit_status == :time_limit
                @info @sprintf("l_2_abs_primal_res:           %.4e", converge_info.l_2_abs_primal_res)
                @info @sprintf("l_2_rel_primal_res:           %.4e", converge_info.l_2_rel_primal_res)
                @info @sprintf("l_inf_abs_primal_res:         %.4e", converge_info.l_inf_abs_primal_res)
                @info @sprintf("l_inf_rel_primal_res:         %.4e", converge_info.l_inf_rel_primal_res)
                @info @sprintf("l_2_abs_dual_res:             %.4e", converge_info.l_2_abs_dual_res)
                @info @sprintf("l_2_rel_dual_res:             %.4e", converge_info.l_2_rel_dual_res)
                @info @sprintf("l_inf_abs_dual_res:           %.4e", converge_info.l_inf_abs_dual_res)
                @info @sprintf("l_inf_rel_dual_res:           %.4e", converge_info.l_inf_rel_dual_res)
                @info @sprintf("rel_gap:                      %.4e", converge_info.rel_gap)
                @info @sprintf("abs_gap:                      %.4e", converge_info.abs_gap)
                @info @sprintf("primal_obj:                   %.4e", converge_info.primal_objective)
                @info @sprintf("dual_obj:                     %.4e", converge_info.dual_objective)
                if info.exit_status == :max_iter
                    @info (@sprintf("max_iter:                     %d", info.iter))
                end
                break
            end
        end
    end
    if info.exit_status == :primal_infeasible_low_acc || info.exit_status == :primal_infeasible_high_acc || info.exit_status == :max_iter
        for infeaInfo in info.infeaInfo
            @info (@sprintf("dual_ray_objective:           %.4e", infeaInfo.dual_ray_objective))
            @info (@sprintf("max_dual_ray_infeasibility:   %.4e", infeaInfo.max_dual_ray_infeasibility))
            break
        end
    end
    if info.exit_status == :dual_infeasible_low_acc || info.exit_status == :dual_infeasible_high_acc || info.exit_status == :max_iter
        for infeaInfo in info.infeaInfo
            @info (@sprintf("primal_ray_objective:         %.4e", infeaInfo.primal_ray_objective))
            @info (@sprintf("max_primal_ray_infeasibility: %.4e", infeaInfo.max_primal_ray_infeasibility))
            break
        end
    end
    @info (@sprintf("time:                         %.4f", info.time))
    @info ("---------------------------------------------------")
end



"""
    rpdhg_cpu_solve()
    the main function of the rpdhg solver


    ## Problem definition
    rpdhg solves a problem of the form
    ```
    minimize    c' * x
    s.t.        G * x - h in K_G
                bl <= x1 <= bu
                x2 in K_{x2}
                x = (x1, x2)
    ```
    where K_G is a product of cone of
    - zero cone
    - nonnegative cone
    - second order cone `{ (t,x) | ||x||_2 ≤ t }`
    - rotated second order cone `{ (x, y, z) | x, y >= 0, 2 * x * y >= ||z||_2^2 }`
    - exponential cone `{ (x, y, z) | y >= x, exp(x) <= y, exp(x) <= z }`
    - dual exponential cone `{ (x, y, z) | -y >= -x, exp(x) <= y, exp(x) <= z }`

    x1 is a box constraint, x2 is a product of cone constraint(only soc, rsoc, exp, dual_exp cone are allowed)

    ## Parameters
    - `n`: the number of variables
    - `m`: the number of constraints
    - `nb`: the number of box constraints
    - `c`: a `Vector` of length `n`
    - `G`: an `AbstractMatrix` with `m` rows and `n` columns
    - `h`: a `Vector` of length `m`
    - `mGzero`: the number of zero constraints
    - `mGnonnegative`: the number of nonnegative constraints
    - `socG`: a `Vector` of sizes of SOCs
    - `rsocG`: a `Vector` of sizes of rotated SOCs
    - `expG`: the number of exponential cones
    - `dual_expG`: the number of dual exponential cones
    - `bl`: the `Vector` of lower bounds for the box cone, length `nb`
    - `bu`: the `Vector` of upper bounds for the box cone, length `nb`
    - `soc_x`: the `Vector` of indices of variables in `x` that belong to SOCs
    - `rsoc_x`: the `Vector` of indices of variables in `x` that belong to rotated SOCs
    - `exp_x`: the number of exponential cones
    - `dual_exp_x`: the number of dual exponential cones
    - `Dl`: the `Vector` of length `m` to scale the constraints
    - `Dr`: the `Vector` of length `n` to scale the variables
    - `rescaling_method`: the method to rescale the data, `:none`, `:ruiz_pock_chambolle` or `:ruiz_pock_more`
    - `use_preconditioner`: whether to use the preconditioner
    - `primal_sol`: a `Vector` to warmstart the primal variables,
    - `dual_sol`: a `Vector` to warmstart the dual variables,
    - `warm_start`: a `Bool` to enable warm start
    - `max_outer_iter`: the maximum number of outer iterations
    - `max_inner_iter`: the maximum number of inner iterations
    - `abs_tol`: the absolute tolerance for the termination criteria
    - `rel_tol`: the relative tolerance for the termination criteria
    - `eps_primal_infeasible_low_acc`: the absolute tolerance for the primal infeasibility
    - `eps_dual_infeasible_low_acc`: the absolute tolerance for the dual infeasibility
    - `eps_primal_infeasible_high_acc`: the absolute tolerance for the primal infeasibility
    - `eps_dual_infeasible_high_acc`: the absolute tolerance for the dual infeasibility
    - `print_freq`: the frequency of printing the information
    - `restart_check_freq`: the frequency of checking the restart criteria
    - `check_terminate_freq`: the frequency of checking the termination criteria
    - `verbose`: the level of verbosity, `0`, `1` or `2`
    - `time_limit`: the time limit for the solver
    - `method`: the method to solve the problem, `:average`, `:halpern`, `:average_no_restart`, `:halpern_no_restart`

    !!! note
        To successfully warmstart the solver `primal_sol`, `dual_sol` and `slack`
        must all be provided **and** `warm_start` option must be set to `true`.
    
    ## Output
    This function returns a `Solution` object, which contains the following fields:
    ```julia
    mutable struct Solution
        x::solVecPrimal
        y::solVecDual
        params::PDHGCLPParameters
        info::PDHGCLPInfo
    end
    ```
    where `x` stores the optimal value of the primal variable, `y` stores the
    optimal value of the dual variable, and `info`
    contains various information about the solve step.
"""

function rpdhg_gpu_solve(;
    n::Integer,
    m::Integer,
    nb::Integer,
    c_cpu::Vector{rpdhg_float},
    G_cpu::AbstractMatrix{rpdhg_float},
    h_cpu::hType,
    mGzero::Integer, # m of Q for zero cone
    mGnonnegative::Integer, # m of Q for positive cone
    socG::Vector{<:Integer},
    rsocG::Vector{<:Integer},
    expG::Integer,
    dual_expG::Integer,
    bl_cpu::Vector{rpdhg_float},
    bu_cpu::Vector{rpdhg_float},
    soc_x::Vector{<:Integer},
    rsoc_x::Vector{<:Integer},
    exp_x::Integer = 0,
    dual_exp_x::Integer = 0,
    Dl_cpu::Vector{rpdhg_float} = ones(m),
    Dr_cpu::Vector{rpdhg_float} = ones(n),
    rescaling_method::Symbol = :none,
    use_preconditioner::Bool = true,
    use_adaptive_restart::Bool = true,
    use_adaptive_step_size_weight::Bool = true,
    use_aggressive::Bool = true,
    use_accelerated::Bool = true,
    use_resolving::Bool = true,
    primal_sol_cpu::Vector{rpdhg_float} = zeros(n),
    dual_sol_cpu::Vector{rpdhg_float} = zeros(m),
    warm_start::Bool = false,
    max_outer_iter::Integer = 10000,
    max_inner_iter::Integer = 500000,
    abs_tol::rpdhg_float = 1e-6,
    rel_tol::rpdhg_float = 1e-6,
    eps_primal_infeasible_low_acc::rpdhg_float = 1e-12,
    eps_dual_infeasible_low_acc::rpdhg_float = 1e-12,
    eps_primal_infeasible_high_acc::rpdhg_float = 1e-16,
    eps_dual_infeasible_high_acc::rpdhg_float = 1e-16,
    print_freq::Integer = 200,
    check_terminate_freq::Integer = 2000,
    verbose::Integer = 1,
    time_limit::Float64 = Inf,
    method::Symbol = :average,
    logfile_name::Union{String, Nothing} = nothing,
    use_kkt_restart::Bool = false,
    kkt_restart_freq::Integer = 500,
    use_duality_gap_restart::Bool = true,
    duality_gap_restart_freq::Integer = 2000
)where {hType<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
}
    if logfile_name === nothing
        if verbose > 0
            println("logfile is set to $logfile_name")
        end
    else
        # 设置日志器并使用
        logfile = open(logfile_name, "w")
        logger = PlainMultiLogger([stdout, logfile], Logging.Info)  # 同时输出到终端和文件
        # 设置全局日志器
        global_logger(logger)
        formatted_time = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        @info "logfile is set to $logfile_name at $formatted_time"
    end
    # Check the assertion
    global num_threads = nthreads()
    global time_proj = 0.0
    global time_iterative = 0.0
    global time_restart_check = 0.0
    global time_exit_check = 0.0
    global time_print_info_cal = 0.0
    global handle = create_cublas_handle()
    if verbose > 0
        @info ("---------------------------------------------------")
        @info ("------------- Parameter Info Summary --------------")
        @info ("---------------------------------------------------")
        @info ("time limit: $time_limit")
        @info ("use_preconditioner: $use_preconditioner")
        @info ("use_adaptive_restart: $use_adaptive_restart")
        @info ("use_adaptive_step_size_weight: $use_adaptive_step_size_weight")
        @info ("use_aggressive: $use_aggressive")
        @info ("use_duality_gap_restart: $use_duality_gap_restart")
        @info ("duality_gap_restart_freq: $duality_gap_restart_freq")
        @info ("use_kkt_restart: $use_kkt_restart")
        @info ("kkt_restart_freq: $kkt_restart_freq")
        @info ("abs_tol: $abs_tol")
        @info ("rel_tol: $rel_tol")
        @info ("eps_primal_infeasible_low_acc: $eps_primal_infeasible_low_acc")
        @info ("eps_dual_infeasible_low_acc: $eps_dual_infeasible_low_acc")
        @info ("eps_primal_infeasible_high_acc: $eps_primal_infeasible_high_acc")
        @info ("eps_dual_infeasible_high_acc: $eps_dual_infeasible_high_acc")
        @info ("print_freq: $print_freq")
        @info ("check_terminate_freq: $check_terminate_freq")
        @info ("---------------------------------------------------")
        @info ("------------- Model Info Summary -----------------")
        @info ("---------------------------------------------------")
        @info ("variable number: $n")
        @info ("constraint number: $m")
        @info ("non-zero elements in G: $(nnz(G_cpu))")
        @info ("Number of box constraints: $nb")
        @info ("Number of equal constraints: $mGzero")
        @info ("Number of nonnegative constraints: $mGnonnegative")
        @info ("Number of soc cone constraints: $(length(socG))")
        @info ("Number of rsoc cone constraints: $(length(rsocG))")
        @info ("Number of exp cone constraints: $expG")
        @info ("Number of dual exp cone constraints: $dual_expG")
        @info ("Number of soc cone variables: $(length(soc_x))")
        @info ("Number of rsoc cone variables: $(length(rsoc_x))")
        @info ("Number of exp cone variables: $exp_x")
        @info ("Number of dual exp cone variables: $dual_exp_x")
        @info ("---------------------------------------------------")
    end
    if m != size(G_cpu, 1)
        error("m does not match the size of G")
    end
    if m != mGzero + mGnonnegative + sum(socG) + sum(rsocG) + (expG + dual_expG) * 3
        error("m does not match the sum of mGzero, mGnonnegative, socG, rsocG, and expG + dual_expG * 3")
    end
    if method != :average && method != :halpern
        throw(ArgumentError("method only available for :average and :halpern"))
    end
    if use_preconditioner
        if verbose > 0
            @info "Using preconditioner"
        end
    else
        if verbose > 0
            @info "Don't use preconditioner"
        end
    end

    Random.seed!(1234)
    if length(primal_sol_cpu) == n && length(dual_sol_cpu) == m
        if !warm_start
            if verbose > 0
                @info ("Warm start not enabled. Ignoring warm start values.")
                @info ("Initializing primal and dual variables to zero.")
            end
            fill!(primal_sol_cpu, 0.0)
            fill!(dual_sol_cpu, 0.0)
        else
            if verbose > 0
                @info ("Warm start!!")
            end
        end
    else
        if warm_start
            throw(ArgumentError("Warmstart doesn't match the problem size"))
        end
        if verbose > 0
            @info ("Initializing primal and dual variables to random.")
        end
        primal_sol_cpu = rand(n)
        dual_sol_cpu = rand(m)
    end
    # make sure rescaling_method do not change initial data
    @assert all(bl_cpu .<= bu_cpu) "Not all elements of bl are less than or equal to the corresponding elements in bu"
    ## set data struct
    coeff = coeffUnion(
        G = G_cpu,
        h = h_cpu,
        m = m,
        n = n,
        d_G = nothing,
        d_h = nothing
    )
    coeffTrans = coeffUnion(
        G = nothing,
        h = nothing,
        m = m,
        n = n,
        d_G = coeff.d_G',
        d_h = coeff.d_h
    )
    x_soc_cone_indices_start, x_soc_cone_indices_end, x_rsoc_cone_indices_start, x_rsoc_cone_indices_end, x_exp_cone_indices_start, x_exp_cone_indices_end, x_dual_exp_cone_indices_start, x_dual_exp_cone_indices_end = 
    recover_soc_cone_rotated_exp_cone_indices(
        zero_indices = 0,
        boxs_indices = length(bl_cpu),
        q = soc_x,
        rq = rsoc_x,
        exp_q = exp_x,
        dual_exp_q = dual_exp_x
    )
    y_soc_cone_indices_start, y_soc_cone_indices_end, y_rsoc_cone_indices_start, y_rsoc_cone_indices_end, y_exp_cone_indices_start, y_exp_cone_indices_end, y_dual_exp_cone_indices_start, y_dual_exp_cone_indices_end = 
    recover_soc_cone_rotated_exp_cone_indices(
        zero_indices = mGzero,
        boxs_indices = mGnonnegative,
        q = socG,
        rq = rsocG,
        exp_q = expG,
        dual_exp_q = dual_expG
    )
    Dl_struct = dualVector(
        y = CuArray(Dl_cpu),
        m = m,
        mGzero = mGzero,
        mGnonnegative = mGnonnegative,
        soc_cone_indices_start = y_soc_cone_indices_start,
        soc_cone_indices_end = y_soc_cone_indices_end,
        rsoc_cone_indices_start = y_rsoc_cone_indices_start,
        rsoc_cone_indices_end = y_rsoc_cone_indices_end,
        exp_cone_indices_start = y_exp_cone_indices_start,
        exp_cone_indices_end = y_exp_cone_indices_end,
        dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = y_dual_exp_cone_indices_end
    )
    Dr_struct = primalVector(
        x = CuArray(Dr_cpu),
        box_index = length(bl_cpu),
        soc_cone_indices_start = x_soc_cone_indices_start,
        soc_cone_indices_end = x_soc_cone_indices_end,
        rsoc_cone_indices_start = x_rsoc_cone_indices_start,
        rsoc_cone_indices_end = x_rsoc_cone_indices_end,
        exp_cone_indices_start = x_exp_cone_indices_start,
        exp_cone_indices_end = x_exp_cone_indices_end,
        dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = x_dual_exp_cone_indices_end
    )
    diagonal_scale = Diagonal_preconditioner(
        Dl = Dl_struct,
        Dr = Dr_struct,
        m = m,
        n = n,
        len_soc_x = length(soc_x),
        len_rsoc_x = length(rsoc_x)
    )
    data = probData(
        m = m,
        n = n,
        nb = nb,
        c_cpu = c_cpu,
        coeff = coeff,
        coeffTrans = coeffTrans,
        GlambdaMax = 0.0, # initialize to zero
        GlambdaMax_flag = 0,
        bl_cpu = bl_cpu,
        bu_cpu = bu_cpu,
        diagonal_scale = diagonal_scale,
        raw_data = nothing
    )
    if use_preconditioner
        raw_data = create_raw_data(
            m = m,
            n = n,
            nb = nb,
            c_cpu = c_cpu,
            coeff = coeff,
            bl_cpu = bl_cpu,
            bu_cpu = bu_cpu,
            hNrm1 = data.hNrm1,
            cNrm1 = data.cNrm1,
            hNrmInf = data.hNrmInf,
            cNrmInf = data.cNrmInf
        )
        data.raw_data = raw_data
    end
    primal_sol_struct = primalVector(
        x = CuArray(primal_sol_cpu),
        box_index = length(bl_cpu),
        soc_cone_indices_start = x_soc_cone_indices_start,
        soc_cone_indices_end = x_soc_cone_indices_end,
        rsoc_cone_indices_start = x_rsoc_cone_indices_start,
        rsoc_cone_indices_end = x_rsoc_cone_indices_end,
        exp_cone_indices_start = x_exp_cone_indices_start,
        exp_cone_indices_end = x_exp_cone_indices_end,
        dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = x_dual_exp_cone_indices_end
    )
    primal_sol_lag_struct = primalVector(
        x = CuArray(primal_sol_cpu),
        box_index = length(bl_cpu),
        soc_cone_indices_start = x_soc_cone_indices_start,
        soc_cone_indices_end = x_soc_cone_indices_end,
        rsoc_cone_indices_start = x_rsoc_cone_indices_start,
        rsoc_cone_indices_end = x_rsoc_cone_indices_end,
        exp_cone_indices_start = x_exp_cone_indices_start,
        exp_cone_indices_end = x_exp_cone_indices_end,
        dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = x_dual_exp_cone_indices_end
    )
    primal_sol_mean_struct = primalVector(
        x = CuArray(primal_sol_cpu),
        box_index = length(bl_cpu),
        soc_cone_indices_start = x_soc_cone_indices_start,
        soc_cone_indices_end = x_soc_cone_indices_end,
        rsoc_cone_indices_start = x_rsoc_cone_indices_start,
        rsoc_cone_indices_end = x_rsoc_cone_indices_end,
        exp_cone_indices_start = x_exp_cone_indices_start,
        exp_cone_indices_end = x_exp_cone_indices_end,
        dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = x_dual_exp_cone_indices_end
    )
    ## set solution struct
    if use_preconditioner
        # println("before primal_sol_struct")
        # CUDA.memory_status()
        x = solVecPrimal(
            primal_sol = primal_sol_struct,
            primal_sol_lag = primal_sol_lag_struct,
            primal_sol_mean = primal_sol_mean_struct,
            box_index = length(bl_cpu),
            bl = data.d_bl,
            bu = data.d_bu,
            soc_cone_indices_start = x_soc_cone_indices_start,
            soc_cone_indices_end = x_soc_cone_indices_end,
            rsoc_cone_indices_start = x_rsoc_cone_indices_start,
            rsoc_cone_indices_end = x_rsoc_cone_indices_end,
            exp_cone_indices_start = x_exp_cone_indices_start,
            exp_cone_indices_end = x_exp_cone_indices_end,
            dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = x_dual_exp_cone_indices_end,
            proj! = x -> println("proj! not implemented"),
            slack_proj! = x -> println("slack_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_primal = solVecPrimalRecovered(deepCopyPrimalVector(primal_sol_struct), deepCopyPrimalVector(primal_sol_struct))
        )
        # println("after primal_sol_struct")
        # CUDA.memory_status()
    else
        x = solVecPrimal(
            primal_sol = primal_sol_struct,
            primal_sol_lag = primal_sol_lag_struct,
            primal_sol_mean = primal_sol_mean_struct,
            box_index = length(bl_cpu),
            bl = data.d_bl,
            bu = data.d_bu,
            soc_cone_indices_start = x_soc_cone_indices_start,
            soc_cone_indices_end = x_soc_cone_indices_end,
            rsoc_cone_indices_start = x_rsoc_cone_indices_start,
            rsoc_cone_indices_end = x_rsoc_cone_indices_end,
            exp_cone_indices_start = x_exp_cone_indices_start,
            exp_cone_indices_end = x_exp_cone_indices_end,
            dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = x_dual_exp_cone_indices_end,
            proj! = x -> println("proj! not implemented"),
            slack_proj! = x -> println("slack_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_primal = nothing
        )
    end
    
    dual_sol_struct = dualVector(
        y = CuArray(dual_sol_cpu),
        m = m,
        mGzero = mGzero,
        mGnonnegative = mGnonnegative,
        soc_cone_indices_start = y_soc_cone_indices_start,
        soc_cone_indices_end = y_soc_cone_indices_end,
        rsoc_cone_indices_start = y_rsoc_cone_indices_start,
        rsoc_cone_indices_end = y_rsoc_cone_indices_end,
        exp_cone_indices_start = y_exp_cone_indices_start,
        exp_cone_indices_end = y_exp_cone_indices_end,
        dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = y_dual_exp_cone_indices_end
    )
    dual_sol_lag_struct = dualVector(
        y = CuArray(dual_sol_cpu),
        m = m,
        mGzero = mGzero,
        mGnonnegative = mGnonnegative,
        soc_cone_indices_start = y_soc_cone_indices_start,
        soc_cone_indices_end = y_soc_cone_indices_end,
        rsoc_cone_indices_start = y_rsoc_cone_indices_start,
        rsoc_cone_indices_end = y_rsoc_cone_indices_end,
        exp_cone_indices_start = y_exp_cone_indices_start,
        exp_cone_indices_end = y_exp_cone_indices_end,
        dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = y_dual_exp_cone_indices_end
    )
    dual_sol_mean_struct = dualVector(
        y = CuArray(dual_sol_cpu),
        m = m,
        mGzero = mGzero,
        mGnonnegative = mGnonnegative,
        soc_cone_indices_start = y_soc_cone_indices_start,
        soc_cone_indices_end = y_soc_cone_indices_end,
        rsoc_cone_indices_start = y_rsoc_cone_indices_start,
        rsoc_cone_indices_end = y_rsoc_cone_indices_end,
        exp_cone_indices_start = y_exp_cone_indices_start,
        exp_cone_indices_end = y_exp_cone_indices_end,
        dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
        dual_exp_cone_indices_end = y_dual_exp_cone_indices_end
    )
    if use_preconditioner
        y = solVecDual(
            dual_sol = dual_sol_struct,
            dual_sol_lag = dual_sol_lag_struct,
            dual_sol_mean = dual_sol_mean_struct,
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            soc_cone_indices_start = y_soc_cone_indices_start,
            soc_cone_indices_end = y_soc_cone_indices_end,
            rsoc_cone_indices_start = y_rsoc_cone_indices_start,
            rsoc_cone_indices_end = y_rsoc_cone_indices_end,
            exp_cone_indices_start = y_exp_cone_indices_start,
            exp_cone_indices_end = y_exp_cone_indices_end,
            dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = y_dual_exp_cone_indices_end,
            slack = solVecPrimal(
                primal_sol = deepCopyPrimalVector(primal_sol_struct),
                primal_sol_lag = deepCopyPrimalVector(primal_sol_lag_struct),
                primal_sol_mean = deepCopyPrimalVector(primal_sol_mean_struct),
                box_index = length(bl_cpu),
                bl = data.d_bl,
                bu = data.d_bu,
                soc_cone_indices_start = x_soc_cone_indices_start,
                soc_cone_indices_end = x_soc_cone_indices_end,
                rsoc_cone_indices_start = x_rsoc_cone_indices_start,
                rsoc_cone_indices_end = x_rsoc_cone_indices_end,
                exp_cone_indices_start = x_exp_cone_indices_start,
                exp_cone_indices_end = x_exp_cone_indices_end,
                dual_exp_cone_indices_start = x_dual_exp_cone_indices_start,
                dual_exp_cone_indices_end = x_dual_exp_cone_indices_end,
                proj! = x -> println("proj! not implemented"),
                slack_proj! = x -> println("slack_proj! not implemented"),
                proj_diagonal! = x -> println("proj_diagonal! not implemented"),
                recovered_primal = nothing
            ),
            proj! = x -> println("proj! not implemented"),
            con_proj! = x -> println("con_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_dual = solVecDualRecovered(deepCopyDualVector(dual_sol_struct), deepCopyDualVector(dual_sol_struct))
        )

        dual_sol_temp = solVecDual(
            dual_sol = deepCopyDualVector(dual_sol_struct),
            dual_sol_lag = deepCopyDualVector(dual_sol_lag_struct),
            dual_sol_mean = deepCopyDualVector(dual_sol_mean_struct),
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            soc_cone_indices_start = y_soc_cone_indices_start,
            soc_cone_indices_end = y_soc_cone_indices_end,
            rsoc_cone_indices_start = y_rsoc_cone_indices_start,
            rsoc_cone_indices_end = y_rsoc_cone_indices_end,
            exp_cone_indices_start = y_exp_cone_indices_start,
            exp_cone_indices_end = y_exp_cone_indices_end,
            dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = y_dual_exp_cone_indices_end,
            slack = nothing,
            proj! = x -> println("proj! not implemented"),
            con_proj! = x -> println("con_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_dual = nothing
        )
    else
        y = solVecDual(
            dual_sol = dual_sol_struct,
            dual_sol_lag = dual_sol_lag_struct,
            dual_sol_mean = dual_sol_mean_struct,
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            soc_cone_indices_start = y_soc_cone_indices_start,
            soc_cone_indices_end = y_soc_cone_indices_end,
            rsoc_cone_indices_start = y_rsoc_cone_indices_start,
            rsoc_cone_indices_end = y_rsoc_cone_indices_end,
            exp_cone_indices_start = y_exp_cone_indices_start,
            exp_cone_indices_end = y_exp_cone_indices_end,
            dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = y_dual_exp_cone_indices_end,
            slack = solVecPrimal(
                primal_sol = deepCopyPrimalVector(primal_sol_struct),
                primal_sol_lag = deepCopyPrimalVector(primal_sol_lag_struct),
                primal_sol_mean = deepCopyPrimalVector(primal_sol_mean_struct),
                box_index = length(bl_cpu),
                bl = data.d_bl,
                bu = data.d_bu,
                soc_cone_indices_start = x_soc_cone_indices_start,
                soc_cone_indices_end = x_soc_cone_indices_end,
                proj! = x -> println("proj! not implemented"),
                slack_proj! = x -> println("slack_proj! not implemented"),
                proj_diagonal! = x -> println("proj_diagonal! not implemented"),
                recovered_primal = nothing
            ),
            proj! = x -> println("proj! not implemented"),
            con_proj! = x -> println("con_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_dual = nothing
        )
        dual_sol_temp = solVecDual(
            dual_sol = deepCopyDualVector(dual_sol_struct),
            dual_sol_lag = deepCopyDualVector(dual_sol_lag_struct),
            dual_sol_mean = deepCopyDualVector(dual_sol_mean_struct),
            mGzero = mGzero,
            mGnonnegative = mGnonnegative,
            soc_cone_indices_start = y_soc_cone_indices_start,
            soc_cone_indices_end = y_soc_cone_indices_end,
            rsoc_cone_indices_start = y_rsoc_cone_indices_start,
            rsoc_cone_indices_end = y_rsoc_cone_indices_end,
            exp_cone_indices_start = y_exp_cone_indices_start,
            exp_cone_indices_end = y_exp_cone_indices_end,
            dual_exp_cone_indices_start = y_dual_exp_cone_indices_start,
            dual_exp_cone_indices_end = y_dual_exp_cone_indices_end,
            slack = nothing,
            proj! = x -> println("proj! not implemented"),
            con_proj! = x -> println("con_proj! not implemented"),
            proj_diagonal! = x -> println("proj_diagonal! not implemented"),
            recovered_dual = nothing
        )
    end

    # set the parameters struct
    params = PDHGCLPParameters(max_outer_iter = max_outer_iter,
                                max_inner_iter = max_inner_iter,
                                rel_tol = rel_tol,
                                abs_tol = abs_tol,
                                eps_primal_infeasible_low_acc = eps_primal_infeasible_low_acc,
                                eps_dual_infeasible_low_acc = eps_dual_infeasible_low_acc,
                                eps_primal_infeasible_high_acc = eps_primal_infeasible_high_acc,
                                eps_dual_infeasible_high_acc = eps_dual_infeasible_high_acc,
                                sigma = 0.0,
                                tau = 0.0,
                                theta = 0.5,
                                verbose = verbose,
                                use_kkt_restart = use_kkt_restart,
                                kkt_restart_freq = kkt_restart_freq,
                                use_duality_gap_restart = use_duality_gap_restart,
                                duality_gap_restart_freq = duality_gap_restart_freq,
                                check_terminate_freq = check_terminate_freq,
                                print_freq = print_freq,
                                time_limit = time_limit);
    # set the info struct
    info = PDHGCLPInfo(iter = 0,
                        convergeInfo = Vector{PDHGCLPConvergeInfo}([]),
                        infeaInfo = Vector{PDHGCLPInfeaInfo}([]),
                        time = 0.0,
                        start_time = time(),
                        restart_used = 0,
                        exit_status = :continue,
                        pObj = 0.0,
                        dObj = 0.0,
                        exit_code = 0);
    sol = Solution(x = x, y = y, params = params, info = info, dual_sol_temp = dual_sol_temp);

    solver = rpdhgSolver(
        data = data,
        sol = sol,
        primalMV! = x -> println("primalMV! not implemented"),
        adjointMV! = x -> println("adjointMV! not implemented"),
        AtAMV! = x -> println("AtAMV! not implemented"),
        addCoeffd! = x-> println("addCoeffd! not implemented"),
        dotCoeffd = x-> println("dotCoeffd not implemented")
    )

    if use_preconditioner
        # scale data
        nrm1G = CUDA.norm(solver.data.coeff.d_G, 1)
        nrmInfG = CUDA.norm(solver.data.coeff.d_G, Inf)
        @info ("initial nrm1G: $(nrm1G), initial nrmInfG: $(nrmInfG)")
        @info ("initial nrm1c: $(CUDA.norm(solver.data.d_c, 1))")
        @info ("initial nrm1h: $(CUDA.norm(solver.data.coeff.d_h, 1))")
        @info ("initial nrmInfc: $(CUDA.norm(solver.data.d_c, Inf))")
        if rescaling_method == :ruiz
            rescale_problem!(
                l_inf_ruiz_iterations = 10,
                pock_chambolle_alpha = nothing,
                data = solver.data,
                Dr_product = solver.data.diagonal_scale.Dr_product.x,
                Dl_product = solver.data.diagonal_scale.Dl_product.y,
                sol = solver.sol.x,
                dual_sol = solver.sol.y
            )
        elseif rescaling_method == :pock_chambolle
            rescale_problem!(
                l_inf_ruiz_iterations = -1,
                pock_chambolle_alpha = 1.0,
                data = solver.data,
                Dr_product = solver.data.diagonal_scale.Dr_product.x,
                Dl_product = solver.data.diagonal_scale.Dl_product.y,
                sol = solver.sol.x,
                dual_sol = solver.sol.y
            )
        elseif rescaling_method == :ruiz_pock_chambolle
            rescale_problem!(
                l_inf_ruiz_iterations = 10,
                pock_chambolle_alpha = 1.0,
                data = solver.data,
                Dr_product = solver.data.diagonal_scale.Dr_product.x,
                Dl_product = solver.data.diagonal_scale.Dl_product.y,
                sol = solver.sol.x,
                dual_sol = solver.sol.y,
                variable_rescaling = solver.data.diagonal_scale.Dr_temp.x,
                constraint_rescaling_G = solver.data.diagonal_scale.Dl_temp.y
            )
        else
            throw(ArgumentError("The rescaling method is not defined, two choices: :ruiz, :pock_chambolle, :ruiz_pock_chambolle"))
        end
        println("scale_preconditioner")
        scale_preconditioner!(
            Dr_product = solver.data.diagonal_scale.Dr_product.x,
            Dl_product = solver.data.diagonal_scale.Dl_product.y,
            Dr_product_inv_normalized = solver.data.diagonal_scale.Dr_product_inv_normalized.x,
            Dr_product_normalized = solver.data.diagonal_scale.Dr_product_normalized.x,
            Dl_product_inv_normalized = solver.data.diagonal_scale.Dl_product_inv_normalized.y,
            Dr_product_inv_normalized_squared = solver.data.diagonal_scale.Dr_product_inv_normalized_squared.x,
            Dr_product_normalized_squared = solver.data.diagonal_scale.Dr_product_normalized_squared.x,
            Dl_product_inv_normalized_squared = solver.data.diagonal_scale.Dl_product_inv_normalized_squared.y,
            primal_sol = solver.sol.x,
            dual_sol = solver.sol.y,
            primalConstScale = solver.data.diagonal_scale.primalConstScale,
            dualConstScale = solver.data.diagonal_scale.dualConstScale
        )
        if verbose == 2
            @info ("max Dr_product:$(CUDA.maximum(solver.data.diagonal_scale.Dr_product.x))")
            @info ("norm Dr_product:$(CUDA.norm(solver.data.diagonal_scale.Dr_product.x, 2))")
            @info ("max Dl_product:$(CUDA.maximum(solver.data.diagonal_scale.Dl_product.y))")
            @info ("norm Dl_product:$(CUDA.norm(solver.data.diagonal_scale.Dl_product.y, 2))")
            @info ("after scaling, norm1 c: $(CUDA.norm(solver.data.d_c, 1))")
            @info ("after scaling, normInf c: $(CUDA.norm(solver.data.d_c, Inf))")
            @info ("after scaling, norm1 G: $(CUDA.norm(solver.data.coeff.d_G, 1))")
            @info ("after scaling, normInf G: $(CUDA.norm(solver.data.coeff.d_G, Inf))")
        end
    end # end if use_preconditioner

    solver_start_time = time()
    setFunctionPointerSolver!(solver)
    # calculate the initial step size, for binary search
    # solver.data.GlambdaMax, solver.data.GlambdaMax_flag = power_method!(solver.data.coeffTrans, solver.data.coeff, solver.AtAMV!, dual_sol_lag_struct)
    # @info ("sqrt(max eigenvalue of GtG): $(solver.data.GlambdaMax)")
    # if solver.data.GlambdaMax_flag == 0
    #     solver.sol.params.sigma = 0.9 / solver.data.GlambdaMax
    #     solver.sol.params.tau = 0.9 / solver.data.GlambdaMax
    # else
    #     solver.sol.params.sigma = 0.8 / solver.data.GlambdaMax
    #     solver.sol.params.tau = 0.8 / solver.data.GlambdaMax
    # end
    solver.sol.params.sigma = 0.9
    solver.sol.params.tau = 0.9

    # new a struct to save convergence information
    push!(sol.info.convergeInfo, PDHGCLPConvergeInfo())
    push!(sol.info.convergeInfo, PDHGCLPConvergeInfo())
    push!(sol.info.infeaInfo, PDHGCLPInfeaInfo()) # infeasibility info for one sequence
    push!(sol.info.infeaInfo, PDHGCLPInfeaInfo()) # infeasibility info for the other sequence
    if use_preconditioner
        recover_solution!(
            data = solver.data,
            Dr_product = solver.data.diagonal_scale.Dr_product.x,
            Dl_product = solver.data.diagonal_scale.Dl_product.y,
            sol = solver.sol.x,
            dual_sol = solver.sol.y
        )
        converge_info_calculation_diagonal!(solver = solver,
                                            primal_sol = sol.x.recovered_primal.primal_sol,
                                            dual_sol = sol.y.recovered_dual.dual_sol,
                                            slack = sol.y.slack,
                                            dual_sol_temp = sol.y,
                                            converge_info = sol.info.convergeInfo[1])
    else
        converge_info_calculation(solver = solver,
                                primal_sol = sol.x.primal_sol,
                                dual_sol = sol.y.dual_sol,
                                slack = sol.y.slack,
                                dual_sol_temp = sol.y,
                                converge_info = sol.info.convergeInfo[1])
    end
    sol.info.start_time = solver_start_time;
    sol.info.time = time() - solver_start_time;
    if sol.params.verbose > 0
        @info ("==================================================")
    end
    # main loop, all data and variables are put in ``solver`` and ``sol`` two structs

    if method == :halpern 
        @info ("Start halpern method, which is not test yet")
    elseif method == :average
        if !use_preconditioner && !use_adaptive_restart && !use_adaptive_step_size_weight
            if verbose > 0
                @info ("Start average method without preconditioner, adaptive restart and adaptive step size weight")
            end
            main_loop! = pdhg_main_iter_average_no_restart!
        elseif use_preconditioner && !use_adaptive_restart && !use_adaptive_step_size_weight
            if verbose > 0
                @info ("Start average method with preconditioner, without adaptive restart and adaptive step size weight")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_no_restarts!
        elseif use_preconditioner && use_adaptive_restart && !use_adaptive_step_size_weight
            if verbose > 0
                @info ("Start average method with preconditioner, adaptive restart and without adaptive step size weight")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_adaptive_restarts!
        elseif use_preconditioner && use_adaptive_restart && use_adaptive_step_size_weight && !use_resolving
            if verbose > 0
                @info ("Start average method with preconditioner, adaptive restart and adaptive step size weight")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight!
        elseif use_preconditioner && use_adaptive_restart && use_adaptive_step_size_weight && use_resolving && !use_accelerated && !use_aggressive
            if verbose > 0
                @info ("Start average method with preconditioner, adaptive restart, adaptive step size weight and resolving")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving!
        elseif use_preconditioner && use_adaptive_restart && use_adaptive_step_size_weight && use_resolving && use_accelerated && !use_aggressive
            if verbose > 0
                @info ("Start average method with preconditioner, adaptive restart, adaptive step size weight and accelerated")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_accelerated!
        elseif use_preconditioner && use_adaptive_restart && use_adaptive_step_size_weight && use_resolving && !use_accelerated && use_aggressive
            if verbose > 0
                @info ("Start average method with preconditioner, adaptive restart, adaptive step size weight, resolving and accelerated")
            end
            main_loop! = pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_aggressive!
        else
            throw(ArgumentError("The combination of options is not supported."))
        end
    end
    if verbose > 0
        printInfo(infoAll = sol.info);
    end
    if verbose > 0
        CUDA.memory_status()
    end
    main_loop!(solver = solver)
    if verbose > 0
        CUDA.memory_status()
    end
    destroy_cublas_handle(handle)
    @info ("===============================================")
    infoSummary(info = sol.info)
    if use_preconditioner
        @info (" norm(sol.x.recovered_primal.primal_sol.x, Inf): $(CUDA.norm(sol.x.recovered_primal.primal_sol.x, Inf))")
        @info (" norm(sol.y.recovered_dual.dual_sol.y, Inf): $(CUDA.norm(sol.y.recovered_dual.dual_sol.y, Inf))")
    else
        @info (" norm(sol.x.primal_sol.x, Inf): $(CUDA.norm(sol.x.primal_sol.x, Inf))")
        @info (" norm(sol.y.dual_sol.y, Inf): $(CUDA.norm(sol.y.dual_sol.y, Inf))")
    end
    @info ("time for projection: $(time_proj)")
    @info ("time for iterative: $(time_iterative)")
    @info ("time for restart check: $(time_restart_check)")
    @info ("time for exit check: $(time_exit_check)")
    @info ("time for print info calculation: $(time_print_info_cal)")
    if logfile_name !== nothing
        close(logfile)
    end
    GC.gc()
    CUDA.reclaim()
    return solver.sol
end # end rpdhg_cpu_solve