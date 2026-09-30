
function approximate_cal_diagonal!(; solver::rpdhgSolver, h1::primalVector, h2::dualVector,
    slack::solVecPrimal, dual_sol_temp::solVecDual, t::rpdhg_float, 
    primal::primalVector, dual::dualVector, 
    tau::rpdhg_float, sigma::rpdhg_float)
    # modify `slack.primal_sol` and `dual_sol_temp.dual_sol`
    # slack.primal_sol.x .= primal.x .+ (t * tau / 2) * h1.x
    axpyz(slack.primal_sol.x, (t * tau / 2), h1.x, primal.x, solver.data.n)
    # here use solver.data.x is different with slack
    projection_start = time_proj
    solver.sol.x.proj_diagonal!(slack.primal_sol, solver.sol.x, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
    global time_proj_dual_slack += time_proj - projection_start
    # dual_sol_temp.dual_sol.y .= dual.y .+ (t * sigma / 2) * h2.y
    axpyz(dual_sol_temp.dual_sol.y, (t * sigma / 2), h2.y, dual.y, solver.data.m)
    projection_start = time_proj
    solver.sol.y.proj_diagonal!(dual_sol_temp.dual_sol, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
    global time_proj_dual_slack += time_proj - projection_start
end

"""
calculate the normalized duality gap
   rho(r, z)
"""
function binary_search_duality_gap_diagonal!(; solver::rpdhgSolver, 
    r::rpdhg_float, primal::primalVector, dual::dualVector, slack::solVecPrimal,
    dual_sol_temp::solVecDual, tau::rpdhg_float, sigma::rpdhg_float,
    maxIter::Integer = 1000, tol::rpdhg_float = 1e-6)
    t = solver.sol.info.binarySearch_t0
    tRight = t
    tLeft = 0.0
    h1 = slack.primal_sol_mean # no copy
    solver.adjointMV!(solver.data.coeff, dual, h1.x)
    h1.x .-= solver.data.d_c
    h2 = dual_sol_temp.dual_sol_mean # no copy
    solver.primalMV!(solver.data.coeff, primal.x, h2)
    h2.y .*= -1.0 
    solver.addCoeffd!(solver.data.coeff, h2, 1.0);
    for k = 1: maxIter
        approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2, 
            slack = slack, dual_sol_temp = dual_sol_temp, 
            t = t, primal = primal, dual = dual, 
            tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x .- primal.x
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y .- dual.y
        value_temp = Mnorm(solver = solver, x = slack.primal_sol_lag.x,
                    y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
                    AxTemp = dual_sol_temp.dual_sol_lag)
        if value_temp > r && value_temp < r + 1e-6
            tRight = t
            tLeft = max(t - 0.01, 0.0)
            approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2, 
                slack = slack, dual_sol_temp = dual_sol_temp, 
                t = tLeft, primal = primal, dual = dual, 
                tau = tau, sigma = sigma)
                slack.primal_sol_lag.x .= slack.primal_sol.x .- primal.x
                dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y .- dual.y
            value_temp = Mnorm(solver = solver, x = slack.primal_sol_lag.x,
                        y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
                        AxTemp = dual_sol_temp.dual_sol_lag)
            if value_temp < r
                # f(tRight) > 0, f(tLeft) < 0 and tRight - tLeft < 0.01
                break
            end
        end
        if value_temp < r && value_temp > r - 1e-6
            tLeft = t
            tRight = t + 0.01
            approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2, 
                slack = slack, dual_sol_temp = dual_sol_temp, 
                t = tRight, primal = primal, dual = dual, 
                tau = tau, sigma = sigma)
            slack.primal_sol_lag.x .= slack.primal_sol.x .- primal.x
            dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y .- dual.y
            value_temp = Mnorm(solver = solver, x = slack.primal_sol_lag.x,
                        y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
                        AxTemp = dual_sol_temp.dual_sol_lag)
            if value_temp > r
                # f(tRight) > 0, f(tLeft) < 0 and tRight - tLeft < 0.01
                break
            end
        end
        if value_temp > r
            if k == 1
                break
            else
                tRight = t, tLeft = t / 2, break
            end
        end
        t *= 2
    end
    tMid = (tRight + tLeft) / 2
    count = 0
    while tRight - tLeft > tol
        tMid = (tRight + tLeft) / 2
        approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2,
                        slack = slack, dual_sol_temp = dual_sol_temp,
                        t = tMid, primal = primal, dual = dual,
                        tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x .- primal.x
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y .- dual.y
        if Mnorm(solver = solver, x = slack.primal_sol_lag.x,
             y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
             AxTemp = dual_sol_temp.dual_sol_lag) < r
            tLeft = tMid
        else
            tRight = tMid
        end
        count += 1
        if count > 1000000
            break
        end
    end
    solver.sol.info.binarySearch_t0 = tMid
    # slack.primal_sol_lag.x .= slack.primal_sol.x .- primal.x
    # dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y .- dual.y
    calculate_diff(dual_sol_temp.dual_sol.y, dual.y, dual_sol_temp.dual_sol_temp.y, solver.data.m, slack.primal_sol.x, primal.x, slack.primal_sol_lag.x, solver.data.n)
    val1 = dot(h1.x, slack.primal_sol_lag.x)
    val2 = dot(h2.y, dual_sol_temp.dual_sol_temp.y)
    val = (val1 + val2) / r
    return val
end

function kkt_error_diagonal!(; omega::rpdhg_float, convergence_info::PDHGCLPConvergeInfo)
    return sqrt(omega^2 * convergence_info.l_2_abs_primal_res^2 + convergence_info.l_2_abs_dual_res^2 / omega^2 + convergence_info.abs_gap^2)
    # return max(max(convergence_info.l_inf_rel_primal_res, convergence_info.l_inf_rel_dual_res), convergence_info.rel_gap)
end



function converge_info_calculation_diagonal!(; solver::rpdhgSolver, primal_sol::primalVector, dual_sol::dualVector, slack::solVecPrimal, dual_sol_temp::solVecDual, converge_info::PDHGCLPConvergeInfo)
    pObj = dot(primal_sol.x, solver.data.raw_data.c)
    solver.adjointMV!(solver.data.raw_data.coeff, dual_sol, slack.primal_sol.x)
    AtyInf = norm(slack.primal_sol.x, Inf)
    AtyNrm1 = norm(slack.primal_sol.x, 1)
    # slack.primal_sol.x .= solver.data.raw_data.c .- slack.primal_sol.x;
    slack.primal_sol.x .*= -1.0
    slack.primal_sol.x .+= solver.data.raw_data.c
    slack.primal_sol_lag.x .= max.(0, slack.primal_sol.x)
    slack.primal_sol_mean.x .= min.(0, slack.primal_sol.x)

    dObj = solver.dotCoeffd(solver.data.raw_data.coeff, dual_sol)
    dObj += (solver.data.raw_data.bl_finite' * slack.primal_sol_lag.xbox + solver.data.raw_data.bu_finite' * slack.primal_sol_mean.xbox)
    abs_gap = abs(pObj - dObj)
    rel_gap = abs_gap / (1 + max(abs(pObj), abs(dObj)));

    # dual_sol_temp.dual_sol_mean = Gx
    solver.primalMV!(solver.data.raw_data.coeff, primal_sol.x, dual_sol_temp.dual_sol_mean);
    AxInf = norm(dual_sol_temp.dual_sol_mean.y, Inf)
    AxNrm1 = norm(dual_sol_temp.dual_sol_mean.y, 1)
    solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
    dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
    projection_start = time_proj
    solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
    global time_proj_dual_slack += time_proj - projection_start
    solver.data.diagonal_scale.Dl_temp.y .= dual_sol_temp.dual_sol_mean.y .- dual_sol_temp.dual_sol_lag.y


    
    slack.primal_sol_lag.x .= slack.primal_sol.x
    projection_start = time_proj
    solver.sol.x.slack_proj!(slack.primal_sol, slack)
    global time_proj_dual_slack += time_proj - projection_start
    sInf = norm(slack.primal_sol.x, Inf)
    sNrm1 = norm(slack.primal_sol.x, 1)

    l_2_abs_primal_res = norm(solver.data.diagonal_scale.Dl_temp.y);
    l_2_rel_primal_res = l_2_abs_primal_res / (1 + max(sNrm1, max(solver.data.raw_data.hNrm1, AxNrm1)));

    l_inf_abs_primal_res = CUDA.maximum(CUDA.abs.(solver.data.diagonal_scale.Dl_temp.y));
    l_inf_rel_primal_res = l_inf_abs_primal_res / (1 + max(sInf, max(solver.data.raw_data.hNrmInf, AxInf)));

    solver.data.diagonal_scale.Dr_temp.x .= slack.primal_sol.x .- slack.primal_sol_lag.x
    l_2_abs_dual_res = norm(solver.data.diagonal_scale.Dr_temp.x);
    l_2_rel_dual_res = l_2_abs_dual_res / (1 + max(solver.data.raw_data.cNrm1, AtyNrm1));
    l_inf_abs_dual_res = CUDA.maximum(CUDA.abs.(solver.data.diagonal_scale.Dr_temp.x));
    l_inf_rel_dual_res = l_inf_abs_dual_res / (1 + max(solver.data.raw_data.cNrmInf, AtyInf));
    
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

function infeasibility_info_calculation_diagonal!(; solver::rpdhgSolver,
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
        # pObj = dot(primal_ray.x, solver.data.raw_data.c)
        pObj = dot(primal_ray.x, solver.data.raw_data.c)
    end
    if l_inf_primal_ray_infeasibility === nothing && pObj < 0.0
        # dual_sol_temp.dual_sol_mean = Gx
        solver.primalMV!(solver.data.raw_data.coeff, primal_ray.x, dual_sol_temp.dual_sol_mean);
        solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
        dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
        projection_start = time_proj
        solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
        global time_proj_dual_slack += time_proj - projection_start
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol_mean.y .- dual_sol_temp.dual_sol_lag.y
        l_inf_primal_ray_infeasibility = CUDA.maximum(CUDA.abs.(dual_sol_temp.dual_sol_temp.y));
    end

    if dObj === nothing && l_inf_dual_ray_infeasibility === nothing
        dObj = solver.dotCoeffd(solver.data.raw_data.coeff, dual_ray)
        solver.adjointMV!(solver.data.raw_data.coeff, dual_ray, slack.primal_sol.x)
        slack.primal_sol.x .= solver.data.raw_data.c .- slack.primal_sol.x;
        slack.primal_sol_lag.x .= CUDA.max.(0.0, slack.primal_sol.x)
        slack.primal_sol_mean.x .= CUDA.min.(0.0, slack.primal_sol.x)
        # primal_lag_part = @view slack.primal_sol_lag.x[1:solver.data.nb]
        # primal_mean_part = @view slack.primal_sol_mean[1:solver.data.nb]
        dObj += (solver.data.raw_data.bl_finite' * slack.primal_sol_lag.xbox + solver.data.raw_data.bu_finite' * slack.primal_sol_mean.xbox)
        if dObj > 0
            slack.primal_sol_lag.x .= slack.primal_sol.x
            projection_start = time_proj
            solver.sol.x.slack_proj!(slack.primal_sol, slack)
            global time_proj_dual_slack += time_proj - projection_start
            solver.data.diagonal_scale.Dr_temp.x .= slack.primal_sol.x .- slack.primal_sol_lag.x
            l_inf_dual_ray_infeasibility = CUDA.maximum(CUDA.abs.(solver.data.diagonal_scale.Dr_temp.x));
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
    infeasibility_info.primal_ray_norm = norm(primal_ray.x, Inf)
    infeasibility_info.dual_ray_norm = norm(dual_ray.y, Inf)
    if rand() < 0.25
        push!(infeasibility_info.primalObj_trend, pObj_seq)
        push!(infeasibility_info.dualObj_trend, dObj_seq)
    end
    return
end



function pdhg_one_iter_diagonal_rescaling!(; solver::rpdhgSolver, x::solVecPrimal, y::solVecDual, tau::rpdhg_float, sigma::rpdhg_float, slack::solVecPrimal, dual_sol_temp::solVecDual)
    _begin_projection_work_iteration!()
    x.primal_sol_lag.x .= x.primal_sol.x
    solver.adjointMV!(solver.data.coeff, y.dual_sol, slack.primal_sol_lag.x)
    x.primal_sol.x .-= tau .* solver.data.d_c
    x.primal_sol.x .+= tau .* slack.primal_sol_lag.x
    projection_start = time_proj
    x.proj_diagonal!(
        x.primal_sol,
        x,
        solver.data.diagonal_scale,
        abs_tol = solver.sol.params.proj_abs_tol,
        rel_tol = solver.sol.params.proj_rel_tol,
    )
    global time_proj_primal += time_proj - projection_start
    x.primal_sol_lag.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
    solver.primalMV!(solver.data.coeff, x.primal_sol_lag.x, dual_sol_temp.dual_sol_lag)
    solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_lag, -1.0)
    y.dual_sol.y .-= sigma .* dual_sol_temp.dual_sol_lag.y
    projection_start = time_proj
    y.proj_diagonal!(
        y.dual_sol,
        solver.data.diagonal_scale,
        abs_tol = solver.sol.params.proj_abs_tol,
        rel_tol = solver.sol.params.proj_rel_tol,
    )
    global time_proj_dual_slack += time_proj - projection_start
    _end_projection_work_iteration!()
    return;
end


function omega_norm_square(; x::CuArray, y::CuArray, omega::rpdhg_float)
    temp1 = omega * norm(x, 2)^2
    temp2 = norm(y, 2)^2 / omega
    return temp1 + temp2
end

function initialize_primal_weight(; solver::rpdhgSolver)
    hNrm2 = norm(solver.data.coeff.d_h, 2)
    cNrm2 = norm(solver.data.d_c, 2)
    # @info ("hNrm2: $(hNrm2), cNrm2: $(cNrm2)")
    if hNrm2 > 1e-10 && cNrm2 > 1e-10
        omega = cNrm2 / hNrm2
    else
        omega = 1.0
    end
    return omega
end


function dynamic_primal_weight_update(; x_diff::CuArray, y_diff::CuArray, omega::rpdhg_float, theta::rpdhg_float, restart_times::Integer, solver::rpdhgSolver)
    x_diff_norm = norm(x_diff, 2)
    y_diff_norm = norm(y_diff, 2)
    @info ("omega:$(omega), x_diff_norm:$(x_diff_norm), y_diff_norm:$(y_diff_norm)")
    # if restart_times % 40 == 0 && restart_times != 0 && (omega > 10000 || omega < 1/10000)
    #     return initialize_primal_weight(solver=solver)
    # end
    if (omega > 10000 || omega < 1/10000)
        if restart_times % 20 == 0 && restart_times != 0
            return initialize_primal_weight(solver=solver)
        end
    end
    if (omega > 30000 || omega < 1/30000)
        if restart_times % 10 == 0 && restart_times != 0
            return initialize_primal_weight(solver=solver)
        end
    end
    if x_diff_norm > 1e-10 && y_diff_norm > 1e-10
        omega_new = exp(theta * log(y_diff_norm / x_diff_norm) + (1-theta) * log(omega))
        return omega_new
    else
        return omega
    end
end

function exit_condition_check_diagonal!(; sol::Solution, solver::rpdhgSolver, dual_sol_temp::solVecDual)
    recover_solution!(data = solver.data, Dr_product = solver.data.diagonal_scale.Dr_product.x,
        Dl_product = solver.data.diagonal_scale.Dl_product.y, sol = sol.x,
        dual_sol = sol.y)
    converge_info_calculation_diagonal!(solver = solver,
        primal_sol = sol.x.recovered_primal.primal_sol,
        dual_sol = sol.y.recovered_dual.dual_sol,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        converge_info = sol.info.convergeInfo[1])
    converge_info_calculation_diagonal!(solver = solver,
        primal_sol = sol.x.recovered_primal.primal_sol_mean,
        dual_sol = sol.y.recovered_dual.dual_sol_mean,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        converge_info = sol.info.convergeInfo[2])
    infeasibility_info_calculation_diagonal!(solver = solver,
        pObj = sol.info.convergeInfo[1].primal_objective,
        dObj = nothing,
        l_inf_primal_ray_infeasibility = nothing,
        l_inf_dual_ray_infeasibility = nothing,
        primal_ray_unscaled = sol.x.recovered_primal.primal_sol,
        dual_ray_unscaled = sol.y.recovered_dual.dual_sol,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        infeasibility_info = sol.info.infeaInfo[1],
        pObj_seq = sol.info.convergeInfo[1].primal_objective,
        dObj_seq = sol.info.convergeInfo[1].dual_objective)
    solver.data.diagonal_scale.Dl_temp.y .= sol.y.recovered_dual.dual_sol.y .- sol.y.recovered_dual.dual_sol_mean.y
    solver.data.diagonal_scale.Dr_temp.x .= sol.x.recovered_primal.primal_sol.x .- sol.x.recovered_primal.primal_sol_mean.x
    infeasibility_info_calculation_diagonal!(solver = solver, 
        pObj = nothing, dObj = nothing,
        l_inf_primal_ray_infeasibility = nothing,
        l_inf_dual_ray_infeasibility = nothing,
        primal_ray_unscaled = solver.data.diagonal_scale.Dr_temp,
        dual_ray_unscaled = solver.data.diagonal_scale.Dl_temp,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        infeasibility_info = sol.info.infeaInfo[2],
        pObj_seq = sol.info.convergeInfo[1].primal_objective,
        dObj_seq = sol.info.convergeInfo[1].dual_objective)
    sol.info.time = time() - sol.info.start_time
    termination_result = check_termination_criteria(
        info = sol.info,
        params = sol.params,
    )

    # `check_termination_criteria` accepts either the current sequence or its
    # running mean.  The old MOI path always exported the current sequence,
    # even when convergenceInfo[2] (the mean) triggered termination.  Select
    # the actual terminating candidate and then recompute its residuals/slack
    # so the returned vectors and reported metrics describe the same point.
    current_info = sol.info.convergeInfo[1]
    mean_info = sol.info.convergeInfo[2]
    current_error = max(
        current_info.l_inf_rel_primal_res,
        current_info.l_inf_rel_dual_res,
        current_info.rel_gap,
    )
    mean_error = max(
        mean_info.l_inf_rel_primal_res,
        mean_info.l_inf_rel_dual_res,
        mean_info.rel_gap,
    )
    use_mean = if termination_result == :optimal
        mean_info.status == :optimal &&
            (current_info.status != :optimal || mean_error < current_error)
    elseif termination_result == :time_limit || termination_result == :max_iter
        mean_error < current_error
    else
        false
    end
    if use_mean
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        sol.x.recovered_primal.primal_sol.x .=
            sol.x.recovered_primal.primal_sol_mean.x
        sol.y.recovered_dual.dual_sol.y .=
            sol.y.recovered_dual.dual_sol_mean.y
    end
    if termination_result == :optimal ||
       termination_result == :time_limit ||
       termination_result == :max_iter
        converge_info_calculation_diagonal!(
            solver = solver,
            primal_sol = sol.x.recovered_primal.primal_sol,
            dual_sol = sol.y.recovered_dual.dual_sol,
            slack = sol.y.slack,
            dual_sol_temp = dual_sol_temp,
            converge_info = sol.info.convergeInfo[1],
        )
        sol.info.pObj = sol.info.convergeInfo[1].primal_objective
        sol.info.dObj = sol.info.convergeInfo[1].dual_objective

        # Candidate selection above changes the point represented by
        # convergeInfo[1].  Its status was assigned before this final residual
        # calculation, so an :optimal status is valid only if the exported
        # point still satisfies the requested tolerances.
        if termination_result == :optimal
            optimality_criteria_met(
                rel_tol = sol.params.rel_tol,
                abs_tol = sol.params.abs_tol,
                info = sol.info.convergeInfo[1],
            )
            final_status = sol.info.convergeInfo[1].status
            if final_status == :optimal
                sol.info.exit_status = :optimal
                sol.info.exit_code = 0
            elseif final_status == :numerical_error_nan
                sol.info.exit_status = :numerical_error_nan
                sol.info.exit_code = 8
                termination_result = :numerical_error_nan
            elseif final_status == :numerical_error_inf
                sol.info.exit_status = :numerical_error_inf
                sol.info.exit_code = 9
                termination_result = :numerical_error_inf
            else
                sol.info.exit_status = :continue
                sol.info.exit_code = 7
                termination_result = :continue
            end
        end
    end
    return termination_result
end

function restart_threshold_calculation_diagonal!(; sol::Solution, solver::rpdhgSolver, primal_sol_0::primalVector, dual_sol_0::dualVector, dual_sol_temp::solVecDual, primal_sol_0_lag::CuArray, dual_sol_0_lag::CuArray, restart_duality_gap_flag::Bool)
    if restart_duality_gap_flag
        # duality gap restart
        solver.data.diagonal_scale.Dr_temp.x .= primal_sol_0.x .- primal_sol_0_lag
        solver.data.diagonal_scale.Dl_temp.y .= dual_sol_0.y .- dual_sol_0_lag
        Mnorm_restart_right = Mnorm(solver = solver,
                                    x = solver.data.diagonal_scale.Dr_temp.x,
                                    y = solver.data.diagonal_scale.Dl_temp.y,
                                    tau = sol.params.tau,
                                    sigma = sol.params.sigma,
                                    AxTemp = dual_sol_temp.dual_sol_mean)
        sol.info.normalized_duality_gap_r = Mnorm_restart_right
        sol.info.normalized_duality_gap_restart_threshold = binary_search_duality_gap_diagonal!(solver = solver,
                    r = Mnorm_restart_right,
                    primal = primal_sol_0,
                    dual = dual_sol_0,
                    slack = sol.y.slack,
                    dual_sol_temp = dual_sol_temp,
                    tau = sol.params.tau,
                    sigma = sol.params.sigma)
        if sol.info.normalized_duality_gap_restart_threshold < 0   
            @info ("use kkt error to calculate restart threshold")
            sol.info.restart_duality_gap_flag = false
        end
        # sol.info.normalized_duality_gap_restart_threshold = rhoVal / exp(1)
    else
        # kkt error restart
        recover_solution!(data = solver.data, Dr_product = solver.data.diagonal_scale.Dr_product.x,
                Dl_product = solver.data.diagonal_scale.Dl_product.y, sol = sol.x,
                dual_sol = sol.y)
        converge_info_calculation_diagonal!(solver = solver,
                primal_sol = sol.x.recovered_primal.primal_sol,
                dual_sol = sol.y.recovered_dual.dual_sol,
                slack = sol.y.slack,
                dual_sol_temp = dual_sol_temp,
                converge_info = sol.info.convergeInfo[1])
        kkt_error_restart_threshold = kkt_error_diagonal!(omega = sol.info.omega, convergence_info = sol.info.convergeInfo[1])
        sol.info.kkt_error_restart_threshold = kkt_error_restart_threshold
    end
end

@inline restart_merit(value::Real) = isfinite(value) ? Float64(value) : Inf

"""Return whether either side of the problem contains an EXP/dual-EXP cone."""
@inline function contains_exponential_cones(sol)
    primal = sol.x.primal_sol
    dual = sol.y.dual_sol
    return !isempty(primal.exp_cone_indices_start) ||
           !isempty(primal.dual_exp_cone_indices_start) ||
           !isempty(dual.exp_cone_indices_start) ||
           !isempty(dual.dual_exp_cone_indices_start)
end

"""
Return the restart candidate index using the legacy tie policy: when enabled,
the running mean wins a tie with the current point, while Halpern must be
strictly better. Candidate switches are eligibility masks; at least one of
current, mean, and Halpern must be enabled.

Indices are current=1, mean=2, and Halpern=3.
"""
function restart_candidate_index(
    current_merit::Real,
    mean_merit::Real,
    halpern_merit::Real,
    use_halpern::Bool,
    use_mean::Bool = true,
    force_mean::Bool = false,
    use_current::Bool = true,
)
    # For EXP problems, paper-era artificial/force restarts kept the running
    # mean as the next epoch's starting point. This override is intentionally
    # independent of the other candidates only when mean itself is enabled.
    force_mean && use_mean && return 2
    !(use_current || use_mean || use_halpern) && throw(ArgumentError(
        "at least one restart candidate must be enabled",
    ))
    selected = 0
    best = Inf
    if use_current
        selected = 1
        best = restart_merit(current_merit)
    end
    if use_mean
        mean = restart_merit(mean_merit)
        # Preserve the legacy current-vs-mean tie rule.
        if selected == 0 || mean <= best
            selected = 2
            best = mean
        end
    end
    halpern = use_halpern ? restart_merit(halpern_merit) : Inf
    if halpern < best
        selected = 3
    end
    return selected
end

"""
Install the best restart candidate into the running-mean buffers. The next
restart epoch already initializes both its anchor and main point from these
buffers, so no additional trajectory mutation is needed here.
"""
function install_restart_candidate!(
    sol::Solution,
    halpern_primal::Union{primalVector,Nothing},
    halpern_dual::Union{dualVector,Nothing},
    merits,
    use_halpern::Bool;
    reason::AbstractString,
    force_mean::Bool = false,
)
    halpern_enabled =
        use_halpern && halpern_primal !== nothing && halpern_dual !== nothing
    selected = restart_candidate_index(
        merits[1],
        merits[2],
        merits[3],
        halpern_enabled,
        sol.params.use_mean_restart_candidate,
        force_mean,
        sol.params.use_current_restart_candidate,
    )
    if selected == 1
        sol.info.restart_trigger_ergodic += 1
        sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
        sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
        selected_name = "current"
    elseif selected == 3
        sol.info.restart_trigger_halpern += 1
        sol.x.primal_sol_mean.x .= halpern_primal.x
        sol.y.dual_sol_mean.y .= halpern_dual.y
        selected_name = "halpern"
    else
        sol.info.restart_trigger_mean += 1
        selected_name = "mean"
    end
    if force_mean
        @info "restart candidate selected" reason selected_name forced_mean=true
    else
        @info "restart candidate selected" reason selected_name current_merit=merits[1] current_enabled=sol.params.use_current_restart_candidate mean_merit=merits[2] mean_enabled=sol.params.use_mean_restart_candidate halpern_merit=(halpern_enabled ? merits[3] : Inf) halpern_enabled
    end
    return selected
end

function enabled_restart_candidate_indices(sol::Solution, use_halpern::Bool)
    indices = Int[]
    sol.params.use_current_restart_candidate && push!(indices, 1)
    sol.params.use_mean_restart_candidate && push!(indices, 2)
    use_halpern && push!(indices, 3)
    isempty(indices) && throw(ArgumentError(
        "no enabled restart candidate is available in this solver path",
    ))
    return indices
end

function enabled_restart_merit(sol::Solution, merits, use_halpern::Bool)
    indices = enabled_restart_candidate_indices(sol, use_halpern)
    return minimum(restart_merit(merits[index]) for index in indices)
end

"""
Evaluate current, running-mean, and optional Halpern candidates with the same
KKT merit. Existing diagonal scratch vectors hold the recovered Halpern point
only for the duration of its metric calculation, avoiding another persistent
pair of GPU vectors.
"""
function evaluate_restart_kkt_candidates!(
    sol::Solution,
    solver::rpdhgSolver,
    dual_sol_temp::solVecDual,
    halpern_primal::Union{primalVector,Nothing},
    halpern_dual::Union{dualVector,Nothing},
    halpern_converge_info::Union{PDHGCLPConvergeInfo,Nothing},
    use_halpern::Bool,
    evaluate_halpern::Bool = use_halpern,
)
    recover_solution!(
        data = solver.data,
        Dr_product = solver.data.diagonal_scale.Dr_product.x,
        Dl_product = solver.data.diagonal_scale.Dl_product.y,
        sol = sol.x,
        dual_sol = sol.y,
    )
    converge_info_calculation_diagonal!(
        solver = solver,
        primal_sol = sol.x.recovered_primal.primal_sol,
        dual_sol = sol.y.recovered_dual.dual_sol,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        converge_info = sol.info.convergeInfo[1],
    )
    converge_info_calculation_diagonal!(
        solver = solver,
        primal_sol = sol.x.recovered_primal.primal_sol_mean,
        dual_sol = sol.y.recovered_dual.dual_sol_mean,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        converge_info = sol.info.convergeInfo[2],
    )
    sol.info.kkt_error[1] = kkt_error_diagonal!(
        omega = sol.info.omega,
        convergence_info = sol.info.convergeInfo[1],
    )
    sol.info.kkt_error[2] = kkt_error_diagonal!(
        omega = sol.info.omega,
        convergence_info = sol.info.convergeInfo[2],
    )

    halpern_enabled =
        use_halpern &&
        evaluate_halpern &&
        halpern_primal !== nothing &&
        halpern_dual !== nothing &&
        halpern_converge_info !== nothing
    if halpern_enabled
        # Reuse the diagonal scratch vector wrappers as recovered candidate
        # inputs. converge_info_calculation_diagonal! consumes each input
        # before reusing the corresponding scratch storage internally.
        solver.data.diagonal_scale.Dr_temp.x .=
            halpern_primal.x ./ solver.data.diagonal_scale.Dr_product.x
        solver.data.diagonal_scale.Dl_temp.y .=
            halpern_dual.y ./ solver.data.diagonal_scale.Dl_product.y
        converge_info_calculation_diagonal!(
            solver = solver,
            primal_sol = solver.data.diagonal_scale.Dr_temp,
            dual_sol = solver.data.diagonal_scale.Dl_temp,
            slack = sol.y.slack,
            dual_sol_temp = dual_sol_temp,
            converge_info = halpern_converge_info,
        )
        sol.info.kkt_error[3] = kkt_error_diagonal!(
            omega = sol.info.omega,
            convergence_info = halpern_converge_info,
        )
    else
        sol.info.kkt_error[3] = Inf
    end
    return sol.info.kkt_error
end

function evaluate_halpern_restart_kkt_candidate!(
    sol::Solution,
    solver::rpdhgSolver,
    dual_sol_temp::solVecDual,
    halpern_primal::Union{primalVector,Nothing},
    halpern_dual::Union{dualVector,Nothing},
    halpern_converge_info::Union{PDHGCLPConvergeInfo,Nothing},
    use_halpern::Bool,
)
    halpern_enabled =
        use_halpern &&
        halpern_primal !== nothing &&
        halpern_dual !== nothing &&
        halpern_converge_info !== nothing
    if !halpern_enabled
        sol.info.kkt_error[3] = Inf
        return Inf
    end
    solver.data.diagonal_scale.Dr_temp.x .=
        halpern_primal.x ./ solver.data.diagonal_scale.Dr_product.x
    solver.data.diagonal_scale.Dl_temp.y .=
        halpern_dual.y ./ solver.data.diagonal_scale.Dl_product.y
    converge_info_calculation_diagonal!(
        solver = solver,
        primal_sol = solver.data.diagonal_scale.Dr_temp,
        dual_sol = solver.data.diagonal_scale.Dl_temp,
        slack = sol.y.slack,
        dual_sol_temp = dual_sol_temp,
        converge_info = halpern_converge_info,
    )
    sol.info.kkt_error[3] = kkt_error_diagonal!(
        omega = sol.info.omega,
        convergence_info = halpern_converge_info,
    )
    return sol.info.kkt_error[3]
end

function restart_condition_check_diagonal!(;
    sol::Solution,
    solver::rpdhgSolver,
    primal_sol_0::primalVector,
    dual_sol_0::dualVector,
    dual_sol_temp::solVecDual,
    inner_iter::Int,
    restart_duality_gap_flag::Bool,
    halpern_primal::Union{primalVector,Nothing} = nothing,
    halpern_dual::Union{dualVector,Nothing} = nothing,
    halpern_converge_info::Union{PDHGCLPConvergeInfo,Nothing} = nothing,
)
    use_halpern =
        sol.params.use_halpern &&
        halpern_primal !== nothing &&
        halpern_dual !== nothing
    if restart_duality_gap_flag
        solver.data.diagonal_scale.Dr_temp.x .= sol.x.primal_sol.x .- primal_sol_0.x
        solver.data.diagonal_scale.Dl_temp.y .= sol.y.dual_sol.y .- dual_sol_0.y
        Mnorm_restart_left = Mnorm(solver = solver,
                            x = solver.data.diagonal_scale.Dr_temp.x,
                            y = solver.data.diagonal_scale.Dl_temp.y,
                            tau = sol.params.tau,
                            sigma = sol.params.sigma,
                            AxTemp = dual_sol_temp.dual_sol_mean)
        rhoVal_left = binary_search_duality_gap_diagonal!(solver = solver,
                                                r = Mnorm_restart_left,
                                                primal = sol.x.primal_sol,
                                                dual = sol.y.dual_sol,
                                                slack = sol.y.slack,
                                                dual_sol_temp = dual_sol_temp,
                                                tau = sol.params.tau,
                                                sigma = sol.params.sigma)
        rhoVal_left_mean = binary_search_duality_gap_diagonal!(solver = solver,
                                                r = Mnorm_restart_left,
                                                primal = sol.x.primal_sol_mean,
                                                dual = sol.y.dual_sol_mean,
                                                slack = sol.y.slack,
                                                dual_sol_temp = dual_sol_temp,
                                                tau = sol.params.tau,   
                                                sigma = sol.params.sigma)
        sol.info.normalized_duality_gap[1] = rhoVal_left
        sol.info.normalized_duality_gap[2] = rhoVal_left_mean
        sol.info.normalized_duality_gap[3] = Inf
        merits = sol.info.normalized_duality_gap
        if use_halpern
            merits[3] = binary_search_duality_gap_diagonal!(
                solver = solver,
                r = Mnorm_restart_left,
                primal = halpern_primal,
                dual = halpern_dual,
                slack = sol.y.slack,
                dual_sol_temp = dual_sol_temp,
                tau = sol.params.tau,
                sigma = sol.params.sigma,
            )
        end
        candidate_merit = enabled_restart_merit(sol, merits, use_halpern)
        # condition 1
        if candidate_merit <
           sol.params.beta_suff * sol.info.normalized_duality_gap_restart_threshold
            sol.info.restart_used = sol.info.restart_used + 1
            install_restart_candidate!(
                sol,
                halpern_primal,
                halpern_dual,
                merits,
                use_halpern;
                reason = "duality-gap condition 1",
            )
            @info "restart due to condition 1"
            return true
        end # end if rhoVal_left
        # condition 2
        if candidate_merit <
           sol.params.beta_necessary * sol.info.normalized_duality_gap_restart_threshold
            sol.info.restart_used = sol.info.restart_used + 1
            install_restart_candidate!(
                sol,
                halpern_primal,
                halpern_dual,
                merits,
                use_halpern;
                reason = "duality-gap condition 2",
            )
            @info "restart due to condition 2"
            return true
        end # end if rhoVal_left
    else
        kkt_error_lag = copy(sol.info.kkt_error)
        merits = evaluate_restart_kkt_candidates!(
            sol,
            solver,
            dual_sol_temp,
            halpern_primal,
            halpern_dual,
            halpern_converge_info,
            use_halpern,
            use_halpern,
        )
        candidate_merit = enabled_restart_merit(sol, merits, use_halpern)
        # condition 1
        if candidate_merit <
           sol.params.beta_suff_kkt * sol.info.kkt_error_restart_threshold
            sol.info.restart_used = sol.info.restart_used + 1
            install_restart_candidate!(
                sol,
                halpern_primal,
                halpern_dual,
                merits,
                use_halpern;
                reason = "KKT condition 1",
            )
            @info "restart due to condition 1 kkt"
            return true
        end # end if rhoVal_left
        # condition 2
        enabled_indices = enabled_restart_candidate_indices(sol, use_halpern)
        if any(
            restart_merit(merits[index]) <
                sol.params.beta_necessary_kkt *
                sol.info.kkt_error_restart_threshold &&
            restart_merit(merits[index]) >
                restart_merit(kkt_error_lag[index])
            for index in enabled_indices
        )
            sol.info.restart_used = sol.info.restart_used + 1
            install_restart_candidate!(
                sol,
                halpern_primal,
                halpern_dual,
                merits,
                use_halpern;
                reason = "KKT condition 2",
            )
            @info "restart due to condition 2 kkt"
            return true
        end # end if rhoVal_left
    end
    # condition 3
    if inner_iter >= sol.info.iter * sol.params.beta_artificial && sol.info.iter > 100000
        @info ("restart due to artificial condition, beta_artificial = $(sol.params.beta_artificial), inner_iter = $inner_iter, iter = $(sol.info.iter)")
        sol.info.restart_used = sol.info.restart_used + 1
        if contains_exponential_cones(sol) &&
           sol.params.use_mean_restart_candidate
            install_restart_candidate!(
                sol,
                nothing,
                nothing,
                sol.info.kkt_error,
                false;
                reason = "artificial condition (EXP paper-era mean)",
                force_mean = true,
            )
        else
            merits = restart_duality_gap_flag ?
                sol.info.normalized_duality_gap :
                sol.info.kkt_error
            if use_halpern && restart_duality_gap_flag
                solver.data.diagonal_scale.Dr_temp.x .=
                    sol.x.primal_sol.x .- primal_sol_0.x
                solver.data.diagonal_scale.Dl_temp.y .=
                    sol.y.dual_sol.y .- dual_sol_0.y
                radius = Mnorm(
                    solver = solver,
                    x = solver.data.diagonal_scale.Dr_temp.x,
                    y = solver.data.diagonal_scale.Dl_temp.y,
                    tau = sol.params.tau,
                    sigma = sol.params.sigma,
                    AxTemp = dual_sol_temp.dual_sol_mean,
                )
                merits[3] = binary_search_duality_gap_diagonal!(
                    solver = solver,
                    r = radius,
                    primal = halpern_primal,
                    dual = halpern_dual,
                    slack = sol.y.slack,
                    dual_sol_temp = dual_sol_temp,
                    tau = sol.params.tau,
                    sigma = sol.params.sigma,
                )
            elseif use_halpern
                evaluate_halpern_restart_kkt_candidate!(
                    sol,
                    solver,
                    dual_sol_temp,
                    halpern_primal,
                    halpern_dual,
                    halpern_converge_info,
                    use_halpern,
                )
            end
            install_restart_candidate!(
                sol,
                halpern_primal,
                halpern_dual,
                merits,
                use_halpern;
                reason = "artificial condition",
            )
        end
        return true
    end # end if inner_iter
    return false
end




function resolving_pessimistic_step!(; solver::rpdhgSolver, 
    primal_sol_0::primalVector, dual_sol_0::dualVector,
    primal_sol_0_lag::CuArray, dual_sol_0_lag::CuArray,
    dual_sol_temp::solVecDual, primal_sol_change::primalVector,
    dual_sol_change::dualVector, primal_sol_diff::primalVector,
    dual_sol_diff::dualVector, random_check::Bool,
    halpern_primal::Union{primalVector,Nothing} = nothing,
    halpern_dual::Union{dualVector,Nothing} = nothing,
    halpern_converge_info::Union{PDHGCLPConvergeInfo,Nothing} = nothing)
    # pdhg + adaptive restart+adaptive stepsize
    sol = solver.sol
    if sol.params.verbose > 0
        @info ("pessimistic step adaptive restart resolving...")
    end
    sol.info.iter_stepsize = 0
    solver.data.GlambdaMax, solver.data.GlambdaMax_flag = power_method!(solver.data.coeffTrans, solver.data.coeff, solver.AtAMV!, dual_sol_change)
    @info ("sqrt(max eigenvalue of GtG): $(solver.data.GlambdaMax)")
    if solver.data.GlambdaMax_flag == 0
        solver.sol.params.sigma = 0.9 / solver.data.GlambdaMax
        solver.sol.params.tau = 0.9 / solver.data.GlambdaMax
    else
        solver.sol.params.sigma = 0.8 / solver.data.GlambdaMax
        solver.sol.params.tau = 0.8 / solver.data.GlambdaMax
    end
    sol.info.omega = initialize_primal_weight(solver = solver)
    sol.info.omega = clamp(sol.info.omega, 0.8, 1.2)
    # use best feasible solution as the initial point
    sol.x.primal_sol_mean.x .= sol.x_best.x
    sol.y.dual_sol_mean.y .= sol.y_best.y
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        time_start = time()
        if sol.params.use_kkt_restart
            restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = false)
        else
            if sol.info.restart_duality_gap_flag
                restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = true)
            else
                restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = false)
            end
        end
        global time_restart_check += time() - time_start
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        while true
            inner_iter += 1
            sol.info.iter += 1
            time_start = time()
            pdhg_one_iter_diagonal_rescaling!(solver = solver, x = sol.x,
                         y = sol.y, tau = sol.params.tau / sol.info.omega,
                         sigma = sol.params.sigma * sol.info.omega, slack = sol.y.slack,
                         dual_sol_temp = dual_sol_temp)
            if sol.params.use_halpern
                halpern_primal === nothing &&
                    error("Halpern-only restart requires a primal candidate buffer")
                halpern_dual === nothing &&
                    error("Halpern-only restart requires a dual candidate buffer")
                # The resolving path has no extrapolation. Reuse the fused
                # update to preserve its legacy arithmetic mean while also
                # forming the anchored restart-only Halpern candidate.
                reflection_update(
                    sol.x.primal_sol.x,
                    sol.x.primal_sol_lag.x,
                    sol.x.primal_sol_mean.x,
                    halpern_primal.x,
                    primal_sol_0.x,
                    sol.y.dual_sol.y,
                    sol.y.dual_sol_lag.y,
                    sol.y.dual_sol_mean.y,
                    halpern_dual.y,
                    dual_sol_0.y,
                    0.0,
                    solver.data.n,
                    solver.data.m,
                    inner_iter,
                    Float64(inner_iter),
                    1.0,
                    sol.params.use_weighted_average,
                    false,
                    true,
                    false,
                )
            else
                average_seq(primal_sol_mean = sol.x.primal_sol_mean.x, primal_sol = sol.x.primal_sol.x, primal_n = solver.data.n, dual_sol_mean = sol.y.dual_sol_mean.y, dual_sol = sol.y.dual_sol.y, dual_n = solver.data.m, inner_iter = inner_iter)
            end
            global time_iterative += time() - time_start
            if sol.info.iter %  sol.params.kkt_restart_freq == 0 && sol.params.use_kkt_restart
                time_start = time()
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false, halpern_primal = halpern_primal, halpern_dual = halpern_dual, halpern_converge_info = halpern_converge_info)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
                global time_restart_check += time() - time_start
            end # end if check restart

            if sol.info.iter %  sol.params.duality_gap_restart_freq == 0 && sol.params.use_duality_gap_restart
                time_start = time()
                if sol.info.restart_duality_gap_flag
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = true, halpern_primal = halpern_primal, halpern_dual = halpern_dual, halpern_converge_info = halpern_converge_info)
                        break
                    end
                else
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false, halpern_primal = halpern_primal, halpern_dual = halpern_dual, halpern_converge_info = halpern_converge_info)
                        break
                    end
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
                global time_restart_check += time() - time_start
            end # end if check restart

            if sol.info.iter %  sol.params.check_terminate_freq == 0 || (sol.params.verbose > 0 && sol.info.iter % sol.params.print_freq == 0) || (random_check && Int(ceil(sol.info.time)) % 3179 == 0)
                time_start = time()
                exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
                if sol.params.adaptive_projection_tolerance !== nothing
                    convergence = sol.info.convergeInfo[1]
                    average_kkt_error = (
                        convergence.l_2_rel_primal_res +
                        convergence.l_2_rel_dual_res +
                        convergence.rel_gap
                    ) / 3
                    sol.params.proj_abs_tol = next_projection_tolerance(
                        sol.params.proj_abs_tol,
                        average_kkt_error,
                        sol.params.adaptive_projection_tolerance,
                        sol.params.proj_base_tol,
                    )
                    sol.params.proj_rel_tol = next_projection_tolerance(
                        sol.params.proj_rel_tol,
                        average_kkt_error,
                        sol.params.adaptive_projection_tolerance,
                        sol.params.proj_base_tol,
                    )
                    push!(
                        sol.params.projection_tolerance_history,
                        sol.params.proj_abs_tol,
                    )
                end
                global time_exit_check += time() - time_start
                if sol.params.verbose > 0   
                    time_start = time()
                    if sol.info.iter % sol.params.print_freq == 0
                        sol.info.time = time() - sol.info.start_time;
                        if sol.params.verbose == 1
                            printInfo(infoAll = sol.info)
                        end # end if verbose
                        if sol.params.verbose == 2
                            primal_sol_change.x .-= sol.x.recovered_primal.primal_sol.x
                            dual_sol_change.y .-= sol.y.recovered_dual.dual_sol.y
                            nrm2_x_recovered = norm(sol.x.recovered_primal.primal_sol.x, 2)
                            nrm2_y_recovered = norm(sol.y.recovered_dual.dual_sol.y, 2)
                            nrmInf_x_recovered = norm(sol.x.recovered_primal.primal_sol.x, Inf)
                            nrmInf_y_recovered = norm(sol.y.recovered_dual.dual_sol.y, Inf)
                            diff_nrm_x_recovered = norm(primal_sol_change.x, 2)
                            diff_nrm_y_recovered = norm(dual_sol_change.y, 2)
                            printInfolevel2(infoAll = sol.info, diff_nrm_x_recovered = diff_nrm_x_recovered, diff_nrm_y_recovered = diff_nrm_y_recovered, nrm2_x_recovered = nrm2_x_recovered, nrm2_y_recovered = nrm2_y_recovered, nrmInf_x_recovered = nrmInf_x_recovered, nrmInf_y_recovered = nrmInf_y_recovered, params = sol.params)
                            primal_sol_change.x .= sol.x.recovered_primal.primal_sol.x
                            dual_sol_change.y .= sol.y.recovered_dual.dual_sol.y
                        end # end if verbose
                    end # end if print_freq
                    global time_print_info_cal += time() - time_start
                end # end if verbose
                if sol.info.exit_status != :continue
                    return
                end
            end
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega, restart_times = sol.info.restart_used, solver = solver)
        sol.info.omega = clamp(sol.info.omega, 0.8, 1.2)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
    sol.info.exit_code = 1
    return;
end

function update_best_sol!(; sol::Solution)
    if sol.info.convergeInfo[1].l_2_rel_primal_res < sol.primal_res_best
        sol.primal_res_best = sol.info.convergeInfo[1].l_2_rel_primal_res
        sol.x_best.x .= sol.x.primal_sol.x
    end
    if sol.info.convergeInfo[1].l_2_rel_dual_res < sol.dual_res_best
        sol.dual_res_best = sol.info.convergeInfo[1].l_2_rel_dual_res
        sol.y_best.y .= sol.y.dual_sol.y
    end
end






function pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(; solver::rpdhgSolver,
    x::solVecPrimal,
    y::solVecDual,
    eta::rpdhg_float,
    omega::rpdhg_float,
    info::PDHGCLPInfo,
    slack::solVecPrimal,
    dual_sol_temp::solVecDual,
    primal_sol_change::primalVector,
    dual_sol_change::dualVector,
    random_check::Bool,
    use_adaptive_step::Bool = DEFAULT_USE_ADAPTIVE_STEP)
    eta = use_adaptive_step ? max(eta, 1e-6) : max(eta, 1e-30)
    x.primal_sol_lag.x .= x.primal_sol.x
    y.dual_sol_lag.y .= y.dual_sol.y

    primal_sol_diff = slack.primal_sol_lag
    primal_sol_temp = slack.primal_sol_mean
    dual_sol_diff = dual_sol_temp.dual_sol_lag

    sol = solver.sol
    decay_step_size_ratio = use_adaptive_step ?
        min(0.9 + 0.1 * log10(sol.info.max_kkt_error * 10.0^5), 1.0) :
        1.0
    # println("decay_step_size_ratio: ", decay_step_size_ratio)
    # decay_step_size_ratio = 1.0
    count = 0
    while true
        _begin_projection_work_iteration!()
        info.iter += 1
        info.iter_stepsize += 1
        ## primal step
        time_start = time()
        solver.adjointMV!(solver.data.coeff, y.dual_sol_lag, primal_sol_diff.x)
        sol.params.tau = eta / omega
        # primal_sol_diff.x .-= solver.data.d_c
        # x.primal_sol.x .= x.primal_sol_lag.x .+ sol.params.tau * primal_sol_diff.x
        # x.primal_sol.x .= x.primal_sol_lag.x .+ sol.params.tau * (primal_sol_diff.x .- solver.data.d_c)
        
        primal_step_size = sol.params.tau * decay_step_size_ratio
        primal_update(x.primal_sol.x, x.primal_sol_lag.x, primal_sol_diff.x, solver.data.d_c, primal_step_size, solver.data.n)
        global time_iterative += time() - time_start
        projection_start = time_proj
        x.proj_diagonal!(x.primal_sol, x, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
        global time_proj_primal += time_proj - projection_start
        ## dual step
        time_start = time()
        # primal_sol_diff.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
        extrapolation_update(primal_sol_diff.x, x.primal_sol.x, x.primal_sol_lag.x, solver.data.n)
        solver.primalMV!(solver.data.coeff, primal_sol_diff.x, dual_sol_diff)
        sol.params.sigma = eta * omega
        # y.dual_sol.y .= y.dual_sol_lag.y .- sol.params.sigma * (dual_sol_diff.y .- solver.data.coeff.d_h)
        dual_step_size = sol.params.sigma * decay_step_size_ratio
        dual_update(y.dual_sol.y, y.dual_sol_lag.y, dual_sol_diff.y, solver.data.coeff.d_h, dual_step_size, solver.data.m)
        global time_iterative += time() - time_start
        projection_start = time_proj
        y.proj_diagonal!(y.dual_sol, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
        global time_proj_dual_slack += time_proj - projection_start
        _end_projection_work_iteration!()
        eta_bar = eta
        eta_prime = eta
        if use_adaptive_step
            ## calculate eta_bar and eta_prime
            time_start = time()
            # dual_sol_diff.y .= y.dual_sol.y - y.dual_sol_lag.y
            # primal_sol_diff.x .= x.primal_sol.x .- x.primal_sol_lag.x
            calculate_diff(y.dual_sol.y, y.dual_sol_lag.y, dual_sol_diff.y, solver.data.m, x.primal_sol.x, x.primal_sol_lag.x, primal_sol_diff.x, solver.data.n)
            eta_bar_numerator = omega_norm_square(x = primal_sol_diff.x, y = dual_sol_diff.y, omega = sol.info.omega)
            solver.adjointMV!(solver.data.coeff, dual_sol_diff, primal_sol_temp.x)
            eta_bar_denominator = 2 * abs(dot(primal_sol_diff.x, primal_sol_temp.x))
            eta_bar = eta_bar_numerator / (eta_bar_denominator + positive_zero)
            eta_prime = min((1 - (info.iter_stepsize+1)^(-0.3)) * eta_bar, (1+(info.iter_stepsize+1)^(-0.6)) * eta)
            global time_iterative += time() - time_start
        end
        flag_random_check = (random_check && Int(ceil(sol.info.time)) % 3179 == 0)
        if sol.info.iter %  sol.params.check_terminate_freq == 0 || (sol.params.verbose > 0 && sol.info.iter % sol.params.print_freq == 0) || flag_random_check
            exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
            sol.info.max_kkt_error = max(max(sol.info.convergeInfo[1].l_2_rel_primal_res, sol.info.convergeInfo[1].l_2_rel_dual_res), sol.info.convergeInfo[1].rel_gap)
            average_kkt_error = (sol.info.convergeInfo[1].l_2_rel_primal_res + sol.info.convergeInfo[1].l_2_rel_dual_res + sol.info.convergeInfo[1].rel_gap) / 3
            solver.sol.params.proj_abs_tol = next_projection_tolerance(
                solver.sol.params.proj_abs_tol,
                average_kkt_error,
                solver.sol.params.adaptive_projection_tolerance,
                solver.sol.params.proj_base_tol,
            )
            solver.sol.params.proj_rel_tol = next_projection_tolerance(
                solver.sol.params.proj_rel_tol,
                average_kkt_error,
                solver.sol.params.adaptive_projection_tolerance,
                solver.sol.params.proj_base_tol,
            )
            push!(
                solver.sol.params.projection_tolerance_history,
                solver.sol.params.proj_abs_tol,
            )
            # @info ("proj_abs_tol: $(solver.sol.params.proj_abs_tol), proj_rel_tol: $(solver.sol.params.proj_rel_tol)")
            if sol.params.verbose > 0
                time_start = time()
                if sol.info.iter % sol.params.print_freq == 0
                    sol.info.time = time() - sol.info.start_time;
                    if sol.params.verbose == 1
                        printInfo(infoAll = sol.info)
                    end # end if verbose
                    if sol.params.verbose == 2
                        primal_sol_change.x .-= sol.x.recovered_primal.primal_sol.x
                        dual_sol_change.y .-= sol.y.recovered_dual.dual_sol.y
                        nrm2_x_recovered = norm(sol.x.recovered_primal.primal_sol.x, 2)
                        nrm2_y_recovered = norm(sol.y.recovered_dual.dual_sol.y, 2)
                        nrmInf_x_recovered = norm(sol.x.recovered_primal.primal_sol.x, Inf)
                        nrmInf_y_recovered = norm(sol.y.recovered_dual.dual_sol.y, Inf)
                        diff_nrm_x_recovered = norm(primal_sol_change.x, 2)
                        diff_nrm_y_recovered = norm(dual_sol_change.y, 2)
                        printInfolevel2(infoAll = sol.info, diff_nrm_x_recovered = diff_nrm_x_recovered, diff_nrm_y_recovered = diff_nrm_y_recovered, nrm2_x_recovered = nrm2_x_recovered, nrm2_y_recovered = nrm2_y_recovered, nrmInf_x_recovered = nrmInf_x_recovered, nrmInf_y_recovered = nrmInf_y_recovered, params = sol.params)
                        primal_sol_change.x .= sol.x.recovered_primal.primal_sol.x
                        dual_sol_change.y .= sol.y.recovered_dual.dual_sol.y
                    end # end if verbose
                end # end if print_freq
                global time_print_info_cal += time() - time_start
            end # end if verbose
            if sol.info.exit_status != :continue
                return eta, eta_prime, false
            end
        end # end if print_freq
        if !use_adaptive_step
            return eta, eta, false
        end
        if eta <= eta_bar
            return eta, eta_prime, false
        end
        eta = eta_prime
        count += 1
        if count > 1000000
            solver.sol.params.proj_abs_tol = stalled_projection_tolerance(
                solver.sol.params.proj_abs_tol,
                solver.sol.params.adaptive_projection_tolerance,
            )
            solver.sol.params.proj_rel_tol = stalled_projection_tolerance(
                solver.sol.params.proj_rel_tol,
                solver.sol.params.adaptive_projection_tolerance,
            )
            push!(
                solver.sol.params.projection_tolerance_history,
                solver.sol.params.proj_abs_tol,
            )
            @info ("proj_abs_tol: $(solver.sol.params.proj_abs_tol), proj_rel_tol: $(solver.sol.params.proj_rel_tol)")
            return eta, eta_prime, true
        end
    end
    return;
end


function pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_aggressive!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepCopyPrimalVector(sol.x.primal_sol)
    dual_sol_0 = deepCopyDualVector(sol.y.dual_sol)
    primal_sol_0_lag = copy(sol.x.primal_sol.x)
    dual_sol_0_lag = copy(sol.y.dual_sol.y)
    dual_sol_temp = sol.dual_sol_temp
    primal_sol_change = deepCopyPrimalVector(sol.x.primal_sol)
    dual_sol_change = deepCopyDualVector(sol.y.dual_sol)
    # The disabled path aliases the read-only restart anchors. The CUDA kernel
    # does not write candidate buffers when use_halpern=false, so the ablation
    # incurs neither extra candidate storage nor global-memory writes.
    halpern_primal = sol.params.use_halpern ?
        deepCopyPrimalVector(sol.x.primal_sol) :
        primal_sol_0
    halpern_dual = sol.params.use_halpern ?
        deepCopyDualVector(sol.y.dual_sol) :
        dual_sol_0
    halpern_converge_info = PDHGCLPConvergeInfo()
    sol.info.omega = initialize_primal_weight(solver = solver)
    primal_sol_diff = deepCopyPrimalVector(sol.x.primal_sol)
    dual_sol_diff = deepCopyDualVector(sol.y.dual_sol)
    GInf = norm(solver.data.coeff.d_G, Inf)
    if sol.params.use_adaptive_step
        eta_prime = 1 / GInf
    else
        solver.data.GlambdaMax, solver.data.GlambdaMax_flag = power_method!(
            solver.data.coeffTrans,
            solver.data.coeff,
            solver.AtAMV!,
            dual_sol_change,
        )
        norm_G = max(solver.data.GlambdaMax, 1e-30)
        eta_prime = (solver.data.GlambdaMax_flag == 0 ? 0.9 : 0.8) / norm_G
        @info "Fixed step enabled: eta=$(eta_prime), estimated_norm_G=$(norm_G), power_method_flag=$(solver.data.GlambdaMax_flag)"
    end
    random_check = false
    for outer_iter = 1 : sol.params.max_outer_iter
        update_best_sol!(sol = sol)
        sol.info.max_kkt_error = max(max(sol.info.convergeInfo[1].l_2_rel_primal_res, sol.info.convergeInfo[1].l_2_rel_dual_res), sol.info.convergeInfo[1].rel_gap)
        sol.info.min_kkt_error = min(min(min(sol.info.convergeInfo[1].l_2_rel_primal_res, sol.info.convergeInfo[1].l_2_rel_dual_res), sol.info.convergeInfo[1].rel_gap), sol.info.min_kkt_error)
        if sol.info.min_kkt_error < 1e-5
            random_check = true
        end
        extra_coeff = -0.1 * log10(sol.info.max_kkt_error) + 0.2
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        if sol.params.use_restart
            time_start = time()
            if sol.params.use_kkt_restart
                restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = false)
            else
                if sol.info.restart_duality_gap_flag
                    restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = true)
                else
                    restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag, restart_duality_gap_flag = false)
                end
            end
            global time_restart_check += time() - time_start
        end
        flag_one = (outer_iter > 10 && sol.info.max_kkt_error > 1e+5)
        flag_two = (outer_iter > 80 && ((sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res) > 1e+10 || sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res < 1e-10))
        flag_three = (outer_iter > 10 && sol.params.tau < 1e-6 && sol.params.sigma < 1e-6)
        flag_four = (sol.info.iter > 1000000 && (sol.info.convergeInfo[1].rel_gap > 0.98 && sol.info.convergeInfo[1].l_2_rel_primal_res > 1.0 && sol.info.convergeInfo[1].l_2_rel_dual_res > 1.0))
        if sol.params.use_restart && sol.params.use_resolving &&
           (flag_one || flag_two || flag_three || flag_four)
            if flag_one
                @info ("resolving pessimistic step adaptive restart resolving... since flag_one")
            elseif flag_two
                @info ("resolving pessimistic step adaptive restart resolving... since flag_two")
                @info ("l_2_rel_primal_res: $(sol.info.convergeInfo[1].l_2_rel_primal_res), l_2_rel_dual_res: $(sol.info.convergeInfo[1].l_2_rel_dual_res)")
                @info ("outer_iter: $(outer_iter)")
            elseif flag_three
                @info ("resolving pessimistic step adaptive restart resolving... since flag_three, step size too small")
            elseif flag_four
                @info ("resolving pessimistic step adaptive restart resolving... since flag_four, it almost fails")
            end 
            resolving_pessimistic_step!(solver = solver,
                primal_sol_0 = primal_sol_0,
                dual_sol_0 = dual_sol_0,
                primal_sol_0_lag = primal_sol_0_lag,
                dual_sol_0_lag = dual_sol_0_lag,
                dual_sol_temp = dual_sol_temp,
                primal_sol_change = primal_sol_change, dual_sol_change = dual_sol_change,
                primal_sol_diff = primal_sol_diff, dual_sol_diff = dual_sol_diff,
                random_check = random_check,
                halpern_primal = halpern_primal,
                halpern_dual = halpern_dual,
                halpern_converge_info = halpern_converge_info)
            if sol.info.exit_status != :continue
                return
            end
        end
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        eta_cum = 0.0
        while true
            inner_iter += 1
            eta, eta_prime, force_restart = pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(
                    solver = solver, x = sol.x, y = sol.y, eta = eta_prime, 
                    omega = sol.info.omega, info = sol.info, slack = sol.y.slack, 
                    dual_sol_temp = dual_sol_temp, primal_sol_change = primal_sol_change, 
                    dual_sol_change = dual_sol_change, random_check = random_check,
                    use_adaptive_step = sol.params.use_adaptive_step
                )
            if sol.info.exit_status != :continue
                return
            end
            eta = max(eta, 1e-30)
            # sol.x.primal_sol.x .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.x.primal_sol.x .- extra_coeff * sol.x.primal_sol_lag.x) + 1 / (inner_iter + 2) * sol.x.primal_sol.x
            # sol.y.dual_sol.y .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.y.dual_sol.y .- extra_coeff * sol.y.dual_sol_lag.y) + 1 / (inner_iter + 2) * sol.y.dual_sol.y
            # # average update
            # sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
            # sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
            reflection_update(
                sol.x.primal_sol.x,
                sol.x.primal_sol_lag.x,
                sol.x.primal_sol_mean.x,
                halpern_primal.x,
                primal_sol_0.x,
                sol.y.dual_sol.y,
                sol.y.dual_sol_lag.y,
                sol.y.dual_sol_mean.y,
                halpern_dual.y,
                dual_sol_0.y,
                extra_coeff,
                solver.data.n,
                solver.data.m,
                inner_iter,
                eta_cum,
                eta,
                sol.params.use_weighted_average,
                sol.params.use_reflection,
                sol.params.use_halpern,
                sol.params.use_inline_halpern,
            )
            eta_cum += eta
            if sol.params.use_restart &&
               ((inner_iter >= sol.info.iter * sol.params.beta_artificial && sol.info.iter > 10000) ||
                (force_restart && inner_iter > 100))
                @info ("restart due to artificial condition, beta_artificial = $(sol.params.beta_artificial), inner_iter = $inner_iter, iter = $(sol.info.iter) or force_restart")
                sol.info.restart_used = sol.info.restart_used + 1
                if contains_exponential_cones(sol) &&
                   sol.params.use_mean_restart_candidate
                    install_restart_candidate!(
                        sol,
                        nothing,
                        nothing,
                        sol.info.kkt_error,
                        false;
                        reason = "force/artificial condition (EXP paper-era mean)",
                        force_mean = true,
                    )
                else
                    merits = evaluate_restart_kkt_candidates!(
                        sol,
                        solver,
                        dual_sol_temp,
                        halpern_primal,
                        halpern_dual,
                        halpern_converge_info,
                        sol.params.use_halpern,
                    )
                    install_restart_candidate!(
                        sol,
                        halpern_primal,
                        halpern_dual,
                        merits,
                        sol.params.use_halpern;
                        reason = "force/artificial condition",
                    )
                end
                break
            end # end if inner_iter
            if sol.params.use_restart &&
               (sol.info.iter % sol.params.kkt_restart_freq == 0) &&
               sol.params.use_kkt_restart
                if restart_condition_check_diagonal!(
                    sol = sol,
                    solver = solver,
                    primal_sol_0 = primal_sol_0,
                    dual_sol_0 = dual_sol_0,
                    dual_sol_temp = dual_sol_temp,
                    inner_iter = inner_iter,
                    restart_duality_gap_flag = false,
                    halpern_primal = halpern_primal,
                    halpern_dual = halpern_dual,
                    halpern_converge_info = halpern_converge_info,
                )
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
            if sol.params.use_restart &&
               (sol.info.iter % sol.params.duality_gap_restart_freq == 0) &&
               sol.params.use_duality_gap_restart
                if sol.info.restart_duality_gap_flag
                    if restart_condition_check_diagonal!(
                        sol = sol,
                        solver = solver,
                        primal_sol_0 = primal_sol_0,
                        dual_sol_0 = dual_sol_0,
                        dual_sol_temp = dual_sol_temp,
                        inner_iter = inner_iter,
                        restart_duality_gap_flag = true,
                        halpern_primal = halpern_primal,
                        halpern_dual = halpern_dual,
                        halpern_converge_info = halpern_converge_info,
                    )
                        break
                    end
                else
                    if restart_condition_check_diagonal!(
                        sol = sol,
                        solver = solver,
                        primal_sol_0 = primal_sol_0,
                        dual_sol_0 = dual_sol_0,
                        dual_sol_temp = dual_sol_temp,
                        inner_iter = inner_iter,
                        restart_duality_gap_flag = false,
                        halpern_primal = halpern_primal,
                        halpern_dual = halpern_dual,
                        halpern_converge_info = halpern_converge_info,
                    )
                        break
                    end
                end
                
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        if sol.params.use_adaptive_step_size_weight
            sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega, restart_times = sol.info.restart_used, solver = solver)
        end
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end
