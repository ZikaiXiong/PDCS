
function approximate_cal_diagonal!(; solver::rpdhgSolver, h1::primalVector, h2::dualVector,
    slack::solVecPrimal, dual_sol_temp::solVecDual, t::rpdhg_float, 
    primal::primalVector, dual::dualVector, 
    tau::rpdhg_float, sigma::rpdhg_float)
    # modify `slack.primal_sol` and `dual_sol_temp.dual_sol`
    slack.primal_sol.x .= primal.x .+ (t * tau / 2) * h1.x
    # here use solver.data.x is different with slack
    solver.sol.x.proj_diagonal!(slack.primal_sol, solver.sol.x, solver.data.diagonal_scale)
    dual_sol_temp.dual_sol.y .= dual.y .+ (t * sigma / 2) * h2.y
    solver.sol.y.proj_diagonal!(dual_sol_temp.dual_sol, solver.data.diagonal_scale)
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
    solver.adjointMV!(solver.data.coeffTrans, dual, h1.x)
    h1.x .-= solver.data.c
    h2 = dual_sol_temp.dual_sol_mean # no copy
    solver.primalMV!(solver.data.coeff, primal.x, h2)
    h2.y .*= -1.0 
    solver.addCoeffd!(solver.data.coeff, h2, 1.0);
    for k = 1: maxIter
        approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2, 
            slack = slack, dual_sol_temp = dual_sol_temp, 
            t = t, primal = primal, dual = dual, 
            tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y - dual.y
        value_temp = Mnorm(solver = solver, x = slack.primal_sol_lag.x,
                    y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
                    AxTemp = dual_sol_temp.dual_sol_lag)
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
    while tRight - tLeft > tol
        tMid = (tRight + tLeft) / 2
        approximate_cal_diagonal!(solver = solver, h1 = h1, h2 = h2,
                        slack = slack, dual_sol_temp = dual_sol_temp,
                        t = tMid, primal = primal, dual = dual,
                        tau = tau, sigma = sigma)
        slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y - dual.y
        if Mnorm(solver = solver, x = slack.primal_sol_lag.x,
             y = dual_sol_temp.dual_sol_temp.y, tau = tau, sigma = sigma,
             AxTemp = dual_sol_temp.dual_sol_lag) < r
            tLeft = tMid
        else
            tRight = tMid
        end
    end
    solver.sol.info.binarySearch_t0 = tMid
    slack.primal_sol_lag.x .= slack.primal_sol.x - primal.x
    dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol.y - dual.y
    val1 = h1.x' * slack.primal_sol_lag.x
    val2 = h2.y' * dual_sol_temp.dual_sol_temp.y
    val = (val1 + val2) / r
    return val
end

function kkt_error_diagonal!(; omega::rpdhg_float, convergence_info::PDHGCLPConvergeInfo)
    # return sqrt(omega^2 * convergence_info.l_2_abs_primal_res^2 + convergence_info.l_2_abs_dual_res^2 / omega^2 + convergence_info.abs_gap^2)
    return max(max(convergence_info.l_inf_rel_primal_res, convergence_info.l_inf_rel_dual_res), convergence_info.rel_gap)
end



function converge_info_calculation_diagonal!(; solver::rpdhgSolver, primal_sol::primalVector, dual_sol::dualVector, slack::solVecPrimal, dual_sol_temp::solVecDual, converge_info::PDHGCLPConvergeInfo)
    pObj = dot(primal_sol.x, solver.data.raw_data.c)

    solver.adjointMV!(solver.data.raw_data.coeffTrans, dual_sol, slack.primal_sol.x)
    slack.primal_sol.x .= solver.data.raw_data.c .- slack.primal_sol.x;
    slack.primal_sol_lag.x .= max.(0, slack.primal_sol.x)
    slack.primal_sol_mean.x .= min.(0, slack.primal_sol.x)

    dObj = solver.dotCoeffd(solver.data.raw_data.coeff, dual_sol)
    dObj += (solver.data.raw_data.bl_finite' * slack.primal_sol_lag.xbox + solver.data.raw_data.bu_finite' * slack.primal_sol_mean.xbox)
    abs_gap = abs(pObj - dObj)
    rel_gap = abs_gap / (1 + abs(pObj) + abs(dObj));

    # dual_sol_temp.dual_sol_mean = Gx
    solver.primalMV!(solver.data.raw_data.coeff, primal_sol.x, dual_sol_temp.dual_sol_mean);
    solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
    dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
    solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
    solver.data.diagonal_scale.Dl_temp.y .= dual_sol_temp.dual_sol_mean.y .- dual_sol_temp.dual_sol_lag.y
    l_2_abs_primal_res = norm(solver.data.diagonal_scale.Dl_temp.y);
    l_2_rel_primal_res = l_2_abs_primal_res / (1 + solver.data.hNrm1);
    l_inf_abs_primal_res = maximum(abs.(solver.data.diagonal_scale.Dl_temp.y));
    l_inf_rel_primal_res = l_inf_abs_primal_res / (1 + solver.data.hNrmInf);
    
    slack.primal_sol_lag.x .= slack.primal_sol.x
    solver.sol.x.slack_proj!(slack.primal_sol, slack)
    solver.data.diagonal_scale.Dr_temp.x .= slack.primal_sol.x .- slack.primal_sol_lag.x
    l_2_abs_dual_res = norm(solver.data.diagonal_scale.Dr_temp.x);
    l_2_rel_dual_res = l_2_abs_dual_res / (1 + solver.data.raw_data.cNrm1);
    l_inf_abs_dual_res = maximum(abs.(solver.data.diagonal_scale.Dr_temp.x));
    l_inf_rel_dual_res = l_inf_abs_dual_res / (1 + solver.data.raw_data.cNrmInf);
    
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
        pObj = dot(primal_ray.x, solver.data.raw_data.c)
    end
    if l_inf_primal_ray_infeasibility === nothing && pObj < 0.0
        # dual_sol_temp.dual_sol_mean = Gx
        solver.primalMV!(solver.data.raw_data.coeff, primal_ray.x, dual_sol_temp.dual_sol_mean);
        solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
        dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
        solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
        dual_sol_temp.dual_sol_temp.y .= dual_sol_temp.dual_sol_mean.y .- dual_sol_temp.dual_sol_lag.y
        l_inf_primal_ray_infeasibility = maximum(abs.(dual_sol_temp.dual_sol_temp.y));
    end

    if dObj === nothing && l_inf_dual_ray_infeasibility === nothing
        dObj = solver.dotCoeffd(solver.data.coeff, dual_ray)
        solver.adjointMV!(solver.data.raw_data.coeffTrans, dual_ray, slack.primal_sol.x)
        slack.primal_sol.x .= solver.data.raw_data.c - slack.primal_sol.x;
        slack.primal_sol_lag.x .= max.(0.0, slack.primal_sol.x)
        slack.primal_sol_mean.x .= min.(0.0, slack.primal_sol.x)
        # primal_lag_part = @view slack.primal_sol_lag.x[1:solver.data.nb]
        # primal_mean_part = @view slack.primal_sol_mean[1:solver.data.nb]
        dObj += (solver.data.raw_data.bl' * slack.primal_sol_lag.xbox + solver.data.raw_data.bu' * slack.primal_sol_mean.xbox)
        if dObj > 0
            slack.primal_sol_lag.x .= slack.primal_sol.x
            solver.sol.x.slack_proj!(slack.primal_sol, slack)
            solver.data.diagonal_scale.Dr_temp.x .= slack.primal_sol.x .- slack.primal_sol_lag.x
            l_inf_dual_ray_infeasibility = maximum(abs.(solver.data.diagonal_scale.Dr_temp.x));
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
    ## debug
    # println("primal_ray_objective:", infeasibility_info.primal_ray_objective)
    # println("dual_ray_objective:", infeasibility_info.dual_ray_objective)
    # println("max_primal_ray_infeasibility:", infeasibility_info.max_primal_ray_infeasibility)
    # println("max_dual_ray_infeasibility:", infeasibility_info.max_dual_ray_infeasibility)
    return
end



function pdhg_one_iter_diagonal_rescaling!(; solver::rpdhgSolver, x::solVecPrimal, y::solVecDual, tau::rpdhg_float, sigma::rpdhg_float, slack::solVecPrimal, dual_sol_temp::solVecDual)
    x.primal_sol_lag.x .= x.primal_sol.x
    solver.adjointMV!(solver.data.coeffTrans, y.dual_sol, slack.primal_sol_lag.x)
    x.primal_sol.x .-= tau * solver.data.c
    x.primal_sol.x .+= tau * slack.primal_sol_lag.x
    x.proj_diagonal!(x.primal_sol, x, solver.data.diagonal_scale)
    x.primal_sol_lag.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
    solver.primalMV!(solver.data.coeff, x.primal_sol_lag.x, dual_sol_temp.dual_sol_lag)
    solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_lag, -1.0)
    y.dual_sol.y .-= sigma .* dual_sol_temp.dual_sol_lag.y
    y.proj_diagonal!(y.dual_sol, solver.data.diagonal_scale)
    return;
end


function omega_norm_square(; x::AbstractVector{rpdhg_float}, y::AbstractVector{rpdhg_float}, omega::rpdhg_float)
    temp1 = omega * norm(x, 2)^2
    temp2 = norm(y, 2)^2 / omega
    return temp1 + temp2
end

function initialize_primal_weight(; solver::rpdhgSolver)
    hNrm2 = norm(solver.data.coeff.h, 2)
    cNrm2 = norm(solver.data.c, 2)
    println("hNrm2:", hNrm2, " cNrm2:", cNrm2)
    if hNrm2 > 1e-10 && cNrm2 > 1e-10
        omega = cNrm2 / hNrm2
    else
        omega = 1.0
    end
    return omega
end


function dynamic_primal_weight_update(; x_diff::AbstractVector{rpdhg_float}, y_diff::AbstractVector{rpdhg_float}, omega::rpdhg_float, theta::rpdhg_float)
    x_diff_norm = norm(x_diff, 2)
    y_diff_norm = norm(y_diff, 2)
    println("omega:", omega, " x_diff_norm:", x_diff_norm, " y_diff_norm:", y_diff_norm)
    if x_diff_norm > 1e-10 && y_diff_norm > 1e-10
        omega_new = exp(theta * log(y_diff_norm / x_diff_norm) + (1-theta) * log(omega))
        return omega_new
    else
        return omega
    end
end

function pdhg_main_iter_halpern_diagonal_rescaling!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol)
    dual_sol_temp = deepcopy(sol.y)
    if sol.params.verbose == 2
        primal_sol_change = deepcopy(sol.x.primal_sol)
        dual_sol_change = deepcopy(sol.y.dual_sol)
    end
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag.x .= primal_sol_0.x
        dual_sol_0_lag.y .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol.x
        dual_sol_0.y .= sol.y.dual_sol.y
        solver.data.diagonal_scale.Dr_temp.x .= primal_sol_0.x - primal_sol_0_lag.x
        solver.data.diagonal_scale.Dl_temp.y .= dual_sol_0.y - dual_sol_0_lag.y
        Mnorm_restart_right = Mnorm(solver = solver,
                                    x = solver.data.diagonal_scale.Dr_temp.x,
                                    y = solver.data.diagonal_scale.Dl_temp.y,
                                    tau = sol.params.tau,
                                    sigma = sol.params.sigma,
                                    AxTemp = dual_sol_temp.dual_sol_mean)
        sol.info.normalized_duality_gap_r = Mnorm_restart_right
        sol.info.normalized_duality_gap_restart_threshold = binary_search_duality_gap(solver = solver,
                    r = Mnorm_restart_right,
                    primal = primal_sol_0,
                    dual = dual_sol_0,
                    slack = sol.y.slack,
                    dual_sol_temp = dual_sol_temp,
                    tau = sol.params.tau,
                    sigma = sol.params.sigma,
                    t0 = sol.info.binarySearch_t0)
        for inner_iter = 1:sol.params.max_inner_iter
            pdhg_one_iter_diagonal_rescaling!(solver = solver, x = sol.x,
                         y = sol.y, tau = sol.params.tau,
                         sigma = sol.params.sigma, slack = sol.y.slack,
                         dual_sol_temp = dual_sol_temp)
            # halpern update
            sol.x.primal_sol.x .= (sol.x.primal_sol.x * (inner_iter + 1) .+ primal_sol_0.x) / (inner_iter + 2)
            sol.y.dual_sol.y .= (sol.y.dual_sol.y * (inner_iter + 1) .+ dual_sol_0.y) / (inner_iter + 2)
            sol.info.iter += 1
            sol.info.iter_stepsize += 1
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
                end
            end # end if print_freq
            if sol.info.iter %  sol.params.check_terminate_freq == 0
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
                solver.data.diagonal_scale.Dl_temp.y .= sol.y.recovered_dual.dual_sol.y .- sol.y.recovered_dual.dual_sol_lag.y
                solver.data.diagonal_scale.Dr_temp.x .= sol.x.recovered_primal.primal_sol.x .- sol.x.recovered_primal.primal_sol_lag.x
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
                check_termination_criteria(info = sol.info, params = sol.params)
                if sol.info.exit_status != :continue
                    return
                end
            end
            if sol.info.iter %  sol.params.restart_check_freq == 0
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
                sol.info.normalized_duality_gap[1] = rhoVal_left
                if rhoVal_left < sol.info.normalized_duality_gap_restart_threshold
                    sol.info.restart_used = sol.info.restart_used + 1
                    sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
                    continue
                else
                    sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
                end # end if rhoVal_left
                if outer_iter == 1
                    continue
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
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
    solver.data.diagonal_scale.Dl_temp.y .= sol.y.recovered_dual.dual_sol.y - sol.y.recovered_dual.dual_sol_lag.y
    solver.data.diagonal_scale.Dr_temp.x .= sol.x.recovered_primal.primal_sol.x - sol.x.recovered_primal.primal_sol_lag.x
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
    check_termination_criteria(info = sol.info, params = sol.params)
end

function restart_threshold_calculation_diagonal!(; sol::Solution, solver::rpdhgSolver, primal_sol_0::primalVector, dual_sol_0::dualVector, dual_sol_temp::solVecDual, primal_sol_0_lag::AbstractVector{rpdhg_float}, dual_sol_0_lag::AbstractVector{rpdhg_float})
    if sol.info.restart_duality_gap_flag
        solver.data.diagonal_scale.Dr_temp.x .= primal_sol_0.x - primal_sol_0_lag
        solver.data.diagonal_scale.Dl_temp.y .= dual_sol_0.y - dual_sol_0_lag
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
            println("use kkt error to calculate restart threshold")
            sol.info.restart_duality_gap_flag = false
        end
        # sol.info.normalized_duality_gap_restart_threshold = rhoVal / exp(1)
    else
        recover_solution!(data = solver.data, Dr_product = solver.data.diagonal_scale.Dr_product.x,
                Dl_product = solver.data.diagonal_scale.Dl_product.y, sol = sol.x,
                dual_sol = sol.y)
        converge_info_calculation_diagonal!(solver = solver,
                primal_sol = sol.x.recovered_primal.primal_sol,
                dual_sol = sol.y.recovered_dual.dual_sol,
                slack = sol.y.slack,
                dual_sol_temp = dual_sol_temp,
                converge_info = sol.info.convergeInfo[1])
        kkt_error_restart_threshold = kkt_error_diagonal!(omega = 1.0, convergence_info = sol.info.convergeInfo[1])
        sol.info.kkt_error_restart_threshold = kkt_error_restart_threshold
    end
end

function restart_condition_check_diagonal!(; sol::Solution, solver::rpdhgSolver, primal_sol_0::primalVector, dual_sol_0::dualVector, dual_sol_temp::solVecDual, inner_iter::Int)
    if sol.info.restart_duality_gap_flag
        solver.data.diagonal_scale.Dr_temp.x .= sol.x.primal_sol.x - primal_sol_0.x
        solver.data.diagonal_scale.Dl_temp.y .= sol.y.dual_sol.y - dual_sol_0.y
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
    else
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
        kkt_error_ergodic = kkt_error_diagonal!(omega = sol.info.omega, convergence_info = sol.info.convergeInfo[1])
        kkt_error_mean = kkt_error_diagonal!(omega = sol.info.omega, convergence_info = sol.info.convergeInfo[2])
        sol.info.kkt_error[1] = kkt_error_ergodic
        sol.info.kkt_error[2] = kkt_error_mean
    end
    # condition 1
    if (sol.info.restart_duality_gap_flag && (rhoVal_left < sol.params.beta_suff * sol.info.normalized_duality_gap_restart_threshold || rhoVal_left_mean < sol.params.beta_suff * sol.info.normalized_duality_gap_restart_threshold)) ||
        (!sol.info.restart_duality_gap_flag && (kkt_error_ergodic < sol.params.beta_suff_kkt * sol.info.kkt_error_restart_threshold || kkt_error_mean < sol.params.beta_suff_kkt * sol.info.kkt_error_restart_threshold))
        sol.info.restart_used = sol.info.restart_used + 1
        if ((sol.info.normalized_duality_gap[1] < sol.info.normalized_duality_gap[2]) && sol.info.restart_duality_gap_flag) || (!sol.info.restart_duality_gap_flag && (sol.info.kkt_error[1] < sol.info.kkt_error[2]))
            sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
            sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
            sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
        else    
            sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
        end
        return true
    end # end if rhoVal_left
    # condition 2
    if (sol.info.restart_duality_gap_flag && (rhoVal_left < sol.params.beta_necessary * sol.info.normalized_duality_gap_restart_threshold || rhoVal_left_mean < sol.params.beta_necessary * sol.info.normalized_duality_gap_restart_threshold)) ||
        (!sol.info.restart_duality_gap_flag && (kkt_error_ergodic < sol.params.beta_necessary_kkt * sol.info.kkt_error_restart_threshold || kkt_error_mean < sol.params.beta_necessary_kkt * sol.info.kkt_error_restart_threshold))
        sol.info.restart_used = sol.info.restart_used + 1
        if ((sol.info.normalized_duality_gap[1] < sol.info.normalized_duality_gap[2]) && sol.info.restart_duality_gap_flag) || (!sol.info.restart_duality_gap_flag && (sol.info.kkt_error[1] < sol.info.kkt_error[2]))
            sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
            sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
            sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
        else
            sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
        end
        return true
    end # end if rhoVal_left
    # condition 3
    if inner_iter >= sol.info.iter * sol.params.beta_artificial
        sol.info.restart_used = sol.info.restart_used + 1
        return true
    end # end if inner_iter
    return false
end


function pdhg_main_iter_average_diagonal_rescaling_no_restarts!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol.x)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol.y)
    dual_sol_temp = deepcopy(sol.y)
    if sol.params.verbose == 2
        primal_sol_change = deepcopy(sol.x.primal_sol)
        dual_sol_change = deepcopy(sol.y.dual_sol)
    end
    for inner_iter = 1 : sol.params.max_outer_iter * sol.params.max_inner_iter
        pdhg_one_iter_diagonal_rescaling!(solver = solver, x = sol.x,
                         y = sol.y, tau = sol.params.tau,
                         sigma = sol.params.sigma, slack = sol.y.slack,
                         dual_sol_temp = dual_sol_temp)
        # average update
        sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * inner_iter .+ sol.x.primal_sol.x) / (inner_iter + 1)
        sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * inner_iter .+ sol.y.dual_sol.y) / (inner_iter + 1)
        sol.info.iter += 1
        sol.info.iter_stepsize += 1
        if sol.info.iter %  sol.params.check_terminate_freq == 0 || sol.info.iter % sol.params.print_freq == 0
            if sol.params.verbose > 0   
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
                end
            end # end if verbose
            exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
            if sol.info.exit_status != :continue
                return
            end
        end
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end



function pdhg_main_iter_average_diagonal_rescaling_adaptive_restarts!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol.x)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol.y)
    dual_sol_temp = deepcopy(sol.y)
    if sol.params.verbose == 2
        primal_sol_change = deepcopy(sol.x.primal_sol)
        dual_sol_change = deepcopy(sol.y.dual_sol)
    end
    # println("beta_suff: ", beta_suff, " beta_necessary: ", beta_necessary, " beta_artificial: ", beta_artificial)
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        # calculate restart threshold
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        inner_iter = 0
        while true
            inner_iter += 1
            pdhg_one_iter_diagonal_rescaling!(solver = solver, x = sol.x,
                         y = sol.y, tau = sol.params.tau,
                         sigma = sol.params.sigma, slack = sol.y.slack,
                         dual_sol_temp = dual_sol_temp)
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * inner_iter .+ sol.x.primal_sol.x) / (inner_iter + 1)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * inner_iter .+ sol.y.dual_sol.y) / (inner_iter + 1)
            sol.info.iter += 1
            sol.info.iter_stepsize += 1
            if sol.info.iter %  sol.params.check_terminate_freq == 0 || sol.info.iter % sol.params.print_freq == 0
                exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
                if sol.info.exit_status != :continue
                    return
                end
                if sol.params.verbose > 0   
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
                        end
                    end # end if print_freq
                end # end if verbose
            end
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
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
                                                                dual_sol_change::dualVector)
    x.primal_sol_lag.x .= x.primal_sol.x
    y.dual_sol_lag.y .= y.dual_sol.y

    primal_sol_diff = slack.primal_sol_lag
    primal_sol_temp = slack.primal_sol_mean
    dual_sol_diff = dual_sol_temp.dual_sol_lag

    sol = solver.sol
    while true  
        info.iter += 1
        info.iter_stepsize += 1
        ## primal step
        solver.adjointMV!(solver.data.coeffTrans, y.dual_sol_lag, primal_sol_diff.x)
        sol.params.tau = eta / omega
        primal_sol_diff.x .-= solver.data.c
        x.primal_sol.x .= x.primal_sol_lag.x .+ sol.params.tau * primal_sol_diff.x
        x.proj_diagonal!(x.primal_sol, x, solver.data.diagonal_scale)

        ## dual step
        primal_sol_diff.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
        solver.primalMV!(solver.data.coeff, primal_sol_diff.x, dual_sol_diff)
        solver.addCoeffd!(solver.data.coeff, dual_sol_diff, -1.0)
        sol.params.sigma = eta * omega
        y.dual_sol.y .= y.dual_sol_lag.y .- sol.params.sigma * dual_sol_diff.y
        y.proj_diagonal!(y.dual_sol, solver.data.diagonal_scale)

        ## calculate eta_bar and eta_prime
        dual_sol_diff.y .= y.dual_sol.y - y.dual_sol_lag.y
        primal_sol_diff.x .= x.primal_sol.x .- x.primal_sol_lag.x
        eta_bar_numerator = omega_norm_square(x = primal_sol_diff.x, y = dual_sol_diff.y, omega = sol.info.omega)
        solver.adjointMV!(solver.data.coeffTrans, dual_sol_diff, primal_sol_temp.x)
        eta_bar_denominator = 2 * abs(dot(primal_sol_diff.x, primal_sol_temp.x))
        eta_bar = eta_bar_numerator / (eta_bar_denominator + positive_zero)
        eta_prime = min((1 - (info.iter_stepsize+1)^(-0.3)) * eta_bar, (1+(info.iter_stepsize+1)^(-0.6)) * eta)

        if sol.info.iter %  sol.params.check_terminate_freq == 0 || (sol.params.verbose > 0 && sol.info.iter % sol.params.print_freq == 0)
            exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
            if sol.params.verbose > 0   
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
                    end
                end # end if print_freq
            end # end if verbose
            if sol.info.exit_status != :continue
                return eta, eta_prime
            end
        end
        if eta <= 0.99 * eta_bar
            # println("eta: ", eta, " eta_bar: ", eta_bar, " eta_prime: ", eta_prime)
            return eta, eta_prime
        end
        eta = eta_prime
    end
    return;
end

function pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol.x)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol.y)
    dual_sol_temp = deepcopy(sol.y)
    primal_sol_change = deepcopy(sol.x.primal_sol)
    dual_sol_change = deepcopy(sol.y.dual_sol)
    sol.info.omega = initialize_primal_weight(solver = solver)
    # println("initial omega: ", omega)
    primal_sol_diff = deepcopy(sol.x.primal_sol)
    dual_sol_diff = deepcopy(sol.y.dual_sol)
    GInf = norm(solver.data.coeff.G, Inf)
    eta_prime = 1 / GInf
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        eta_cum = 0.0
        while true
            inner_iter += 1
            eta, eta_prime = pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(
                    solver = solver, x = sol.x, y = sol.y, eta = eta_prime, 
                    omega = sol.info.omega, info = sol.info, slack = sol.y.slack, 
                    dual_sol_temp = dual_sol_temp, primal_sol_change = primal_sol_change, 
                    dual_sol_change = dual_sol_change
                )
            if sol.info.exit_status != :continue
                return 
            end
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
            eta_cum += eta
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end

function resolving_pessimistic_step!(; solver::rpdhgSolver, 
    primal_sol_0::primalVector, dual_sol_0::dualVector,
    primal_sol_0_lag::AbstractVector{rpdhg_float}, dual_sol_0_lag::AbstractVector{rpdhg_float},
    dual_sol_temp::solVecDual, primal_sol_change::primalVector,
    dual_sol_change::dualVector, primal_sol_diff::primalVector, dual_sol_diff::dualVector)
    # pdhg + adaptive restart+adaptive stepsize
    sol = solver.sol
    if sol.params.verbose > 0
        println("pessimistic step adaptive restart resolving...")
    end
    sol.info.iter_stepsize = 0
    solver.data.GlambdaMax, solver.data.GlambdaMax_flag = power_method!(solver.data.coeffTrans, solver.data.coeff, solver.AtAMV!, dual_sol_change)
    println("sqrt(max eigenvalue of GtG):", solver.data.GlambdaMax)
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
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        while inner_iter < sol.params.max_inner_iter
            inner_iter += 1
            sol.info.iter += 1
            pdhg_one_iter_diagonal_rescaling!(solver = solver, x = sol.x,
                         y = sol.y, tau = sol.params.tau / sol.info.omega,
                         sigma = sol.params.sigma * sol.info.omega, slack = sol.y.slack,
                         dual_sol_temp = dual_sol_temp)
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * inner_iter .+ sol.x.primal_sol.x) / (inner_iter + 1)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * inner_iter .+ sol.y.dual_sol.y) / (inner_iter + 1)
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
            if sol.info.iter %  sol.params.check_terminate_freq == 0 || (sol.params.verbose > 0 && sol.info.iter % sol.params.print_freq == 0)
                exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
                if sol.params.verbose > 0   
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
                end # end if verbose
                if sol.info.exit_status != :continue
                    return
                end
            end
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega)
        sol.info.omega = clamp(sol.info.omega, 0.8, 1.2)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
    return;
end

function resolving_adaptive_restart!(; solver::rpdhgSolver, 
    primal_sol_0::primalVector, dual_sol_0::dualVector,
    primal_sol_0_lag::AbstractVector{rpdhg_float}, dual_sol_0_lag::AbstractVector{rpdhg_float},
    dual_sol_temp::solVecDual, primal_sol_change::primalVector,
    dual_sol_change::dualVector, primal_sol_diff::primalVector, dual_sol_diff::dualVector)
    # pdhg + adaptive restart+adaptive stepsize
    sol = solver.sol
    if sol.params.verbose > 0
        println("adaptive restart resolving...")
    end
    recover_solution_resolving!(
        data = solver.data,
        Dr_product = solver.data.diagonal_scale.Dr_product.x,
        Dl_product = solver.data.diagonal_scale.Dl_product.y,
        sol = solver.sol.x,
        dual_sol = solver.sol.y
    )
    recover_data!(
        data = solver.data
    )

    block_diagonal_preconditioner!(
        problem = solver.data,
        diag_precond = solver.data.diagonal_scale,
        sol = solver.sol.x,
        dual_sol = solver.sol.y
    )
    setFunctionPointerSolver!(solver)
    sol.info.iter_stepsize = 0
    solver.data.GlambdaMax, solver.data.GlambdaMax_flag = power_method!(solver.data.coeffTrans, solver.data.coeff, solver.AtAMV!, dual_sol_change)
    println("sqrt(max eigenvalue of GtG):", solver.data.GlambdaMax)
    if solver.data.GlambdaMax_flag == 0
        solver.sol.params.sigma = 0.9 / solver.data.GlambdaMax
        solver.sol.params.tau = 0.9 / solver.data.GlambdaMax
    else
        solver.sol.params.sigma = 0.8 / solver.data.GlambdaMax
        solver.sol.params.tau = 0.8 / solver.data.GlambdaMax
    end
    sol.info.omega = initialize_primal_weight(solver = solver)
    GInf = norm(solver.data.coeff.G, Inf)
    eta_prime = 1 / GInf
    kkt_error_restart_threshold = 0;
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        eta_cum = 0.0
        while inner_iter < sol.params.max_inner_iter
            inner_iter += 1
            eta, eta_prime = pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(
                    solver = solver, x = sol.x, y = sol.y, eta = eta_prime, 
                    omega = sol.info.omega, info = sol.info, slack = sol.y.slack, 
                    dual_sol_temp = dual_sol_temp, primal_sol_change = primal_sol_change, 
                    dual_sol_change = dual_sol_change
                )
            if sol.info.exit_status != :continue
                return
            end
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
            eta_cum += eta
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega)
    end # end outer_iter loop

    sol.info.exit_status = :max_iter
    return;
end

function pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol.x)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol.y)
    dual_sol_temp = deepcopy(sol.y)
    primal_sol_change = deepcopy(sol.x.primal_sol)
    dual_sol_change = deepcopy(sol.y.dual_sol)
    sol.info.omega = initialize_primal_weight(solver = solver)
    primal_sol_diff = deepcopy(sol.x.primal_sol)
    dual_sol_diff = deepcopy(sol.y.dual_sol)
    GInf = norm(solver.data.coeff.G, Inf)
    eta_prime = 1 / GInf
    for outer_iter = 1 : sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        if outer_iter > 100
            resolving_pessimistic_step!(solver = solver,
                primal_sol_0 = primal_sol_0,
                dual_sol_0 = dual_sol_0,
                primal_sol_0_lag = primal_sol_0_lag,
                dual_sol_0_lag = dual_sol_0_lag,
                dual_sol_temp = dual_sol_temp,
                primal_sol_change = primal_sol_change, dual_sol_change = dual_sol_change,
                primal_sol_diff = primal_sol_diff, dual_sol_diff = dual_sol_diff)
            if sol.info.exit_status != :continue
                return
            end
        end
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        eta_cum = 0.0
        while inner_iter < sol.params.max_inner_iter
            inner_iter += 1
            eta, eta_prime = pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(
                    solver = solver, x = sol.x, y = sol.y, eta = eta_prime, 
                    omega = sol.info.omega, info = sol.info, slack = sol.y.slack, 
                    dual_sol_temp = dual_sol_temp, primal_sol_change = primal_sol_change, 
                    dual_sol_change = dual_sol_change
                )
            if sol.info.exit_status != :continue
                return
            end
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
            eta_cum += eta
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end

function update_best_sol(; sol::Solution)
    if sol.info.convergeInfo[1].l_2_rel_primal_res < sol.primal_res_best
        sol.primal_res_best = sol.info.convergeInfo[1].l_2_rel_primal_res
        sol.x_best.x .= sol.x.primal_sol.x
    end
    if sol.info.convergeInfo[1].l_2_rel_dual_res < sol.dual_res_best
        sol.dual_res_best = sol.info.convergeInfo[1].l_2_rel_dual_res
        sol.y_best.y .= sol.y.dual_sol.y
    end
end

function pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_aggressive!(; solver::rpdhgSolver)
    sol = solver.sol
    primal_sol_0 = deepcopy(sol.x.primal_sol)
    dual_sol_0 = deepcopy(sol.y.dual_sol)
    primal_sol_0_lag = deepcopy(sol.x.primal_sol.x)
    dual_sol_0_lag = deepcopy(sol.y.dual_sol.y)
    dual_sol_temp = deepcopy(sol.y)
    primal_sol_change = deepcopy(sol.x.primal_sol)
    dual_sol_change = deepcopy(sol.y.dual_sol)
    sol.info.omega = initialize_primal_weight(solver = solver)
    primal_sol_diff = deepcopy(sol.x.primal_sol)
    dual_sol_diff = deepcopy(sol.y.dual_sol)
    GInf = norm(solver.data.coeff.G, Inf)
    eta_prime = 1 / GInf
    for outer_iter = 1 : sol.params.max_outer_iter
        max_kkt_error = max(max(sol.info.convergeInfo[1].l_2_rel_primal_res, sol.info.convergeInfo[1].l_2_rel_dual_res), sol.info.convergeInfo[1].rel_gap)
        extra_coeff = -0.1 * log10(max_kkt_error) + 0.2
        primal_sol_0_lag .= primal_sol_0.x
        dual_sol_0_lag .= dual_sol_0.y
        primal_sol_0.x .= sol.x.primal_sol_mean.x
        dual_sol_0.y .= sol.y.dual_sol_mean.y
        sol.x.primal_sol.x .= sol.x.primal_sol_mean.x
        sol.y.dual_sol.y .= sol.y.dual_sol_mean.y
        restart_threshold_calculation_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, primal_sol_0_lag = primal_sol_0_lag, dual_sol_0_lag = dual_sol_0_lag)
        flag_one = (outer_iter > 10 && max_kkt_error > 1e+5)
        flag_two = (outer_iter > 80 && ((sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res) > 1e+10 || sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res < 1e-10))
        if  flag_one || flag_two
            if flag_one
                println("resolving pessimistic step adaptive restart resolving... since flag_one")
            else
                println("resolving pessimistic step adaptive restart resolving... since flag_two")
                println("l_2_rel_primal_res: ", sol.info.convergeInfo[1].l_2_rel_primal_res, " l_2_rel_dual_res: ", sol.info.convergeInfo[1].l_2_rel_dual_res)
                println("outer_iter: ", outer_iter)
            end 
            resolving_pessimistic_step!(solver = solver,
                primal_sol_0 = primal_sol_0,
                dual_sol_0 = dual_sol_0,
                primal_sol_0_lag = primal_sol_0_lag,
                dual_sol_0_lag = dual_sol_0_lag,
                dual_sol_temp = dual_sol_temp,
                primal_sol_change = primal_sol_change, dual_sol_change = dual_sol_change,
                primal_sol_diff = primal_sol_diff, dual_sol_diff = dual_sol_diff)
            if sol.info.exit_status != :continue
                return
            end
        end
        inner_iter = 0
        primal_sol_diff.x .= sol.x.primal_sol.x
        dual_sol_diff.y .= sol.y.dual_sol.y
        eta_cum = 0.0
        while inner_iter < sol.params.max_inner_iter
            inner_iter += 1
            eta, eta_prime = pdhg_one_iter_diagonal_rescaling_adaptive_step_size!(
                    solver = solver, x = sol.x, y = sol.y, eta = eta_prime, 
                    omega = sol.info.omega, info = sol.info, slack = sol.y.slack, 
                    dual_sol_temp = dual_sol_temp, primal_sol_change = primal_sol_change, 
                    dual_sol_change = dual_sol_change
                )
            if sol.info.exit_status != :continue
                return
            end
            sol.x.primal_sol.x .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.x.primal_sol.x .- extra_coeff * sol.x.primal_sol_lag.x) + 1 / (inner_iter + 2) * sol.x.primal_sol.x
            sol.y.dual_sol.y .= (inner_iter + 1) / (inner_iter + 2) * ((1 + extra_coeff) * sol.y.dual_sol.y .- extra_coeff * sol.y.dual_sol_lag.y) + 1 / (inner_iter + 2) * sol.y.dual_sol.y
            # average update
            sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * eta_cum .+ sol.x.primal_sol.x * eta) / (eta_cum + eta)
            sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * eta_cum .+ sol.y.dual_sol.y * eta) / (eta_cum + eta)
            eta_cum += eta
            if sol.info.iter %  sol.params.restart_check_freq == 0
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter)
                    # println("outer_iter: ", outer_iter, " sol.info.restart_used: ", sol.info.restart_used)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
        end # end inner_iter loop
        # println("outer_iter: ", outer_iter, " sol.info.restart_used: ", sol.info.restart_used)
        primal_sol_diff.x .-= sol.x.primal_sol.x
        dual_sol_diff.y .-= sol.y.dual_sol.y
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end



