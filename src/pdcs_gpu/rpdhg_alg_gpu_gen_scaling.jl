
function approximate_cal_diagonal!(; solver::rpdhgSolver, h1::primalVector, h2::dualVector,
    slack::solVecPrimal, dual_sol_temp::solVecDual, t::rpdhg_float, 
    primal::primalVector, dual::dualVector, 
    tau::rpdhg_float, sigma::rpdhg_float)
    # modify `slack.primal_sol` and `dual_sol_temp.dual_sol`
    # slack.primal_sol.x .= primal.x .+ (t * tau / 2) * h1.x
    axpyz(slack.primal_sol.x, (t * tau / 2), h1.x, primal.x, solver.data.n)
    # here use solver.data.x is different with slack
    solver.sol.x.proj_diagonal!(slack.primal_sol, solver.sol.x, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
    # dual_sol_temp.dual_sol.y .= dual.y .+ (t * sigma / 2) * h2.y
    axpyz(dual_sol_temp.dual_sol.y, (t * sigma / 2), h2.y, dual.y, solver.data.m)
    solver.sol.y.proj_diagonal!(dual_sol_temp.dual_sol, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
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
    val1 = CUDA.dot(h1.x, slack.primal_sol_lag.x)
    val2 = CUDA.dot(h2.y, dual_sol_temp.dual_sol_temp.y)
    val = (val1 + val2) / r
    return val
end

function kkt_error_diagonal!(; omega::rpdhg_float, convergence_info::PDHGCLPConvergeInfo)
    return sqrt(omega^2 * convergence_info.l_2_abs_primal_res^2 + convergence_info.l_2_abs_dual_res^2 / omega^2 + convergence_info.abs_gap^2)
    # return max(max(convergence_info.l_inf_rel_primal_res, convergence_info.l_inf_rel_dual_res), convergence_info.rel_gap)
end



function converge_info_calculation_diagonal!(; solver::rpdhgSolver, primal_sol::primalVector, dual_sol::dualVector, slack::solVecPrimal, dual_sol_temp::solVecDual, converge_info::PDHGCLPConvergeInfo)
    pObj = CUDA.dot(primal_sol.x, solver.data.raw_data.c)
    solver.adjointMV!(solver.data.raw_data.coeff, dual_sol, slack.primal_sol.x)
    AtyInf = CUDA.norm(slack.primal_sol.x, Inf)
    AtyNrm1 = CUDA.norm(slack.primal_sol.x, 1)
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
    AxInf = CUDA.norm(dual_sol_temp.dual_sol_mean.y, Inf)
    AxNrm1 = CUDA.norm(dual_sol_temp.dual_sol_mean.y, 1)
    solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
    dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
    solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
    solver.data.diagonal_scale.Dl_temp.y .= dual_sol_temp.dual_sol_mean.y .- dual_sol_temp.dual_sol_lag.y


    
    slack.primal_sol_lag.x .= slack.primal_sol.x
    solver.sol.x.slack_proj!(slack.primal_sol, slack)
    sInf = CUDA.norm(slack.primal_sol.x, Inf)
    sNrm1 = CUDA.norm(slack.primal_sol.x, 1)

    l_2_abs_primal_res = CUDA.norm(solver.data.diagonal_scale.Dl_temp.y);
    l_2_rel_primal_res = l_2_abs_primal_res / (1 + max(sNrm1, max(solver.data.raw_data.hNrm1, AxNrm1)));

    l_inf_abs_primal_res = CUDA.maximum(CUDA.abs.(solver.data.diagonal_scale.Dl_temp.y));
    l_inf_rel_primal_res = l_inf_abs_primal_res / (1 + max(sInf, max(solver.data.raw_data.hNrmInf, AxInf)));

    solver.data.diagonal_scale.Dr_temp.x .= slack.primal_sol.x .- slack.primal_sol_lag.x
    l_2_abs_dual_res = CUDA.norm(solver.data.diagonal_scale.Dr_temp.x);
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
        pObj = CUDA.dot(primal_ray.x, solver.data.raw_data.c)
    end
    if l_inf_primal_ray_infeasibility === nothing && pObj < 0.0
        # dual_sol_temp.dual_sol_mean = Gx
        solver.primalMV!(solver.data.raw_data.coeff, primal_ray.x, dual_sol_temp.dual_sol_mean);
        solver.addCoeffd!(solver.data.raw_data.coeff, dual_sol_temp.dual_sol_mean, -1.0);
        dual_sol_temp.dual_sol_lag.y .= dual_sol_temp.dual_sol_mean.y;
        solver.sol.y.con_proj!(dual_sol_temp.dual_sol_mean)
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
            solver.sol.x.slack_proj!(slack.primal_sol, slack)
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
    infeasibility_info.primal_ray_norm = CUDA.norm(primal_ray.x, Inf)
    infeasibility_info.dual_ray_norm = CUDA.norm(dual_ray.y, Inf)
    if rand() < 0.25
        push!(infeasibility_info.primalObj_trend, pObj_seq)
        push!(infeasibility_info.dualObj_trend, dObj_seq)
    end
    return
end



function pdhg_one_iter_diagonal_rescaling!(; solver::rpdhgSolver, x::solVecPrimal, y::solVecDual, tau::rpdhg_float, sigma::rpdhg_float, slack::solVecPrimal, dual_sol_temp::solVecDual)
    x.primal_sol_lag.x .= x.primal_sol.x
    solver.adjointMV!(solver.data.coeff, y.dual_sol, slack.primal_sol_lag.x)
    x.primal_sol.x .-= tau .* solver.data.d_c
    x.primal_sol.x .+= tau .* slack.primal_sol_lag.x
    x.proj_diagonal!(x.primal_sol, x, solver.data.diagonal_scale, abs_tol = 1e-16, rel_tol = 1e-12)
    x.primal_sol_lag.x .= 2 .* x.primal_sol.x .- x.primal_sol_lag.x
    solver.primalMV!(solver.data.coeff, x.primal_sol_lag.x, dual_sol_temp.dual_sol_lag)
    solver.addCoeffd!(solver.data.coeff, dual_sol_temp.dual_sol_lag, -1.0)
    y.dual_sol.y .-= sigma .* dual_sol_temp.dual_sol_lag.y
    y.proj_diagonal!(y.dual_sol, solver.data.diagonal_scale, abs_tol = 1e-16, rel_tol = 1e-12)
    return;
end


function omega_norm_square(; x::CuArray, y::CuArray, omega::rpdhg_float)
    temp1 = omega * CUDA.norm(x, 2)^2
    temp2 = CUDA.norm(y, 2)^2 / omega
    return temp1 + temp2
end

function initialize_primal_weight(; solver::rpdhgSolver)
    hNrm2 = CUDA.norm(solver.data.coeff.d_h, 2)
    cNrm2 = CUDA.norm(solver.data.d_c, 2)
    @info ("hNrm2: $(hNrm2), cNrm2: $(cNrm2)")
    if hNrm2 > 1e-10 && cNrm2 > 1e-10
        omega = cNrm2 / hNrm2
    else
        omega = 1.0
    end
    return omega
end


function dynamic_primal_weight_update(; x_diff::CuArray, y_diff::CuArray, omega::rpdhg_float, theta::rpdhg_float, restart_times::Integer, solver::rpdhgSolver)
    x_diff_norm = CUDA.norm(x_diff, 2)
    y_diff_norm = CUDA.norm(y_diff, 2)
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
    check_termination_criteria(info = sol.info, params = sol.params)
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

function restart_condition_check_diagonal!(; sol::Solution, solver::rpdhgSolver, primal_sol_0::primalVector, dual_sol_0::dualVector, dual_sol_temp::solVecDual, inner_iter::Int, restart_duality_gap_flag::Bool)
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
        # condition 1
        if ((rhoVal_left < sol.params.beta_suff * sol.info.normalized_duality_gap_restart_threshold || rhoVal_left_mean < sol.params.beta_suff * sol.info.normalized_duality_gap_restart_threshold))
            sol.info.restart_used = sol.info.restart_used + 1
            if ((sol.info.normalized_duality_gap[1] < sol.info.normalized_duality_gap[2]) && sol.info.restart_duality_gap_flag)
                sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
                sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
                sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
            else    
                sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
            end
            println("restart due to condition 1")
            return true
        end # end if rhoVal_left
        # condition 2
        if (rhoVal_left < sol.params.beta_necessary * sol.info.normalized_duality_gap_restart_threshold || rhoVal_left_mean < sol.params.beta_necessary * sol.info.normalized_duality_gap_restart_threshold) 
            sol.info.restart_used = sol.info.restart_used + 1
            if ((sol.info.normalized_duality_gap[1] < sol.info.normalized_duality_gap[2]) && sol.info.restart_duality_gap_flag)
                sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
                sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
                sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
            else
                sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
            end
            println("restart due to condition 2")
            return true
        end # end if rhoVal_left
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
        kkt_error_ergodic_lag = sol.info.kkt_error[1]
        kkt_error_mean_lag = sol.info.kkt_error[2]
        kkt_error_ergodic = kkt_error_diagonal!(omega = sol.info.omega, convergence_info = sol.info.convergeInfo[1])
        kkt_error_mean = kkt_error_diagonal!(omega = sol.info.omega, convergence_info = sol.info.convergeInfo[2])
        sol.info.kkt_error[1] = kkt_error_ergodic
        sol.info.kkt_error[2] = kkt_error_mean
        # condition 1
        if ((kkt_error_ergodic < sol.params.beta_suff_kkt * sol.info.kkt_error_restart_threshold || kkt_error_mean < sol.params.beta_suff_kkt * sol.info.kkt_error_restart_threshold))
            sol.info.restart_used = sol.info.restart_used + 1
            if (sol.info.kkt_error[1] < sol.info.kkt_error[2])
                sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
                sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
                sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
            else    
                sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
            end
            println("restart due to condition 1 kkt")
            return true
        end # end if rhoVal_left
        # condition 2
        if (kkt_error_ergodic < sol.params.beta_necessary_kkt * sol.info.kkt_error_restart_threshold && kkt_error_ergodic > kkt_error_ergodic_lag) ||
             (kkt_error_mean < sol.params.beta_necessary_kkt * sol.info.kkt_error_restart_threshold && kkt_error_mean > kkt_error_mean_lag)
            sol.info.restart_used = sol.info.restart_used + 1
            if (sol.info.kkt_error[1] < sol.info.kkt_error[2])
                sol.info.restart_trigger_ergodic = sol.info.restart_trigger_ergodic + 1
                sol.x.primal_sol_mean.x .= sol.x.primal_sol.x
                sol.y.dual_sol_mean.y .= sol.y.dual_sol.y
            else
                sol.info.restart_trigger_mean = sol.info.restart_trigger_mean + 1
            end
            println("restart due to condition 2 kkt")
            return true
        end # end if rhoVal_left
    end
    # condition 3
    if inner_iter >= sol.info.iter * sol.params.beta_artificial && sol.info.iter > 100000
        @info ("restart due to artificial condition, beta_artificial = $(sol.params.beta_artificial), inner_iter = $inner_iter, iter = $(sol.info.iter)")
        sol.info.restart_used = sol.info.restart_used + 1
        return true
    end # end if inner_iter
    return false
end




function resolving_pessimistic_step!(; solver::rpdhgSolver, 
    primal_sol_0::primalVector, dual_sol_0::dualVector,
    primal_sol_0_lag::CuArray, dual_sol_0_lag::CuArray,
    dual_sol_temp::solVecDual, primal_sol_change::primalVector,
    dual_sol_change::dualVector, primal_sol_diff::primalVector, dual_sol_diff::dualVector, random_check::Bool)
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
            # average update
            # sol.x.primal_sol_mean.x .= (sol.x.primal_sol_mean.x * inner_iter .+ sol.x.primal_sol.x) / (inner_iter + 1)
            # sol.y.dual_sol_mean.y .= (sol.y.dual_sol_mean.y * inner_iter .+ sol.y.dual_sol.y) / (inner_iter + 1)
            average_seq(primal_sol_mean = sol.x.primal_sol_mean.x, primal_sol = sol.x.primal_sol.x, primal_n = solver.data.n, dual_sol_mean = sol.y.dual_sol_mean.y, dual_sol = sol.y.dual_sol.y, dual_n = solver.data.m, inner_iter = inner_iter)
            global time_iterative += time() - time_start
            if sol.info.iter %  sol.params.kkt_restart_freq == 0 && sol.params.use_kkt_restart
                time_start = time()
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false)
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
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = true)
                        break
                    end
                else
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false)
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
                            nrm2_x_recovered = CUDA.norm(sol.x.recovered_primal.primal_sol.x, 2)
                            nrm2_y_recovered = CUDA.norm(sol.y.recovered_dual.dual_sol.y, 2)
                            nrmInf_x_recovered = CUDA.norm(sol.x.recovered_primal.primal_sol.x, Inf)
                            nrmInf_y_recovered = CUDA.norm(sol.y.recovered_dual.dual_sol.y, Inf)
                            diff_nrm_x_recovered = CUDA.norm(primal_sol_change.x, 2)
                            diff_nrm_y_recovered = CUDA.norm(dual_sol_change.y, 2)
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
    random_check::Bool)
    eta = max(eta, 1e-6)
    x.primal_sol_lag.x .= x.primal_sol.x
    y.dual_sol_lag.y .= y.dual_sol.y

    primal_sol_diff = slack.primal_sol_lag
    primal_sol_temp = slack.primal_sol_mean
    dual_sol_diff = dual_sol_temp.dual_sol_lag

    sol = solver.sol
    decay_step_size_ratio = min(0.9 + 0.1 * log10(sol.info.max_kkt_error * 10.0^5), 1.0)
    # println("decay_step_size_ratio: ", decay_step_size_ratio)
    # decay_step_size_ratio = 1.0
    count = 0
    while true
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
        x.proj_diagonal!(x.primal_sol, x, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
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
        y.proj_diagonal!(y.dual_sol, solver.data.diagonal_scale, abs_tol = solver.sol.params.proj_abs_tol, rel_tol = solver.sol.params.proj_rel_tol)
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
        flag_random_check = (random_check && Int(ceil(sol.info.time)) % 3179 == 0)
        if sol.info.iter %  sol.params.check_terminate_freq == 0 || (sol.params.verbose > 0 && sol.info.iter % sol.params.print_freq == 0) || flag_random_check
            exit_condition_check_diagonal!(; sol = sol, solver = solver, dual_sol_temp = dual_sol_temp)
            sol.info.max_kkt_error = max(max(sol.info.convergeInfo[1].l_2_rel_primal_res, sol.info.convergeInfo[1].l_2_rel_dual_res), sol.info.convergeInfo[1].rel_gap)
            average_kkt_error = (sol.info.convergeInfo[1].l_2_rel_primal_res + sol.info.convergeInfo[1].l_2_rel_dual_res + sol.info.convergeInfo[1].rel_gap) / 3
            solver.sol.params.proj_abs_tol = max(min(solver.sol.params.proj_abs_tol, min(average_kkt_error * solver.sol.params.proj_base_tol, 1e-7)), 5e-16)
            solver.sol.params.proj_rel_tol = max(min(solver.sol.params.proj_rel_tol, min(average_kkt_error * solver.sol.params.proj_base_tol, 1e-7)), 5e-16)
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
                        nrm2_x_recovered = CUDA.norm(sol.x.recovered_primal.primal_sol.x, 2)
                        nrm2_y_recovered = CUDA.norm(sol.y.recovered_dual.dual_sol.y, 2)
                        nrmInf_x_recovered = CUDA.norm(sol.x.recovered_primal.primal_sol.x, Inf)
                        nrmInf_y_recovered = CUDA.norm(sol.y.recovered_dual.dual_sol.y, Inf)
                        diff_nrm_x_recovered = CUDA.norm(primal_sol_change.x, 2)
                        diff_nrm_y_recovered = CUDA.norm(dual_sol_change.y, 2)
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
        if eta <= eta_bar
            return eta, eta_prime, false
        end
        eta = eta_prime
        count += 1
        if count > 1000000
            solver.sol.params.proj_abs_tol *= 0.1
            solver.sol.params.proj_rel_tol *= 0.1
            solver.sol.params.proj_abs_tol = max(solver.sol.params.proj_abs_tol, 1e-22)
            solver.sol.params.proj_rel_tol = max(solver.sol.params.proj_rel_tol, 1e-22)
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
    sol.info.omega = initialize_primal_weight(solver = solver)
    primal_sol_diff = deepCopyPrimalVector(sol.x.primal_sol)
    dual_sol_diff = deepCopyDualVector(sol.y.dual_sol)
    GInf = CUDA.norm(solver.data.coeff.d_G, Inf)
    eta_prime = 1 / GInf
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
        flag_one = (outer_iter > 10 && sol.info.max_kkt_error > 1e+5)
        flag_two = (outer_iter > 80 && ((sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res) > 1e+10 || sol.info.convergeInfo[1].l_2_rel_primal_res / sol.info.convergeInfo[1].l_2_rel_dual_res < 1e-10))
        flag_three = (outer_iter > 10 && sol.params.tau < 1e-6 && sol.params.sigma < 1e-6)
        flag_four = (sol.info.iter > 1000000 && (sol.info.convergeInfo[1].rel_gap > 0.98 && sol.info.convergeInfo[1].l_2_rel_primal_res > 1.0 && sol.info.convergeInfo[1].l_2_rel_dual_res > 1.0))
        if  flag_one || flag_two || flag_three || flag_four
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
                random_check = random_check)
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
                    dual_sol_change = dual_sol_change, random_check = random_check
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
            reflection_update(sol.x.primal_sol.x, sol.x.primal_sol_lag.x, sol.x.primal_sol_mean.x, sol.y.dual_sol.y, sol.y.dual_sol_lag.y, sol.y.dual_sol_mean.y, extra_coeff, solver.data.n, solver.data.m, inner_iter, eta_cum, eta)
            eta_cum += eta
            if (inner_iter >= sol.info.iter * sol.params.beta_artificial && sol.info.iter > 10000) || (force_restart && inner_iter > 100)
                @info ("restart due to artificial condition, beta_artificial = $(sol.params.beta_artificial), inner_iter = $inner_iter, iter = $(sol.info.iter) or force_restart")
                sol.info.restart_used = sol.info.restart_used + 1
                break
            end # end if inner_iter
            if (sol.info.iter %  sol.params.kkt_restart_freq == 0) && sol.params.use_kkt_restart
                if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false)
                    break
                end
                if outer_iter == 1
                    break
                end # end if outer_iter
            end # end if check restart
            if (sol.info.iter %  sol.params.duality_gap_restart_freq == 0) && sol.params.use_duality_gap_restart
                if sol.info.restart_duality_gap_flag
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = true)
                        break
                    end
                else
                    if restart_condition_check_diagonal!(sol = sol, solver = solver, primal_sol_0 = primal_sol_0, dual_sol_0 = dual_sol_0, dual_sol_temp = dual_sol_temp, inner_iter = inner_iter, restart_duality_gap_flag = false)
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
        sol.info.omega = dynamic_primal_weight_update(theta = sol.params.theta, x_diff = primal_sol_diff.x, y_diff = dual_sol_diff.y, omega = sol.info.omega, restart_times = sol.info.restart_used, solver = solver)
    end # end outer_iter loop
    sol.info.exit_status = :max_iter
end