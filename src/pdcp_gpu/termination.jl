"A description of solver termination criteria. Refer to https://github.com/jinwen-yang/cuHPDLP"
mutable struct TerminationCriteria
    """
    Absolute tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_absolute::Float64

    
    """
    Relative tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_relative::Float64

    """
    If the following two conditions hold we say that we have obtained an
    approximate dual ray, which is an approximate certificate of primal
    infeasibility.
    (1) dual_ray_objective > 0.0,
    (2) max_dual_ray_infeasibility / dual_ray_objective <=
        eps_primal_infeasible.
    """
    eps_primal_infeasible::Float64

    """
    If the following three conditions hold we say we have obtained an
    approximate primal ray, which is an approximate certificate of dual
    infeasibility.
    (1) primal_ray_objective < 0.0,
    (2) max_primal_ray_infeasibility / (-primal_ray_objective) <=
        eps_dual_infeasible,
    """
    eps_dual_infeasible::Float64

    """
    Maximum number of iterations.
    """
    iteration_limit::Integer
    

    function TerminationCriteria(; eps_optimal_absolute::Float64=1e-6,
                                  eps_optimal_relative::Float64=1e-6,
                                  eps_primal_infeasible::Float64=1e-5,
                                  eps_dual_infeasible::Float64=1e-5,
                                  iteration_limit::Int32=1000)
        return new(eps_optimal_absolute, eps_optimal_relative,
                   eps_primal_infeasible, eps_dual_infeasible,
                   iteration_limit)
    end # function
end # module

function validate_termination_criteria(criteria::TerminationCriteria)
    if criteria.eps_primal_infeasible < 0
        error("eps_primal_infeasible must be nonnegative")
    end
    if criteria.eps_optimal_absolute < 0
        error("eps_optimal_absolute must be nonnegative")
    end
    if criteria.eps_dual_infeasible < 0
        error("eps_dual_infeasible must be nonnegative")
    end
    if criteria.eps_optimal_relative < 0
        error("eps_optimal_relative must be nonnegative")
    end
    if criteria.iteration_limit <= 0
        error("iteration_limit must be positive")
    end
end

"""
Check if the algorithm should terminate declaring the optimal solution is found.
"""
function optimality_criteria_met(;
    rel_tol::Float64,
    abs_tol::Float64,
    info::PDHGCLPConvergeInfo,
)
    if max(info.l_inf_rel_primal_res, info.l_inf_rel_dual_res, info.rel_gap) < rel_tol
        info.status = :optimal;
    end
    if max(info.l_inf_abs_primal_res, info.l_inf_abs_dual_res, info.abs_gap) < abs_tol
        info.status = :optimal;
    end
    if isnan(info.rel_gap)
        info.status = :numerical_error_nan
    end
    if isinf(info.rel_gap)
        info.status = :numerical_error_inf
    end
end

"""
Check if the algorithm should terminate declaring the primal is infeasible.
"""
function primal_infeasibility_criteria_met(;
    eps_primal_infeasible_low_acc::Float64,
    eps_primal_infeasible_high_acc::Float64,
    infeasibility_info::PDHGCLPInfeaInfo,
)
    if infeasibility_info.dual_ray_objective <= 0.0
        return;
    end
    # println("infeasibility_info.dual_ray_objective: $(infeasibility_info.dual_ray_objective), infeasibility_info.primal_ray_objective: $(infeasibility_info.primal_ray_objective)")
    # println("infeasibility_info.max_dual_ray_infeasibility: $(infeasibility_info.max_dual_ray_infeasibility), infeasibility_info.max_primal_ray_infeasibility: $(infeasibility_info.max_primal_ray_infeasibility)")
    # println("infeasibility_inf.dual_ray_norm: $(infeasibility_info.dual_ray_norm)")
    # heuristic to exit
    if (infeasibility_info.dual_ray_objective - infeasibility_info.primal_ray_objective) / (1 + abs(infeasibility_info.dual_ray_objective) + abs(infeasibility_info.primal_ray_objective)) > 0.99999 && infeasibility_info.dual_ray_objective > 0.0
        if infeasibility_info.max_primal_ray_infeasibility > 1e+3
            if is_monotonically_increasing(infeasibility_info.dualObj_trend, infeasibility_info.trend_len) &&
                infeasibility_info.max_dual_ray_infeasibility > 1e4 && infeasibility_info.dual_ray_objective > 1e+14 &&
                infeasibility_info.dual_ray_norm > 1e8 # inf norm
                infeasibility_info.status = :primal_infeasible_low_acc;
            end
        end
    end


    res = max(infeasibility_info.max_dual_ray_infeasibility, 1.0) / infeasibility_info.dual_ray_objective
    # println("infeasibility_info.max_dual_ray_infeasibility: $(infeasibility_info.max_dual_ray_infeasibility), infeasibility_info.dual_ray_objective: $(infeasibility_info.dual_ray_objective), res: $res", "eps_primal_infeasible_low_acc: $eps_primal_infeasible_low_acc, eps_primal_infeasible_high_acc: $eps_primal_infeasible_high_acc")
    if  res <= eps_primal_infeasible_low_acc && infeasibility_info.dual_ray_objective > 0.0
        if res <= eps_primal_infeasible_high_acc
            infeasibility_info.status = :primal_infeasible_high_acc;
        end
        if infeasibility_info.max_dual_ray_infeasibility > 1e8
            # println("max_dual_ray_infeasibility > 1e8")
            infeasibility_info.status = :primal_infeasible_low_acc;
        end
        if is_monotonically_increasing(infeasibility_info.dualObj_trend, infeasibility_info.trend_len)
            infeasibility_info.status = :primal_infeasible_low_acc;
        end
    end
end

"""
Check if the algorithm should terminate declaring the dual is infeasible.
"""
function dual_infeasibility_criteria_met(;
    eps_dual_infeasible_low_acc::Float64,
    eps_dual_infeasible_high_acc::Float64,
    infeasibility_info::PDHGCLPInfeaInfo,
)
    if infeasibility_info.primal_ray_objective >= 0.0
        return;
    end
    # println("infeasibility_info.primal_ray_objective: $(infeasibility_info.primal_ray_objective), infeasibility_info.dual_ray_objective: $(infeasibility_info.dual_ray_objective)")
    # println("infeasibility_info.max_dual_ray_infeasibility: $(infeasibility_info.max_dual_ray_infeasibility), infeasibility_info.max_primal_ray_infeasibility: $(infeasibility_info.max_primal_ray_infeasibility)")
    # println("infeasibility_inf.primal_ray_norm: $(infeasibility_info.primal_ray_norm)")

    # heuristic to exit
    if (infeasibility_info.primal_ray_objective - infeasibility_info.dual_ray_objective) / (1 + abs(infeasibility_info.primal_ray_objective) + abs(infeasibility_info.dual_ray_objective)) > 0.99999 && infeasibility_info.primal_ray_objective < 0.0
        if infeasibility_info.max_dual_ray_infeasibility > 1e+3
            if is_monotonically_decreasing(infeasibility_info.primalObj_trend, infeasibility_info.trend_len) &&
                infeasibility_info.max_primal_ray_infeasibility > 1e4 && -infeasibility_info.primal_ray_objective > 1e+14 &&
                infeasibility_info.primal_ray_norm > 1e8 # inf norm
                infeasibility_info.status = :dual_infeasible_low_acc;
            end
        end
    end

    res = max(infeasibility_info.max_primal_ray_infeasibility, 1.0) / (-infeasibility_info.primal_ray_objective)
    if  res <= eps_dual_infeasible_low_acc && infeasibility_info.primal_ray_objective < 0.0
        if res <= eps_dual_infeasible_high_acc
            infeasibility_info.status = :dual_infeasible_high_acc;
        end
        if infeasibility_info.max_primal_ray_infeasibility > 1e8
            # println("max_primal_ray_infeasibility > 1e8")
            infeasibility_info.status = :dual_infeasible_low_acc;
        end
        if is_monotonically_decreasing(infeasibility_info.primalObj_trend, infeasibility_info.trend_len)
            infeasibility_info.status = :dual_infeasible_low_acc;
        end
    end
end

"""
Checks if the given iteration_stats satisfy the termination criteria. Returns
a TerminationReason if so, and false otherwise.
"""

function check_termination_criteria(;
    info::PDHGCLPInfo,
    params::PDHGCLPParameters
    )

    for c_info in info.convergeInfo
        optimality_criteria_met(rel_tol=params.rel_tol, abs_tol = params.abs_tol, info=c_info)
        if c_info.status == :optimal
            info.exit_status = :optimal
            info.exit_code = 0
            info.pObj = c_info.primal_objective
            info.dObj = c_info.dual_objective
            return :optimal
        end
        if c_info.status == :numerical_error_nan
            info.exit_status = :numerical_error_nan
            info.exit_code = 8
            return :numerical_error_nan
        end
        if c_info.status == :numerical_error_inf
            info.exit_status = :numerical_error_inf
            info.exit_code = 9
            return :numerical_error_inf
        end
    end

    for i_info in info.infeaInfo
        primal_infeasibility_criteria_met(eps_primal_infeasible_low_acc=params.eps_primal_infeasible_low_acc,
                                        eps_primal_infeasible_high_acc=params.eps_primal_infeasible_high_acc,
                                        infeasibility_info=i_info)
        if i_info.status == :primal_infeasible_low_acc && info.iter > 2000
            info.exit_status = :primal_infeasible_low_acc
            info.exit_code = 2
            info.pObj = i_info.primal_ray_objective
            info.dObj = i_info.dual_ray_objective
            return :primal_infeasible_low_acc
        end
        if i_info.status == :primal_infeasible_high_acc && info.iter > 500
            info.exit_status = :primal_infeasible_high_acc
            info.exit_code = 3
            info.pObj = i_info.primal_ray_objective
            info.dObj = i_info.dual_ray_objective
            return :primal_infeasible_high_acc
        end
        dual_infeasibility_criteria_met(eps_dual_infeasible_low_acc=params.eps_dual_infeasible_low_acc,
                                        eps_dual_infeasible_high_acc=params.eps_dual_infeasible_high_acc,
                                        infeasibility_info=i_info)
        if i_info.status == :dual_infeasible_low_acc && info.iter > 2000
            info.exit_status = :dual_infeasible_low_acc
            info.exit_code = 4
            info.pObj = i_info.primal_ray_objective
            info.dObj = i_info.dual_ray_objective
            return :dual_infeasible_low_acc
        end
        if i_info.status == :dual_infeasible_high_acc && info.iter > 500
            info.exit_status = :dual_infeasible_high_acc
            info.exit_code = 5
            info.pObj = i_info.primal_ray_objective
            info.dObj = i_info.dual_ray_objective
            return :dual_infeasible_high_acc
        end
    end
    if info.iter >= params.max_outer_iter * params.max_inner_iter
        info.exit_status = :max_iter
        info.exit_code = 1
        for c_info in info.convergeInfo
            info.pObj = c_info.primal_objective
            info.dObj = c_info.dual_objective
            break
        end
        return :max_iter
    end
    if info.time > params.time_limit
        info.exit_status = :time_limit
        info.exit_code = 6
        return :time_limit
    end
    info.exit_status = :continue
    info.exit_code = 7
    return :continue
end