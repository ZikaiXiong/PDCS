

"""
recover the primal and dual solution from the product of the dual variables and the primal variables

recover the primal solution
    x ./= Dr_product
"""

function recover_solution!(;
    data::probData,
    Dr_product::Vector{Float64},
    Dl_product::Vector{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    sol.recovered_primal.primal_sol.x .=  sol.primal_sol.x ./ Dr_product
    sol.recovered_primal.primal_sol_lag.x .= sol.primal_sol_lag.x ./ Dr_product
    sol.recovered_primal.primal_sol_mean.x .= sol.primal_sol_mean.x ./ Dr_product

    dual_sol.recovered_dual.dual_sol.y .= dual_sol.dual_sol.y ./ Dl_product
    dual_sol.recovered_dual.dual_sol_lag.y .= dual_sol.dual_sol_lag.y ./ Dl_product
    dual_sol.recovered_dual.dual_sol_mean.y .= dual_sol.dual_sol_mean.y ./ Dl_product
end # recover_solution

function recover_solution_resolving!(;
    data::probData,
    Dr_product::Vector{Float64},
    Dl_product::Vector{Float64},
    sol::solVecPrimal,
    dual_sol::solVecDual
)
    sol.primal_sol.x .=  sol.primal_sol.x ./ Dr_product
    sol.primal_sol_lag.x .= sol.primal_sol_lag.x ./ Dr_product
    sol.primal_sol_mean.x .= sol.primal_sol_mean.x ./ Dr_product

    dual_sol.dual_sol.y .= dual_sol.dual_sol.y ./ Dl_product
    dual_sol.dual_sol_lag.y .= dual_sol.dual_sol_lag.y ./ Dl_product
    dual_sol.dual_sol_mean.y .= dual_sol.dual_sol_mean.y ./ Dl_product
end # recover_solution_resolving

function calculate_slack_solution(
    solver::rpdhgSolver,
    dual_sol::solVecDual,
    slack::solVecPrimal
)
    # slack variable = c - Q * yQ - A * yA
    solver.adjointMV!(solver.data.coeffTrans, dual_sol.y, slack.primal_sol)
    slack.primal_sol .= solver.data.c - slack.primal_sol;
end


function recover_data!(;
    data::probData
)
    data.c .= data.raw_data.c
    data.coeff.G.nzval .= data.raw_data.coeff.G.nzval
    # data.coeff.G = data.raw_data.coeff.G
    data.coeffTrans.G = transpose(data.coeff.G)
    data.coeff.h .= data.raw_data.coeff.h
    if data.nb > 0
        data.bl .= data.raw_data.bl
        data.bu .= data.raw_data.bu
        data.bl_finite .= data.raw_data.bl_finite
        data.bu_finite .= data.raw_data.bu_finite
    end
end # recover_data