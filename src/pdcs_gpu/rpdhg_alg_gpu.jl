"""
rpdhg_alg_cpu.jl
"""

function recover_soc_cone_indices(; l, bu, q)
    soc_cone_indices_start = []
    soc_cone_indices_end = []
    start_idx = l + length(bu) + 1
    for i in eachindex(q)
        push!(soc_cone_indices_start, start_idx)
        push!(soc_cone_indices_end, start_idx + q[i] - 1)
        start_idx += q[i]
    end
    soc_cone_indices_start = Vector{Int}(soc_cone_indices_start)
    soc_cone_indices_end = Vector{Int}(soc_cone_indices_end)
    return soc_cone_indices_start, soc_cone_indices_end
end

function setInfo!(; info::PDHGCLPInfo, iter::Integer, primal_res::rpdhg_float, dual_res::rpdhg_float, primal_obj::rpdhg_float, dual_obj::rpdhg_float, gap::rpdhg_float, time::rpdhg_float, restart_times::Integer)
    info.iter = iter
    info.primal_res = primal_res
    info.dual_res = dual_res
    info.primal_obj = primal_obj
    info.dual_obj = dual_obj
    info.gap = gap
    info.time = time
    info.restart_times = restart_times
end

function Mnorm(; solver::RPDHGSolver, x::Vector{rpdhg_float}, y::Vector{rpdhg_float}, tau::rpdhg_float, sigma::rpdhg_float)
    yAx = y' * solver.A * x
    return x' * x / tau - 2 * yAx + y' * y / sigma
end

function approximate_cal(; h1::Vector{rpdhg_float}, h2::h2Type,
                    slack::solVecPrimal, dual_sol_temp::Vector{rpdhg_float}, t::rpdhg_float,
                    primal::Vector{rpdhg_float}, dual::Vector{rpdhg_float},
                    tau::rpdhg_float, sigma::rpdhg_float) where h2Type<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
    #   TODO: check modify the slack.primal_sol and dual_sol_temp really??
    # h1 = solver.At * dual - solver.c
    slack.primal_sol .= primal .+ (t * tau / 2) * h1
    slack.proj!(slack)
    # h2 = solver.b - solver.A * primal
    dual_sol_temp .= dual .+ (t * sigma / 2) * h2
end

function binary_search_duality_gap(;solver::RPDHGSolver,
     r::rpdhg_float, primal::Vector{rpdhg_float},
     dual::Vector{rpdhg_float}, slack::solVecPrimal,
     dual_sol_temp::Vector{rpdhg_float}, tau::rpdhg_float, sigma::rpdhg_float,
     maxIter::Integer = 1000, tol::rpdhg_float = 1e-6)
     """
     calculate the normalized duality gap
        rho(r, z)
     """
    t = 10.0
    tRight = t
    tLeft = 0.0
    h1 = (solver.At * dual - solver.c)
    h2 = (solver.b - solver.A * primal)
    for k = 1: maxIter
        ## approximate calculation
        approximate_cal(h1 = h1, h2 = h2,
                        slack = slack, dual_sol_temp = dual_sol_temp,
                        t = t, primal = primal, dual = dual,
                        tau = tau, sigma = sigma)
        # `slcak.primal_sol` and `dual_sol_temp` are used as temporary variables
        if Mnorm(solver = solver, x = slack.primal_sol, y = dual_sol_temp, tau = tau, sigma = sigma) > r
            if k == 1
                break
            else
                tRight = t, tLeft = t / 2, break
            end
        end
        t *= 2
    end
    # binary function search_element(arr::Vector{T}, target::T) where T
    while tRight - tLeft > tol
        tMid = (tRight + tLeft) / 2
        approximate_cal(h1 = h1, h2 = h2,
                        slack = slack, dual_sol_temp = dual_sol_temp,
                        t = tMid, primal = primal, dual = dual,
                        tau = tau, sigma = sigma)
        if Mnorm(solver = solver, x = slack.primal_sol, y = dual_sol_temp, tau = tau, sigma = sigma) > r
            tLeft = tMid
        else
            tRight = tMid
        end
    end
    return (h1' * slack.primal_sol + h2' * dual_sol_temp) / r
end

function info_calculation(;solver::RPDHGSolver, primal_sol::Vector{rpdhg_float}, dual_sol::Vector{rpdhg_float}, slack::solVecPrimal)
    pObj = dot(solver.c, primal_sol)
    dObj = dot(solver.b, dual_sol)
    gap = abs(pObj - dObj) / (1 + abs(pObj) + abs(dObj))
    primal_res = norm(solver.A * primal_sol - solver.b) / (1 + solver.bNrm1)
    slack.primal_sol .= solver.c - (solver.At * dual_sol)
    slack.primal_sol_lag .= slack.primal_sol
    slack.proj!(slack)
    dual_res = norm(slack.primal_sol_lag - slack.primal_sol) / (1 + solver.cNrm1)
    return pObj, dObj, gap, primal_res, dual_res
end

function pdhg_one_iter!(;solver::RPDHGSolver, x::solVecPrimal, y::Vector{rpdhg_float}, tau::rpdhg_float, sigma::rpdhg_float)
    x.primal_sol_lag .= x.primal_sol
    x.primal_sol .= x.primal_sol .- tau * (solver.c - solver.At * y)
    x.proj!(x)
    y .+= sigma * (solver.b - solver.A * (2 * x.primal_sol - x.primal_sol_lag))
end

function pdhg_main_iter_halpern!(;solver::RPDHGSolver, sol::Solution)
    start_time = time()
    primal_sol_0 = copy(sol.x.primal_sol)
    dual_sol_0 = copy(sol.y.dual_sol)
    primal_sol_0_lag = copy(sol.x.primal_sol)
    dual_sol_0_lag = copy(sol.y.dual_sol)
    dual_sol_temp = copy(sol.y.dual_sol)
    for outer_iter = 1:sol.params.max_outer_iter
        primal_sol_0_lag .= primal_sol_0
        dual_sol_0_lag .= dual_sol_0
        primal_sol_0 .= sol.x.primal_sol
        dual_sol_0 .= sol.y.dual_sol
        Mnorm_restart_right = Mnorm(solver = solver,
                            x = (primal_sol_0 - primal_sol_0_lag),
                            y = (dual_sol_0 - dual_sol_0_lag),
                            tau = sol.params.tau, sigma = sol.params.sigma)
        rhoVal = binary_search_duality_gap(solver = solver,
                            r = Mnorm_restart_right,
                            primal = primal_sol_0,
                            dual = dual_sol_0,
                            slack = sol.y.slack,
                            dual_sol_temp = dual_sol_temp,
                            tau = sol.params.tau, sigma = sol.params.sigma)
        rho_restart_cond = rhoVal / exp(1)
        for  inner_iter = 1:sol.params.max_inner_iter
            pdhg_one_iter!(solver = solver, x = sol.x, y = sol.y.dual_sol, tau = sol.params.tau, sigma = sol.params.sigma)
            # halpern update
            sol.x.primal_sol .= sol.x.primal_sol * (inner_iter + 1) / (inner_iter + 2) + primal_sol_0 / (inner_iter + 2)
            sol.y.dual_sol .= sol.y.dual_sol * (inner_iter + 1) / (inner_iter + 2) + dual_sol_0 / (inner_iter + 2)
            sol.info.iter = sol.info.iter + 1
            if sol.params.verbose && inner_iter % sol.params.print_freq == 0
                pObj, dObj, gap, primal_res, dual_res = info_calculation(
                    solver = solver,
                    primal_sol = sol.x.primal_sol,
                    dual_sol = sol.y.dual_sol,
                    slack = sol.y.slack);
                setInfo!(info = sol.info,
                        iter = sol.info.iter,
                        primal_res = primal_res,
                        dual_res = dual_res,
                        primal_obj = pObj,
                        dual_obj = dObj,
                        gap = gap,
                        time = time() - start_time,
                        restart_times = sol.info.restart_times);
                printInfo(info = sol.info);
                if max(primal_res, dual_res, gap) < sol.params.tol
                    @info ("Converged!");
                    sol.info.status = :Converged;
                    return;
                end
            end
            if inner_iter % sol.params.restart_check_freq == 0
                # check restart condition
                Mnorm_restart_left = Mnorm(solver = solver,
                                    x = (sol.x.primal_sol - primal_sol_0),
                                    y = (sol.y.dual_sol - dual_sol_0),
                                    tau = sol.params.tau, sigma = sol.params.sigma)
                rhoVal_left = binary_search_duality_gap(solver = solver,
                                                r = Mnorm_restart_left,
                                                primal = sol.x.primal_sol,
                                                dual = sol.y.dual_sol,
                                                slack = sol.y.slack,
                                                dual_sol_temp = dual_sol_temp,
                                                tau = sol.params.tau, sigma = sol.params.sigma)
                if rhoVal_left < rho_restart_cond
                    sol.info.re_times_cond = sol.info.re_times_cond + 1
                    break
                end
                if outer_iter == 1
                    break
                end
            end
        end
        sol.info.restart_times = sol.info.restart_times + 1
    end
    sol.info.status = :maxIter
end


function pdhg_main_iter_halpern_no_restart!(;solver::RPDHGSolver, sol::Solution)
    start_time = time()
    primal_sol_0 = copy(sol.x.primal_sol)
    dual_sol_0 = copy(sol.y.dual_sol)
    for outer_iter = 1:sol.params.max_outer_iter
        primal_sol_0 .= sol.x.primal_sol
        dual_sol_0 .= sol.y.dual_sol
        for  inner_iter = 1:sol.params.max_inner_iter
            pdhg_one_iter!(solver = solver, x = sol.x, y = sol.y.dual_sol, tau = sol.params.tau, sigma = sol.params.sigma)
            # halpern update
            sol.x.primal_sol .= sol.x.primal_sol * (inner_iter + 1) / (inner_iter + 2) + primal_sol_0 / (inner_iter + 2)
            sol.y.dual_sol .= sol.y.dual_sol * (inner_iter + 1) / (inner_iter + 2) + dual_sol_0 / (inner_iter + 2)
            sol.info.iter = sol.info.iter + 1
            if sol.params.verbose && inner_iter % sol.params.print_freq == 0
                pObj, dObj, gap, primal_res, dual_res = info_calculation(solver = solver, primal_sol = sol.x.primal_sol, dual_sol = sol.y.dual_sol, slack = sol.y.slack);
                setInfo!(info = sol.info,
                        iter = sol.info.iter,
                        primal_res = primal_res,
                        dual_res = dual_res,
                        primal_obj = pObj,
                        dual_obj = dObj,
                        gap = gap,
                        time = time() - start_time,
                        restart_times = sol.info.restart_times)
                printInfo(info = sol.info)
                if max(primal_res, dual_res, gap) < sol.params.tol
                    @info ("Converged!");
                    sol.info.status = :Converged;
                    return;
                end
            end
        end
        sol.info.restart_times = sol.info.restart_times + 1
    end
    sol.info.status = :maxIter;
end

function pdhg_main_iter_average!(;solver::RPDHGSolver, sol::Solution)
    start_time = time()
    primal_sol_0 = copy(sol.x.primal_sol)
    dual_sol_0 = copy(sol.y.dual_sol)
    primal_sol_0_lag = copy(sol.x.primal_sol)
    dual_sol_0_lag = copy(sol.y.dual_sol)
    dual_sol_temp = copy(sol.y.dual_sol)
    for outer_iter = 1:sol.params.max_outer_iter
        sol.x.primal_sol .= sol.x.primal_sol_mean
        sol.y.dual_sol .= sol.y.dual_sol_mean
        primal_sol_0_lag .= primal_sol_0
        dual_sol_0_lag .= dual_sol_0
        primal_sol_0 .= sol.x.primal_sol
        dual_sol_0 .= sol.y.dual_sol
        Mnorm_restart_right = Mnorm(solver = solver,
                                x = (primal_sol_0 - primal_sol_0_lag),
                                y = (dual_sol_0 - dual_sol_0_lag),
                                tau = sol.params.tau, sigma = sol.params.sigma)
        rhoVal = binary_search_duality_gap(solver = solver,
                r = Mnorm_restart_right,
                primal = primal_sol_0,
                dual = dual_sol_0,
                slack = sol.y.slack,
                dual_sol_temp = dual_sol_temp,
                tau = sol.params.tau, sigma = sol.params.sigma)
        rho_restart_cond = rhoVal / exp(1)
        for  inner_iter = 1:sol.params.max_inner_iter
            pdhg_one_iter!(solver = solver, x = sol.x, y = sol.y.dual_sol, tau = sol.params.tau, sigma = sol.params.sigma)
            # halpern update
            sol.x.primal_sol_mean .= (sol.x.primal_sol_mean * inner_iter .+ sol.x.primal_sol) / (inner_iter + 1)
            sol.y.dual_sol_mean .= (sol.y.dual_sol_mean * inner_iter .+ sol.y.dual_sol) / (inner_iter + 1)
            sol.info.iter = sol.info.iter + 1
            if sol.params.verbose && inner_iter % sol.params.print_freq == 0
                pObj, dObj, gap, primal_res, dual_res = info_calculation(solver = solver, primal_sol = sol.x.primal_sol_mean, dual_sol = sol.y.dual_sol_mean, slack = sol.y.slack);
                setInfo!(info = sol.info,
                        iter = sol.info.iter,
                        primal_res = primal_res,
                        dual_res = dual_res,
                        primal_obj = pObj,
                        dual_obj = dObj,
                        gap = gap,
                        time = time() - start_time,
                        restart_times = sol.info.restart_times)
                printInfo(info = sol.info)
                if max(primal_res, dual_res, gap) < sol.params.tol
                    @info ("Converged!")
                    sol.info.status = :Converged;
                    return;
                end
            end
            if inner_iter % sol.params.restart_check_freq == 0
                # check restart condition
                Mnorm_restart_left = Mnorm(solver = solver,
                                    x = (sol.x.primal_sol_mean - primal_sol_0),
                                    y = (sol.y.dual_sol_mean - dual_sol_0),
                                    tau = sol.params.tau, sigma = sol.params.sigma)
                rhoVal_left = binary_search_duality_gap(solver = solver,
                                                r = Mnorm_restart_left,
                                                primal = sol.x.primal_sol_mean,
                                                dual = sol.y.dual_sol_mean,
                                                slack = sol.y.slack,
                                                dual_sol_temp = dual_sol_temp,
                                                tau = sol.params.tau, sigma = sol.params.sigma)
                if rhoVal_left < rho_restart_cond
                    sol.info.re_times_cond = sol.info.re_times_cond + 1
                    break
                end
                if outer_iter == 1
                    break
                end
            end
        end
        sol.info.restart_times = sol.info.restart_times + 1
    end
    sol.info.status = :maxIter
end

function pdhg_main_iter_average_no_restart!(;solver::RPDHGSolver, sol::Solution)
    start_time = time()
    primal_sol_0 = copy(sol.x.primal_sol)
    dual_sol_0 = copy(sol.y.dual_sol)
    for outer_iter = 1:sol.params.max_outer_iter
        sol.x.primal_sol .= sol.x.primal_sol_mean
        sol.y.dual_sol .= sol.y.dual_sol_mean
        primal_sol_0 .= sol.x.primal_sol
        dual_sol_0 .= sol.y.dual_sol
        for  inner_iter = 1:sol.params.max_inner_iter
            pdhg_one_iter!(solver = solver, x = sol.x, y = sol.y.dual_sol, tau = sol.params.tau, sigma = sol.params.sigma)
            # halpern update
            sol.x.primal_sol_mean .= (sol.x.primal_sol_mean * inner_iter .+ sol.x.primal_sol) / (inner_iter + 1)
            sol.y.dual_sol_mean .= (sol.y.dual_sol_mean * inner_iter .+ sol.y.dual_sol) / (inner_iter + 1)
            sol.info.iter = sol.info.iter + 1
            if sol.params.verbose && inner_iter % sol.params.print_freq == 0
                pObj, dObj, gap, primal_res, dual_res = info_calculation(solver = solver, primal_sol = sol.x.primal_sol_mean, dual_sol = sol.y.dual_sol_mean, slack = sol.y.slack);
                setInfo!(info = sol.info,
                        iter = sol.info.iter,
                        primal_res = primal_res,
                        dual_res = dual_res,
                        primal_obj = pObj,
                        dual_obj = dObj,
                        gap = gap,
                        time = time() - start_time,
                        restart_times = sol.info.restart_times)
                printInfo(info = sol.info)
                if max(primal_res, dual_res, gap) < sol.params.tol
                    @info ("Converged!");
                    sol.info.status = :Converged;
                    return;
                end
            end
        end
        sol.info.restart_times = sol.info.restart_times + 1
    end
    sol.info.status = :maxIter;
end



function printInfo(;info::PDHGCLPInfo)
    @info (@sprintf("iter: %d primal_res: %.2e dual_res: %.2e primal_obj: %.2e dual_obj: %.2e pd_gap: %.2e time: %.2f restart_times: %d",
    info.iter, info.primal_res, info.dual_res, info.primal_obj, info.dual_obj, info.gap, info.time, info.restart_times))
end

function infoSummary(;info::PDHGCLPInfo)
@info ("----------------------------------------")
@info ("------ Solver Info Summary -------------")
@info ("----------------------------------------")
@info (@sprintf("iter_num:             %d", info.iter))
@info (@sprintf("exit_status:          %s", info.status))
@info (@sprintf("primal_res:           %.4e", info.primal_res))
@info (@sprintf("dual_res:             %.4e", info.dual_res))
@info (@sprintf("pd_gap:               %.4e", info.gap))
@info (@sprintf("primal_obj:           %.4e", info.primal_obj))
@info (@sprintf("dual_obj:             %.4e", info.dual_obj))
@info (@sprintf("time:                 %.4f", info.time))
@info ("----------------------------------------")
end

"""
    rpdhg_cpu_solve()
    the main function of the rpdhg solver


    ## Problem definition
    rpdhg solves a problem of the form
    ```
    minimize  c' * x
    s.t.      A * x = b
              x in K
    ```
    where K is a product of cone of
    - linear cone positive orthant `{ x | x ≥ 0 }`
    - box cone `{ (t,x) | t*l ≤ x ≤ t*u }`
    - second order cone `{ (t,x) | ||x||_2 ≤ t }`

    ## Parameters
    - `m`: the number of affine constraints
    - `n`: the number of variables
    - `A`: an `AbstractMatrix` with `m` rows and `n` columns
    - `b`: a `Vector` of length `m`
    - `c`: a `Vector` of length `n`
    - `l`: the number of linear cones
    - `bu`: the `Vector` of upper bounds for the box cone 
    - `bl`: the `Vector` of lower bounds for the box cone
    - `q`: the `Vector` of SOCs sizes

    Provide a warm start to rpdhg by overriding:
    - `primal_sol = zeros(n)`: a `Vector` to warmstart the primal variables,
    - `dual_sol = zeros(m)`: a `Vector` to warmstart the dual variables,
    - `warm_start = false`: a `Bool` to enable warm start
    - `options...`: a list of options to pass to the solver

    !!! note
        To successfully warmstart the solver `primal_sol`, `dual_sol` and `slack`
        must all be provided **and** `warm_start` option must be set to `true`.
    
    ## Output
    This function returns a `Solution` object, which contains the following fields:
    ```julia
    mutable struct Solution{T}
        x::Vector{Float64}
        y::Vector{Float64}
        params::PDHGCLPParameters{T}
        info::PDHGCLPInfo{T}
    end
    ```
    where `x` stores the optimal value of the primal variable, `y` stores the
    optimal value of the dual variable, and `info`
    contains various information about the solve step.
"""
function rpdhg_cpu_solve(;
    m::Integer,
    n::Integer,
    A::AbstractMatrix,
    b::bT,
    c::Vector{rpdhg_float},
    l::Integer,
    bu::Vector{rpdhg_float},
    bl::Vector{rpdhg_float},
    q::Vector{<:Integer},
    primal_sol::Vector{rpdhg_float}=zeros(n),
    dual_sol::Vector{rpdhg_float}=zeros(m),
    warm_start::Bool=false,
    max_outer_iter::Integer=10000,
    max_inner_iter::Integer=10000,
    tol::rpdhg_float=1e-6,
    print_freq::Integer=100,
    restart_check_freq::Integer=200,
    verbose::Bool=true,
    plot::Bool=false,
    method::Symbol=:halpern,
)where bT<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}}
    # set random seed
    Random.seed!(1234)
    if length(primal_sol) == n && length(dual_sol) == m
        if !warm_start
            @info ("Warm start not enabled. Ignoring warm start values.")
            @info ("Initializing primal and dual variables to zero.")
            fill!(primal_sol, 0.0)
            fill!(dual_sol, 0.0)
        end
    else
        if warm_start
            throw(ArgumentError("Warmstart doesn't match the problem size"))
        end
        @info ("Initializing primal and dual variables to random.")
        # primal_sol, dual_sol = zeros(n), zeros(m)
        # random initialization
        # Random.seed!(1234)
        primal_sol = rand(n)
        dual_sol = rand(m)
    end
    # create solver and set data
    solver = RPDHGSolver(
        m = m,
        n = n,
        A = A,
        b = b,
        c = c,
        AlambdaMax = 0.0,
        At = transpose(A), # return a new view of A, not copy
        bNrm1 = 0.0,
        cNrm1 = 0.0
    )
    AtA = transpose(solver.A) * solver.A
    ## precalculate 
    solver.AlambdaMax = AlambdaMax_cal!(AtA)
    solver.bNrm1 = norm(b, 1)
    solver.cNrm1 = norm(c, 1)
    # create solution variables
    # copy the primal and dual variables
    primal_sol_lag = copy(primal_sol);
    primal_sol_mean = copy(primal_sol);
    soc_cone_indices_start, soc_cone_indices_end = recover_soc_cone_indices(l = l, bu = bu, q = q)
    if length(bl) != 0
        box_cone_index = Vector{Int}([l + 1, l + length(bl)])
    else
        box_cone_index = Vector{Int}([])
    end
    # initialize the primal solution vector

    x = solVecPrimal(primal_sol = primal_sol, 
                    primal_sol_lag = primal_sol_lag,
                    primal_sol_mean = primal_sol_mean,
                    linear_cone_index = l,
                    box_cone_index = box_cone_index,
                    bl = bl,
                    bu = bu,
                    soc_cone_indices_start = soc_cone_indices_start,
                    soc_cone_indices_end = soc_cone_indices_end,
                    proj = x -> println("The projection function is not defined."));
    setProperty!(x)
    retcode = solVecPrimalVerify(x);
    if !retcode
        throw(ArgumentError("The primal solution vector is not valid."))
    end

    # initialize the dual solution vector
    dual_sol = zeros(m);
    dual_sol_mean = copy(dual_sol);
    slack_sol = zeros(n);
    slack_sol_lag = copy(slack_sol);
    slack = solVecPrimal(primal_sol = slack_sol, 
                    primal_sol_lag = slack_sol_lag,
                    primal_sol_mean = Vector{rpdhg_float}([]),
                    linear_cone_index = l,
                    box_cone_index = box_cone_index,
                    bl = bl,
                    bu = bu,
                    soc_cone_indices_start = soc_cone_indices_start,
                    soc_cone_indices_end = soc_cone_indices_end,
                    proj = x -> println("The projection function is not defined."));
    setProperty!(slack);
    y = solVecDual(dual_sol = dual_sol,
                 dual_sol_mean = dual_sol_mean,
                 len = m,
                 slack = slack);
    
    # set the parameters
    params = PDHGCLPParameters(max_outer_iter = max_outer_iter,
                                max_inner_iter = max_inner_iter,
                                tol = tol,
                                sigma = 0.9/solver.AlambdaMax,
                                tau = 0.9/solver.AlambdaMax,
                                verbose = verbose,
                                restart_check_freq = restart_check_freq,
                                print_freq = print_freq,
                                plot = plot);

    pObj, dObj, gap, primal_res, dual_res = info_calculation(solver = solver, primal_sol = x.primal_sol, dual_sol = y.dual_sol, slack = y.slack);
    if plot
        primal_res_history = Vector{rpdhg_float}([primal_res])
        dual_res_history = Vector{rpdhg_float}([dual_res])
        gap_history = Vector{rpdhg_float}([gap])
    else
        primal_res_history = Vector{rpdhg_float}([])
        dual_res_history = Vector{rpdhg_float}([])
        gap_history = Vector{rpdhg_float}([])
    end
    # set the info
    info = PDHGCLPInfo(iter = 0,
                       primal_res = primal_res,
                       dual_res = dual_res,
                       primal_obj = pObj,
                       dual_obj = dObj,
                       gap = gap,
                       time = 0.0,
                       restart_times = 0,
                       re_times_cond = 0,
                       status = :NotYet,
                       primal_res_history = primal_res_history,
                       dual_res_history = dual_res_history,
                       gap_history = gap_history);

    sol = Solution(x = x, y = y, params = params, info = info);
    if verbose
        @info ("===============================================")
        printInfo(info = sol.info);
    end
    # main loop, all data and variables are put in ``solver`` and ``sol`` two structs
    if method == :halpern
        if plot
            pdhg_main_iter_halpern_plot!(solver = solver, sol = sol)
        else
            pdhg_main_iter_halpern!(solver = solver, sol = sol)
        end
    elseif method == :halpern_no_restart
        if plot
            pdhg_main_iter_halpern_no_restart_plot!(solver = solver, sol = sol)
        else
            pdhg_main_iter_halpern_no_restart!(solver = solver, sol = sol)
        end
    elseif method == :average
        if plot
            pdhg_main_iter_average_plot!(solver = solver, sol = sol)
        else
            pdhg_main_iter_average!(solver = solver, sol = sol)
        end
    elseif method == :average_no_restart
        if plot
            pdhg_main_iter_average_no_restart_plot!(solver = solver, sol = sol)
        else
            pdhg_main_iter_average_no_restart!(solver = solver, sol = sol)
        end
    else
        throw(ArgumentError("The method is not defined."))
    end
    @info ("===============================================")
    infoSummary(info = sol.info)
    return sol
end
