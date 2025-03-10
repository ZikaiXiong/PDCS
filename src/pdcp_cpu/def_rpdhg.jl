
"""
def_rpdhg.jl
"""


"""
solVec is a struct that holds the solution vector and the indices of the linear, box, and SOC cones.
    - primal_sol: the solution vector
    - primal_sol_lag: the previous solution vector
    - primal_sol_mean: the mean of the previous inner primal_sol
    - linear_cone_index: the index of the linear cone (1:linear_cone_index)
    - box_cone_index: the index of the box cone ([box_cone_index[1]:box_cone_index[2]])
    - bl: the lower bounds of the box cone
    - bu: the upper bounds of the box cone
    - soc_cone_indices_start: the start indices of the SOC cones
    - soc_cone_indices_end: the end indices of the SOC cones
    -- primal_sol[1:linear_cone_index] is the linear cone
    -- primal_sol[box_cone_index[1]:box_cone_index[2]] is the box cone
    -- primal_sol[soc_cone_indices_start[i]:soc_cone_indices_end[i]] is the i-th SOC cone
"""
mutable struct solVecPrimal
    primal_sol::Vector{rpdhg_float}
    primal_sol_lag::Vector{rpdhg_float}
    primal_sol_mean::Vector{rpdhg_float}
    linear_cone_index::Integer
    box_cone_index::Vector{<:Integer}
    bl::Vector{rpdhg_float}
    bu::Vector{rpdhg_float}
    soc_cone_indices_start::Vector{<:Integer}
    soc_cone_indices_end::Vector{<:Integer}
    proj!::Function
    function solVecPrimal(; primal_sol, primal_sol_lag, primal_sol_mean, linear_cone_index, box_cone_index, bl, bu, soc_cone_indices_start, soc_cone_indices_end, proj)
        new(primal_sol, primal_sol_lag, primal_sol_mean, linear_cone_index, box_cone_index, bl, bu, soc_cone_indices_start, soc_cone_indices_end, proj)
    end
end

mutable struct solVecDual
    dual_sol::Vector{rpdhg_float}
    dual_sol_mean::Vector{rpdhg_float}
    len::Integer
    slack::solVecPrimal
    function solVecDual(; dual_sol, dual_sol_mean, len, slack)
        new(dual_sol, dual_sol_mean, len, slack)
    end
end

function solVecPrimalVerify(sol::solVecPrimal)
    if length(sol.primal_sol) != sol.soc_cone_indices_end[end]
        return false
    end
    if length(sol.bl) != 0 && length(sol.bu) != 0
        if length(sol.bl) != length(sol.bu) != sol.box_cone_index[2] - sol.box_cone_index[1] + 1
            println("Lower and upper bounds have different lengths or box cone index is different.")
            return false
        end
    else
        println("length of lower or upper bounds is zero.")
    end
    return true
end

function soc_proj!(sol::AbstractVector{rpdhg_float})
    t = sol[1]
    x = @view sol[2:end]
    nrm_x = LinearAlgebra.norm(x)
    if nrm_x <= -t
        x .= 0.0
        sol[1] = 0.0
    elseif nrm_x <= t
        return
    else
        c = (1.0 + t/nrm_x)/2.0
        sol[1] = nrm_x * c
        x .= x * c
    end
end

function linear_cone_proj!(sol::solVecPrimal)
    linear_cone_val = @view sol.primal_sol[1:sol.linear_cone_index]
    linear_cone_val .= max.(linear_cone_val , 0.0)
end

function box_cone_proj!(sol::solVecPrimal)
    box_cone_val = @view sol.primal_sol[sol.box_cone_index[1]: sol.box_cone_index[2]]
    box_cone_val .= clamp.(box_cone_val, sol.bl, sol.bu)
end

function soc_cone_proj!(sol::solVecPrimal)
    for (start_idx, end_idx) in zip(sol.soc_cone_indices_start, sol.soc_cone_indices_end)
        soc_proj!(@view sol.primal_sol[start_idx:end_idx])
    end
end

function all_cone_proj!(sol::solVecPrimal)
    linear_cone_proj!(sol)
    box_cone_proj!(sol)
    soc_cone_proj!(sol)
end

function linear_box_proj!(sol::solVecPrimal)
    linear_cone_proj!(sol)
    box_cone_proj!(sol)
end

function linear_soc_proj!(sol::solVecPrimal)
    linear_cone_proj!(sol)
    soc_cone_proj!(sol)
end

function box_soc_proj!(sol::solVecPrimal)
    box_cone_proj!(sol)
    soc_cone_proj!(sol)
end


function setProperty!(sol::solVecPrimal)
    if sol.linear_cone_index == 0
        if length(sol.box_cone_index) == 0
            if length(sol.soc_cone_indices_start) == 0
                # no cone is defined
                throw(ArgumentError("No cone is defined."))
            else
                # only SOC cone is defined
                sol.proj! = soc_cone_proj!
            end
        else
            if length(sol.soc_cone_indices_start) == 0
                # only box cone is defined
                sol.proj! = box_cone_proj!
            else
                # box and SOC cones are defined
                sol.proj! = box_soc_proj!
            end
        end
    else
        if length(sol.box_cone_index) == 0
            if length(sol.soc_cone_indices_start) == 0
                # only linear cone is defined
                sol.proj! = linear_cone_proj!
            else
                # linear and SOC cones are defined
                sol.proj! = linear_soc_proj!
            end
        else
            if length(sol.soc_cone_indices_start) == 0
                # linear and box cones are defined
                sol.proj! = linear_box_proj!
            else
                # all cones are defined
                sol.proj! = all_cone_proj!
            end
        end
    end
end


mutable struct PDHGCLPInfo
    # results
    iter::Integer
    primal_res::rpdhg_float
    dual_res::rpdhg_float
    primal_obj::rpdhg_float
    dual_obj::rpdhg_float
    gap::rpdhg_float
    time::Float64
    restart_times::Integer
    re_times_cond::Integer # restart times when meet restart condition
    status::Symbol
    primal_res_history::Vector{rpdhg_float}
    dual_res_history::Vector{rpdhg_float}
    gap_history::Vector{rpdhg_float}
    function PDHGCLPInfo(; iter, primal_res, dual_res, primal_obj, dual_obj, gap, time, restart_times, re_times_cond, status, primal_res_history, dual_res_history, gap_history)
        new(iter, primal_res, dual_res, primal_obj, dual_obj, gap, time, restart_times, re_times_cond, status, primal_res_history, dual_res_history, gap_history)
    end
end

mutable struct PDHGCLPParameters
    # parameters
    max_outer_iter::Integer
    max_inner_iter::Integer
    tol::rpdhg_float
    sigma::rpdhg_float
    tau::rpdhg_float
    restart_check_freq::Integer
    verbose::Bool
    print_freq::Integer
    plot::Bool
    function PDHGCLPParameters(; max_outer_iter, max_inner_iter, tol, sigma, tau, restart_check_freq, verbose, print_freq,plot)
        new(max_outer_iter, max_inner_iter, tol, sigma, tau, restart_check_freq, verbose, print_freq,plot)
    end
end

mutable struct Solution
    x::solVecPrimal
    y::solVecDual
    params::PDHGCLPParameters
    info::PDHGCLPInfo
    function Solution(; x, y, params, info)
        new(x, y, params, info)
    end
end


"""
RPDHGSolver is a struct that holds the data for the RPDHG solver.
    - m: the number of rows of the matrix A
    - n: the number of columns of the matrix A
    - A: the matrix A
    - b: the vector b
    - c: the vector c
    - AlambdaMax: the maximum eigenvalue of the matrix AtA
    - At: the transpose of the matrix A, At = transpose(A), use the same memory as A, different view
"""
mutable struct RPDHGSolver{
        AT<:AbstractMatrix{rpdhg_float},
        bT<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
        cT<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
        AtT<:AbstractMatrix{rpdhg_float}
    }
    m::Integer
    n::Integer
    A::AT
    b::bT
    c::cT
    AlambdaMax::rpdhg_float
    At::AtT
    bNrm1::rpdhg_float
    cNrm1::rpdhg_float
    function RPDHGSolver(;
            m::Integer, n::Integer,
            A::AT,
            b::bT, c::cT, AlambdaMax::rpdhg_float,
            At::AtT,
            bNrm1::rpdhg_float, cNrm1::rpdhg_float
         ) where{
            AT<:AbstractMatrix{rpdhg_float},
            bT<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
            cT<:Union{Vector{rpdhg_float}, SparseVector{rpdhg_float}},
            AtT<:AbstractMatrix{rpdhg_float}
            }
        new{AT, bT, cT, AtT}(m, n, A, b, c, AlambdaMax, At, bNrm1, cNrm1)
    end
end



function power_method!(A::AbstractMatrix; tol=1e-8, maxiter=1000)
    n = size(A, 1)
    b = rand(n)
    b = b / norm(b)

    lambda_old = 0.0
    for iter in 1:maxiter
        b = A * b 
        b = b / norm(b) 
        lambda = dot(b, A * b)
        if abs(lambda - lambda_old) < tol
            return sqrt(lambda)
        end
        lambda_old = lambda
    end
    error("power method cannot converge at $maxiter iterations.")
end

function AlambdaMax_cal!(A::AbstractMatrix)
    val, _ = eigsolve(A, 1, :LM)
    return sqrt(val[1])
end

