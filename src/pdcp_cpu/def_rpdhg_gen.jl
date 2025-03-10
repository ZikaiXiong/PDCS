"""
def_rpdhg_gen.jl
"""

function gen_lambd(bl::Vector{rpdhg_float}, bu::Vector{rpdhg_float})
    lambd_l = ifelse.(bl .== -Inf,
        ifelse.(bu .== Inf, 0.0, -Inf),
        ifelse.(bu .== Inf, 0.0, -Inf)
    )

    # 修正后的 lambd_u
    lambd_u = ifelse.(bu .== Inf,
        ifelse.(bl .== -Inf, 0.0, Inf),
        ifelse.(bl .== -Inf, 0.0, Inf)
    )
    return lambd_l, lambd_u
end

"""
function for primal diagonal projection
"""

function box_proj_diagonal!(x::T, dummy1::T, dummy2::T, dummy3::T, dummy5::T, dummy6::T, dummy7::T, dummy8::T, sol::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    x .= clamp.(x, sol.bl, sol.bu)
end
function soc_cone_proj_diagonal!(x::T, x_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy3::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    soc_proj_diagonal!(x, x_part, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function soc_cone_proj_const_scale_diagonal!(x::T, x_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy3::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    soc_proj_const_scale!(x, x_part, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function rsoc_cone_proj_diagonal!(x::T, x_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy1::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    rsoc_proj_diagonal!(x, x_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function rsoc_cone_proj_const_scale_diagonal!(x::T, x_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy1::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    rsoc_proj_const_scale!(x, x_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function EXP_proj_diagonal!(x::T, dummy1::T, dummy2::T, dummy3::T, dummy5::T, dummy6::T, dummy7::T, D::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    exponent_proj_diagonal!(x, D)
end

function DUALEXPonent_proj_diagonal!(x::T,  dummy1::T, Dinv::T, dummy2::T, dummy4::T, temp::T, dummy5::T, dummy6::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj_diagonal!(x, Dinv, temp)
end

function primal_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner)
    time_start = time()
    @threads for i in 1:x.blkLen
        x.x_slice_proj_diagonal![i](x.x_slice[i],
                                    x.x_slice_part[i],
                                    diag_precond.Dr_product_inv_normalized.x_slice[i],
                                    diag_precond.Dr_product_inv_normalized.x_slice_part[i],
                                    diag_precond.Dr_product_inv_normalized_squared.x_slice_part[i],
                                    diag_precond.Dr.x_slice_part[i],
                                    diag_precond.Dr_temp.x_slice_part[i],
                                    diag_precond.Dr_product.x_slice_part[i],
                                    sol,
                                    x.t_warm_start,
                                    i,
                                    diag_precond.primalProjInfo[i])
    
    end
    time_end = time()
    global time_proj += time_end - time_start
end

function slack_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner)
    time_start = time()
    @threads for i in 1:x.blkLen
         x.x_slice_proj_diagonal![i](x.x_slice[i],
                                    x.x_slice_part[i],
                                    diag_precond.Dr_product_inv_normalized.x_slice[i],
                                    diag_precond.Dr_product_inv_normalized_squared.x_slice[i],
                                    diag_precond.Dr.x_slice[i],
                                    diag_precond.Dr_temp.x_slice[i],
                                    sol,
                                    diag_precond.Dr_product.x_slice[i],
                                    x.t_warm_start,
                                    i,
                                    diag_precond.slackProjInfo[i])
    end
    time_end = time()
    global time_proj += time_end - time_start
end

"""
function for primal projection
"""
function box_proj!(x::T, sol::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    x .= clamp.(x, sol.bl, sol.bu)
end

function slack_box_proj!(x::T, sol::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    x .= clamp.(x, sol.lambd_l, sol.lambd_u)
end

function soc_cone_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    soc_proj!(x)
end

function rsoc_cone_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}    
    rsoc_proj!(x)
end

function EXP_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    exponent_proj!(x)
end

function slack_EXP_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj!(x)
end

function DUALEXP_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj!(x)
end

function slack_DUALEXP_proj!(x::T, solDummy::solVecPrimal) where T<:AbstractVector{rpdhg_float}
    exponent_proj!(x)
end

function primal_proj!(x::primalVector, sol::solVecPrimal)
    time_start = time()
    @threads for i in 1:x.blkLen
         x.x_slice_proj![i](x.x_slice[i], sol)
    end
    time_end = time()
    global time_proj += time_end - time_start
end

function slack_proj!(x::primalVector, sol::solVecPrimal)
    time_start = time()
    @threads for i in 1:x.blkLen
         x.x_slice_proj_slack![i](x.x_slice[i], sol)
    end
    time_end = time()
    global time_proj += time_end - time_start
end


function setFunctionPointerPrimal!(sol::solVecPrimal, primalConstScale::Vector{Bool})
    for i in 1:sol.primal_sol.blkLen
        if sol.primal_sol.x_slice_func_symbol[i] == :proj_box!
            sol.primal_sol.x_slice_proj![i] = box_proj!
            sol.primal_sol.x_slice_proj_diagonal![i] = box_proj_diagonal!
            sol.primal_sol.x_slice_proj_slack![i] = slack_box_proj!
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_soc_cone!
            sol.primal_sol.x_slice_proj![i] = soc_cone_proj!
            if primalConstScale[i]
                sol.primal_sol.x_slice_proj_diagonal![i] = soc_cone_proj_const_scale!
            else
                sol.primal_sol.x_slice_proj_diagonal![i] = soc_cone_proj_diagonal!
            end
            sol.primal_sol.x_slice_proj_slack![i] = soc_cone_proj!
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_rsoc_cone!
            sol.primal_sol.x_slice_proj![i] = rsoc_cone_proj!
            if primalConstScale[i]
                sol.primal_sol.x_slice_proj_diagonal![i] = rsoc_cone_proj_const_scale!
            else
                sol.primal_sol.x_slice_proj_diagonal![i] = rsoc_cone_proj_diagonal!
            end
            sol.primal_sol.x_slice_proj_slack![i] = rsoc_cone_proj!
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_exp_cone!
            sol.primal_sol.x_slice_proj![i] = EXP_proj!
            sol.primal_sol.x_slice_proj_diagonal![i] = EXP_proj_diagonal!
            sol.primal_sol.x_slice_proj_slack![i] = DUALEXP_proj!
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_dual_exp_cone!
            sol.primal_sol.x_slice_proj![i] = DUALEXP_proj!
            sol.primal_sol.x_slice_proj_diagonal![i] = DUALEXPonent_proj_diagonal!
            sol.primal_sol.x_slice_proj_slack![i] = EXP_proj!
        end
    end
    sol.proj! = primal_proj!
    sol.slack_proj! = slack_proj!
    sol.proj_diagonal! = primal_proj_diagonal!
end


"""
function for dual diagonal projection
"""
function dual_free_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    return
end

function dual_positive_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    y .= max.(y, 0.0)
end

function dual_soc_proj_diagonal!(y::T, y_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    soc_proj_diagonal!(y, y_part, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end

function dual_soc_proj_const_scale_diagonal!(y::T, y_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    soc_proj_const_scale!(y, y_part, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end

function dual_rsoc_proj_diagonal!(y::T, y_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    rsoc_proj_diagonal!(y, y_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end

function dual_rsoc_proj_const_scale_diagonal!(y::T, y_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    rsoc_proj_const_scale!(y, y_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end


function dual_EXP_proj_diagonal!(y::T, dummy::T, D_scaled::T, dummy2::T, dummy3::T, dummy4::T, temp::T, dummy5::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj_diagonal!(y, D_scaled, temp)
end

function dual_DUALEXP_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dl_product::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo) where T<:AbstractVector{rpdhg_float}
    exponent_proj_diagonal!(y, Dl_product)
end

function dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner)
    time_start = time()
    @threads for i in 1:y.blkLen
         y.y_slice_proj_diagonal![i](y.y_slice[i],
                                    y.y_slice_part[i],
                                    diag_precond.Dl_product_inv_normalized.y_slice[i],
                                    diag_precond.Dl_product_inv_normalized.y_slice_part[i],
                                    diag_precond.Dl_product_inv_normalized_squared.y_slice_part[i],
                                    diag_precond.Dl.y_slice_part[i],
                                    diag_precond.Dl_temp.y_slice_part[i],
                                    diag_precond.Dl_product.y_slice[i],
                                    y.t_warm_start,
                                    i,
                                    diag_precond.dualProjInfo[i])
    end
    time_end = time()
    global time_proj += time_end - time_start
end

"""
function for dual projection
"""
function dual_free_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    return
end

function con_zero_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    y .= 0.0
end

function dual_positive_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    y .= max.(y, 0.0)
end

function dual_soc_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    soc_proj!(y)
end

function dual_rsoc_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    rsoc_proj!(y)
end

function dual_EXP_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj!(y)
end

function dual_DUALEXP_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    exponent_proj!(y)
end

function con_EXP_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    exponent_proj!(y)
end

function con_DUALEXP_proj!(y::T) where T<:AbstractVector{rpdhg_float}
    dualExponent_proj!(y)
end

function dual_proj!(y::dualVector)
    time_start = time()
    @threads for i in 1:y.blkLen
         y.y_slice_proj![i](y.y_slice[i])
    end
    time_end = time()
    global time_proj += time_end - time_start
end

function con_proj!(y::dualVector)
    time_start = time()
    @threads for i in 1:y.blkLen
         y.y_slice_con_proj![i](y.y_slice[i])
    end
    time_end = time()
    global time_proj += time_end - time_start
end



function setFunctionPointerDual!(dualSol::solVecDual, primalConstScale::Vector{Bool}, dualConstScale::Vector{Bool})
    for i in 1:dualSol.dual_sol.blkLen
        if dualSol.dual_sol.y_slice_func_symbol[i] == :dual_free_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_zero_proj!
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_positive_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_positive_proj!
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_soc_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_soc_proj!
            if dualConstScale[i]
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
            else
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
            end
            dualSol.dual_sol.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_soc_proj!
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_rsoc_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_rsoc_proj!
            if dualConstScale[i]
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
            else
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
            end
            dualSol.dual_sol.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_rsoc_proj!
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_exp_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_EXP_proj!
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_DUALEXP_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_DUALEXP_proj!
        end
    end
    dualSol.proj! = dual_proj!
    dualSol.proj_diagonal! = dual_proj_diagonal!
    dualSol.con_proj! = con_proj!
end



function is_monotonically_decreasing(buffer::CircularBuffer{rpdhg_float}, len::Integer)
    if length(buffer) < len
        return false
    end
    for i in 2:len
        if buffer[i] > buffer[i-1]
            return false
        end
    end
    return true
end

function is_monotonically_increasing(buffer::CircularBuffer{rpdhg_float}, len::Integer)
    if length(buffer) < len
        return false
    end
    for i in 2:len
        if buffer[i] < buffer[i-1]
            return false
        end
    end
    return true
end


function unionPrimalMV!(coeff::coeffUnion, x::Vector{rpdhg_float}, Ax::dualVector)
    # Ax .= A.G * x;
    # most time-consuming part
    mul!(Ax.y, coeff.G, x)
end


function union_adjoint!(coeffTrans::coeffUnion, y::dualVector, Aty::Vector{rpdhg_float})
    # Aty .= coeffTrans.G' * y;
    # most time-consuming part
    mul!(Aty, coeffTrans.G, y.y)
end

function AtAMVUnion!(coeffTrans::coeffTransType, coeff::coeffType, x::Vector{rpdhg_float}, Ax::dualVector, AtAx::Vector{rpdhg_float}) where{
    coeffTransType<:Union{coeffUnion}, coeffType<:Union{coeffUnion}}
    unionPrimalMV!(coeff, x, Ax);
    mul!(AtAx, coeffTrans.G, Ax.y);
end

function addCoeffdUnion!(coeff::coeffUnion, Gx::dualVector, w::rpdhg_float)
    # mA = size(coeff.A, 1);
    Gx.y .+= w * coeff.h
end

function dotCoeffdUnion(coeff::coeffUnion, y::dualVector)
    val = dot(coeff.h, y.y)
    return val
end


function power_method!(coeffTrans::coeffTransType, coeff::coeffType, AtAMV!::Function, Ab::dualVector; tol = 1e-3, maxiter = 1000) where
    {coeffType<:Union{coeffSplitQA, coeffUnion},
    coeffTransType<:Union{coeffSplitQA, coeffUnion}}
    println("Start power method for estimating the largest eigenvalue of the matrix.")
    n = coeff.n;
    b = normalize!(rand(n))
    lambda_old = 0.0;
    AtAb = similar(b);
    lambda = 0.0;
    for iter in 1:maxiter
        # AtAMV!(coeffTrans, coeff, b, Ab, AtAb);
        mul!(Ab.y, coeff.G, b)
        mul!(AtAb, coeffTrans.G, Ab.y)
        b .= AtAb;
        normalize!(b)
        # AtAMV!(coeffTrans, coeff, b, Ab, AtAb);
        mul!(Ab.y, coeff.G, b)
        mul!(AtAb, coeffTrans.G, Ab.y)
        lambda = dot(b, AtAb);
        if abs(lambda - lambda_old) < tol
            return sqrt(lambda), 0;
        end
        lambda_old = lambda;
        # println("iter: ", iter, " lambda: ", lambda)
   end
   println("Power method did not converge, quit since maxiter.")
   return sqrt(lambda), 1;
end

function AlambdaMax_cal!(A::AbstractMatrix)
    val, _ = eigsolve(A, 1, :LM)
    return sqrt(val[1])
end


function setFunctionPointerSolver!(solver::rpdhgSolver)
    setFunctionPointerPrimal!(solver.sol.x, solver.data.diagonal_scale.primalConstScale)
    setFunctionPointerPrimal!(solver.sol.y.slack, solver.data.diagonal_scale.dualConstScale)
    setFunctionPointerDual!(solver.sol.y, solver.data.diagonal_scale.primalConstScale, solver.data.diagonal_scale.dualConstScale)
    setFunctionPointer(solver)
end

function setFunctionPointer(solver::rpdhgSolver)
    @match solver.data.coeff begin
        _::coeffSplitQA => begin
            error("Not used, discard it, old function");
            solver.primalMV! = splitQAPrimalMV!
            solver.adjointMV! = splitQA_adjoint!
            solver.AtAMV! = AtAMVSplitQA!
            solver.addCoeffd! = addCoeffdSplitQA!
            solver.dotCoeffd = dotCoeffdSplitQA
        end
        _::coeffUnion => begin
            solver.primalMV! = unionPrimalMV!
            solver.adjointMV! = union_adjoint!
            solver.AtAMV! = AtAMVUnion!
            solver.addCoeffd! = addCoeffdUnion!
            solver.dotCoeffd = dotCoeffdUnion
        end
    end
end


function create_raw_data(;
    m::Integer,
    n::Integer,
    nb::Integer,
    c::Vector{rpdhg_float},
    coeff::coeffType,
    bl::Vector{rpdhg_float},
    bu::Vector{rpdhg_float},
    hNrm1::rpdhg_float,
    cNrm1::rpdhg_float,
    hNrmInf::rpdhg_float,
    cNrmInf::rpdhg_float,
)where coeffType<:Union{coeffSplitQA, coeffUnion}
    coeffCopy = deepcopy(coeff)
    if isa(coeffCopy, coeffSplitQA)
        error("Not used, discard it, old function");
        coeffCopyTrans = coeffSplitQA(
            Q = transpose(coeffCopy.Q),
            A = transpose(coeffCopy.A),
            h = coeffCopy.h,
            b = coeffCopy.b,
            m = coeffCopy.m,
            n = coeffCopy.n
        )
    else
        coeffCopyTrans = coeffUnion(
            G = transpose(coeffCopy.G),
            h = coeffCopy.h,
            m = coeffCopy.m,
            n = coeffCopy.n
        )
    end
    raw_data = rpdhgRawData(
        m = m,
        n = n,
        nb = nb,  
        c = deepcopy(c),
        coeff = coeffCopy,
        coeffTrans = coeffCopyTrans,
        bl = deepcopy(bl),
        bu = deepcopy(bu),
        hNrm1 = hNrm1,
        cNrm1 = cNrm1,
        hNrmInf = hNrmInf,
        cNrmInf = cNrmInf
    )
    return raw_data
end