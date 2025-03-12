"""
def_rpdhg_gen.jl
"""

function gen_lambd(bl::CuArray, bu::CuArray)
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

function box_proj_diagonal!(x::T, dummy1::T, dummy2::T, dummy3::T, dummy5::T, dummy6::T, dummy7::T, dummy8::T, sol::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 18
    x .= clamp.(x, sol.bl, sol.bu)
end
function soc_cone_proj_diagonal!(x::T, x_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy3::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 22
    soc_proj_diagonal!(x, x_part, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function soc_cone_proj_const_scale_diagonal!(x::T, x_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy3::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 21
    soc_proj_const_scale!(x, x_part, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function rsoc_cone_proj_diagonal!(x::T, x_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy1::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 25
    rsoc_proj_diagonal!(x, x_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function rsoc_cone_proj_const_scale_diagonal!(x::T, x_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_all_part::T, temp_all2_part::T, dummy1::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 24
    rsoc_proj_const_scale!(x, x_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_all_part, temp_all2_part, t_warm_start, i)
end

function EXP_proj_diagonal!(x::T, dummy1::T, dummy2::T, dummy3::T, dummy5::T, dummy6::T, dummy7::T, D::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 27
    exponent_proj_diagonal!(x, D)
end

function DUALEXPonent_proj_diagonal!(x::T,  dummy1::T, Dinv::T, dummy2::T, dummy4::T, temp::T, dummy5::T, dummy6::T, solDummy::solVecPrimal, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 29
    dualExponent_proj_diagonal!(x, Dinv, temp)
end

function massive_primal_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    massive_block_proj(x.x, sol.bl, sol.bu, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


function moderate_primal_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    moderate_block_proj(x.x, sol.bl, sol.bu, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


function sufficient_primal_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    sufficient_block_proj(x.x, sol.bl, sol.bu, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


function few_primal_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #     x.x_slice_proj_diagonal![i](x.x_slice[i],
    #                                 x.x_slice_part[i],
    #                                 diag_precond.Dr_product_inv_normalized.x_slice[i],
    #                                 diag_precond.Dr_product_inv_normalized.x_slice_part[i],
    #                                 diag_precond.Dr_product_inv_normalized_squared.x_slice_part[i],
    #                                 diag_precond.Dr.x_slice_part[i],
    #                                 diag_precond.Dr_temp.x_slice_part[i],
    #                                 diag_precond.Dr_product.x_slice_part[i],
    #                                 sol,
    #                                 x.t_warm_start,
    #                                 i,
    #                                 diag_precond.primalProjInfo[i],
    #                                 x.x_slice_length[i])
    # end
    few_block_proj(x.x, sol.bl, sol.bu, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start_cpu, x.x_slice_length, x.x_slice_length_cpu, x.blkLen, x.x_slice_proj_kernel_diagonal, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function massive_slack_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj_diagonal![i](x.x_slice[i],
    #                                 x.x_slice_part[i],
    #                                 diag_precond.Dr_product_inv_normalized.x_slice[i],
    #                                 diag_precond.Dr_product_inv_normalized_squared.x_slice[i],
    #                                 diag_precond.Dr.x_slice[i],
    #                                 diag_precond.Dr_temp.x_slice[i],
    #                                 sol,
    #                                 diag_precond.Dr_product.x_slice[i],
    #                                 x.t_warm_start,
    #                                 i,
    #                                 diag_precond.slackProjInfo[i])
    # end
    massive_block_proj(x.x, sol.lambd_l, sol.lambd_l, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_slack_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    moderate_block_proj(x.x, sol.lambd_l, sol.lambd_l, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function sufficient_slack_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    sufficient_block_proj(x.x, sol.lambd_l, sol.lambd_l, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_slack_proj_diagonal!(x::primalVector, sol::solVecPrimal, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj_diagonal![i](x.x_slice[i],
    #                                 x.x_slice_part[i],
    #                                 diag_precond.Dr_product_inv_normalized.x_slice[i],
    #                                 diag_precond.Dr_product_inv_normalized_squared.x_slice[i],
    #                                 diag_precond.Dr.x_slice[i],
    #                                 diag_precond.Dr_temp.x_slice[i],
    #                                 sol,
    #                                 diag_precond.Dr_product.x_slice[i],
    #                                 x.t_warm_start,
    #                                 i,
    #                                 diag_precond.slackProjInfo[i])
    # end
    few_block_proj(x.x, sol.lambd_l, sol.lambd_l, diag_precond.Dr_product_inv_normalized.x, diag_precond.Dr_product_inv_normalized_squared.x, diag_precond.Dr.x, diag_precond.Dr_temp.x, x.t_warm_start_device, x.cone_index_start_cpu, x.x_slice_length, x.x_slice_length_cpu, x.blkLen, x.x_slice_proj_kernel_slack, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

"""
function for primal projection
"""
function box_proj!(x::T, sol::solVecPrimal) where T<:CuArray
    # code: 17
    x .= clamp.(x, sol.bl, sol.bu)
end

function slack_box_proj!(x::T, sol::solVecPrimal) where T<:CuArray
    # code: 19
    x .= clamp.(x, sol.lambd_l, sol.lambd_u)
end

function soc_cone_proj!(x::T, solDummy::solVecPrimal) where T<:CuArray
    # code: 20
    soc_proj!(x)
end

function rsoc_cone_proj!(x::T, solDummy::solVecPrimal) where T<:CuArray    
    # code: 23
    rsoc_proj!(x)
end

function EXP_proj!(x::T, solDummy::solVecPrimal) where T<:CuArray
    # code: 26
    exponent_proj!(x)
end

function slack_EXP_proj!(x::T, solDummy::solVecPrimal) where T<:CuArray
    # code: 28
    dualExponent_proj!(x)
end

function DUALEXP_proj!(x::T, solDummy::solVecPrimal) where T<:CuArray
    # code: 28
    dualExponent_proj!(x)
end

function massive_primal_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj![i](x.x_slice[i], sol)
    # end
    massive_block_proj(x.x, sol.bl, sol.bu, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_primal_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj![i](x.x_slice[i], sol)
    # end
    moderate_block_proj(x.x, sol.bl, sol.bu, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


function sufficient_primal_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj![i](x.x_slice[i], sol)
    # end
    sufficient_block_proj(x.x, sol.bl, sol.bu, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_primal_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:x.blkLen
    #      x.x_slice_proj![i](x.x_slice[i], sol)
    # end
    few_block_proj(x.x, sol.bl, sol.bu, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start_cpu, x.x_slice_length, x.x_slice_length_cpu, x.blkLen, x.x_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function massive_slack_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    massive_block_proj(x.x, sol.lambd_l, sol.lambd_u, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_slack_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    moderate_block_proj(x.x, sol.lambd_l, sol.lambd_u, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function sufficient_slack_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    sufficient_block_proj(x.x, sol.lambd_l, sol.lambd_u, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start, x.x_slice_length, x.blkLen, x.x_slice_proj_kernel_slack_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_slack_proj!(x::primalVector, sol::solVecPrimal; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    few_block_proj(x.x, sol.lambd_l, sol.lambd_u, x.x, x.x, x.x, x.x, x.t_warm_start_device, x.cone_index_start_cpu, x.x_slice_length, x.x_slice_length_cpu, x.blkLen, x.x_slice_proj_kernel_slack, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

"""
function for setting the function pointer
    17: box_proj!
    18: box_proj_diagonal!
    19: slack_box_proj!
    20: soc_cone_proj!
    21: soc_cone_proj_const_scale!
    22: soc_cone_proj_diagonal!
    23: rsoc_cone_proj!
    24: rsoc_cone_proj_const_scale!
    25: rsoc_cone_proj_diagonal!
    26: EXP_proj!
    27: EXP_proj_diagonal!
    28: DUALEXP_proj!
    29: DUALEXPonent_proj_diagonal!
"""

function setFunctionPointerPrimal!(sol::solVecPrimal, primalConstScale::Vector{Bool})
    for i in 1:sol.primal_sol.blkLen
        if sol.primal_sol.x_slice_func_symbol[i] == :proj_box!
            sol.primal_sol.x_slice_proj![i] = box_proj!
            sol.primal_sol.x_slice_proj_diagonal![i] = box_proj_diagonal!
            sol.primal_sol.x_slice_proj_slack![i] = slack_box_proj!
            sol.primal_sol.x_slice_proj_kernel[i] = 17
            sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 18
            sol.primal_sol.x_slice_proj_kernel_slack[i] = 19
            if all(sol.bl .== -Inf) && all(sol.bu .== Inf)
                sol.primal_sol.x_slice_proj_kernel[i] = 0
                sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 1
            end
            if all(sol.lambd_l .== -Inf) && all(sol.lambd_u .== Inf)
                sol.primal_sol.x_slice_proj_kernel_slack[i] = 0
            end
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_soc_cone!
            sol.primal_sol.x_slice_proj![i] = soc_cone_proj!
            sol.primal_sol.x_slice_proj_kernel![i] = 20
            if primalConstScale[i]
                sol.primal_sol.x_slice_proj_diagonal![i] = soc_cone_proj_const_scale!
                sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 21
            else
                sol.primal_sol.x_slice_proj_diagonal![i] = soc_cone_proj_diagonal!
                sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 22
            end
            sol.primal_sol.x_slice_proj_slack![i] = soc_cone_proj!
            sol.primal_sol.x_slice_proj_kernel_slack[i] = 20
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_rsoc_cone!
            sol.primal_sol.x_slice_proj![i] = rsoc_cone_proj!
            sol.primal_sol.x_slice_proj_kernel[i] = 23
            if primalConstScale[i]
                sol.primal_sol.x_slice_proj_diagonal![i] = rsoc_cone_proj_const_scale!
                sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 24
            else
                sol.primal_sol.x_slice_proj_diagonal![i] = rsoc_cone_proj_diagonal!
                sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 25
            end
            sol.primal_sol.x_slice_proj_slack![i] = rsoc_cone_proj!
            sol.primal_sol.x_slice_proj_kernel_slack[i] = 23
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_exp_cone!
            sol.primal_sol.x_slice_proj![i] = EXP_proj!
            sol.primal_sol.x_slice_proj_kernel[i] = 26
            sol.primal_sol.x_slice_proj_diagonal![i] = EXP_proj_diagonal!
            sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 27
            sol.primal_sol.x_slice_proj_slack![i] = DUALEXP_proj!
            sol.primal_sol.x_slice_proj_kernel_slack[i] = 28
        elseif sol.primal_sol.x_slice_func_symbol[i] == :proj_dual_exp_cone!
            sol.primal_sol.x_slice_proj![i] = DUALEXP_proj!
            sol.primal_sol.x_slice_proj_kernel[i] = 28
            sol.primal_sol.x_slice_proj_diagonal![i] = DUALEXPonent_proj_diagonal!
            sol.primal_sol.x_slice_proj_kernel_diagonal[i] = 29
            sol.primal_sol.x_slice_proj_slack![i] = EXP_proj!
            sol.primal_sol.x_slice_proj_kernel_slack[i] = 26
        end
    end
    sol.primal_sol.x_slice_proj_kernel_device = CuArray{Int64}(sol.primal_sol.x_slice_proj_kernel)
    sol.primal_sol.x_slice_proj_kernel_diagonal_device = CuArray{Int64}(sol.primal_sol.x_slice_proj_kernel_diagonal)
    sol.primal_sol.x_slice_proj_kernel_slack_device = CuArray{Int64}(sol.primal_sol.x_slice_proj_kernel_slack)

    if sol.primal_sol.blkLen <= 3 && !any(k -> k in (8, 9, 10, 23, 24, 25), sol.primal_sol.x_slice_proj_kernel)
        println("few primal proj");
        sol.proj! = few_primal_proj!
        sol.slack_proj! = few_slack_proj!
        sol.proj_diagonal! = few_primal_proj_diagonal!
    elseif sol.primal_sol.blkLen <= 1000 || maximum(sol.primal_sol.x_slice_length_cpu[3:end]) >= 2000
        println("moderate primal proj");
        sol.proj! = moderate_primal_proj!
        sol.slack_proj! = moderate_slack_proj!
        sol.proj_diagonal! = moderate_primal_proj_diagonal!
    elseif sol.primal_sol.blkLen <= 60000 || maximum(sol.primal_sol.x_slice_length_cpu[3:end]) >= 150
        println("sufficient primal proj");
        sol.proj! = sufficient_primal_proj!
        sol.slack_proj! = sufficient_slack_proj!
        sol.proj_diagonal! = sufficient_primal_proj_diagonal!
    else
        println("massive primal proj");
        sol.proj! = massive_primal_proj!
        sol.slack_proj! = massive_slack_proj!
        sol.proj_diagonal! = massive_primal_proj_diagonal!
    end
end


"""
function for dual diagonal projection
"""
function dual_free_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 1
    return
end

function dual_positive_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 4
    y .= CUDA.max.(y, 0.0)
end

function dual_soc_proj_diagonal!(y::T, y_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 6
    soc_proj_diagonal!(y, y_part, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo, vector_length)
end

function dual_soc_proj_const_scale_diagonal!(y::T, y_part::T, dummy::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 7
    soc_proj_const_scale!(y, y_part, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end

function dual_rsoc_proj_diagonal!(y::T, y_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 9
    rsoc_proj_diagonal!(y, y_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end

function dual_rsoc_proj_const_scale_diagonal!(y::T, y_part::T, D_scaled::T, D_scaled_part::T, D_scaled_squared_part::T, temp_part::T, temp2_part::T, Dummy::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 10
    rsoc_proj_const_scale!(y, y_part, D_scaled, D_scaled_part, D_scaled_squared_part, temp_part, temp2_part, t_warm_start, i, projInfo)
end


function dual_EXP_proj_diagonal!(y::T, dummy::T, D_scaled::T, dummy2::T, dummy3::T, dummy4::T, temp::T, dummy5::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 12
    dualExponent_proj_diagonal!(y, D_scaled, temp)
end

function dual_DUALEXP_proj_diagonal!(y::T, dummy::T, dummy1::T, dummy2::T, dummy3::T, dummy4::T, dummy5::T, Dl_product::T, t_warm_start::Vector{rpdhg_float}, i::Integer, projInfo::timesInfo, vector_length::Integer) where T<:CuArray
    # code: 15
    exponent_proj_diagonal!(y, Dl_product)
end

function massive_dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    massive_block_proj(y.y, y.y, y.y, diag_precond.Dl_product_inv_normalized.y, diag_precond.Dl_product_inv_normalized_squared.y, diag_precond.Dl.y, diag_precond.Dl_temp.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    moderate_block_proj(y.y, y.y, y.y, diag_precond.Dl_product_inv_normalized.y, diag_precond.Dl_product_inv_normalized_squared.y, diag_precond.Dl.y, diag_precond.Dl_temp.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


function sufficient_dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    sufficient_block_proj(y.y, y.y, y.y, diag_precond.Dl_product_inv_normalized.y, diag_precond.Dl_product_inv_normalized_squared.y, diag_precond.Dl.y, diag_precond.Dl_temp.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_diagonal_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    few_block_proj(y.y, y.y, y.y, diag_precond.Dl_product_inv_normalized.y, diag_precond.Dl_product_inv_normalized_squared.y, diag_precond.Dl.y, diag_precond.Dl_temp.y, y.t_warm_start_device, y.cone_index_start_cpu, y.y_slice_length, y.y_slice_length_cpu, y.blkLen, y.y_slice_proj_kernel_diagonal, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

"""
function for dual projection
"""
function dual_free_proj!(y::T) where T<:CuArray
    # code: 0
    return
end

function con_zero_proj!(y::T) where T<:CuArray
    # code: 2
    y .= 0.0
end

function dual_positive_proj!(y::T) where T<:CuArray
    # code: 3
    y .= max.(y, 0.0)
end

function dual_soc_proj!(y::T) where T<:CuArray
    # code: 5
    soc_proj!(y)
end

function dual_rsoc_proj!(y::T) where T<:CuArray
    # code: 8
    rsoc_proj!(y)
end

function dual_EXP_proj!(y::T) where T<:CuArray
    # code: 11
    dualExponent_proj!(y)
end

function dual_DUALEXP_proj!(y::T) where T<:CuArray
    # code: 14
    exponent_proj!(y)
end

function con_EXP_proj!(y::T) where T<:CuArray
    # code: 13
    exponent_proj!(y)
end

function con_DUALEXP_proj!(y::T) where T<:CuArray
    # code: 16
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


function dual_proj_diagonal!(y::dualVector, diag_precond::Diagonal_preconditioner)
    time_start = time()
    for i in 1:y.blkLen
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
                                    diag_precond.dualProjInfo[i],
                                    y.y_slice_length[i])
    end
    time_end = time()
    global time_proj += time_end - time_start
end

function massive_dual_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:y.blkLen
    #      y.y_slice_proj![i](y.y_slice[i])
    # end
    massive_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_dual_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:y.blkLen
    #      y.y_slice_proj![i](y.y_slice[i])
    # end
    moderate_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function sufficient_dual_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:y.blkLen
    #      y.y_slice_proj![i](y.y_slice[i])
    # end
    sufficient_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_dual_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:y.blkLen
    #      y.y_slice_proj![i](y.y_slice[i])
    # end
    few_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start_cpu, y.y_slice_length, y.y_slice_length_cpu, y.blkLen, y.y_slice_proj_kernel, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function massive_con_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    # @threads for i in 1:y.blkLen
    #      y.y_slice_con_proj![i](y.y_slice[i])
    # end
    massive_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_con_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function moderate_con_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    moderate_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_con_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function sufficient_con_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    sufficient_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start, y.y_slice_length, y.blkLen, y.y_slice_con_proj_kernel_device, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end

function few_con_proj!(y::dualVector; abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12)
    time_start = time()
    few_block_proj(y.y, y.y, y.y, y.y, y.y, y.y, y.y, y.t_warm_start_device, y.cone_index_start_cpu, y.y_slice_length, y.y_slice_length_cpu, y.blkLen, y.y_slice_con_proj_kernel, abs_tol, rel_tol)
    time_end = time()
    global time_proj += time_end - time_start
end


"""
function for setting function pointers
    0: dual_free_proj!
    1: dual_free_proj_diagonal!
    2: con_zero_proj!
    3: dual_positive_proj!
    4: dual_positive_proj_diagonal!
    5: dual_soc_proj!
    6: dual_soc_proj_diagonal!
    7: dual_soc_proj_const_scale_diagonal!
    8: dual_rsoc_proj!
    9: dual_rsoc_proj_diagonal!
    10: dual_rsoc_proj_const_scale_diagonal!
    11: dual_EXP_proj!
    12: dual_EXP_proj_diagonal!
    13: con_EXP_proj!
    14: dual_DUALEXP_proj!
    15: dual_DUALEXP_proj_diagonal!
    16: con_DUALEXP_proj!
"""
function setFunctionPointerDual!(dualSol::solVecDual, primalConstScale::Vector{Bool}, dualConstScale::Vector{Bool})
    for i in 1:dualSol.dual_sol.blkLen
        if dualSol.dual_sol.y_slice_func_symbol[i] == :dual_free_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_free_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 0
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 0
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 0
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 0
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_free_proj_diagonal!
            dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 1
            dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 1
            dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 1
            dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 1
            dualSol.dual_sol.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_zero_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 2
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 2
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 2
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 2
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_positive_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_positive_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 3
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 3
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 3
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 3
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_positive_proj_diagonal!
            dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 4
            dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 4
            dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 4
            dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 4
            dualSol.dual_sol.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_positive_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 3
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 3
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 3
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 3
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_soc_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_soc_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 5
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 5
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 5
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 5
            if dualConstScale[i]
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_soc_proj_const_scale_diagonal!
                dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 7
                dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 7
                dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 7
                dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 7
            else
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_soc_proj_diagonal!
                dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 6
                dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 6
                dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 6
                dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 6
            end
            dualSol.dual_sol.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_soc_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 5
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 5
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 5
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 5
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_rsoc_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 8
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 8
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 8
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 8
            if dualConstScale[i]
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_rsoc_proj_const_scale_diagonal!
                dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 10
                dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 10
                dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 10
                dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 10
            else
                dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_rsoc_proj_diagonal!
                dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 9
                dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 9
                dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 9
                dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 9
            end
            dualSol.dual_sol.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = dual_rsoc_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 8
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 8
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 8
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 8
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_exp_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_EXP_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 11
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 11
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 11
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 11
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_EXP_proj_diagonal!
            dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 12
            dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 12
            dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 12
            dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 12
            dualSol.dual_sol.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_EXP_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 13
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 13
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 13
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 13
        elseif dualSol.dual_sol.y_slice_func_symbol[i] == :dual_DUALEXP_proj!
            dualSol.dual_sol.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_lag.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_mean.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol_temp.y_slice_proj![i] = dual_DUALEXP_proj!
            dualSol.dual_sol.y_slice_proj_kernel[i] = 14
            dualSol.dual_sol_lag.y_slice_proj_kernel[i] = 14
            dualSol.dual_sol_mean.y_slice_proj_kernel[i] = 14
            dualSol.dual_sol_temp.y_slice_proj_kernel[i] = 14
            dualSol.dual_sol.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_lag.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_mean.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol_temp.y_slice_proj_diagonal![i] = dual_DUALEXP_proj_diagonal!
            dualSol.dual_sol.y_slice_proj_kernel_diagonal[i] = 15
            dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal[i] = 15
            dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal[i] = 15
            dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal[i] = 15
            dualSol.dual_sol.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_lag.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_mean.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol_temp.y_slice_con_proj![i] = con_DUALEXP_proj!
            dualSol.dual_sol.y_slice_con_proj_kernel[i] = 16
            dualSol.dual_sol_lag.y_slice_con_proj_kernel[i] = 16
            dualSol.dual_sol_mean.y_slice_con_proj_kernel[i] = 16
            dualSol.dual_sol_temp.y_slice_con_proj_kernel[i] = 16
        end
    end
    dualSol.dual_sol.y_slice_proj_kernel_device = CuArray{Int64}(dualSol.dual_sol.y_slice_proj_kernel)
    dualSol.dual_sol.y_slice_proj_kernel_diagonal_device = CuArray{Int64}(dualSol.dual_sol.y_slice_proj_kernel_diagonal)
    dualSol.dual_sol.y_slice_con_proj_kernel_device = CuArray{Int64}(dualSol.dual_sol.y_slice_con_proj_kernel)
    dualSol.dual_sol_lag.y_slice_proj_kernel_device = dualSol.dual_sol.y_slice_proj_kernel_device
    dualSol.dual_sol_lag.y_slice_proj_kernel_diagonal_device = dualSol.dual_sol.y_slice_proj_kernel_diagonal_device
    dualSol.dual_sol_lag.y_slice_con_proj_kernel_device = dualSol.dual_sol.y_slice_con_proj_kernel_device
    dualSol.dual_sol_mean.y_slice_proj_kernel_device = dualSol.dual_sol.y_slice_proj_kernel_device
    dualSol.dual_sol_mean.y_slice_proj_kernel_diagonal_device = dualSol.dual_sol.y_slice_proj_kernel_diagonal_device
    dualSol.dual_sol_mean.y_slice_con_proj_kernel_device = dualSol.dual_sol.y_slice_con_proj_kernel_device
    dualSol.dual_sol_temp.y_slice_proj_kernel_device = dualSol.dual_sol.y_slice_proj_kernel_device
    dualSol.dual_sol_temp.y_slice_proj_kernel_diagonal_device = dualSol.dual_sol.y_slice_proj_kernel_diagonal_device
    dualSol.dual_sol_temp.y_slice_con_proj_kernel_device = dualSol.dual_sol.y_slice_con_proj_kernel_device

    if dualSol.dual_sol.blkLen <= 3 && !any(k -> k in (8, 9, 10, 23, 24, 25), dualSol.dual_sol.y_slice_proj_kernel)
        println("few dual proj");
        dualSol.proj! = few_dual_proj!
        dualSol.proj_diagonal! = few_dual_proj_diagonal!
        dualSol.con_proj! = few_con_proj!
    elseif dualSol.dual_sol.blkLen <= 1000 || maximum(dualSol.dual_sol.y_slice_length_cpu[3:end]) >= 2000
        println("moderate dual proj");
        dualSol.proj! = moderate_dual_proj!
        dualSol.proj_diagonal! = moderate_dual_proj_diagonal!
        dualSol.con_proj! = moderate_con_proj!
    elseif dualSol.dual_sol.blkLen <= 60000 || maximum(dualSol.dual_sol.y_slice_length_cpu[3:end]) >= 150
        println("sufficient dual proj");
        dualSol.proj! = sufficient_dual_proj!
        dualSol.proj_diagonal! = sufficient_dual_proj_diagonal!
        dualSol.con_proj! = sufficient_con_proj!
    else
        println("massive dual proj");
        dualSol.proj! = massive_dual_proj!
        dualSol.proj_diagonal! = massive_dual_proj_diagonal!
        dualSol.con_proj! = massive_con_proj!
    end
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


function unionPrimalMV!(coeff::coeffUnion, x::CuArray, Ax::dualVector)
    # Ax .= A.G * x;
    # most time-consuming part
    # mul!(Ax.y, coeff.d_G, x)
    CUDA.CUSPARSE.mv!('N', 1, coeff.d_G, x, 0, Ax.y, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end


function union_adjoint!(coeff::coeffUnion, y::dualVector, Aty::CuArray)
    # Aty .= coeffTrans.G' * y;
    # most time-consuming part
    # mul!(Aty, coeffTrans.d_G, y.y)
    CUDA.CUSPARSE.mv!('T', 1, coeff.d_G, y.y, 0, Aty, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function AtAMVUnion!(coeffTrans::coeffTransType, coeff::coeffType, x::CuArray, Ax::dualVector, AtAx::CuArray) where{
    coeffTransType<:Union{coeffUnion}, coeffType<:Union{coeffUnion}}
    unionPrimalMV!(coeff, x, Ax);
    # mul!(AtAx, coeffTrans.d_G, Ax.y);
    CUDA.CUSPARSE.mv!('T', 1, coeff.d_G, Ax.y, 0, AtAx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function addCoeffdUnion!(coeff::coeffUnion, Gx::dualVector, w::rpdhg_float)
    # mA = size(coeff.A, 1);
    # Gx.y .+= w * coeff.d_h
    axpyz(Gx.y, w, coeff.d_h, Gx.y, coeff.m)
end

function dotCoeffdUnion(coeff::coeffUnion, y::dualVector)
    # val = dot(coeff.d_h, y.y)
    val = CUDA.dot(coeff.d_h, y.y)
    return val
end


function power_method!(coeffTrans::coeffTransType, coeff::coeffType, AtAMV!::Function, Ab::dualVector; tol = 1e-3, maxiter = 1000) where
    {coeffType<:Union{ coeffUnion},
    coeffTransType<:Union{ coeffUnion}}
    @info "Start power method for estimating the largest eigenvalue of the matrix."
    n = coeff.n;
    b = normalize!(rand(n))
    b = CuArray(b)
    lambda_old = 0.0;
    AtAb = similar(b);
    lambda = 0.0;
    for iter in 1:maxiter
        # AtAMV!(coeffTrans, coeff, b, Ab, AtAb);
        CUDA.CUSPARSE.mv!('N', 1, coeff.d_G, b, 0, Ab.y, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('T', 1, coeff.d_G, Ab.y, 0, AtAb, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        b .= AtAb;
        normalize!(b)
        # AtAMV!(coeffTrans, coeff, b, Ab, AtAb);
        CUDA.CUSPARSE.mv!('N', 1, coeff.d_G, b, 0, Ab.y, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('T', 1, coeff.d_G, Ab.y, 0, AtAb, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda = CUDA.dot(b, AtAb);
        if abs(lambda - lambda_old) < tol
            return sqrt(lambda), 0;
        end
        lambda_old = lambda;
    end
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
    setFunctionPointerDual!(solver.sol.dual_sol_temp, solver.data.diagonal_scale.primalConstScale, solver.data.diagonal_scale.dualConstScale)
    setFunctionPointer(solver)
end

function setFunctionPointer(solver::rpdhgSolver)
    @match solver.data.coeff begin
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
    c_cpu::AbstractVector{rpdhg_float},
    coeff::coeffType,
    bl_cpu::AbstractVector{rpdhg_float},
    bu_cpu::AbstractVector{rpdhg_float},
    hNrm1::rpdhg_float, cNrm1::rpdhg_float, 
    hNrmInf::rpdhg_float, cNrmInf::rpdhg_float) where coeffType<:Union{coeffUnion}
    # println("before coeffCopy")
    # CUDA.memory_status()
    coeffCopy = deepcopy(coeff)
    # println("after coeffCopy")
    # CUDA.memory_status()
    coeffCopyTrans = coeffUnion(
        G = nothing,
        h = nothing,
        m = coeffCopy.m,
        n = coeffCopy.n,
        d_G = coeffCopy.d_G,
        d_h = coeffCopy.d_h,
    )
    # println("before raw_data")
    # CUDA.memory_status()
    raw_data = rpdhgRawData(
        m = m,
        n = n,
        nb = nb,  
        c = CuArray(c_cpu),
        coeff = coeffCopy,
        coeffTrans = coeffCopyTrans,
        bl = CuArray(bl_cpu),
        bu = CuArray(bu_cpu),
        hNrm1 = hNrm1,
        cNrm1 = cNrm1,
        hNrmInf = hNrmInf,
        cNrmInf = cNrmInf,
    )
    # println("after raw_data")
    # CUDA.memory_status()
    return raw_data
end
