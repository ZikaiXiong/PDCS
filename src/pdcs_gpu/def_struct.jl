
mutable struct primalVector
    x::CuArray
    xbox::CuArray
    x_slice::Vector{CuArray}
    t_warm_start::Vector{rpdhg_float}
    t_warm_start_device::CuArray
    x_slice_part::Vector{CuArray}
    x_slice_proj!::Vector{Function}
    x_slice_proj_kernel::Vector{Int64}
    x_slice_proj_kernel_device::CuArray{Int64}
    x_slice_proj_diagonal!::Vector{Function}
    x_slice_proj_kernel_diagonal::Vector{Int64}
    x_slice_proj_kernel_diagonal_device::CuArray{Int64}
    x_slice_proj_slack!::Vector{Function}
    x_slice_proj_kernel_slack::Vector{Int64}
    x_slice_proj_kernel_slack_device::CuArray{Int64}
    x_slice_func_symbol::Vector{Symbol}
    blkLen::Integer
    box_index::Integer
    soc_cone_indices_start::Vector{Integer}
    soc_cone_indices_end::Vector{Integer}
    rsoc_cone_indices_start::Vector{Integer}
    rsoc_cone_indices_end::Vector{Integer}
    exp_cone_indices_start::Vector{Integer}
    exp_cone_indices_end::Vector{Integer}
    dual_exp_cone_indices_start::Vector{Integer}
    dual_exp_cone_indices_end::Vector{Integer}
    x_slice_length::CuArray{Int64}
    x_slice_length_cpu::Vector{Int64}
    cone_index_start::CuArray{Int64}
    cone_index_start_cpu::Vector{Int64}
    function primalVector(; x, box_index, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end, x_slice_length_cpu = nothing, x_slice_length = nothing, cone_index_start_cpu = nothing, cone_index_start = nothing, place_holder = false)
        if !place_holder
            blkLen = length(soc_cone_indices_start) + length(rsoc_cone_indices_start) + length(exp_cone_indices_start) + length(dual_exp_cone_indices_start)
            first_init = true
            if box_index > 0
                blkLen += 1
            end
            if blkLen > 0
                x_slice = Vector{CuArray}(undef, blkLen)
                x_slice_part = Vector{CuArray}(undef, blkLen)
                x_slice_proj! = Vector{Function}(undef, blkLen)
                x_slice_proj_diagonal! = Vector{Function}(undef, blkLen)
                x_slice_proj_kernel = Vector{Int64}(undef, blkLen)
                x_slice_proj_kernel_device = CuArray{Int64}([])
                x_slice_proj_kernel_diagonal = Vector{Int64}(undef, blkLen)
                x_slice_proj_kernel_diagonal_device = CuArray{Int64}([])
                x_slice_proj_slack! = Vector{Function}(undef, blkLen)
                x_slice_proj_kernel_slack = Vector{Int64}(undef, blkLen)
                x_slice_proj_kernel_slack_device = CuArray{Int64}([])
                x_slice_func_symbol = Vector{Symbol}(undef, blkLen)
                t_warm_start = Vector{rpdhg_float}(undef, blkLen)
                t_warm_start .= 1.0
                if x_slice_length_cpu === nothing
                    x_slice_length_cpu = Vector{Int64}(undef, blkLen)
                else
                    first_init = false
                end
                if cone_index_start_cpu === nothing
                    cone_index_start_cpu = Vector{Int64}(undef, blkLen)
                else
                    first_init = false
                end
            else
                x_slice = Vector{CuArray}([])
                x_slice_proj! = Vector{Function}([])
                x_slice_proj_kernel = Vector{Int64}([])
                x_slice_proj_kernel_device = CuArray{Int64}([])
                x_slice_proj_diagonal! = Vector{Function}([])
                x_slice_proj_kernel_diagonal = Vector{Int64}([])
                x_slice_proj_kernel_diagonal_device = CuArray{Int64}([])
                x_slice_proj_slack! = Vector{Function}([])
                x_slice_proj_kernel_slack = Vector{Int64}([])
                x_slice_proj_kernel_slack_device = CuArray{Int64}([])
                x_slice_func_symbol = Vector{Symbol}([])
                t_warm_start = Vector{rpdhg_float}([])
                if x_slice_length_cpu === nothing
                    x_slice_length_cpu = Vector{Int64}([])
                else
                    first_init = false
                end
                if cone_index_start === nothing
                    cone_index_start = Vector{Int64}([])
                else
                    first_init = false
                end
            end
            baseIndex = 1
            if box_index > 0
                xbox = @view x[1:box_index]
                x_slice[baseIndex] = xbox
                x_slice_proj![baseIndex] = x -> println("proj_box! not implemented");
                x_slice_proj_diagonal![baseIndex] = x -> println("proj_box_diagonal! not implemented"); 
                x_slice_part[baseIndex] = @view x[1:box_index]
                x_slice_func_symbol[baseIndex] = :proj_box!
                x_slice_proj_slack![baseIndex] = x -> println("proj_box_slack! not implemented");
                if first_init
                    x_slice_length_cpu[baseIndex] = box_index
                    cone_index_start_cpu[baseIndex] = 0
                end
                baseIndex += 1
            else
                xbox = CuArray([])
            end
            if length(soc_cone_indices_start) > 0
                for (start_idx, end_idx) in zip(soc_cone_indices_start, soc_cone_indices_end)
                    x_slice[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_proj![baseIndex] = x -> println("proj_soc_cone! not implemented");
                    x_slice_proj_diagonal![baseIndex] = x -> println("proj_soc_cone_diagonal! not implemented");
                    x_slice_part[baseIndex] = @view x[start_idx+1:end_idx]
                    x_slice_func_symbol[baseIndex] = :proj_soc_cone!
                    x_slice_proj_slack![baseIndex] = x -> println("proj_soc_cone_slack! not implemented");
                    if first_init
                        x_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                        cone_index_start_cpu[baseIndex] = start_idx - 1
                    end
                    baseIndex += 1
                end
            end
            if length(rsoc_cone_indices_start) > 0
                for (start_idx, end_idx) in zip(rsoc_cone_indices_start, rsoc_cone_indices_end)
                    x_slice[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_proj![baseIndex] = x -> println("proj_rsoc_cone! not implemented");
                    x_slice_proj_diagonal![baseIndex] = x -> println("proj_rsoc_cone_diagonal! not implemented");
                    x_slice_part[baseIndex] = @view x[start_idx+2:end_idx]
                    x_slice_func_symbol[baseIndex] = :proj_rsoc_cone!
                    x_slice_proj_slack![baseIndex] = x -> println("proj_rsoc_cone_slack! not implemented");
                    if first_init
                        x_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                        cone_index_start_cpu[baseIndex] = start_idx - 1
                    end
                    baseIndex += 1
                end
            end
            if length(exp_cone_indices_start) > 0
                for (start_idx, end_idx) in zip(exp_cone_indices_start, exp_cone_indices_end)
                    x_slice[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_proj![baseIndex] = x -> println("proj_exp_cone! not implemented");
                    x_slice_proj_diagonal![baseIndex] = x -> println("proj_exp_cone_diagonal! not implemented");
                    x_slice_part[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_func_symbol[baseIndex] = :proj_exp_cone!
                    x_slice_proj_slack![baseIndex] = x -> println("proj_exp_cone_slack! not implemented");
                    if first_init
                        x_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                        cone_index_start_cpu[baseIndex] = start_idx - 1
                    end
                    baseIndex += 1
                end
            end
            if length(dual_exp_cone_indices_start) > 0
                for (start_idx, end_idx) in zip(dual_exp_cone_indices_start, dual_exp_cone_indices_end)
                    x_slice[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_proj![baseIndex] = x -> println("proj_dual_exp_cone! not implemented");
                    x_slice_proj_diagonal![baseIndex] = x -> println("proj_dual_exp_cone_diagonal! not implemented");
                    x_slice_part[baseIndex] = @view x[start_idx:end_idx]
                    x_slice_func_symbol[baseIndex] = :proj_dual_exp_cone!
                    x_slice_proj_slack![baseIndex] = x -> println("proj_dual_exp_cone_slack! not implemented");
                    if first_init
                        x_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                        cone_index_start_cpu[baseIndex] = start_idx - 1
                    end
                    baseIndex += 1
                end
            end
            if first_init
                x_slice_length = CuArray(x_slice_length_cpu)
                cone_index_start = CuArray(cone_index_start_cpu)
            end
            new(x, xbox, x_slice, t_warm_start, CuArray(t_warm_start), x_slice_part, x_slice_proj!, x_slice_proj_kernel, x_slice_proj_kernel_device, x_slice_proj_diagonal!, x_slice_proj_kernel_diagonal, x_slice_proj_kernel_diagonal_device, x_slice_proj_slack!, x_slice_proj_kernel_slack, x_slice_proj_kernel_slack_device, x_slice_func_symbol, blkLen, box_index, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end, x_slice_length, x_slice_length_cpu, cone_index_start, cone_index_start_cpu)
        else
            x = CuArray([0.0])
            xbox = CuArray([0.0])
            x_slice = Vector{CuArray}(undef, 2)
            x_slice[1] = CuArray([0.0])
            x_slice[2] = CuArray([0.0])
            t_warm_start = Vector{rpdhg_float}([0.0])
            t_warm_start_device = CuArray([0.0])
            x_slice_part = Vector{CuArray}(undef, 2)
            x_slice_part[1] = CuArray([0.0])
            x_slice_part[2] = CuArray([0.0])
            x_slice_proj! = Vector{Function}([])
            x_slice_proj_kernel = Vector{Int64}([])
            x_slice_proj_kernel_device = CuArray{Int64}([0])
            x_slice_proj_diagonal! = Vector{Function}([])
            x_slice_proj_kernel_diagonal = Vector{Int64}([])
            x_slice_proj_kernel_diagonal_device = CuArray{Int64}([0])
            x_slice_proj_slack! = Vector{Function}([])
            x_slice_proj_kernel_slack = Vector{Int64}([])
            x_slice_proj_kernel_slack_device = CuArray{Int64}([0])
            x_slice_func_symbol = Vector{Symbol}([])
            blkLen = 0
            box_index = 0
            soc_cone_indices_start = Vector{Integer}([])
            soc_cone_indices_end = Vector{Integer}([])
            rsoc_cone_indices_start = Vector{Integer}([])
            rsoc_cone_indices_end = Vector{Integer}([])
            exp_cone_indices_start = Vector{Integer}([])
            exp_cone_indices_end = Vector{Integer}([])
            dual_exp_cone_indices_start = Vector{Integer}([])
            dual_exp_cone_indices_end = Vector{Integer}([])
            x_slice_length = CuArray{Int64}([0])
            x_slice_length_cpu = Vector{Int64}([])
            cone_index_start = CuArray{Int64}([0])
            cone_index_start_cpu = Vector{Int64}([])
            new(x, xbox, x_slice, t_warm_start, t_warm_start_device, x_slice_part, x_slice_proj!, x_slice_proj_kernel, x_slice_proj_kernel_device, x_slice_proj_diagonal!, x_slice_proj_kernel_diagonal, x_slice_proj_kernel_diagonal_device, x_slice_proj_slack!, x_slice_proj_kernel_slack, x_slice_proj_kernel_slack_device, x_slice_func_symbol, blkLen, box_index, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end, x_slice_length, x_slice_length_cpu, cone_index_start, cone_index_start_cpu)
        end
    end
end


function deepCopyPrimalVector(primal_sol::primalVector)
    primal_sol_copy = primalVector(x = copy(primal_sol.x),
                                    box_index = primal_sol.box_index,
                                    soc_cone_indices_start = primal_sol.soc_cone_indices_start,
                                    soc_cone_indices_end = primal_sol.soc_cone_indices_end,
                                    rsoc_cone_indices_start = primal_sol.rsoc_cone_indices_start,
                                    rsoc_cone_indices_end = primal_sol.rsoc_cone_indices_end,
                                    exp_cone_indices_start = primal_sol.exp_cone_indices_start,
                                    exp_cone_indices_end = primal_sol.exp_cone_indices_end,
                                    dual_exp_cone_indices_start = primal_sol.dual_exp_cone_indices_start,
                                    dual_exp_cone_indices_end = primal_sol.dual_exp_cone_indices_end,
                                    x_slice_length = primal_sol.x_slice_length,
                                    x_slice_length_cpu = primal_sol.x_slice_length_cpu,
                                    cone_index_start = primal_sol.cone_index_start,
                                    cone_index_start_cpu = primal_sol.cone_index_start_cpu)
    return primal_sol_copy
end

function deepCopyPrimalVector_null(primal_sol::primalVector)
    primal_sol_copy = primalVector(x = CuArray([0.0]),
                                    box_index = primal_sol.box_index,
                                    soc_cone_indices_start = primal_sol.soc_cone_indices_start,
                                    soc_cone_indices_end = primal_sol.soc_cone_indices_end,
                                    rsoc_cone_indices_start = primal_sol.rsoc_cone_indices_start,
                                    rsoc_cone_indices_end = primal_sol.rsoc_cone_indices_end,
                                    exp_cone_indices_start = primal_sol.exp_cone_indices_start,
                                    exp_cone_indices_end = primal_sol.exp_cone_indices_end,
                                    dual_exp_cone_indices_start = primal_sol.dual_exp_cone_indices_start,
                                    dual_exp_cone_indices_end = primal_sol.dual_exp_cone_indices_end,
                                    x_slice_length = primal_sol.x_slice_length,
                                    x_slice_length_cpu = primal_sol.x_slice_length_cpu,
                                    cone_index_start = primal_sol.cone_index_start,
                                    cone_index_start_cpu = primal_sol.cone_index_start_cpu,
                                    place_holder = true)
    return primal_sol_copy
end


mutable struct dualVector
    y::CuArray
    y_slice::Vector{CuArray}
    y_slice_part::Vector{CuArray}
    t_warm_start::Vector{rpdhg_float}
    t_warm_start_device::CuArray
    y_slice_proj!::Vector{Function}
    y_slice_proj_kernel::Vector{Int64}
    y_slice_proj_kernel_device::CuArray{Int64}
    y_slice_proj_diagonal!::Vector{Function}
    y_slice_proj_kernel_diagonal::Vector{Int64}
    y_slice_proj_kernel_diagonal_device::CuArray{Int64}
    y_slice_con_proj!::Vector{Function}
    y_slice_con_proj_kernel::Vector{Int64}
    y_slice_con_proj_kernel_device::CuArray{Int64}
    y_slice_func_symbol::Vector{Symbol}
    blkLen::Integer
    m::Integer
    mGzero::Integer
    mGnonnegative::Integer
    soc_cone_indices_start::Vector{Integer}
    soc_cone_indices_end::Vector{Integer}
    rsoc_cone_indices_start::Vector{Integer}
    rsoc_cone_indices_end::Vector{Integer}
    exp_cone_indices_start::Vector{Integer}
    exp_cone_indices_end::Vector{Integer}
    dual_exp_cone_indices_start::Vector{Integer}
    dual_exp_cone_indices_end::Vector{Integer}
    y_slice_length::CuArray{Int64}
    y_slice_length_cpu::Vector{Int64}
    cone_index_start::CuArray{Int64}
    cone_index_start_cpu::Vector{Int64}
    function dualVector(; y, m, mGzero, mGnonnegative, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end, y_slice_length_cpu = nothing, y_slice_length = nothing, cone_index_start_cpu = nothing, cone_index_start = nothing)
        blkLen = length(soc_cone_indices_start) + length(rsoc_cone_indices_start) + length(exp_cone_indices_start) + length(dual_exp_cone_indices_start)
        first_init = true
        if mGzero > 0
            blkLen += 1
        end
        if mGnonnegative > 0
            blkLen += 1
        end
        if blkLen > 0
            y_slice = Vector{CuArray}(undef, blkLen)
            y_slice_part = Vector{CuArray}(undef, blkLen)
            y_slice_proj! = Vector{Function}(undef, blkLen)
            y_slice_proj_diagonal! = Vector{Function}(undef, blkLen)
            y_slice_proj_kernel = Vector{Int64}(undef, blkLen)
            y_slice_proj_kernel_device = CuArray{Int64}([])
            y_slice_proj_kernel_diagonal = Vector{Int64}(undef, blkLen)
            y_slice_proj_kernel_diagonal_device = CuArray{Int64}([])
            y_slice_con_proj! = Vector{Function}(undef, blkLen)
            y_slice_con_proj_kernel = Vector{Int64}(undef, blkLen)
            y_slice_con_proj_kernel_device = CuArray{Int64}([])
            y_slice_func_symbol = Vector{Symbol}(undef, blkLen)
            t_warm_start = Vector{rpdhg_float}(undef, blkLen)
            if y_slice_length_cpu === nothing
                y_slice_length_cpu = Vector{Int64}(undef, blkLen)
            else
                first_init = false
            end
            if cone_index_start_cpu === nothing
                cone_index_start_cpu = Vector{Int64}(undef, blkLen)
            else
                first_init = false
            end
            t_warm_start .= 1.0
        else
            y_slice = Vector{CuArray}([])
            y_slice_proj! = Vector{Function}([])
            y_slice_part = Vector{CuArray}([])
            y_slice_proj_diagonal! = Vector{Function}([])
            y_slice_proj_kernel = Vector{Int64}([])
            y_slice_proj_kernel_device = CuArray{Int64}([])
            y_slice_proj_kernel_diagonal = Vector{Int64}([])
            y_slice_proj_kernel_diagonal_device = CuArray{Int64}([])
            y_slice_con_proj! = Vector{Function}([])
            y_slice_con_proj_kernel = Vector{Int64}([])
            y_slice_con_proj_kernel_device = CuArray{Int64}([])
            y_slice_func_symbol = Vector{Symbol}([])
            t_warm_start = Vector{rpdhg_float}([])
            y_slice_length_cpu = Vector{Int64}([])
            cone_index_start_cpu = Vector{Int64}([])
        end
        baseIndex = 1
        if mGzero > 0
            y_free = @view y[1:mGzero];
            y_slice[baseIndex] = @view y[1:mGzero];
            y_slice_proj![baseIndex] = x -> println("dual_zero_proj! not implemented");
            y_slice_proj_diagonal![baseIndex] = x -> println("dual_zero_proj_diagonal! not implemented");
            y_slice_con_proj![baseIndex] = x -> println("dual_free_proj_con! not implemented");
            y_slice_func_symbol[baseIndex] = :dual_free_proj!
            y_slice_part[baseIndex] = @view y[1:mGzero]
            if first_init
                y_slice_length_cpu[baseIndex] = mGzero
                cone_index_start_cpu[baseIndex] = 0
            end
            baseIndex += 1
        end
        if mGnonnegative > 0
            y_pos = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice[baseIndex] = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice_proj![baseIndex] = x -> println("dual_positive_proj! not implemented");
            y_slice_proj_diagonal![baseIndex] = x -> println("dual_positive_proj_diagonal! not implemented");
            y_slice_part[baseIndex] = @view y[mGzero + 1 : mGzero + mGnonnegative]
            y_slice_func_symbol[baseIndex] = :dual_positive_proj!
            y_slice_con_proj![baseIndex] = x -> println("dual_positive_proj_con! not implemented");
            if first_init
                y_slice_length_cpu[baseIndex] = mGnonnegative
                cone_index_start_cpu[baseIndex] = mGzero
            end
            baseIndex += 1
        end
        if length(soc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(soc_cone_indices_start, soc_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_soc_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_soc_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_soc_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx+1:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_soc_proj!
                if first_init
                    y_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                    cone_index_start_cpu[baseIndex] = start_idx - 1
                end
                baseIndex += 1
            end
        end
        if length(rsoc_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(rsoc_cone_indices_start, rsoc_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_rsoc_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_rsoc_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_rsoc_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx+2:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_rsoc_proj!
                if first_init
                    y_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                    cone_index_start_cpu[baseIndex] = start_idx - 1
                end
                baseIndex += 1
            end
        end
        if length(exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(exp_cone_indices_start, exp_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_exp_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_exp_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_exp_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_exp_proj!
                if first_init
                    y_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                    cone_index_start_cpu[baseIndex] = start_idx - 1
                end
                baseIndex += 1
            end
        end
        if length(dual_exp_cone_indices_start) > 0
            for (start_idx, end_idx) in zip(dual_exp_cone_indices_start, dual_exp_cone_indices_end)
                y_slice[baseIndex] = @view y[start_idx:end_idx]
                y_slice_proj![baseIndex] = x -> println("dual_dual_exp_proj! not implemented");
                y_slice_proj_diagonal![baseIndex] = x -> println("dual_dual_exp_proj_diagonal! not implemented");
                y_slice_con_proj![baseIndex] = x -> println("dual_dual_exp_proj_con! not implemented");
                y_slice_part[baseIndex] = @view y[start_idx:end_idx]
                y_slice_func_symbol[baseIndex] = :dual_DUALEXP_proj!
                if first_init
                    y_slice_length_cpu[baseIndex] = end_idx - start_idx + 1
                    cone_index_start_cpu[baseIndex] = start_idx - 1
                end
                baseIndex += 1
            end
        end
        if first_init
            y_slice_length = CuArray(y_slice_length_cpu)
            cone_index_start = CuArray(cone_index_start_cpu)
        end
        new(y, y_slice, y_slice_part,t_warm_start, CuArray(t_warm_start), y_slice_proj!, y_slice_proj_kernel, y_slice_proj_kernel_device, y_slice_proj_diagonal!, y_slice_proj_kernel_diagonal, y_slice_proj_kernel_diagonal_device, y_slice_con_proj!, y_slice_con_proj_kernel, y_slice_con_proj_kernel_device, y_slice_func_symbol, blkLen, m, mGzero, mGnonnegative, soc_cone_indices_start, soc_cone_indices_end, rsoc_cone_indices_start, rsoc_cone_indices_end, exp_cone_indices_start, exp_cone_indices_end, dual_exp_cone_indices_start, dual_exp_cone_indices_end, y_slice_length, y_slice_length_cpu, cone_index_start, cone_index_start_cpu)
    end
end

function deepCopyDualVector(dual_sol::dualVector)
    dual_sol_copy = dualVector(y = copy(dual_sol.y),
                                m = dual_sol.m,
                                mGzero = dual_sol.mGzero,
                                mGnonnegative = dual_sol.mGnonnegative,
                                soc_cone_indices_start = dual_sol.soc_cone_indices_start,
                                soc_cone_indices_end = dual_sol.soc_cone_indices_end,
                                rsoc_cone_indices_start = dual_sol.rsoc_cone_indices_start,
                                rsoc_cone_indices_end = dual_sol.rsoc_cone_indices_end,
                                exp_cone_indices_start = dual_sol.exp_cone_indices_start,
                                exp_cone_indices_end = dual_sol.exp_cone_indices_end,
                                dual_exp_cone_indices_start = dual_sol.dual_exp_cone_indices_start,
                                dual_exp_cone_indices_end = dual_sol.dual_exp_cone_indices_end,
                                y_slice_length = dual_sol.y_slice_length,
                                y_slice_length_cpu = dual_sol.y_slice_length_cpu,
                                cone_index_start = dual_sol.cone_index_start,
                                cone_index_start_cpu = dual_sol.cone_index_start_cpu)
    return dual_sol_copy
end

mutable struct timesInfo
    interior::Integer
    boundary::Integer
    zero::Integer
    status::Symbol
    function timesInfo()
        new(0, 0, 0, :unknown)
    end
end

""" Diagonal_preconditioner for the PDHGCLP solver
    DQl: the diagonal of the matrix Ql
    DAl: the diagonal of the matrix Al
    Dr: the diagonal of the matrix r
    DQl_product: the product of the diagonal of the matrix Ql and the vector
    DAl_product: the product of the diagonal of the matrix Al and the vector
    Dr_product: the product of the diagonal of the matrix r and the vector
    Dr_product_inv_normalized: the product of the inverse of the diagonal of the matrix r and the vector
    DQl_product_normalized: the product of the inverse of the diagonal of the matrix Ql and the vector

    new_c = c ./ Dr_product
    new_Q = diag(DQl^{-1}) * Q * diag(Dr^{-1})
    new_A = diag(DAl^{-1}) * A * diag(Dr^{-1})
    new_h = diag(DQl^{-1}) * h
    new_b = diag(DAl^{-1}) * b
    new_bl = Dr[1:data.nb] * l
    new_bu = Dr[1:data.nb] * u
"""

mutable struct Diagonal_preconditioner
    Dl::dualVector
    Dr::primalVector

    Dl_temp::dualVector
    Dr_temp::primalVector

    m::Integer
    n::Integer

    Dl_product::dualVector
    Dr_product::primalVector

    Dr_product_inv_normalized::primalVector # primal variable projection Dr[1]./Dr -- primal variable projection
    Dr_product_normalized::primalVector # primal variable projection Dr./Dr[1]  -- slack variable projection
    Dl_product_inv_normalized::dualVector # dual variable projection Dl[1]./Dr -- dual variable projection

    Dr_product_inv_normalized_squared::primalVector
    Dr_product_normalized_squared::primalVector
    Dl_product_inv_normalized_squared::dualVector

    primalConstScale::Vector{Bool}
    dualConstScale::Vector{Bool}

    primalProjInfo::Vector{timesInfo}
    dualProjInfo::Vector{timesInfo}
    slackProjInfo::Vector{timesInfo}

    function Diagonal_preconditioner(; Dl, Dr, m, n, len_soc_x, len_rsoc_x, Dl_product = deepCopyDualVector(Dl), Dr_product = deepCopyPrimalVector(Dr), Dl_temp = deepCopyDualVector(Dl),
        Dl_product_inv_normalized = deepCopyDualVector(Dl), Dl_product_inv_normalized_squared = deepCopyDualVector(Dl))

        if len_soc_x > 0 || len_rsoc_x > 0
            Dr_product_inv_normalized = deepCopyPrimalVector(Dr)
            Dr_product_normalized = deepCopyPrimalVector(Dr)
            Dr_product_inv_normalized_squared = deepCopyPrimalVector(Dr)
            Dr_product_normalized_squared = deepCopyPrimalVector(Dr)
            Dr_temp.x .= 1.0
            Dr_product_inv_normalized.x .= 1.0
            Dr_product_normalized.x .= 1.0
            Dr_product_inv_normalized_squared.x .= 1.0
            Dr_product_normalized_squared.x .= 1.0
        else
            # placeholder not used
            Dr_product_inv_normalized = deepCopyPrimalVector_null(Dr)
            Dr_product_normalized = deepCopyPrimalVector_null(Dr)
            Dr_product_inv_normalized_squared = deepCopyPrimalVector_null(Dr)
            Dr_product_normalized_squared = deepCopyPrimalVector_null(Dr)
        end
        Dr_temp = deepCopyPrimalVector(Dr)
        Dl_product.y .= 1.0
        Dr_product.x .= 1.0
        Dl_product_inv_normalized.y .= 1.0
        Dl_product_inv_normalized_squared.y .= 1.0
        primalConstScale = Vector{Bool}(undef, Dr.blkLen)
        dualConstScale = Vector{Bool}(undef, Dl.blkLen)
        primalConstScale .= false
        dualConstScale .= false
        primalProjInfo = Vector{timesInfo}(undef, Dr.blkLen)
        dualProjInfo = Vector{timesInfo}(undef, Dl.blkLen)
        slackProjInfo = Vector{timesInfo}(undef, Dr.blkLen)
        for i in 1:Dr.blkLen
            primalProjInfo[i] = timesInfo()
            slackProjInfo[i] = timesInfo()
        end
        for i in 1:Dl.blkLen
            dualProjInfo[i] = timesInfo()
        end
        new(Dl, Dr, Dl_temp, Dr_temp, m, n, Dl_product, Dr_product,
        Dr_product_inv_normalized, Dr_product_normalized, Dl_product_inv_normalized, Dr_product_inv_normalized_squared,
        Dr_product_normalized_squared, Dl_product_inv_normalized_squared, primalConstScale, dualConstScale, primalProjInfo, dualProjInfo, slackProjInfo)
    end
end




mutable struct solVecPrimalRecovered
    primal_sol::primalVector
    primal_sol_mean::primalVector
end




"""
solVecPrimal is a struct that stores the primal solution of the optimization problem.
    - primal_sol: the primal solution vector
    - primal_sol_lag: the previous primal solution vector
    - primal_sol_mean: the mean of the previous primal solution vector
    - box_index: the index of the box cone ([1:box_index])
    - bl: the lower bounds of the box cone
    - bu: the upper bounds of the box cone
    - soc_cone_indices_start: the start indices of the SOC cones
    - soc_cone_indices_end: the end indices of the SOC cones
    - rsoc_cone_indices_start: the start indices of the rotated SOC cones
    - rsoc_cone_indices_end: the end indices of the rotated SOC cones
    -- primal_sol[1:box_index] is the box cone
    -- primal_sol[soc_cone_indices_start[i]:soc_cone_indices_end[i]] is the i-th SOC cone
    -- primal_sol[rsoc_cone_indices_start[i]:rsoc_cone_indices_end[i]] is the i-th rotated SOC cone
"""
mutable struct solVecPrimal
    primal_sol::primalVector
    primal_sol_lag::primalVector
    primal_sol_mean::primalVector
    box_index::Integer
    bl::CuArray
    bu::CuArray
    soc_cone_indices_start::Vector{<:Integer}
    soc_cone_indices_end::Vector{<:Integer}
    rsoc_cone_indices_start::Vector{<:Integer}
    rsoc_cone_indices_end::Vector{<:Integer}
    exp_cone_indices_start::Vector{<:Integer}
    exp_cone_indices_end::Vector{<:Integer}
    dual_exp_cone_indices_start::Vector{<:Integer}
    dual_exp_cone_indices_end::Vector{<:Integer}
    proj!::Function
    proj_diagonal!::Function
    lambd_l::CuArray
    lambd_u::CuArray
    slack_proj!::Function
    recovered_primal::Union{solVecPrimalRecovered,Nothing}
    function solVecPrimal(; primal_sol, primal_sol_lag, primal_sol_mean, box_index,
        bl, bu, soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        proj!, proj_diagonal!,
        slack_proj!, recovered_primal)
        lambd_l, lambd_u = gen_lambd(bl, bu)
        new(primal_sol, primal_sol_lag, primal_sol_mean, box_index,
        bl, bu, soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        proj!, proj_diagonal!, lambd_l, lambd_u, slack_proj!, recovered_primal)
    end
end

mutable struct solVecDualRecovered
    dual_sol::dualVector
    dual_sol_mean::dualVector
end

"""
solVecDual is a struct that stores the dual solution of the optimization problem.
    - dual_sol: the dual solution vector
    - dual_sol_mean: the mean of the previous dual solution vector
    - len: the length of the dual solution vector
    - m1: the number of rows of matrix Q1, >=0, positive orthant
    - soc_cone_indices_start: the start indices of the SOC cones
    - soc_cone_indices_end: the end indices of the SOC cones
    - rsoc_cone_indices_start: the start indices of the rotated SOC cones
    - rsoc_cone_indices_end: the end indices of the rotated SOC cones
    - slack: the primal solution vector, slack variables
"""
mutable struct solVecDual
    dual_sol::dualVector
    dual_sol_lag::dualVector
    dual_sol_mean::dualVector
    dual_sol_temp::dualVector
    mGzeroIndices::Vector{<:Integer}
    mGnonnegativeIndices::Vector{<:Integer}
    soc_cone_indices_start::Vector{<:Integer}
    soc_cone_indices_end::Vector{<:Integer}
    rsoc_cone_indices_start::Vector{<:Integer}
    rsoc_cone_indices_end::Vector{<:Integer}
    exp_cone_indices_start::Vector{<:Integer}
    exp_cone_indices_end::Vector{<:Integer}
    dual_exp_cone_indices_start::Vector{<:Integer}
    dual_exp_cone_indices_end::Vector{<:Integer}
    slack::Union{solVecPrimal,Nothing}
    proj!::Function
    con_proj!::Function # constraint projection
    proj_diagonal!::Function
    recovered_dual::Union{solVecDualRecovered,Nothing}
    function solVecDual(; dual_sol, dual_sol_lag, dual_sol_mean, dual_sol_temp = deepCopyDualVector(dual_sol), mGzero, mGnonnegative,
        soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        slack, proj!, con_proj!, proj_diagonal!, recovered_dual)
        if mGzero > 0
            mGzeroIndices = Vector{Int}([1,mGzero])
        else
            mGzeroIndices = Vector{Int}([])
        end
        if mGnonnegative > 0
            mGnonnegativeIndices = Vector{Int}([mGzero + 1, mGzero + mGnonnegative])
        else
            mGnonnegativeIndices = Vector{Int}([])
        end
        new(dual_sol, dual_sol_lag, dual_sol_mean, dual_sol_temp, mGzeroIndices, mGnonnegativeIndices,
        soc_cone_indices_start, soc_cone_indices_end,
        rsoc_cone_indices_start, rsoc_cone_indices_end,
        exp_cone_indices_start, exp_cone_indices_end,
        dual_exp_cone_indices_start, dual_exp_cone_indices_end,
        slack, proj!, con_proj!, proj_diagonal!, recovered_dual)
    end
end


mutable struct PDHGCLPConvergeInfo
    primal_objective::rpdhg_float
    dual_objective::rpdhg_float
    abs_gap::rpdhg_float
    rel_gap::rpdhg_float
    l_inf_rel_primal_res::rpdhg_float
    l_inf_rel_dual_res::rpdhg_float
    l_2_rel_primal_res::rpdhg_float
    l_2_rel_dual_res::rpdhg_float
    l_inf_abs_primal_res::rpdhg_float
    l_inf_abs_dual_res::rpdhg_float
    l_2_abs_primal_res::rpdhg_float
    l_2_abs_dual_res::rpdhg_float
    status::Symbol
    function PDHGCLPConvergeInfo(; primal_objective = 1e+30, dual_objective= 1e+30, abs_gap= 1e+30, rel_gap= 1e+30,
        l_inf_rel_primal_res= 1e+30, l_inf_rel_dual_res= 1e+30, l_2_rel_primal_res= 1e+30, l_2_rel_dual_res= 1e+30,
        l_inf_abs_primal_res= 1e+30, l_inf_abs_dual_res= 1e+30, l_2_abs_primal_res= 1e+30, l_2_abs_dual_res= 1e+30,
        status= :continue)
        new(primal_objective, dual_objective, abs_gap, rel_gap,
        l_inf_rel_primal_res, l_inf_rel_dual_res, l_2_rel_primal_res, l_2_rel_dual_res,
        l_inf_abs_primal_res, l_inf_abs_dual_res, l_2_abs_primal_res, l_2_abs_dual_res,
        status)
    end
end



"""
Information measuring how close a point is to establishing primal or dual
infeasibility (i.e. has no solution); see also TerminationCriteria.
"""
mutable struct PDHGCLPInfeaInfo
    max_primal_ray_infeasibility::rpdhg_float
    primal_ray_objective::rpdhg_float
    max_dual_ray_infeasibility::rpdhg_float
    dual_ray_objective::rpdhg_float
    trend_len::Integer
    primalObj_trend::CircularBuffer{rpdhg_float}
    dualObj_trend::CircularBuffer{rpdhg_float}
    primal_ray_norm::rpdhg_float
    dual_ray_norm::rpdhg_float
    status::Symbol
    function PDHGCLPInfeaInfo(; max_primal_ray_infeasibility = 1e+30,
         primal_ray_objective = 1e+30, max_dual_ray_infeasibility = 1e+30,
         dual_ray_objective = 1e+30, trend_len = 20, primalObj_trend = CircularBuffer{rpdhg_float}(trend_len),
         dualObj_trend = CircularBuffer{rpdhg_float}(trend_len), status = :continue)
         primal_ray_norm = 0.0
         dual_ray_norm = 0.0
        new(max_primal_ray_infeasibility, primal_ray_objective,
         max_dual_ray_infeasibility, dual_ray_objective,
         trend_len, primalObj_trend, dualObj_trend, primal_ray_norm, dual_ray_norm, status)
    end
end


"""
exit_status:
    :optimal 0
    :max_iter 1
    :primal_infeasible_low_acc 2
    :primal_infeasible_high_acc 3
    :dual_infeasible_low_acc 4
    :dual_infeasible_high_acc 5
    :time_limit 6   
    :continue 7
"""

mutable struct PDHGCLPInfo
    # results
    iter::Integer
    iter_stepsize::Integer
    convergeInfo::Vector{PDHGCLPConvergeInfo} # multi sequence to check convergence
    infeaInfo::Vector{PDHGCLPInfeaInfo} # multi sequence to check infeasibility
    time::Float64
    start_time::Float64
    restart_used::Integer
    restart_trigger_mean::Integer
    restart_trigger_ergodic::Integer
    exit_status::Symbol
    pObj::rpdhg_float
    dObj::rpdhg_float
    exit_code::Int
    normalized_duality_gap::Vector{rpdhg_float}
    normalized_duality_gap_restart_threshold::rpdhg_float
    normalized_duality_gap_r::rpdhg_float
    kkt_error::Vector{rpdhg_float}
    kkt_error_restart_threshold::rpdhg_float
    restart_duality_gap_flag::Bool
    binarySearch_t0::rpdhg_float
    omega::rpdhg_float
    max_kkt_error::rpdhg_float
    min_kkt_error::rpdhg_float
    function PDHGCLPInfo(; iter, convergeInfo, infeaInfo, time, start_time, restart_used = 0, restart_trigger_mean = 0, restart_trigger_ergodic = 0, exit_status = :continue, pObj = 1e+30, dObj = 1e+30, exit_code = 7, normalized_duality_gap = Vector{rpdhg_float}(undef, 2), normalized_duality_gap_restart_threshold = 0, kkt_error = Vector{rpdhg_float}(undef, 2), kkt_error_restart_threshold = 0)
        normalized_duality_gap[1] = 1e+30
        normalized_duality_gap[2] = 1e+30
        kkt_error[1] = 1e+30
        kkt_error[2] = 1e+30    
        iter_stepsize = 0
        normalized_duality_gap_r = 1e+30
        restart_duality_gap_flag = true
        binarySearch_t0 = 1.0
        omega = 1.0
        max_kkt_error = 1e+30
        min_kkt_error = 1e+30
        new(iter, iter_stepsize, convergeInfo, infeaInfo, time, start_time, restart_used, restart_trigger_mean, restart_trigger_ergodic, exit_status, pObj, dObj, exit_code, normalized_duality_gap, normalized_duality_gap_restart_threshold, normalized_duality_gap_r, kkt_error, kkt_error_restart_threshold, restart_duality_gap_flag, binarySearch_t0, omega, max_kkt_error, min_kkt_error)
    end
end




mutable struct PDHGCLPParameters
    # parameters
    max_outer_iter::Integer
    max_inner_iter::Integer
    rel_tol::rpdhg_float
    abs_tol::rpdhg_float
    eps_primal_infeasible_low_acc::rpdhg_float
    eps_dual_infeasible_low_acc::rpdhg_float
    eps_primal_infeasible_high_acc::rpdhg_float
    eps_dual_infeasible_high_acc::rpdhg_float
    sigma::rpdhg_float
    tau::rpdhg_float
    theta::rpdhg_float
    use_kkt_restart::Bool
    kkt_restart_freq::Integer
    use_duality_gap_restart::Bool
    duality_gap_restart_freq::Integer    
    check_terminate_freq::Integer
    verbose::Integer
    print_freq::Integer
    time_limit::rpdhg_float
    beta_suff::rpdhg_float
    beta_necessary::rpdhg_float
    beta_suff_kkt::rpdhg_float
    beta_necessary_kkt::rpdhg_float
    beta_artificial::rpdhg_float
    proj_base_tol::rpdhg_float
    proj_abs_tol::rpdhg_float
    proj_rel_tol::rpdhg_float
    function PDHGCLPParameters(;
         max_outer_iter, max_inner_iter, rel_tol, abs_tol,
         eps_primal_infeasible_low_acc, eps_dual_infeasible_low_acc,
         eps_primal_infeasible_high_acc, eps_dual_infeasible_high_acc,
         sigma, tau, theta,
         use_kkt_restart, kkt_restart_freq, use_duality_gap_restart, duality_gap_restart_freq, check_terminate_freq, verbose, print_freq, time_limit)
         beta_suff = 0.4
         beta_necessary = 0.8
         beta_suff_kkt = 0.4
         beta_necessary_kkt = 0.8
         beta_artificial = 0.223
         proj_base_tol = 1e-9
         proj_abs_tol = 1e-11
         proj_rel_tol = 1e-11
        new(max_outer_iter, max_inner_iter, rel_tol, abs_tol,
        eps_primal_infeasible_low_acc, eps_dual_infeasible_low_acc,
        eps_primal_infeasible_high_acc, eps_dual_infeasible_high_acc,
        sigma, tau, theta,
        use_kkt_restart, kkt_restart_freq, use_duality_gap_restart, duality_gap_restart_freq, check_terminate_freq, verbose, print_freq, time_limit,
        beta_suff, beta_necessary, beta_suff_kkt, beta_necessary_kkt, beta_artificial, proj_base_tol, proj_abs_tol, proj_rel_tol)
    end
end

mutable struct Solution
    x::solVecPrimal
    y::solVecDual
    x_best::primalVector
    y_best::dualVector
    primal_res_best::rpdhg_float
    dual_res_best::rpdhg_float
    params::PDHGCLPParameters
    info::PDHGCLPInfo
    dual_sol_temp::Union{solVecDual, Nothing}
    function Solution(; x, y, params, info, dual_sol_temp = nothing)
        x_best = deepCopyPrimalVector(x.primal_sol)
        y_best = deepCopyDualVector(y.dual_sol)
        primal_res_best = 1e+30
        dual_res_best = 1e+30
        new(x, y, x_best, y_best, primal_res_best, dual_res_best, params, info, dual_sol_temp)
    end
end


mutable struct coeffUnion{
    dhType<:Union{CuVector{Float64},CuArray, Nothing},
    dGType<:Union{CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}, Adjoint{Float64, CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}}, CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int64}, Adjoint{Float64, CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int64}}}
}    
    d_G::dGType
    d_h::dhType
    m::Integer
    n::Integer
    function coeffUnion(; G::GType,
        h::hType, m::Integer, n::Integer, d_G=nothing, d_h=nothing) where{
           hType<:Union{Vector{rpdhg_float}, Nothing},
           GType<:Union{AbstractMatrix{rpdhg_float}, Nothing}
        }
        if d_G === nothing
            d_G = CUDA.CUSPARSE.CuSparseMatrixCSR(G)
            d_h = CuArray(h)
        end
        new{typeof(d_h), typeof(d_G)}(d_G, d_h, m, n)       
   end
end

mutable struct rpdhgRawData{
    coeffType<:Union{ coeffUnion},
    coeffTransType<:Union{ coeffUnion}
}
    m::Integer
    n::Integer
    nb::Integer
    c::CuArray
    coeff::coeffType
    coeffTrans::coeffTransType
    bl::CuArray
    bu::CuArray
    bl_finite::CuArray # avoid 0.0 * -Inf
    bu_finite::CuArray # avoid 0.0 * Inf
    hNrm1::rpdhg_float
    cNrm1::rpdhg_float
    hNrmInf::rpdhg_float
    cNrmInf::rpdhg_float
    function rpdhgRawData(; m::Integer, n::Integer, nb::Integer,
        c::CuArray, coeff::coeffType, coeffTrans::coeffTransType,
        bl::CuArray, bu::CuArray,
        hNrm1::rpdhg_float, cNrm1::rpdhg_float, 
        hNrmInf::rpdhg_float, cNrmInf::rpdhg_float) where{
        coeffType<:Union{coeffUnion},
        coeffTransType<:Union{coeffUnion},
        }
        bl_finite = deepcopy(bl)
        bu_finite = deepcopy(bu)
        println("consider bl_finite: ")
        n = length(bl)
        if n > 0
            # CUDA.@allowscalar bl_finite = replace(bl_finite, -Inf=>0.0)
            # CUDA.@allowscalar bu_finite = replace(bu_finite, Inf=>0.0)
            replace_inf_with_zero(bl_finite, bu_finite, n)
        end
        new{coeffType, coeffTransType}(m, n, nb, c, coeff, coeffTrans, bl, bu, bl_finite, bu_finite, hNrm1, cNrm1, hNrmInf, cNrmInf)
    end
end


"""
probData is a struct that stores the data for the optimization problem.
    - m: the number of rows of the matrix A
    - n: the number of columns of the matrix A
    - c: the vector c
    - coeff: the data coefficient
    - coeffTrans: the transpose of the data coefficient, no vector
    - GlambdaMax: the maximum eigenvalue of the matrix G
    - hNrm1: the 1-norm of the vector d
    - cNrm1: the 1-norm of the vector c
"""

function cal_constant(; c, h)
    hNrm1 = norm(h, 1)
    hNrm2 = norm(h, 2)
    cNrm1 = norm(c, 1)
    cNrm2 = norm(c, 2)
    hNrmInf = norm(h, Inf)
    cNrmInf = norm(c, Inf)
    return hNrm1, hNrm2, cNrm1, cNrm2, hNrmInf, cNrmInf
end # end cal_constant

mutable struct probData{
    cType<:Union{CuArray, CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}, CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int64}},
    coeffType<:Union{coeffUnion},
    coeffTransType<:Union{coeffUnion}
}
    m::Integer
    n::Integer
    nb::Integer
    d_c::cType
    coeff::coeffType
    coeffTrans::coeffTransType
    GlambdaMax::rpdhg_float
    GlambdaMax_flag::Integer
    d_bl::CuArray
    d_bu::CuArray
    d_bl_finite::CuArray # avoid 0.0 * -Inf
    d_bu_finite::CuArray # avoid 0.0 * Inf
    hNrm1::rpdhg_float
    hNrm2::rpdhg_float
    cNrm1::rpdhg_float
    cNrm2::rpdhg_float
    hNrmInf::rpdhg_float
    cNrmInf::rpdhg_float
    diagonal_scale::Diagonal_preconditioner
    raw_data::Union{rpdhgRawData,Nothing}
    function probData(; m::Integer, n::Integer, nb::Integer,
         c_cpu::AbstractVector{rpdhg_float}, coeff::coeffType, coeffTrans::coeffTransType,
         GlambdaMax::rpdhg_float, GlambdaMax_flag::Integer, bl_cpu::AbstractVector{rpdhg_float}, bu_cpu::AbstractVector{rpdhg_float},
         diagonal_scale::Diagonal_preconditioner, raw_data::Union{rpdhgRawData,Nothing}) where{
            coeffType<:Union{coeffUnion},
            coeffTransType<:Union{coeffUnion},
         }
        bl_finite = deepcopy(bl_cpu)
        bu_finite = deepcopy(bu_cpu)
        if length(bl_cpu) > 0
            bl_finite = replace(bl_finite, -Inf=>0.0)
            bu_finite = replace(bu_finite, Inf=>0.0)
        end
        d_c = CuArray(c_cpu)
        d_bl = CuArray(bl_cpu)
        d_bu = CuArray(bu_cpu)
        (hNrm1, hNrm2, cNrm1, cNrm2, hNrmInf, cNrmInf) = cal_constant(; c = d_c, h = coeff.d_h)
        new{CuArray, coeffType, coeffTransType}(m, n, nb, d_c, coeff, coeffTrans, GlambdaMax, GlambdaMax_flag, d_bl, d_bu, bl_finite, bu_finite, hNrm1, hNrm2, cNrm1, cNrm2, hNrmInf, cNrmInf, diagonal_scale, raw_data)
    end
end

mutable struct rpdhgSolver
    data::probData
    sol::Solution
    primalMV!::Function
    adjointMV!::Function
    AtAMV!::Function
    addCoeffd!::Function
    dotCoeffd::Function
    function rpdhgSolver(; data, sol, primalMV!, adjointMV!, AtAMV!, addCoeffd!, dotCoeffd)
        new(data, sol, primalMV!, adjointMV!, AtAMV!, addCoeffd!, dotCoeffd)
    end
end

