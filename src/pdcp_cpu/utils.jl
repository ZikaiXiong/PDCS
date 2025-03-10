"""
utils.jl
"""

function write_info_to_csv(;file_path::String, fileName::String, m::Integer, n::Integer, socNum::Integer, info::PDHGCLPInfo, method::String)
    data = DataFrame(m = [m], n = [n],
                 socNum = [socNum], objective_value = [info.primal_obj],
                primal_res = [info.primal_res],
                dual_res = [info.dual_res],
                pd_gap = [info.gap],
                iteration = [info.iter],
                restart_times = [info.restart_times],
                re_times_cond = [info.re_times_cond],
                solve_time = [info.time],
                solver = [method])
    if !isdir(file_path)
        mkpath(file_path)
    end
    output_csv = joinpath(file_path, fileName)
    if !isfile(output_csv)
        # create an empty file
        touch(output_csv)
        # write header
        header = DataFrame(m = Integer[], n = Integer[],
                        socNum = Integer, objective_value = Float64[],
                        primal_res = Float64[], dual_res = Float64[],
                        pd_gap = Float64[], iteration = Integer[],
                        restart_times = Integer[], re_times_cond = Integer[],
                        solve_time = Float64[], solver = String[])
        CSV.write(output_csv, header)
    end
    CSV.write(output_csv, data, append=true)
end


function write_history_to_csv(;file_path::String, m::Integer, n::Integer, socNum::Integer, info::PDHGCLPInfo, method::String)
    if !isdir(file_path)
        mkpath(file_path)
    end
    fileName = string(method, "_", m, "_", n, "_", socNum, "_history.csv")
    output_csv = joinpath(file_path, fileName)
    if !isfile(output_csv)
        touch(output_csv)
        header = DataFrame(primal_res = Float64[], dual_res = Float64[], pd_gap = Float64[])
        CSV.write(output_csv, header)
    end
    CSV.write(output_csv, DataFrame(primal_res = info.primal_res_history, dual_res = info.dual_res_history, pd_gap = info.gap_history), append=true)
end