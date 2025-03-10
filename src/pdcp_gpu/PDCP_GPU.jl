module PDCP_GPU

using Random, SparseArrays, LinearAlgebra
using Printf
using Match
using DataStructures
using Base.Threads
using JuMP
using Polynomials
using Statistics
using CUDA
using Libdl
using JLD2 # for debugging
using Logging
using Dates
# Logging.with_logger(Logging.NullLogger()) do
#     CUDA.allowscalar(true)
# end

const rpdhg_float = Float64
const rpdhg_int = Int32
const positive_zero = 1e-20
const negative_zero = -1e-20
const proj_rel_tol = 1e-12
const proj_abs_tol = 1e-16
const ThreadPerBlock = 256

const MODULE_DIR = @__DIR__
CUDA.seed!(1)


## standard formulation of the optimization problem ##

# def var solver and methods
# include("./def_rpdhg.jl")
# include("./def_rpdhg_gen.jl")

# main algorithm
# include("./rpdhg_alg_gpu.jl")
# include("./rpdhg_alg_gpu_plot.jl")



struct PlainMultiLogger <: AbstractLogger
    io_list::Vector{IO}  
    level::Logging.LogLevel  
end


Logging.min_enabled_level(logger::PlainMultiLogger) = logger.level



function Logging.shouldlog(logger::PlainMultiLogger, level, _module, group, id)
    return level >= logger.level  
end


function Logging.handle_message(logger::PlainMultiLogger, level, message, _module, group, id, file, line)
    if level < logger.level  
        return
    end
    for io in logger.io_list
        println(io, message)  
        if io isa IOStream && io!= stdout  
            flush(io)
        end
    end
end

## general formulation of the optimization problem ##
include("./gpu_kernel.jl")
include("./def_struct.jl")
include("./exp_proj.jl")
include("./soc_rsoc_proj.jl")
include("./def_rpdhg_gen.jl")
include("./preprocess.jl")
include("./postprocess.jl")

# # # main algorithm
include("./termination.jl")
include("rpdhg_alg_gpu_gen_scaling.jl")
include("./rpdhg_alg_gpu_gen.jl")

# include("./rpdhg_alg_gpu_plot_gen.jl")


include("./utils.jl")
include("./MOI_wrapper/MOI_wrapper.jl")

export rpdhg_gpu_solve;


end