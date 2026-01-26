__precompile__()
module PDCS_GPU

using Random, SparseArrays, LinearAlgebra
using Printf
using Match
using DataStructures
using Base.Threads
using JuMP
using Polynomials
using Statistics
using CUDA, PythonCall
using CUDA.CUSPARSE
using Libdl
using Logging
using Dates
import Base: unsafe_convert
using Base.Threads: SpinLock
using SnoopPrecompile

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

const _kernlib_ref = Ref{Ptr{Cvoid}}(C_NULL)
const few_block_proj_ptr = Ref{Ptr{Cvoid}}(C_NULL)


function __init__()
    # Open your own kernel library (NOT libcublas)
    # Replace with the actual .so path in your project
    libpath = joinpath(joinpath(MODULE_DIR, "cuda/libfew_block_proj.so"))

    _kernlib_ref[] = Libdl.dlopen(libpath)

    # IMPORTANT: symbol name must match EXACTLY what is exported by the .so
    few_block_proj_ptr[] = Libdl.dlsym(_kernlib_ref[], :few_block_proj)

    few_block_proj_ptr[] != C_NULL || error("Cannot find symbol `few_block_proj` in $libpath")
end



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
include("./cvxpy_wrapper/py2jl.jl")
include("./cvxpy_wrapper/data_updating.jl")

include("./precompile.jl")
redirect_stdout(devnull) do; 
    SnoopPrecompile.@precompile_all_calls begin
        if CUDA.has_cuda()
            __init__()
            @info "============precompile PDCS_GPU============"
            __precompile_gpu()
            __precompile_gpu_clean_pointer()
            @info "============precompile PDCS_GPU done============"
        else
            @info "============ PDCS_GPU need cuda to precompile ============"
        end
    end
end

export rpdhg_gpu_solve;


end