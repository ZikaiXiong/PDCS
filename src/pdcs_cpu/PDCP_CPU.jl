module PDCP_CPU

using Random, SparseArrays, LinearAlgebra
using Printf
using Match
using DataStructures
using Base.Threads
using JuMP
using Polynomials
using Statistics

const rpdhg_float = Float64
const rpdhg_int = Int32
const positive_zero = 1e-20
const negative_zero = -1e-20
const proj_rel_tol = 1e-12
const proj_abs_tol = 1e-16


## standard formulation of the optimization problem ##

# def var solver and methods
# include("./def_rpdhg.jl")
# include("./def_rpdhg_gen.jl")

# main algorithm
# include("./rpdhg_alg_cpu.jl")
# include("./rpdhg_alg_cpu_plot.jl")


## general formulation of the optimization problem ##
include("./def_struct.jl")
include("./exp_proj.jl")
include("./soc_rsoc_proj.jl")
include("./def_rpdhg_gen.jl")
include("./preprocess.jl")
include("./postprocess.jl")

# # # main algorithm
include("./termination.jl")
include("rpdhg_alg_cpu_gen_scaling.jl")
include("./rpdhg_alg_cpu_gen.jl")

# include("./rpdhg_alg_cpu_plot_gen.jl")


include("./utils.jl")
include("./MOI_wrapper/MOI_wrapper.jl")

export rpdhg_cpu_solve;


end