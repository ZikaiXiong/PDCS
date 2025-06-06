using Pkg
Pkg.activate("pdcs_env")
include("../src/pdcs_gpu/PDCS_GPU.jl")
include("../src/pdcs_cpu/PDCS_CPU.jl")
using .PDCS_GPU
using .PDCS_CPU
using LinearAlgebra
using JuMP
using Random, SparseArrays
import MathOptInterface as MOI
rng = Random.MersenneTwister(2)
basedim = Int64(100)
n = 2 * basedim # number of variables
m = 5 * basedim # number of constraints
m_zero = 1 * basedim # number of zero constraints
m_nonnegative = 1 * basedim # number of nonnegative constraints
m_soc = m - m_zero - m_nonnegative # number of second-order cone constraints
c = ones(n) * 10
# sparse matrix with 10% nonzeros
density = 1.0
A = sprand(m, n, density)
x_fea = rand(n)
x_fea .= max.(x_fea, 0.0)
b = A * x_fea
bCopy = deepcopy(b)
bCopy[1:m_zero] .= 0.0
bCopy[m_zero+1:m_zero+m_nonnegative] .= max.(b[m_zero+1:m_zero+m_nonnegative], 0.0)
PDCS_CPU.soc_proj!(@view(bCopy[m_zero+m_nonnegative+1:end]))
b .-= bCopy
model = Model(PDCS_GPU.Optimizer)
set_optimizer_attribute(model, "time_limit_secs", 1000.0)
set_optimizer_attribute(model, "verbose", 2)
@variable(model, x[1:n] >= 0)
@objective(model, Min, c' * x)
# (A * x - b)[1:m_zero] == 0
@constraint(model, (A * x - b)[1:m_zero] .== 0)
# (A * x - b)[m_zero+1:m_zero+m_nonnegative] >= 0
@constraint(model, (A * x - b)[m_zero+1:m_zero+m_nonnegative] .>= 0)
# (A * x - b)[m_zero+m_nonnegative+1:end] in SOC
@constraint(model, (A * x - b)[m_zero+m_nonnegative+1:end] in SecondOrderCone())
optimize!(model)