using Pkg
Pkg.activate("pdcs_env")
using PDCS: PDCS_GPU, PDCS_CPU
using LinearAlgebra
using JuMP
using Random, SparseArrays
import MathOptInterface as MOI
using CUDA


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



sol_res = PDCS_GPU.rpdhg_gpu_solve(
    n = n,
    m = m,
    nb = n,
    c = c,
    G = A,
    h = b,
    mGzero = m_zero,
    mGnonnegative = m_nonnegative,
    socG = Vector{Integer}([m - m_zero - m_nonnegative]),
    rsocG = Vector{Integer}([]),
    expG = 0,
    dual_expG = 0,
    bl = zeros(n),
    bu = ones(n) * Inf,
    soc_x = Vector{Integer}([]),
    rsoc_x = Vector{Integer}([]),
    exp_x = 0,
    dual_exp_x = 0,
    use_preconditioner = true,
    method = :average
)



G_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
sol_res = PDCS_GPU.rpdhg_gpu_solve_input_gpu_data(
    n = n,
    m = m,
    nb = n,
    c = CuArray(c),
    G = G_gpu,
    h = CuArray(b),
    mGzero = m_zero,
    mGnonnegative = m_nonnegative,
    socG = Vector{Integer}([m - m_zero - m_nonnegative]),
    rsocG = Vector{Integer}([]),
    expG = 0,
    dual_expG = 0,
    bl = CuArray(zeros(n)),
    bu = CuArray(ones(n) * Inf),
    soc_x = Vector{Integer}([]),
    rsoc_x = Vector{Integer}([]),
    exp_x = 0,
    dual_exp_x = 0,
    use_preconditioner = true,
    method = :average
)