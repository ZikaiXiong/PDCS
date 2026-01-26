using Pkg
Pkg.activate("pdcs_env")
using PDCS: PDCS_GPU, PDCS_CPU
using LinearAlgebra
using JuMP
using Random, SparseArrays
using CUDA

import MathOptInterface as MOI
rng = Random.MersenneTwister(1)
basedim = Int64(100)
n = 2 * basedim # number of variables
m = 5 * basedim # number of constraints
m_zero = 1 * basedim # number of zero constraints
m_nonnegative = 1 * basedim # number of nonnegative constraints
m_exp = m - m_zero - m_nonnegative # number of second-order cone constraints
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
for i in 0:(Int(m_exp / 3) - 1)
    PDCS_CPU.exponent_proj!(@view(bCopy[m_zero+m_nonnegative + i * 3 + 1:m_zero+m_nonnegative+(i+1) * 3]))
end
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
for i in 0:(Int(m_exp / 3) - 1)
    @constraint(model, (A * x - b)[m_zero+m_nonnegative + i * 3 + 1:m_zero+m_nonnegative+(i+1) * 3] in MOI.ExponentialCone())
end
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
    socG = Vector{Integer}([]),
    rsocG = Vector{Integer}([]),
    expG = Int(m_exp / 3),
    dual_expG = 0,
    bl = zeros(n),
    bu = ones(n) * Inf,
    soc_x = Vector{Integer}([]),
    rsoc_x = Vector{Integer}([]),
    exp_x = 0,
    dual_exp_x = 0,
    use_preconditioner = true,
    method = :average,
    print_freq = 2000,
    time_limit = 1000.0,
    use_adaptive_restart = true,
    use_adaptive_step_size_weight = true,
    use_resolving = true,
    use_accelerated = false,
    use_aggressive = true,
    verbose = 2,
    rel_tol = 1e-6,
    abs_tol = 1e-6,
    kkt_restart_freq = 2000,
    duality_gap_restart_freq = 2000,
    use_kkt_restart = false,
    use_duality_gap_restart = true,
    logfile_name = nothing,
    # max_outer_iter = 3,
    # max_inner_iter = 10,
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
    socG = Vector{Integer}([]),
    rsocG = Vector{Integer}([]),
    expG = Int(m_exp / 3),
    dual_expG = 0,
    bl = CuArray(zeros(n)),
    bu = CuArray(ones(n) * Inf),
    soc_x = Vector{Integer}([]),
    rsoc_x = Vector{Integer}([]),
    exp_x = 0,
    dual_exp_x = 0,
    use_preconditioner = true,
    method = :average,
    print_freq = 2000,
    time_limit = 1000.0,
    use_adaptive_restart = true,
    use_adaptive_step_size_weight = true,
    use_resolving = true,
    use_accelerated = false,
    use_aggressive = true,
    verbose = 2,
    rel_tol = 1e-6,
    abs_tol = 1e-6,
    kkt_restart_freq = 2000,
    duality_gap_restart_freq = 2000,
    use_kkt_restart = false,
    use_duality_gap_restart = true,
    logfile_name = nothing,
    # max_outer_iter = 3,
    # max_inner_iter = 10,
)