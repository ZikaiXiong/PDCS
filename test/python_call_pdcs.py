## root path to run this script: PDCS_fork/..
from juliacall import Main as jl
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

jl.seval('using Pkg')
jl.seval("Pkg.activate(\"./pdcs_env\")")
# jl.seval('Pkg.add("PythonCall")')
# jl.seval('using PythonCall')
jl.seval('using LinearAlgebra, SparseArrays')
jl.seval('using CUDA, CUDA.CUSPARSE')
jl.seval('include("./PDCS_fork/src/pdcs_gpu/PDCS_GPU.jl")')
jl.seval('include("./PDCS_fork/src/pdcs_cpu/PDCS_CPU.jl")')


jl.seval('''
using .PDCS_GPU
using .PDCS_CPU
using LinearAlgebra
using JuMP
using Random, SparseArrays
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

solver = PDCS_GPU.PDCS_GPU_Solver(
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
)
''')


# Update P matrix
# Define a new CSR sparse matrix on GPU
Ppy = csr_matrix(cp.array([
    [3.0, 0.5, 0.0],
    [0.5, 2.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=cp.float64))

# Extract the pointers (as integers)
data_ptr    = int(Ppy.data.data.ptr)
indices_ptr = int(Ppy.indices.data.ptr)
indptr_ptr  = int(Ppy.indptr.data.ptr)

n_rows, n_cols = Ppy.shape
nnz = Ppy.nnz

Pjl = jl.PDCS_GPU.cupy_to_cucsrmat(
    jl.Float64, data_ptr, indices_ptr, indptr_ptr, n_rows, n_cols, nnz
)


jl.PDCS_GPU.solve_with_solver(jl.solver)