# Optional bridge smoke test: requires PythonCall and a Python environment with
# CVXPY, but does not load CUDA or require a GPU.
using PythonCall

pyexec(raw"""
import sys
sys.path.insert(0, wrapper_path)
from juliacall import Main as jl
from pdcs_cvxpy import normalize_solver_options

options, _ = normalize_solver_options({
    "use_adaptive_step": False,
    "max_inner_iter": 7,
    "abs_tol": 1e-8,
    "rescaling_method": "ruiz",
    "logfile": 'a "quoted" $name.log',
    "adaptive_projection_tolerance": None,
    "time_limit": float("inf"),
})
options["rescaling_method"] = jl.Symbol(options["rescaling_method"])
check = jl.seval('''(;use_adaptive_step::Bool, max_inner_iter::Integer,
    abs_tol::Float64, rescaling_method::Symbol, logfile_name::Union{String,Nothing},
    adaptive_projection_tolerance::Union{Bool,Nothing}, time_limit::Float64,
    verbose::Integer) -> (!use_adaptive_step && max_inner_iter == 7 &&
    abs_tol == 1e-8 && rescaling_method == :ruiz &&
    adaptive_projection_tolerance === nothing && isinf(time_limit) && verbose == 0,
    logfile_name)
''')
assert tuple(check(**options)) == (True, 'a "quoted" $name.log')
options["logfile_name"] = None
assert tuple(check(**options)) == (True, None)
assert list(jl.seval("Vector{Int}")([3, 4])) == [3, 4]
assert list(jl.seval("Vector{Int}")([])) == []
print("JuliaCall typed keyword and vector conversion passed", flush=True)
""", Main, (wrapper_path=joinpath(@__DIR__, "..", "src", "pdcs_gpu", "cvxpy_wrapper"),))
