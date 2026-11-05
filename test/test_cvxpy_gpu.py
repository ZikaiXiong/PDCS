"""Real CVXPY/PDCS GPU regression with analytically known solutions.

Requires CuPy, JuliaCall, CVXPY, and compiled PDCS CUDA artifacts. By default,
load the installed PDCS Julia package. --julia-source loads PDCS_GPU.jl directly
from a checkout using an existing Julia dependency environment instead.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "src/pdcs_gpu/cvxpy_wrapper"))
import pdcs_cvxpy


def model(kind):
    constraints = []
    objective = 0
    variables = {}
    target = cp.Parameter(2, value=np.array([3.0, 4.0]))
    if kind in ("soc", "mixed"):
        x = cp.Variable(2)
        t = cp.Variable()
        constraints += [x == target, cp.SOC(t, x)]
        variables.update(x=x, t=t)
        objective += t
    if kind in ("exp", "mixed"):
        a = cp.Variable()
        z = cp.Variable()
        constraints += [a == 0, cp.ExpCone(a, 1.0, z)]
        variables.update(a=a, z=z)
        objective += z
    return cp.Problem(cp.Minimize(objective), constraints), variables, target


def check_solution(problem, variables, target):
    expected = 0.0
    residual = 0.0
    if "x" in variables:
        x = variables["x"].value
        t = float(variables["t"].value)
        expected += float(np.linalg.norm(target.value))
        residual = max(residual, float(np.max(np.abs(x - target.value))),
                       float(np.linalg.norm(x)) - t)
    if "z" in variables:
        a = float(variables["a"].value)
        z = float(variables["z"].value)
        expected += 1.0
        residual = max(residual, abs(a), math.exp(a) - z)
    assert problem.status == cp.OPTIMAL, problem.status
    assert math.isfinite(problem.value), problem.value
    assert abs(problem.value - expected) <= 1e-4 * max(1.0, expected), (
        problem.value, expected)
    assert residual <= 1e-4, residual
    for constraint in problem.constraints:
        for value in constraint.dual_value if isinstance(constraint.dual_value, list) else [constraint.dual_value]:
            assert value is not None and np.all(np.isfinite(value)), constraint
    return expected, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia-source", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case", choices=("soc", "exp", "mixed"), action="append")
    args = parser.parse_args()

    import cupy
    from juliacall import Main as jl

    assert cupy.cuda.runtime.getDeviceCount() > 0, "A GPU allocation is required"
    source = args.julia_source
    if source is not None:
        # Only package discovery is replaced. All arrays, kernels, constructors,
        # solves, and CVXPY result inversion use the real production code.
        jl.seval("using CUDA, CUDA.CUSPARSE")
        jl.include(str(source.resolve()))
        pdcs_cvxpy._load_julia = lambda env: jl
    else:
        jl = pdcs_cvxpy._load_julia(None)
    pdcs_cvxpy.register()
    metadata = {
        "cvxpy": cp.__version__, "cupy": cupy.__version__,
        "julia": str(jl.VERSION), "gpu": str(jl.CUDA.name(jl.CUDA.device())),
        "cuda_jl": str(jl.Base.pkgversion(jl.CUDA)),
        "julia_source": str(source.resolve()) if source else "installed PDCS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    print("GPU_ENV " + json.dumps(metadata), flush=True)
    results = []
    common = dict(abs_tol=1e-6, rel_tol=1e-6, check_terminate_freq=25,
                  max_inner_iter=20000, max_outer_iter=100, time_limit_secs=120.0)
    configurations = {
        "default": {},
        "adaptive": {"use_adaptive_step": True},
        "no_scaling": {"use_scaling": False},
        "fixed_no_restart": {"use_adaptive_step": False, "use_restart": False},
        "ruiz": {"rescaling_method": "ruiz"},
    }

    def run(name, problem, variables, target, **settings):
        start = time.monotonic()
        record = {"name": name, "settings": settings}
        try:
            problem.solve(solver=cp.PDCS, **common, **settings)
            expected, residual = check_solution(problem, variables, target)
            record.update(passed=True, status=problem.status,
                          objective=float(problem.value), expected=expected,
                          violation=residual, iterations=problem.solver_stats.num_iters)
        except Exception:
            record.update(passed=False, error=traceback.format_exc())
        record["wall_seconds"] = time.monotonic() - start
        results.append(record)
        print("GPU_RESULT " + json.dumps(record), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"environment": metadata, "results": results}, indent=2))
        return record["passed"]

    for kind in args.case or ("soc", "exp", "mixed"):
        for name, settings in configurations.items():
            problem, variables, target = model(kind)
            if not run(f"{kind}/{name}", problem, variables, target,
                       warm_start=False, **settings):
                return 1
        problem, variables, target = model(kind)
        if not run(f"{kind}/warm_seed", problem, variables, target, warm_start=True):
            return 1
        assert pdcs_cvxpy._CACHE_KEY in problem._solver_cache
        if "x" in variables:
            target.value = np.array([6.0, 8.0])
        if not run(f"{kind}/warm_reuse", problem, variables, target,
                   warm_start=True, use_adaptive_step=False):
            return 1
        if not run(f"{kind}/cold_repeat", problem, variables, target, warm_start=False):
            return 1
    print(f"GPU_SUMMARY {len(results)}/{len(results)} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
