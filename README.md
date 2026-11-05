# PDCS

<p align="center">
  <img src="./pdcs_assets/PDCS_logo.png" alt="PDCS: Primal Dual Conic Programming Solver" width="70%">
</p>

[![CPU tests](https://github.com/ZikaiXiong/PDCS/actions/workflows/ci.yml/badge.svg)](https://github.com/ZikaiXiong/PDCS/actions/workflows/ci.yml)

Authors: Zhenwei Lin, Zikai Xiong, Dongdong Ge, and Yinyu Ye.

PDCS is a Julia solver for large-scale conic optimization, with CPU and optional
CUDA GPU implementations of a primal-dual algorithm. It supports second-order
cones (SOC), exponential cones, and dual exponential cones through JuMP and
MathOptInterface.

Project home: https://github.com/ZikaiXiong/PDCS

Julia package UUID: `9123d4a1-5282-4e19-bc2a-6f2650421a93`.

Current project version: `0.1.1`. See [RELEASE.md](RELEASE.md) for the version
and tag procedure.

Project manager: [Zikai Xiong](https://github.com/ZikaiXiong).
Email: [zikai.xiong@northwestern.edu](mailto:zikai.xiong@northwestern.edu).

## Install and run

Requires Julia 1.10 or newer. From the root of this checkout, the recommended
end-to-end CPU command is:

```sh
MODE=cpu bash build.sh
```

It instantiates the environment, runs the small SOC example, and executes the
portable test suite. The equivalent individual commands are:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/soc.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

The CPU installation needs no GPU, CUDA compiler, external datasets, Python
installation, or commercial solver license. See [INSTALL](INSTALL) for use from
another Julia environment and optional GPU setup.

```julia
using JuMP
using PDCS: PDCS_CPU

model = Model(PDCS_CPU.Optimizer)
set_silent(model)
@variable(model, t)
@constraint(model, [t, 3.0, 4.0] in SecondOrderCone())
@objective(model, Min, t)
optimize!(model)
println(objective_value(model)) # approximately 5.0
```

For GPU use, install and load CUDA explicitly before importing the GPU solver:

```julia
using CUDA
using PDCS: PDCS_GPU
model = Model(PDCS_GPU.Optimizer)
```

The complete GPU build and hardware test can be run with:

```sh
CUDA_HOME=/usr/local/cuda GPU_ARCH=sm_80 MODE=gpu bash build.sh
```

Select `GPU_ARCH` for the allocated device; `sm_80` is appropriate for A100 and
`sm_90` for H100. See [INSTALL](INSTALL) for the full prerequisites.

Native GPU projection build requirements and diagnostics are documented in
[src/pdcs_gpu/cuda/README.md](src/pdcs_gpu/cuda/README.md).

## CVXPY parameter interface

The maintained adapter is
[`pdcs_cvxpy.py`](src/pdcs_gpu/cvxpy_wrapper/pdcs_cvxpy.py). It extends CVXPY's
PDCS adapter and requires a CVXPY version containing that adapter (validated
with 1.9.3), CuPy, JuliaCall, and a Julia environment containing CUDA and this
PDCS checkout. From the repository root, expose the adapter to Python:

```bash
export PYTHONPATH="$PWD/src/pdcs_gpu/cvxpy_wrapper${PYTHONPATH:+:$PYTHONPATH}"
```

Register it once, before the first solve in each Python process:

```python
import cvxpy as cp
from pdcs_cvxpy import register

register()
t = cp.Variable()
prob = cp.Problem(cp.Minimize(t), [cp.SOC(t, cp.Constant([3.0, 4.0]))])
prob.solve(
    solver=cp.PDCS,
    abs_tol=1e-6,
    rel_tol=1e-6,
    time_limit_secs=60.0,
    use_adaptive_step=False,
    use_scaling=True,
    rescaling_method="ruiz_pock_chambolle",
    max_inner_iter=100000,
    verbose=True,
)
```

`register()` selects this checkout's adapter for `cp.PDCS` in the current
process. The `examples/cvxpy_parameters.py` example registers it automatically.
If JuliaCall's active project does not contain this checkout, pass
`julia_env="/absolute/path/to/your/julia/project"`; that project must already
have PDCS and CUDA installed.

JuMP sets options with `set_optimizer_attribute(model, "use_adaptive_step", false)`;
CVXPY passes `prob.solve(solver=cp.PDCS, use_adaptive_step=False)`; the Julia
direct API accepts `use_adaptive_step=false` as a keyword argument to
`rpdhg_gpu_solve` or `PDCS_GPU_Solver`, alongside the problem data.

| CVXPY keyword | Julia constructor keyword | Meaning |
| --- | --- | --- |
| `time_limit_secs` or `time_limit` | `time_limit` | Time limit in seconds; `float("inf")` removes the limit |
| `logfile` or `logfile_name` | `logfile_name` | Plain path string / `PathLike`, or `None` to disable file logging |
| `use_scaling` or `use_preconditioner` | `use_preconditioner` | Enable diagonal preconditioning |
| `rescaling_method` | `rescaling_method` | Python string converted to a Julia `Symbol` |
| `pdcs_method` | `method` | Currently `"average"`; `method` itself is reserved by CVXPY |
| `verbose` / `solver_verbose` | `verbose` | CVXPY controls solver verbosity; forwarded as `0` or `1` |
| `warm_start` | `warm_start`, `primal_sol`, `dual_sol` | Reuse the previous optimal solution when the cone dimensions match |
| `julia_env` | Project activation | Select the Julia project before loading PDCS |

The following algorithm options are forwarded under their Julia names:

| Group | Keywords |
| --- | --- |
| Accuracy | `abs_tol`, `rel_tol`, `eps_primal_infeasible_low_acc`, `eps_dual_infeasible_low_acc`, `eps_primal_infeasible_high_acc`, `eps_dual_infeasible_high_acc` |
| Iterations and reporting | `max_outer_iter`, `max_inner_iter`, `print_freq`, `check_terminate_freq` |
| Scaling and step size | `scalar_cone_rescaling`, `use_adaptive_diagonal_scalar_rescaling`, `use_adaptive_step`, `use_adaptive_step_size_weight`, `adaptive_projection_tolerance` |
| Iterate updates | `use_aggressive`, `use_reflection`, `use_weighted_average`, `use_halpern`, `use_inline_halpern`, `use_resolving` |
| Restart | `use_adaptive_restart`, `use_restart`, `use_current_restart_candidate`, `use_mean_restart_candidate`, `use_kkt_restart`, `kkt_restart_freq`, `use_duality_gap_restart`, `duality_gap_restart_freq` |

Unspecified algorithm options use the Julia constructor's defaults, including
defaults derived from other options. The CVXPY time limit remains 1000 seconds
unless overridden. Boolean options require Python booleans;
`adaptive_projection_tolerance` and `use_inline_halpern` also accept `None`
for automatic selection. Iteration limits and frequencies must be positive
integers; tolerances must be finite and nonnegative.

Supported `rescaling_method` strings are `"none"`, `"ruiz"`,
`"pock_chambolle"`, `"ruiz_pock_chambolle"`, and
`"ruiz_pock_chambolle_scalar_cone"`. Use `use_halpern` or
`use_inline_halpern` for the supported Halpern controls; the legacy
`pdcs_method="halpern"` path is rejected. The compatibility option
`use_accelerated` issues a deprecation warning and remains ignored by Julia.

Unknown options, invalid types, and conflicting alias values raise an error.
CVXPY supplies matrix data and cone dimensions, so direct-API data keywords
cannot be overridden here. The CuPy pointer bridge uses Int32 sparse indices;
use the Julia direct API for matrices exceeding that range.

Parameter-forwarding regression tests can run without a GPU. The Python tests
use real CVXPY canonicalization with a simulated GPU backend; the Julia tests
exercise both branches of the actual solver dispatcher:

```bash
python test/test_cvxpy_options.py
julia --startup-file=no test/test_solver_keyword_forwarding.jl
```

An additional bridge test checks real Python-to-Julia type conversion without
a GPU. Run `test/test_cvxpy_juliacall.jl` in a Julia project containing
PythonCall, with `JULIA_PYTHONCALL_EXE` pointing to the Python environment
containing CVXPY.

For actual GPU validation, run the deterministic SOC, exponential-cone, and
mixed-cone regression after configuring CuPy, JuliaCall, and the CUDA artifacts:

```bash
python test/test_cvxpy_gpu.py --report gpu-results.json
```

It checks analytic objective values, feasibility, and finite dual solutions in
24 solves covering default settings, adaptive steps, disabled scaling, fixed
steps without restart, Ruiz scaling, warm starts, and repeated cold starts. If the Julia
environment contains solver dependencies but not an installed PDCS package,
add `--julia-source src/pdcs_gpu/PDCS_GPU.jl` to load this checkout directly.
This mode tests the actual GPU solver and CVXPY bridge but does not test Julia
package installation/discovery.

## Dependencies

Julia's package manager installs the dependencies listed in [Project.toml](Project.toml):
JuMP, MathOptInterface, DataStructures, Match, Polynomials, SnoopPrecompile,
Statistics, PythonCall, and Julia standard libraries. PythonCall is used by the
GPU CVXPY bridge; the CPU solver does not import it. CUDA is an optional dependency.

## Tests and contributions

`Pkg.test()` runs the portable test module in [test/runtests.jl](test/runtests.jl).
It checks CPU imports, cone projections, SOC/exponential/dual-exponential solves
against analytic answers, bulk-cache structure and validation, and GPU utilities
that do not require a device. It excludes linear-programming and rotated-SOC
solve tests. GPU hardware regressions are run separately as described in the
native GPU documentation. See [CONTRIBUTING.md](CONTRIBUTING.md) for adding tests.

The `src/` and `ext/` directories contain the solver, `examples/` contains a small
runnable example, and `test/` contains tests.

The active GitHub Actions workflow runs the CPU suite with Julia 1.10 and the
current stable Julia release on Linux, macOS, and Windows. Its manual dispatch
also exposes an opt-in GPU job for a self-hosted runner labelled `nvidia-gpu`;
hosted GitHub runners are not claimed to provide NVIDIA hardware.

## Support and license

Contact project manager Zikai Xiong, report bugs, and request features through
https://github.com/ZikaiXiong/PDCS/issues. Include your Julia version, package
versions, a small reproducing example, and GPU details when applicable.

PDCS is distributed under the Apache License 2.0; see [LICENSE](LICENSE).
See [AUTHORS](AUTHORS) for attribution.
For community participation, see the [COIN-OR Code of Conduct](https://www.coin-or.org/code-of-conduct/).

### Convergence Criteria

The solver employs three convergence criteria to assess solution quality:

1. **Primal infeasibility**: 
   $$\frac{\|(Gx - h) - \text{proj}\_{\mathcal{K}_d}(Gx - h)\|\_{\infty}}{1+\max(\|h\|\_{\infty}, \|Gx\|\_{\infty}, \|\text{proj}\_{\mathcal{K}_d}(Gx - h)\|\_{\infty})}$$

2. **Dual infeasibility**: 
   $$\frac{\max\\{\|\lambda_1-\text{proj}\_{\Lambda_1}(\lambda_1)\|\_{\infty},\|\lambda_2-\text{proj}\_{\mathcal{K}_p^*}(\lambda_2)\|\_{\infty}\\}}{1+\max\\{\|c\|\_{\infty},\|G^\top y\|\_{\infty}\\}}$$

3. **Objective value accuracy**: 
   $$\frac{|c^{\top}x-(y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-})|}{1+\max\{|c^{\top}x|, |y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-}|\}}$$

where $\lambda=c-G^{\top}y=[\lambda_{1}^{\top},\lambda_{2}^{\top}]^{\top}$, with $\lambda_1\in \Lambda_1 \subseteq \mathbb{R}^{n_1}$ and $\lambda_2\in \mathbb{R}^{n_2}$.



### Citation

If you use PDCS in your research, please cite the following paper:

```bibtex
@misc{PDCS,
      title={PDCS: A Primal-Dual Large-Scale Conic Programming Solver with GPU Enhancements}, 
      author={Zhenwei Lin and Zikai Xiong and Dongdong Ge and Yinyu Ye},
      year={2025},
      eprint={2505.00311},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2505.00311}, 
}
```
