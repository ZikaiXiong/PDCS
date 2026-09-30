# PDCS

Authors: Zhenwei Lin, Zikai Xiong, Dongdong Ge, and Yinyu Ye.

PDCS is a Julia solver for large-scale conic optimization, with CPU and optional
CUDA GPU implementations of a primal-dual algorithm. It supports second-order
cones (SOC), exponential cones, and dual exponential cones through JuMP and
MathOptInterface.

Project home: https://github.com/ZikaiXiong/PDCS

Project manager: [Zikai Xiong](https://github.com/ZikaiXiong).
Email: [zikai.xiong@northwestern.edu](mailto:zikai.xiong@northwestern.edu).

## Install and run

Requires Julia 1.10 or newer. From the root of this checkout:

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

Native GPU projection build requirements and diagnostics are documented in
[src/pdcs_gpu/cuda/README.md](src/pdcs_gpu/cuda/README.md).

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
