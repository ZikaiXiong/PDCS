<p align="center">
  <img src="./pdcs_assets/PDCS_logo.png" width="70%">
</p>

PDCS is a high-performance Julia and CUDA implementation of a primal-dual algorithm for solving large-scale conic optimization problems.

### Overview

This software package implements a primal-dual algorithm for solving conic optimization problems of the following form:

$$\min_{x=(x_1,x_2),x_1\in \mathbb{R}^{n_1}, x_2\in \mathbb{R}^{n_2}} c^{\top} x\ \  \text{s.t.} Gx-h\in \mathcal{K}_d, l\leq x_1\leq u, x_2 \in \mathcal{K}_p,$$

where $\mathcal{K}_d$ and $\mathcal{K}_p$ are closed convex cones.

### Features

The solver supports the following cone types:
- **Second-order cone** (SOC) and **rotated second-order cone** (RSOC)
- **Exponential cone** and **dual exponential cone**

The implementation provides both CPU and GPU-accelerated solvers, with the GPU version leveraging CUDA for enhanced computational performance on large-scale problems.

### Installation

#### Prerequisites

1. Clone the repository:
```bash
git clone https://github.com/ZikaiXiong/PDCS.git
cd PDCS
```

2. Compile the CUDA code:
```bash
cd src/pdcs_gpu/cuda
make
```
Note: The default compute architecture is `sm_90`. Modify the Makefile if a different architecture is required.

3. Return to the project root:
```bash
cd ../../..
```

#### Julia Package Installation

Install the `PDCS` package in Julia using the following command:

```julia
using Pkg
Pkg.develop(path="PDCS")
```

Note: The installation process will execute a small demonstration for precompilation purposes. Verbose logging will only occur during the initial installation.

### Usage
```julia
julia ./test/install.jl    ## Install package
```

Example code is provided in the `./test/` directory. The following commands demonstrate how to execute the test suites:

#### CPU Solver Tests
```julia
julia ./test/test_exp.jl         # Test exponential cone solver (CPU)
julia ./test/test_soc.jl         # Test second-order cone solver (CPU)
```

#### GPU Solver Tests
```julia
julia ./test/test_exp_gpu.jl     # Test exponential cone solver (GPU)
julia ./test/test_soc_gpu.jl     # Test second-order cone solver (GPU)
```

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
