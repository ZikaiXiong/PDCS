@def title = "Franklin Example"
@def tags = ["syntax", "code"]


\tableofcontents

![](/assets/PDCS.png)


## Overview

This project implements the algorithm for solving conic optimization problems of the form:

$$\min_{x=(x_1,x_2),x_1\in \mathbb{R}^{n_1}, x_2\in \mathbb{R}^{n_2}} c^{\top} x\ \  \text{s.t.} Gx-h\in \mathcal{K}_d, l\leq x_1\leq u, x_2 \in \mathcal{K}_p,$$

where $\mathcal{K}_G$ is a closed convex cone, and $\mathcal{K}_x$ is a closed convex cone.

## Features

- **Multiple Cone Types Support**:
    - Second-order cone (recommended but not the rsoc)
    - Rotated second-order cone
    - Exponential cone and dual exponential cone

## Installation

```bash
git clone https://github.com/ZikaiXiong/PDCS.git
cd PDCS
cd src/pdcs_gpu/cuda
make # compile the cuda code, default ARCH = sm_90
cd ../../..
```

## Usage

```julia
julia ./test/env.jl # install the dependencies
julia ./test/test_exp.jl # test the exponential cone cpu
julia ./test/test_soc.jl # test the second-order cone cpu
julia ./test/test_rsoc.jl # test the rotated second-order cone cpu

julia ./test/test_exp_gpu.jl # test the exponential cone gpu
julia ./test/test_soc_gpu.jl # test the second-order cone gpu
julia ./test/test_rsoc_gpu.jl # test the rotated second-order cone gpu
```

## Convergence Criteria

Three criteria are considered for the performance:

1. **Primal infeasibility**:
   $$\frac{\|(Gx - h) - \text{proj}_{\mathcal{K}_G}(Gx - h)\|_{\infty}}{1+\max(\|h\|_{\infty}, \|Gx\|_{\infty}, \|\text{proj}_{\mathcal{K}_G}(Gx - h)\|_{\infty})}$$

2. **Dual infeasibility**:
   $$\frac{\max\{\|\lambda_1-\text{proj}_{\Lambda_1}(\lambda_1)\|_{\infty},\|\lambda_2-\text{proj}_{\mathcal{K}_x^*}(\lambda_2)\|_{\infty}\}}{1+\max\{\|c\|_{\infty},\|G^\top y\|_{\infty}\}}$$

3. **Objective value accuracy**:
   $$\frac{|c^{\top}x-(y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-})|}{1+\max\{|c^{\top}x|, |y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-}|\}}$$

where $\lambda=c-G^{\top}y=[\lambda_{1}^{\top},\lambda_{2}^{\top}]^{\top},\lambda_1\in \Lambda_1 \subseteq \mathbb{R}^{n_1}, \lambda_2\in \mathbb{R}^{n_2}$.

## Paper
```
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

## LICENSE
PDCS are licensed under the Apache License 2.0