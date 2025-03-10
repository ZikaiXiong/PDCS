# PDCP
Primal Dual Algorithm for Conic Programming
- A Julia and CUDA implementation of the Primal-Dual algorithm for conic programming

### Overview
This project implements the algorithm for solving conic optimization problems of the form:

$$\min_{x=(x_1,x_2),x_1\in \mathbb{R}^{n_1}, x_2\in \mathbb{R}^{n_2}} c^{\top} x\ \  \text{s.t.} Gx-h\in \mathcal{K}_d, l\leq x_1\leq u, x_2 \in \mathcal{K}_p,$$

where $\mathcal{K}_G$ is a closed convex cone, and $\mathcal{K}_x$ is a closed convex cone.



### Features
- Supports multiple cone types:
    - Second-order cone (recommended but not the rsoc) and Rotated second-order cone
    - Exponential cone and dual exponential cone


### Installation
```bash
git clone https://github.com/zhenweilin/PDCP.git
cd PDCP
cd src/pdcp_gpu/cuda
make # compile the cuda code, default ARCH = sm_90, if necessary please modify makefile
cd ../../..
```


### Usage
- Given the example code in `./test/`, you can run the following commands to test the code.
```julia
julia ./test/env.jl # install the dependencies
julia ./test/test_exp.jl # test the exponential cone cpu
julia ./test/test_soc.jl # test the second-order cone cpu
julia ./test/test_rsoc.jl # test the rotated second-order cone cpu

julia ./test/test_exp_gpu.jl # test the exponential cone gpu
julia ./test/test_soc_gpu.jl # test the second-order cone gpu
julia ./test/test_rsoc_gpu.jl # test the rotated second-order cone gpu
```


### Converge Criteria
Three criteria are considered for the performance:
- Primal infeasibility: $\frac{\|(Gx - h) - \text{proj}\_{\mathcal{K}_G}(Gx - h)\|\_{\infty}}{1+\max(\|h\|\_{\infty}, \|Gx\|\_{\infty}, \|\text{proj}\_{\mathcal{K}_G}(Gx - h)\|\_{\infty})}$
- Dual infeasibility: $\frac{\max\\{\|\lambda_1-\text{proj}\_{\Lambda_1}(\lambda_1)\|\_{\infty},\|\lambda_2-\text{proj}\_{\mathcal{K}_x^*}(\lambda_2)\|\_{\infty}\\}}{1+\max\\{\|c\|\_{\infty},\|G^\top y\|\_{\infty}\\}}$, 
- Objective value accuracy: $\frac{|c^{\top}x-(y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-})|}{1+\max\{|c^{\top}x|, |y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-}|\}}$

where $\lambda=c-G^{\top}y=[\lambda_{1}^{\top},\lambda_{2}^{\top}]^{\top},\lambda_1\in \Lambda_1 \subseteq \mathbb{R}^{n_1}, \lambda_2\in \mathbb{R}^{n_2}$.




### Citation
- Lin Zhenwei and Zikai Xiong. (2025). zhenweilin/PDCP: v1.0 (v1.0). Zenodo. https://doi.org/10.5281/zenodo.14954645

### Paper
- Coming soon
