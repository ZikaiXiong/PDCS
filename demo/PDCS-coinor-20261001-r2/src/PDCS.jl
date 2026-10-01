"""
PDCS: A Primal-Dual Large-Scale Conic Programming Solver with GPU Enhancements

This package provides both CPU and GPU implementations of the Primal-Dual algorithm
for conic programming.

# Usage

```julia
using PDCS: PDCS_CPU

# GPU support is loaded only after CUDA is requested.
using CUDA
using PDCS: PDCS_GPU
```
"""

__precompile__()
module PDCS

# The CPU implementation is always available. The GPU implementation is loaded
# by PDCSGPUExt only after CUDA itself is loaded.
include("pdcs_cpu/PDCS_CPU.jl")

function _register_gpu!(gpu_module::Module)
    isdefined(@__MODULE__, :PDCS_GPU) && return getfield(@__MODULE__, :PDCS_GPU)
    Core.eval(@__MODULE__, :(const PDCS_GPU = $gpu_module))
    return gpu_module
end

end # module PDCS
