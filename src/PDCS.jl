"""
PDCS: A Primal-Dual Large-Scale Conic Programming Solver with GPU Enhancements

This package provides both CPU and GPU implementations of the Primal-Dual algorithm
for conic programming.

# Usage

```julia
using PDCS
using PDCS.PDCS_GPU  # or PDCS.PDCS_CPU

# Use PDCS_GPU or PDCS_CPU modules as before
```
"""
module PDCS

# Load submodules - they will be available as PDCS.PDCS_GPU and PDCS.PDCS_CPU
include("pdcs_gpu/PDCS_GPU.jl")
include("pdcs_cpu/PDCS_CPU.jl")

# Export submodules for easier access
# Users can do: using PDCS: PDCS_GPU, PDCS_CPU
# or: using PDCS.PDCS_GPU

end # module PDCS
