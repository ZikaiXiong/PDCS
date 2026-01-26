using Pkg
Pkg.activate("pdcs_env")
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.resolve()
using PDCS: PDCS_CPU, PDCS_GPU