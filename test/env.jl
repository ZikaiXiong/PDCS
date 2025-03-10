
using Pkg
Pkg.activate("pdcp_env")

Pkg.add(Pkg.PackageSpec(name="JuMP", version="1.22.2"))
Pkg.add(Pkg.PackageSpec(name="CodecZlib", version="0.7.5"))
Pkg.add(Pkg.PackageSpec(name="MathOptInterface", version="1.31.0"))
Pkg.add(Pkg.PackageSpec(name="Roots", version="2.1.0"))
Pkg.add(Pkg.PackageSpec(name="PolynomialRoots", version="1.0.0"))
Pkg.add(Pkg.PackageSpec(name="Polynomials", version="3.1.0"))
Pkg.add("JLD2")
Pkg.add("CSV")
Pkg.add("BlockArrays")
Pkg.add("DataStructures")
Pkg.add("CUDA")
Pkg.add("DataFrames")
Pkg.add("Match")