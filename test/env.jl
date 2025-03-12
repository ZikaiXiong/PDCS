
using Pkg
Pkg.activate("pdcs_env")

Pkg.add(Pkg.PackageSpec(name="JuMP", version="1.22.2"))
Pkg.add(Pkg.PackageSpec(name="CodecZlib", version="0.7.5"))
Pkg.add(Pkg.PackageSpec(name="MathOptInterface", version="1.31.0"))
Pkg.add(Pkg.PackageSpec(name="Roots", version="2.1.0"))
Pkg.add(Pkg.PackageSpec(name="PolynomialRoots", version="1.0.0"))
Pkg.add(Pkg.PackageSpec(name="Polynomials", version="3.1.0"))
Pkg.add(Pkg.PackageSpec(name="JLD2", version="0.5.11"))
Pkg.add(Pkg.PackageSpec(name="CSV", version="0.10.15"))
Pkg.add(Pkg.PackageSpec(name="BlockArrays", version="1.4.0"))
Pkg.add(Pkg.PackageSpec(name="DataStructures", version="0.18.20"))
Pkg.add(Pkg.PackageSpec(name="DataFrames", version="1.7.0"))
Pkg.add(Pkg.PackageSpec(name="Match", version="2.4.0"))
Pkg.add(Pkg.PackageSpec(name="CUDA", version="5.6.1"))
