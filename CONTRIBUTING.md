# Contributing to PDCS

Discuss bugs and proposed changes with the maintainers through
https://github.com/ZikaiXiong/PDCS/issues and submit a pull request with a small
reproducing example and relevant test coverage.

## Running and extending tests

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The default suite is `test/runtests.jl`, wrapped in the `PDCSTests` module. Add
small deterministic `@testset` cases to the relevant included file, or include a
new file explicitly in the runner. Check statuses, analytic objective values,
and feasibility for solver regressions; avoid timing assertions and comparisons
that require another solver. Do not activate or modify environments from tests.

The default suite covers SOC, exponential and dual exponential cones. Linear
programming and rotated second-order cone solves are outside its scope.
`test_bulk_cache_structure.jl` tests the cache without solving an LP.

CPU tests must run without CUDA or external datasets. Add hardware-dependent
regressions separately and document prerequisites in `src/pdcs_gpu/cuda/README.md`.
The hardware entry point is `test/runtests_gpu.jl`. On an allocated GPU, build
the native artifacts as described there, set `PDCS_CUDA_PROJECTION_ARTIFACT_DIR`,
and run it from a separate environment containing this checkout, CUDA, JuMP,
and MathOptInterface. For example, from the repository root:

```julia
using Pkg
Pkg.activate(temp=true)
Pkg.add(["CUDA", "JuMP", "MathOptInterface"])
Pkg.develop(path=pwd())
include("test/runtests_gpu.jl")
```

This suite requires working hardware and fails if no GPU is available. It checks
device arithmetic, native SOC projection with workspace aliasing, lazy cuBLAS
handle creation, and SOC/exponential/dual-exponential solves with analytic answers.

The GitHub Actions workflow runs the CPU suite with Julia 1.10 and the current
stable Julia release on Linux, macOS, and Windows. A manual workflow dispatch
can also request the GPU suite on a self-hosted runner labelled `nvidia-gpu`.
The runner must provide `CUDA_HOME`, `GPU_ARCH`, a compatible driver and an
allocated NVIDIA device. Hosted CPU CI does not claim GPU coverage.

## Releases

Keep the version in `Project.toml` synchronized with the root `VERSION` file.
Follow [RELEASE.md](RELEASE.md) before creating a release tag.

## COIN-OR submission preparation

See the [submission guidelines](https://www.coin-or.org/contributing/code/#submissions)
and [project management checklist](https://www.coin-or.org/management/).
The root README, AUTHORS, INSTALL, and LICENSE provide the package overview,
attribution, installation/testing procedure, and existing Apache 2.0 license.

The project authors are Zhenwei Lin, Zikai Xiong, Dongdong Ge, and Yinyu Ye,
as listed in AUTHORS and in the paper citation.
The upstream LICENSE also names Zhenwei Lin and Zikai Xiong in its copyright
notice and licenses the code under Apache 2.0.
The project manager is [Zikai Xiong](https://github.com/ZikaiXiong).
Email: [zikai.xiong@northwestern.edu](mailto:zikai.xiong@northwestern.edu), as listed
in the [Northwestern faculty directory](https://research.mccormick.northwestern.edu/research-faculty/directory/profiles/xiong-zikai.html).
Report bugs and request features through https://github.com/ZikaiXiong/PDCS/issues.
Before submitting, the maintainers still need to complete the required ownership
statements and provide
the proposed project description/classification. The permanent Julia package UUID
is `9123d4a1-5282-4e19-bc2a-6f2650421a93`; use this UUID in registries and
downstream environments.
The submission guidelines identify CSRO and CSOL ownership documentation, with
individual OCL confirmations recommended. These must be completed by the relevant
people; they are not generated or signed by the package tests.

After reviewing and committing the changes, a source archive can be made with:

```sh
git archive --format=tar.gz --prefix=PDCS/ --output=PDCS-source.tar.gz HEAD
```

The submission branch contains the standalone package, without research datasets
or cluster workflows. Test the extracted archive with the commands in INSTALL
before sending it. An archive of HEAD contains only committed changes. This
repository preparation does not submit the project or assert COIN-OR acceptance.
