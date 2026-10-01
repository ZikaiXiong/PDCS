# PDCS validation — submission package 20261001-r2

This package is based on a 74-file PDCS source preparation snapshot at commit
`966385736b25bea85f29be81bff8984f35c3b7af`. AUTHORS, Project.toml, README.md
and CONTRIBUTING.md now list all four paper authors. The other 70 base files,
including every solver, extension, test and example file, are unchanged.
Dependency declarations are unchanged. This folder adds packaging and
administrative material. `validation/SUMMARY.json` records the checks.

The permanent Julia package UUID is
`9123d4a1-5282-4e19-bc2a-6f2650421a93`. Historical validation logs and
`validation/preparation_changes.patch` predate this assignment and therefore
show the former placeholder UUID; they are retained as test provenance.
Release-only repository and local-path labels in the historical patch and CI
annotation metadata were normalized without changing runtime source.

After assigning the permanent UUID, the release folder was copied to an
isolated `/tmp` directory and checked with Julia 1.12.5. With
`JULIA_PKG_OFFLINE=true` and the existing local Julia depot, `Pkg.instantiate()`,
`examples/soc.jl`, and `Pkg.test()` completed with exit code 0. Julia identified
the package as `[9123d4a1] PDCS v0.1.0`; the SOC objective was
`4.9999999996627995`, and the CPU suite passed **89/89** assertions. No package
or artifact was downloaded during this check.

## Completed source and runtime checks

- A fresh clone established the 74-file upstream base before the four declared
  metadata overrides. All 41 literal Julia includes, three reviewed nonliteral
  includes and 23 local CUDA includes resolve to included files. Imported Julia
  packages are declared in `Project.toml` or belong to Base/the package itself.
- On Linux with Julia 1.11.4, a fresh dependency depot and
  `JULIA_LOAD_PATH=@:@stdlib`, installation, the documented SOC example and
  `Pkg.test()` succeeded. The default CPU suite passed **89/89** assertions.
  No previous user depot, startup file or research dataset was used.
- The source archive was extracted without `.git`, loaded from another working
  directory and checked against three analytic problems. Each solve returned
  **OPTIMAL** and **FEASIBLE_POINT**.
- CUDA Toolkit 12.6.77 rebuilt all four PTX files and `libfew_block_proj.so` for
  A100 (`sm_80`) from source. The library loads, its required symbols exist,
  its ABI is **2**, and all linked libraries resolve.
- Prior A100 hardware testing passed **47/47** assertions with Julia 1.10.4 and
  CUDA.jl 5.11.3. All 74 files in that test's source tree were compared with this
  release base and match exactly. This evidence is separate from the fresh-depot
  CPU run and the queued CUDA.jl 6.4.1 run.

| Extracted-archive CPU case | Objective | Analytic optimum | Status |
|---|---:|---:|---|
| SOC | 4.9999999996627995 | 5 | OPTIMAL |
| Exponential | 2.718281828289473 | exp(1) | OPTIMAL |
| Dual exponential | 0.36787944102817904 | exp(-1) | OPTIMAL |

Absolute objective errors are below `4e-10`. The regression acceptance tolerance
is `2e-4`; tests also check feasible primal status and the lower bound on `t`.
The test suite deliberately excludes LP and rotated-SOC solves.

## Evidence and limits

`validation/build-and-tests.log` records the new folder's CPU `build.sh` run: **exit 0, 89/89 passed**.
`cpu-fresh.log` records the separate clean-depot installation test;
`archive-solves.log` records the explicit extracted-archive statuses;
`prior-same-source-gpu-a100.log` records the completed hardware suite;
`native-build.log` and `native-linkage.txt` record native build/linkage checks.

The documented GPU installation also resolved successfully to CUDA.jl 6.4.1 in
another fresh depot, and its dependencies precompiled. Hardware job `7972329`
was still queued when this package was prepared. It is **not counted as passed**.
The convenience wrapper therefore defaults to the previously tested CUDA.jl
5.11.3; it permits an explicit `CUDA_VERSION` override. Its GPU/all wrapper modes
were not newly executed for this release folder.

GitHub Actions run `36734318016` did not start because of an account billing
lock. Its annotation is retained in `ci-annotations.json`. It supplies no test
result; Windows and macOS were not exercised here. The optional Python/CVXPY
bridge was not tested. These are installation and small-instance correctness
checks, not performance benchmarks or proof for all solver inputs.

The package is self-contained with respect to PDCS source and test data.
Julia dependencies are downloaded by Pkg, and GPU mode needs an external
NVIDIA driver/toolkit, compatible host compiler and GNU make. Validation logs
contain the original machine paths as evidence; they are not runtime inputs.
