# CUDA build and hardware tests

CPU installation does not require these artifacts. GPU native projection needs
an NVIDIA device and driver, a CUDA toolkit with `nvcc`, GNU make, and a C++ host
compiler supported by that toolkit.

## Build

Run from the PDCS repository root. Set `CUDA_HOME` to your installed toolkit:

```bash
artifact_dir="$PWD/build/cuda"
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME=/usr/local/cuda ARCH=sm_90 OUTPUT_DIR="$artifact_dir"
```

Use `ARCH=sm_90` for H100 or `ARCH=sm_80` for A100. Rebuild for the device and
toolkit you will actually use. The build produces four PTX files and a library:

```text
moderate_block_proj.ptx
massive_block_proj.ptx
sufficient_block_proj.ptx
utils.ptx
libfew_block_proj.so
```

`make print-config` prints the compiler and build options. `make rebuild-profile`
produces separate diagnostic kernels for the solver's projection profiling API;
these are optional and are not used by the standard hardware tests.

## Install the GPU test environment

From the repository root, create an isolated Julia environment:

```bash
julia --startup-file=no -e 'using Pkg; Pkg.activate("build/gpu-env"); Pkg.add(["CUDA", "JuMP", "MathOptInterface"]); Pkg.develop(path=pwd())'
```

Set runtime options before starting Julia:

```bash
export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$PWD/build/cuda"
export PDCS_SKIP_GPU_PRECOMPILE=1
export PDCS_GRIDWISE_MODE=native
export CUBLAS_WORKSPACE_CONFIG=:4096:8
julia --startup-file=no --project=build/gpu-env test/runtests_gpu.jl
```

On a cluster, run this command inside a scheduler allocation with one GPU.
Keep the scheduler's device visibility settings. The test fails if CUDA is not
functional. It checks GPU arithmetic, native SOC projection and workspace
aliasing, cuBLAS handle creation/reuse, and SOC, exponential, and dual exponential
solves against analytic answers. Linear-programming and rotated-SOC solves are
not included.

## Native runtime and diagnostics

The native library creates and destroys its own cuBLAS handle, uses the handle's
stream, and checks ABI version 2. Julia checks device ownership, storage types,
cone layouts, and tolerances before entering native code. Calls sharing the
native handle and workspace are serialized. The first native projection also
runs a SOC self-test, including the case where input and workspace alias.

The native build and CUDA.jl runtime may use different toolkit versions. The
library records its toolkit library directory as an RPATH. Avoid putting a
conflicting toolkit on Julia's library search path. On Quest, the hardware suite
passed with normal Julia compiled modules and `LD_LIBRARY_PATH` removed only for
the Julia process:

```bash
env -u LD_LIBRARY_PATH julia --startup-file=no \
  --project=build/gpu-env test/runtests_gpu.jl
```

The verified Quest environment was H100 80 GB, driver 610.43.02, native CUDA
12.6.77, Julia 1.10.4, CUDA.jl 5.11.3, and CUDA.jl runtime artifact 13.2.
The hardware suite passed 47 assertions. This is validation of that environment,
not a claim that every CUDA/toolkit combination has been tested.

`PDCS_GRIDWISE_MODE=native` is the default: missing artifacts, stale ABI, failed
self-tests, or native errors stop the solve. `auto` permits a compatibility
fallback, and `block` disables the native grid-wise path. Use `native` for the
hardware suite; a fallback does not validate the native library.

Inspect runtime state after importing CUDA and PDCS:

```julia
using CUDA
using PDCS: PDCS_GPU
println(PDCS_GPU.gridWise_runtime_status())
PDCS_GPU.check_gridWise_runtime!()
```

For a bug report, include the test output, `nvidia-smi`, `nvcc --version`,
`julia --version`, `CUDA.versioninfo()`, `Pkg.status()`, the build command, runtime
options, and `ldd` output for `libfew_block_proj.so`.
