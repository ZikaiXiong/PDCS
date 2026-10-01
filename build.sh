#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

julia_bin="${JULIA:-julia}"
mode="${MODE:-cpu}"
case "$mode" in
    cpu|gpu|all) ;;
    *) echo "MODE must be cpu, gpu or all" >&2; exit 2 ;;
esac

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

run_julia() {
    env -u LD_LIBRARY_PATH "$julia_bin" --startup-file=no \
        --threads="${JULIA_NUM_THREADS:-2}" "$@"
}

if [[ "$mode" == cpu || "$mode" == all ]]; then
    run_julia --project=. -e \
        'using Pkg; Pkg.instantiate(); include("examples/soc.jl"); Pkg.test()'
fi

if [[ "$mode" == gpu || "$mode" == all ]]; then
    : "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit directory}"
    : "${GPU_ARCH:?Set GPU_ARCH to sm_80 for A100 or sm_90 for H100}"
    export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$PWD/build/cuda"
    make -C src/pdcs_gpu/cuda rebuild-gpu \
        CUDA_HOME="$CUDA_HOME" ARCH="$GPU_ARCH" \
        OUTPUT_DIR="$PDCS_CUDA_PROJECTION_ARTIFACT_DIR"
    export PDCS_SKIP_GPU_PRECOMPILE=1
    export PDCS_GRIDWISE_MODE=native
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PDCS_TEST_CUDA_VERSION="${CUDA_VERSION:-5.11.3}"
    run_julia -e \
        'using Pkg; Pkg.activate("build/gpu-env"); Pkg.add([Pkg.PackageSpec(name="CUDA", version=ENV["PDCS_TEST_CUDA_VERSION"]), Pkg.PackageSpec(name="JuMP"), Pkg.PackageSpec(name="MathOptInterface")]); Pkg.develop(path=pwd())'
    run_julia --project=build/gpu-env test/runtests_gpu.jl
fi

echo "PDCS $mode installation checks passed."
