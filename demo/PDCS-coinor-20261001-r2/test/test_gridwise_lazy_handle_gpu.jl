using Test
using CUDA
using PDCS: PDCS_GPU

CUDA.functional() || error("A functional CUDA device is required")

@testset "gridWise lazy cuBLAS handle" begin
    dimension = 10
    count = 3
    total = dimension * count
    input = collect(Float64, 1:total) ./ total
    starts = Int64.(0:dimension:total-dimension)
    sizes = fill(Int64(dimension), count)
    types = fill(Int64(20), count)
    zeros_gpu = CUDA.zeros(Float64, total)
    x = CuArray(input)
    warm = CUDA.ones(Float64, count)

    args = (x, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu,
            zeros_gpu, warm, starts, CuArray(sizes), sizes, Int64(count), types)

    # Loading PDCS must not eagerly create a CUDA/cuBLAS handle. The first
    # grid-wise call creates it even when PDCS_SKIP_GPU_PRECOMPILE=1.
    existing = PDCS_GPU._gridWise_cublas_handle[]
    existing === nothing || PDCS_GPU.destroy_cublas_handle(existing)
    PDCS_GPU._gridWise_cublas_handle[] = nothing
    @test !isdefined(PDCS_GPU, :handle)
    @test PDCS_GPU._gridWise_cublas_handle[] === nothing
    PDCS_GPU.gridWise_block_proj(args...)
    first_handle = PDCS_GPU._gridWise_cublas_handle[]
    @test first_handle !== nothing
    @test first_handle.handle != C_NULL
    @test all(isfinite, Array(x))
    runtime = PDCS_GPU.gridWise_runtime_status()
    @test runtime.native_enabled
    @test runtime.state == :passed
    @test runtime.configuration !== nothing

    # A second projection reuses the process-owned handle.
    copyto!(x, input)
    PDCS_GPU.gridWise_block_proj(args...)
    @test PDCS_GPU._gridWise_cublas_handle[] === first_handle
    @test all(isfinite, Array(x))
end
