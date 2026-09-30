module PDCSGPUTests

using Test
using LinearAlgebra
using CUDA
using JuMP
import PDCS
import MathOptInterface as MOI
using PDCS: PDCS_GPU

CUDA.functional() || error("A functional, allocated CUDA GPU is required")
CUDA.versioninfo()

@testset "PDCS GPU hardware" begin
    @test realpath(pkgdir(PDCS)) == realpath(joinpath(@__DIR__, ".."))
    @test Base.get_extension(PDCS, :PDCSGPUExt) !== nothing

    @testset "Device arithmetic" begin
        x = CuArray(collect(1.0:1024.0))
        y = 2 .* x .+ 1
        CUDA.synchronize()
        @test Array(y) == 2 .* collect(1.0:1024.0) .+ 1
        @test sum(x) ≈ 1024 * 1025 / 2
    end

    include("test_gridwise_lazy_handle_gpu.jl")

    @testset "Native SOC projection and workspace aliasing" begin
        dimension = 10_002
        tail = fill(0.8 / sqrt(dimension - 1), dimension - 1)
        for (input, expected) in (
            ([0.9; tail], [0.9; tail]),
            ([0.2; tail], [0.5; (0.5 / 0.8) .* tail]),
            ([-0.9; tail], zeros(dimension)),
        ), alias_workspace in (false, true)
            x = CuArray(input)
            dummy = CUDA.zeros(Float64, dimension)
            workspace = alias_workspace ? x : CUDA.zeros(Float64, dimension)
            warm = CUDA.zeros(Float64, 1)
            sizes = Int64[dimension]
            for repetition in 1:3
                PDCS_GPU.gridWise_block_proj(
                    x, dummy, dummy, dummy, dummy, dummy, workspace, warm,
                    Int64[0], CuArray(sizes), sizes, Int64(1), Int64[20],
                )
                CUDA.synchronize()
                @test maximum(abs, Array(x) .- expected) <= 64eps(Float64)
            end
        end
    end

    @testset "Conic solves" begin
        for (name, cone, constants, expected) in (
            ("SOC", MOI.SecondOrderCone(3), [3.0, 4.0], 5.0),
            ("Exponential", MOI.ExponentialCone(), [1.0, 1.0], exp(1.0)),
            ("Dual exponential", MOI.DualExponentialCone(), [-1.0, 0.0], exp(-1.0)),
        )
            @testset "$name" begin
                model = Model(PDCS_GPU.Optimizer)
                set_silent(model)
                set_time_limit_sec(model, 60.0)
                set_optimizer_attribute(model, "rel_tol", 1e-7)
                set_optimizer_attribute(model, "abs_tol", 1e-7)
                @variable(model, t)
                if name == "SOC"
                    @constraint(model, [t, constants[1], constants[2]] in cone)
                else
                    @constraint(model, [constants[1], constants[2], t] in cone)
                end
                @objective(model, Min, t)
                optimize!(model)
                CUDA.synchronize()
                @test termination_status(model) == MOI.OPTIMAL
                @test primal_status(model) == MOI.FEASIBLE_POINT
                @test objective_value(model) ≈ expected atol=2e-4
                @test value(t) >= expected - 2e-4
                println("GPU_SOLVE cone=$name objective=$(objective_value(model)) expected=$expected")
            end
        end
    end

    runtime = PDCS_GPU.check_gridWise_runtime!()
    @test runtime.native_enabled
    @test runtime.state == :passed
    @test isempty(runtime.missing_artifacts)
    println("GPU_NATIVE_RUNTIME=$runtime")
end

end
