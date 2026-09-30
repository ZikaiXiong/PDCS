using Test
using LinearAlgebra
using PDCS: PDCS_CPU

@testset "SOC projection" begin
    for (input, expected) in (
        ([2.0, 1.0, 0.0], [2.0, 1.0, 0.0]),
        ([-2.0, 1.0, 0.0], zeros(3)),
        ([0.0, 3.0, 4.0], [2.5, 1.5, 2.0]),
        (zeros(3), zeros(3)),
        ([5.0, 3.0, 4.0], [5.0, 3.0, 4.0]),
    )
        projected = copy(input)
        PDCS_CPU.soc_proj!(projected)
        @test projected ≈ expected atol=1e-12
        @test norm(projected[2:end]) <= projected[1] + 1e-12
        again = copy(projected)
        PDCS_CPU.soc_proj!(again)
        @test again ≈ projected atol=1e-12
    end
    # Projection must respect view boundaries.
    storage = [99.0, 0.0, 3.0, 4.0, 99.0]
    PDCS_CPU.soc_proj!(@view storage[2:4])
    @test storage ≈ [99.0, 2.5, 1.5, 2.0, 99.0]
end

@testset "Exponential projection" begin
    for point in ([0.0, 1.0, 2.0], [-1.0, 0.0, 2.0], zeros(3))
        projected = copy(point)
        PDCS_CPU.exponent_proj!(projected)
        @test projected ≈ point atol=1e-10
    end
    projected = [1.0, 1.0, 1.0]
    PDCS_CPU.exponent_proj!(projected)
    @test all(isfinite, projected)
    @test projected[2] > 0
    @test projected[2] * exp(projected[1] / projected[2]) <= projected[3] + 1e-8
    again = copy(projected)
    PDCS_CPU.exponent_proj!(again)
    @test again ≈ projected atol=1e-8
end
