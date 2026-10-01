using Test
using PDCS: PDCS_CPU

@testset "diagonal exponential projection accepts collapsed root brackets" begin
    rho = PDCS_CPU.newton_rootsearch_diagonal(
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    @test rho == 0.0
end
