using Test

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "projection_strategy.jl"))

@testset "GPU projection strategy heuristic" begin
    @test select_projection_strategy([8], [20]) == :gridWise
    @test select_projection_strategy([1, 1, 3], [2, 3, 20]) == :threadWise
    @test select_projection_strategy([1, 1, 8], [2, 3, 20]) == :warpWise
    @test select_projection_strategy([1, 1, 128], [2, 3, 20]) == :blockWise
    @test select_projection_strategy([1, 1, 32_768], [2, 3, 20]) == :gridWise
    @test select_projection_strategy([1, 1, 8], [2, 3, 23]) == :blockWise
    @test select_projection_strategy(vcat([1, 1], fill(3, 1_001)), vcat([2, 3], fill(20, 1_001))) == :threadWise
    @test select_projection_strategy(vcat([1, 1], fill(8, 8_191)), vcat([2, 3], fill(20, 8_191))) == :warpWise
    @test select_projection_strategy(vcat([1, 1], fill(8, 8_192)), vcat([2, 3], fill(20, 8_192))) == :threadWise
    @test select_projection_strategy(vcat([1, 1], fill(16, 16_383)), vcat([2, 3], fill(20, 16_383))) == :warpWise
    @test select_projection_strategy(vcat([1, 1], fill(16, 16_384)), vcat([2, 3], fill(20, 16_384))) == :threadWise
    @test select_projection_strategy(vcat([1, 1], fill(32, 60_001)), vcat([2, 3], fill(20, 60_001))) == :threadWise
    @test select_projection_strategy(vcat([1, 1], fill(201, 998)), vcat([2, 3], fill(20, 998))) == :blockWise
    @test select_projection_strategy(vcat([1, 1], fill(201, 999)), vcat([2, 3], fill(20, 999))) == :warpWise
    @test select_projection_strategy(vcat([1, 1], fill(1_024, 4_096)), vcat([2, 3], fill(20, 4_096))) == :blockWise
    @test select_projection_strategy([1, 1, 3, 3], [2, 3, 20, 27]) == :blockWise
    @test_throws DimensionMismatch select_projection_strategy([2], [20, 20])

    withenv("PDCS_PROJECTION_STRATEGY_OVERRIDE" => "warp") do
        @test select_projection_strategy([1, 1, 3], [2, 3, 20]) == :warpWise
    end
    withenv("PDCS_PROJECTION_STRATEGY_OVERRIDE" => "threadWise") do
        @test select_projection_strategy([1, 1, 128], [2, 3, 20]) == :threadWise
    end
    withenv("PDCS_PROJECTION_STRATEGY_OVERRIDE" => "invalid") do
        @test_throws ArgumentError select_projection_strategy([1, 1, 3], [2, 3, 20])
    end
end
