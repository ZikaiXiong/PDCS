module PDCSTests

using Test

@testset "PDCS" begin
    include("test_cpu_only_import.jl")
    include("test_projections_cpu.jl")
    include("test_solver_cpu.jl")
    include("test_exp_projection_regression.jl")
    include("test_bulk_cache_structure.jl")
    # These GPU utilities are pure Julia and need no CUDA device.
    include("test_projection_strategy.jl")
    include("test_plain_multi_logger.jl")
end

end
