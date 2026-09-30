using Test
using Logging

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const LOGGER_PATH = joinpath(PROJECT_ROOT, "src", "pdcs_gpu", "plain_multi_logger.jl")
include(LOGGER_PATH)

@testset "PlainMultiLogger preserves structured metadata" begin
    output = IOBuffer()
    logger = PlainMultiLogger(IO[output], Logging.Info)

    logging_error = try
        with_logger(logger) do
            @info "restart candidate selected" reason="adaptive restart" selected_name="mean" current_merit=1.25 mean_enabled=true
        end
        nothing
    catch error
        error
    end

    @test logging_error === nothing
    if logging_error === nothing
        logged = String(take!(output))
        @test occursin("restart candidate selected", logged)
        @test occursin("reason = \"adaptive restart\"", logged)
        @test occursin("selected_name = \"mean\"", logged)
        @test occursin("current_merit = 1.25", logged)
        @test occursin("mean_enabled = true", logged)
    end
end
