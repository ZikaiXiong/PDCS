using Test

@testset "PDCS CPU import does not load CUDA" begin
    project = dirname(Base.active_project())
    script = """
    using PDCS: PDCS_CPU
    loaded_names = string.(nameof.(values(Base.loaded_modules)))
    @assert !("CUDA" in loaded_names)
    @assert isdefined(PDCS_CPU, :Optimizer)
    """
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`
    @test success(pipeline(command; stdout=devnull, stderr=stderr))
end
