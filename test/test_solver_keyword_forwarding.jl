module SolverKeywordForwardingTests

using Test

# Exercise the production dispatcher without a GPU. Only the receiving solve
# functions and the data container are stand-ins; the dispatch body and field
# names come from the maintained source files.
module ForwardingHarness
struct CuArray end
const PDCS_GPU_Solver = NamedTuple
rpdhg_gpu_solve(; kwargs...) = (; kwargs...)
rpdhg_gpu_solve_input_gpu_data(; kwargs...) = (; kwargs...)
end

gpu_source = joinpath(@__DIR__, "..", "src", "pdcs_gpu")
definitions = Meta.parseall(read(joinpath(gpu_source, "def_struct.jl"), String))
solver_struct = only(filter(definitions.args) do expression
    expression isa Expr && expression.head == :struct &&
        expression.args[2] == :PDCS_GPU_Solver
end)
fields = Symbol[
    declaration.args[1] for declaration in solver_struct.args[3].args
    if declaration isa Expr && declaration.head == :(::)
]
algorithms = Meta.parseall(read(joinpath(gpu_source, "rpdhg_alg_gpu_gen.jl"), String))
dispatcher = only(filter(algorithms.args) do expression
    expression isa Expr && expression.head == :function &&
        expression.args[1] isa Expr && expression.args[1].head == :call &&
        expression.args[1].args[1] == :solve_with_solver
end)
Core.eval(ForwardingHarness, dispatcher)

@testset "solver struct forwards every stored option" begin
    for input in ([1.0], ForwardingHarness.CuArray())
        # Each field has a distinct value so swapping or dropping fields fails.
        values = NamedTuple{Tuple(fields)}(Tuple(fields))
        solver = merge(values, (; c=input))
        forwarded = ForwardingHarness.solve_with_solver(solver)
        @test Set(keys(forwarded)) == Set(fields)
        for name in fields
            @test getproperty(forwarded, name) === getproperty(solver, name)
        end
    end
end

end
