include("test_bulk_cache_structure.jl")

@testset "PDCS bulk cache optimizes without a second generic copy" begin
    data = (
        objective_sense=MOI.MIN_SENSE,
        objective_constant=0.0,
        objective_coefficients=[1.0],
        variable_lower=[-Inf],
        variable_upper=[Inf],
        num_rows=1,
        num_variables=1,
        colptr=[1, 2],
        rowval=[1],
        nzval=[1.0],
        affine_constants=[-1.0],
        cone_blocks=[
            (set_type=MOI.Zeros, dimension=1, first_row=1, source_block=1),
        ],
        layout=:one_equality,
        timings=(total_seconds=0.0,),
    )
    optimizer = PDCS_CPU.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("verbose"), 0)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("time_limit_secs"), 10.0)
    model = PDCS_CPU.model_from_conic_data(data; optimizer)
    backend = JuMP.backend(model)
    cache_before = backend.model_cache
    JuMP.optimize!(model)

    @test backend.model_cache === cache_before
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ 1.0 atol=1e-5

    limited_optimizer = PDCS_CPU.Optimizer()
    for (name, value) in (
        "verbose" => 0,
        "time_limit_secs" => 10.0,
        "max_outer_iter" => 1,
        "max_inner_iter" => 1,
        "check_terminate_freq" => 1,
        "print_freq" => 1,
    )
        MOI.set(
            limited_optimizer,
            MOI.RawOptimizerAttribute(name),
            value,
        )
    end
    limited_model = PDCS_CPU.model_from_conic_data(
        data;
        optimizer=limited_optimizer,
    )
    JuMP.optimize!(limited_model)

    @test JuMP.termination_status(limited_model) == MOI.ITERATION_LIMIT
    @test MOI.get(limited_optimizer, PDCS_CPU.PDHGIterations()) == 1
end
