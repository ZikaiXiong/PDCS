using Test
using JuMP
using PDCS: PDCS_CPU
import MathOptInterface as MOI

function small_cpu_model()
    model = Model(PDCS_CPU.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 30.0)
    set_optimizer_attribute(model, "rel_tol", 1e-7)
    set_optimizer_attribute(model, "abs_tol", 1e-7)
    return model
end

function check_solution(model, expected)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test objective_value(model) ≈ expected atol=2e-4
end

@testset "CPU solver through JuMP" begin
    @testset "Second-order cone" begin
        model = small_cpu_model()
        @variable(model, t)
        @constraint(model, [t, 3.0, 4.0] in SecondOrderCone())
        @objective(model, Min, t)
        check_solution(model, 5.0)
        @test value(t) >= 5.0 - 2e-4
    end
    @testset "Exponential cone" begin
        model = small_cpu_model()
        @variable(model, t)
        @constraint(model, [1.0, 1.0, t] in MOI.ExponentialCone())
        @objective(model, Min, t)
        check_solution(model, exp(1.0))
        @test value(t) >= exp(1.0) - 2e-4
    end
    @testset "Dual exponential cone" begin
        model = small_cpu_model()
        @variable(model, t)
        @constraint(model, [-1.0, 0.0, t] in MOI.DualExponentialCone())
        @objective(model, Min, t)
        check_solution(model, exp(-1.0))
        @test value(t) >= exp(-1.0) - 2e-4
    end
end
