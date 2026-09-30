using JuMP
using PDCS: PDCS_CPU
import MathOptInterface as MOI

model = Model(PDCS_CPU.Optimizer)
set_silent(model)
set_time_limit_sec(model, 30.0)
@variable(model, t)
@constraint(model, [t, 3.0, 4.0] in SecondOrderCone())
@objective(model, Min, t)
optimize!(model)
termination_status(model) == MOI.OPTIMAL || error("SOC example did not converge")
println("SOC optimum: ", objective_value(model), " (expected 5.0)")
