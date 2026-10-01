using Test
using JuMP
import MathOptInterface as MOI
using PDCS: PDCS_CPU

function mock_conic_data()
    return (
        objective_sense=MOI.MIN_SENSE,
        objective_constant=2.0,
        objective_coefficients=[1.0, 0.0, -3.0],
        variable_lower=[-Inf, 0.0, -1.0],
        variable_upper=[Inf, Inf, 2.0],
        num_rows=6,
        num_variables=3,
        colptr=[1, 3, 5, 7],
        rowval=[1, 4, 2, 5, 3, 6],
        nzval=[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        affine_constants=[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        cone_blocks=[
            (set_type=MOI.Zeros, dimension=1, first_row=1, source_block=1),
            (set_type=MOI.Nonnegatives, dimension=2, first_row=2, source_block=2),
            (set_type=MOI.SecondOrderCone, dimension=3, first_row=4, source_block=3),
        ],
        layout=:mock_layout,
        timings=(total_seconds=0.1,),
    )
end

function cache_from_model(model)
    backend = JuMP.backend(model)
    return backend.model_cache.model
end

@testset "PDCS reuses Int64 sparse indices" begin
    int64_indices = Int64[1, 3, 5]
    int32_indices = Int32[1, 3, 5]
    @test PDCS_CPU._as_int64_indices(int64_indices) === int64_indices
    @test PDCS_CPU._as_int64_indices(int32_indices) == int64_indices
    @test PDCS_CPU._as_int64_indices(int32_indices) !== int32_indices
end

@testset "PDCS bulk conic cache construction" begin
    data = mock_conic_data()
    model = PDCS_CPU.model_from_conic_data(data)
    cache = cache_from_model(model)
    coefficients = cache.constraints.coefficients

    @test JuMP.num_variables(model) == data.num_variables
    @test JuMP.objective_sense(model) == MOI.MIN_SENSE
    @test MOI.get(
        JuMP.backend(model),
        MOI.NumberOfConstraints{
            MOI.VectorAffineFunction{Float64},
            MOI.SecondOrderCone,
        }(),
    ) == 1
    @test model.ext[:PDCS_conic_data] === data
    @test model.ext[:JumpRW_CBF_layout] === data.layout
    @test model.ext[:PDCS_optimizer_cache] === JuMP.backend(model).model_cache
    @test model.ext[:PDCS_cache_build_seconds] >= 0.0
    @test coefficients.colptr === data.colptr
    @test coefficients.rowval === data.rowval
    @test coefficients.nzval === data.nzval
    @test cache.constraints.constants.b === data.affine_constants
    @test cache.variables.lower === data.variable_lower
    @test cache.variables.upper === data.variable_upper
    @test cache.variables.set_mask == UInt16[0x0000, 0x0002, 0x0008]
    @test cache.constraints.final_touch

    objective = MOI.get(
        cache,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    )
    @test objective.constant == 2.0
    @test [(term.variable.value, term.coefficient) for term in objective.terms] == [
        (1, 1.0),
        (3, -3.0),
    ]
end

@testset "PDCS bulk conic cache validation" begin
    data = mock_conic_data()
    @test_throws ArgumentError PDCS_CPU.model_from_conic_data(
        merge(data, (colptr=[2, 3, 5, 7],)),
    )
    @test_throws DimensionMismatch PDCS_CPU.model_from_conic_data(
        merge(data, (affine_constants=zeros(5),)),
    )
    @test_throws ArgumentError PDCS_CPU.model_from_conic_data(
        merge(data, (rowval=[4, 1, 2, 5, 3, 6],)),
    )
    @test_throws ArgumentError PDCS_CPU.model_from_conic_data(
        merge(data, (variable_lower=[-Inf, NaN, -1.0],)),
    )
    @test_throws ArgumentError PDCS_CPU.model_from_conic_data(
        merge(data, (cone_blocks=reverse(data.cone_blocks),)),
    )
    @test_throws ArgumentError PDCS_CPU.model_from_conic_data((num_rows=1,))
end
