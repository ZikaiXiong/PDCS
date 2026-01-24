
### update solver's data

function update_G!(solver::PDCS_GPU_Solver, G::Union{CuArray, CUDA.CUSPARSE.CuSparseMatrixCSR, Vector{rpdhg_float}})
    
    if solver.G isa CUDA.CUSPARSE.CuSparseMatrixCSR
        if G isa CUDA.CUSPARSE.CuSparseMatrixCSR
            @assert length(solver.G.nzVal) == length(G.nzVal) && 
                    size(solver.G) == size(G) && 
                    solver.G.colVal == G.colVal && 
                    solver.G.rowPtr == G.rowPtr "Matrix sparsity pattern must match"
            solver.G.nzVal .= G.nzVal
        elseif G isa CuArray
            @assert length(solver.G.nzVal) == length(G)
            solver.G.nzVal .= G
        elseif G isa Vector{rpdhg_float}
            @assert length(solver.G.nzVal) == length(G)
            solver.G.nzVal .= CuArray(G)
        else
            error("Unsupported A type: $(typeof(G))")
        end
    elseif solver.G isa CUDA.CUSPARSE.CuSparseMatrixCSC
        if G isa CuArray
            @assert length(solver.G.nzVal) == length(G)
            solver.G.nzVal .= G
        elseif G isa CUDA.CUSPARSE.CuSparseMatrixCSC
            @assert length(solver.G.nzVal) == length(G.nzVal) && size(solver.G) == size(G)
            solver.G.nzVal .= G.nzVal
        elseif G isa Vector{rpdhg_float}
            @assert length(solver.G.nzVal) == length(G)
            solver.G.nzVal .= CuArray(G)
        else
            error("Unsupported A type: $(typeof(G))")
        end
    else
        error("Unsupported solver.G type: $(typeof(solver.G)). Only CuSparseMatrixCSR/CSC supported for update.")
    end
end


function update_vector!(target::Union{CuArray, Vector{rpdhg_float}}, vector::Union{CuArray, Vector{rpdhg_float}})
    if target isa CuArray
        if vector isa CuArray
            target .= vector
        elseif vector isa Vector{rpdhg_float}
            target .= CuArray(vector)
        else
            error("vector must be CuArray for update. Current type: $(typeof(vector))")
        end
    elseif target isa Vector{rpdhg_float}
        if vector isa Vector{rpdhg_float}
            target .= vector
        else
            error("vector must be Vector{rpdhg_float} for update. Current type: $(typeof(vector))")
        end
    else
        error("Unsupported target type: $(typeof(target)). Expected CuArray or Vector.")
    end
end



function update_solver!(
    solver::PDCS_GPU_Solver,
    G::Union{CuArray, CUDA.CUSPARSE.CuSparseMatrixCSR},
    c::Union{CuArray, Vector{rpdhg_float}},
    h::Union{CuArray, Vector{rpdhg_float}},
    bl::Union{CuArray, Vector{rpdhg_float}},
    bu::Union{CuArray, Vector{rpdhg_float}},
)
    update_A!(solver, G)
    update_vector!(solver.c, c)
    update_vector!(solver.h, h)
    update_vector!(solver.bl, bl)
    update_vector!(solver.bu, bu)
end