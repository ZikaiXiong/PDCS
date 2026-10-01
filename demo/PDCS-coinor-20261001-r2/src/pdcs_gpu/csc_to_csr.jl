const _SPARSE_INDEX_TYPE_ERROR =
    "sparse_index_type must be :auto, Int32, or Int64 " *
    "(aliases :int32, :int64, \"auto\", \"int32\", and \"int64\" are accepted)"

@inline function _sparse_unsigned_thread_index(
    ::Type{Ti},
    block_index::Integer,
    block_dimension::Integer,
    thread_index::Integer,
) where {Ti<:Union{Int32,Int64}}
    Tu = unsigned(Ti)
    return (Tu(block_index) - one(Tu)) * Tu(block_dimension) + Tu(thread_index)
end

function _normalize_sparse_index_type(option)
    option === :auto && return :auto
    option === "auto" && return :auto
    option === Int32 && return Int32
    option === :int32 && return Int32
    option === "int32" && return Int32
    option === Int64 && return Int64
    option === :int64 && return Int64
    option === "int64" && return Int64
    throw(ArgumentError(_SPARSE_INDEX_TYPE_ERROR))
end

function _fits_int32_sparse_indices(m::Integer, n::Integer, stored_entries::Integer)
    m >= 0 || throw(ArgumentError("matrix row count must be nonnegative"))
    n >= 0 || throw(ArgumentError("matrix column count must be nonnegative"))
    stored_entries >= 0 || throw(ArgumentError("matrix nnz must be nonnegative"))
    limit = typemax(Int32)
    return m <= limit && n <= limit && stored_entries <= limit - 1
end

function _resolve_sparse_index_type(
    option,
    m::Integer,
    n::Integer,
    stored_entries::Integer,
)
    normalized = _normalize_sparse_index_type(option)
    fits_int32 = _fits_int32_sparse_indices(m, n, stored_entries)
    normalized === :auto && return fits_int32 ? Int32 : Int64
    if normalized === Int32 && !fits_int32
        throw(ArgumentError(
            "matrix with m=$m, n=$n, and nnz=$stored_entries does not fit " *
            "in Int32 sparse indices; use sparse_index_type=Int64 or :auto",
        ))
    end
    return normalized
end

function _csc_to_gpu_csr(
    G::SparseMatrixCSC{Float64,<:Integer},
    sparse_index_type = :auto,
)
    Ti = _resolve_sparse_index_type(
        sparse_index_type,
        size(G, 1),
        size(G, 2),
        nnz(G),
    )
    return _upload_csc_to_gpu_csr(G, Ti)
end

function _csc_to_gpu_csr(G::AbstractMatrix{Float64}, sparse_index_type = :auto)
    return _csc_to_gpu_csr(sparse(G), sparse_index_type)
end

_as_sparse_index_vector(values::Vector{Ti}, ::Type{Ti}) where {
    Ti<:Union{Int32,Int64}
} = values

function _as_sparse_index_vector(
    values::AbstractVector{<:Integer},
    ::Type{Ti},
) where {Ti<:Union{Int32,Int64}}
    return Ti.(values)
end

function _csc_to_csr_components(
    G::SparseMatrixCSC{Tv,<:Integer},
    ::Type{Ti},
) where {Tv,Ti<:Union{Int32,Int64}}
    transposed = copy(transpose(G))
    return (;
        rowptr = _as_sparse_index_vector(transposed.colptr, Ti),
        colval = _as_sparse_index_vector(transposed.rowval, Ti),
        nzval = transposed.nzval,
        dims = size(G),
    )
end
