"""
    select_projection_strategy(block_sizes, projection_types) -> Symbol

Select the GPU hierarchy used to project a collection of cone blocks.  The
return value is one of `:gridWise`, `:blockWise`, `:warpWise`, or
`:threadWise`.

`block_sizes` and `projection_types` use the solver's complete block layout.
The first two blocks are scalar/simple cones in that layout, so only entries
from the third block onward determine the largest structured-cone dimension.

The rule is fitted from same-input grid/block/warp/thread measurements.
Mixed cone families and internal RSOC layouts use the robust block mapping. For pure SOC layouts, dimensions 1--4 favor a thread, 5--64 a
warp, dimensions 65--1023 switch from block to warp near 1,000 cones, and
dimensions at least 1024 favor a block. A single SOC switches from block to
the grid implementation at dimension 32768. The small-SOC high-count crossover
is handled separately because warp occupancy eventually costs more than doing
the short reduction in one thread.

For controlled end-to-end A/B experiments, `PDCS_PROJECTION_STRATEGY_OVERRIDE`
may be set to `grid`, `block`, `warp`, or `thread`.  It is intentionally unset
in production runs, where the measured automatic rule below is used.
"""
function projection_strategy_override()::Union{Nothing,Symbol}
    raw = lowercase(strip(get(ENV, "PDCS_PROJECTION_STRATEGY_OVERRIDE", "")))
    isempty(raw) && return nothing
    raw in ("grid", "gridwise") && return :gridWise
    raw in ("block", "blockwise") && return :blockWise
    raw in ("warp", "warpwise") && return :warpWise
    raw in ("thread", "threadwise") && return :threadWise
    throw(ArgumentError(
        "PDCS_PROJECTION_STRATEGY_OVERRIDE must be grid, block, warp, or thread",
    ))
end

function select_projection_strategy(
    block_sizes::AbstractVector{<:Integer},
    projection_types::AbstractVector{<:Integer},
)::Symbol
    length(block_sizes) == length(projection_types) ||
        throw(DimensionMismatch("block_sizes and projection_types must have equal length"))

    override = projection_strategy_override()
    override === nothing || return override

    block_count = length(block_sizes)
    # Kernels used for block/warp mappings assume the solver's two leading
    # simple blocks. Preserve the legacy safe path for direct underspecified
    # calls, which are useful in downstream code even though PDCS pads layouts.
    block_count < 3 && return :gridWise

    structured_types = @view projection_types[3:end]
    structured_sizes = @view block_sizes[3:end]
    is_soc(code) = code in (5, 6, 7, 20, 21, 22)
    is_rsoc(code) = code in (8, 9, 10, 23, 24, 25)
    has_rsoc = any(is_rsoc, structured_types)
    has_soc = any(is_soc, structured_types)
    pure_soc = has_soc && all(is_soc, structured_types)

    (has_rsoc || !pure_soc) && return :blockWise

    soc_dimensions = (structured_sizes[i] for i in eachindex(structured_sizes)
                      if is_soc(structured_types[i]))
    largest_soc = maximum(soc_dimensions)
    structured_count = block_count - 2

    if structured_count == 1
        largest_soc <= 3 && return :threadWise
        largest_soc <= 64 && return :warpWise
        largest_soc >= 32_768 && return :gridWise
        return :blockWise
    elseif largest_soc <= 4
        return :threadWise
    elseif largest_soc <= 64
        # The measured crossover follows the work balance closely: one-warp
        # mapping wins until roughly 1024 cones per SOC coordinate, after
        # which its 32-lane occupancy cost exceeds the serial short reduction.
        return structured_count >= 1_024 * largest_soc ?
            :threadWise : :warpWise
    elseif largest_soc >= 1_024
        return :blockWise
    elseif structured_count <= 998
        return :blockWise
    else
        return :warpWise
    end
end
