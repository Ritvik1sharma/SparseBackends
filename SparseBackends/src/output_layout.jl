# ─────────────────────────────────────────────────────────────────────────────
# OutputLayout — the single way to request how a contraction's OUTPUT is laid out.
#
#   OutputLayout(sparse = [...], dense = [...])
#
# `sparse` are the indices that form the aliased/block-sparse KEY (prefix), `dense`
# the indices of each dense BLOCK (tail). The order inside each list is the storage
# order. For a dense output there is no key: use `sparse = Index[]` and put the whole
# output order in `dense`.
#
# `layout = nothing` (the default everywhere) means the result's stock layout: the
# key/dense split is inherited from the operands and the order is output_inds'
# fixed grouping (one documented exception: aliased × dense with a dense output uses
# its own stock order, see wrapped_contract_aliased).
#
# A layout that cannot be honoured is an error (OutputLayoutError), never a silent
# fallback.
# ─────────────────────────────────────────────────────────────────────────────

import ITensors

struct OutputLayout
    sparse::Vector{ITensors.Index}
    dense::Vector{ITensors.Index}
end
OutputLayout(; sparse, dense) =
    OutputLayout(collect(ITensors.Index, sparse), collect(ITensors.Index, dense))

struct OutputLayoutError <: Exception
    msg::String
end
Base.showerror(io::IO, e::OutputLayoutError) = print(io, "OutputLayoutError: ", e.msg)

"""All output indices of `L` in storage order (key first, then dense block)."""
layout_inds(L::OutputLayout) = vcat(L.sparse, L.dense)

"""The same order with no key: for requesting a dense output."""
as_dense(L::OutputLayout) = OutputLayout(ITensors.Index[], layout_inds(L))

_layout_tag(I::ITensors.Index) = string(ITensors.dim(I), "|", ITensors.tags(I), "|p", ITensors.plev(I))

"""
    validate_layout(L, indsA, indsB)

Throw `OutputLayoutError` unless `L` lists exactly the uncontracted indices of A and B
(full Index equality, prime level included), each once.
"""
function validate_layout(L::OutputLayout, indsA, indsB)
    sA, sB = Set(indsA), Set(indsB)
    expected = ITensors.Index[I for I in indsA if !(I in sB)]
    append!(expected, ITensors.Index[I for I in indsB if !(I in sA)])
    got = layout_inds(L)
    if !allunique(got)
        dup = unique(I for I in got if count(==(I), got) > 1)
        throw(OutputLayoutError("index listed more than once: " * join(_layout_tag.(dup), ", ")))
    end
    sg, se = Set(got), Set(expected)
    if sg != se
        missing = [I for I in expected if !(I in sg)]
        extra   = [I for I in got if !(I in se)]
        throw(OutputLayoutError("layout does not match the output's indices; missing: [" *
            join(_layout_tag.(missing), ", ") * "], not in the output: [" *
            join(_layout_tag.(extra), ", ") * "]"))
    end
    return nothing
end

"""Layout of an `output_inds` result (`indsC` = [key..., dense...], `denseC` = dense)."""
function _layout_from_output_inds(indsC, denseC)
    nd = length(denseC)
    tail = collect(ITensors.Index, indsC[end-nd+1:end])
    tail == collect(ITensors.Index, denseC) ||
        error("_layout_from_output_inds: indsC does not end with denseC")
    return OutputLayout(collect(ITensors.Index, indsC[1:end-nd]), tail)
end
