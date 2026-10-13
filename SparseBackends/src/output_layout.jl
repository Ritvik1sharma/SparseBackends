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

# Entries are Index objects, or ROLES resolved against the two operands of the
# contraction (A = first operand, B = second, as written by the caller):
#   (op, kind, k)    op   :A or :B
#                    kind :l   link to the previous site (tag l = n-1)
#                         :r   link to the next site     (tag l = n)
#                         :s   site index, prime level 0
#                         :sp  site index, prime level 1
#                    k    the k-th index of that kind in op's stored order (the key comes
#                         first, so on an aliased ψ = P·core, (:B,:l,1) is P's left link
#                         in the key and (:B,:l,2) core's left link in the dense block)
#   (op, kind)       short for (op, kind, 1)
# n is the operand's site number (its Site index tag n=). A role absent at this site
# (e.g. (:A,:l) on the first site, or a contracted index) is skipped, so one layout
# serves every site of an MPO × MPS contraction. show_roles prints a tensor's roles.
const Role = Union{Tuple{Symbol,Symbol},Tuple{Symbol,Symbol,Int}}
const LayoutEntry = Union{ITensors.Index,Role}

struct OutputLayout
    sparse::Vector{LayoutEntry}
    dense::Vector{LayoutEntry}
end
OutputLayout(; sparse, dense) =
    OutputLayout(collect(LayoutEntry, sparse), collect(LayoutEntry, dense))

"""True when every entry is an Index (no roles left to resolve)."""
isresolved(L::OutputLayout) = all(x -> x isa ITensors.Index, L.sparse) &&
                              all(x -> x isa ITensors.Index, L.dense)

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
    isresolved(L) || throw(OutputLayoutError(
        "validate_layout: layout still contains roles; resolve it first (resolve_layout)"))
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
