# edited_packages/utils.jl
# ─────────────────────────────────────────────────────────────────────────────
# Shared tensor-layout utilities for the edited packages.
#
# `reorder_to_roles(T, roles)` — permute an ITensor's indices into a target order
# specified SYMBOLICALLY by role, so callers need not know the concrete Index ids
# (which vary run-to-run). Layout-only: a pure `ITensors.permute`, contraction math
# unchanged.
#
# Role vocabulary (for DMRG-local tensors carrying Links + physical Sites):
#   :l  :m  :r        Link indices in ASCENDING "l=" number  (left, middle, right)
#   :s  :s2 :s3 …     physical Site indices, plev 0 (ket), ASCENDING "n=" number
#   :sp :s2p :s3p …   physical Site indices, plev 1 (bra), ASCENDING "n=" number
#
# The target `roles` list may name roles the tensor doesn't have (e.g. `:r` on an
# edge tensor with a single link) — missing roles are skipped, so ONE target can
# serve bulk and edge bonds. After skipping, the surviving roles must cover every
# index of T exactly once (else it's an error — catches a mis-specified target).
#
# Examples (φ = psi[b]*psi[b+1], inds {Link l=1, Site n=2, Site n=3, Link l=3}):
#   reorder_to_roles(φ, [:l, :s, :s2, :r])  → [Link l=1, Site n=2, Site n=3, Link l=3]
#   reorder_to_roles(φ, [:l, :r, :s2, :s])  → [Link l=1, Link l=3, Site n=3, Site n=2]
# ─────────────────────────────────────────────────────────────────────────────

_layout_num(I, key::String) = begin
    m = match(Regex(key * raw"=(\d+)"), string(ITensors.tags(I)))
    m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end
_layout_islink(I) = occursin("Link", string(ITensors.tags(I)))
_layout_issite(I) = occursin("Site", string(ITensors.tags(I)))

# Map each index of `inds` to its role symbol (see vocabulary above).
function _layout_role_map(inds)
    links = sort([I for I in inds if _layout_islink(I)]; by = I -> _layout_num(I, "l"))
    ket   = sort([I for I in inds if _layout_issite(I) && ITensors.plev(I) == 0]; by = I -> _layout_num(I, "n"))
    bra   = sort([I for I in inds if _layout_issite(I) && ITensors.plev(I) == 1]; by = I -> _layout_num(I, "n"))
    rm = Dict{Symbol,ITensors.Index}()
    if length(links) == 2
        rm[:l] = links[1]; rm[:r] = links[2]
    elseif length(links) == 3
        rm[:l] = links[1]; rm[:m] = links[2]; rm[:r] = links[3]
    else
        for (i, I) in enumerate(links); rm[i == 1 ? :l : Symbol("link", i)] = I; end
    end
    for (i, I) in enumerate(ket); rm[i == 1 ? :s  : Symbol("s", i)]       = I; end
    for (i, I) in enumerate(bra); rm[i == 1 ? :sp : Symbol("s", i, "p")]  = I; end
    return rm
end

function reorder_to_roles(T::ITensors.ITensor, roles::AbstractVector{Symbol})::ITensors.ITensor
    it = collect(ITensors.inds(T))
    rm = _layout_role_map(it)
    chosen = ITensors.Index[rm[r] for r in roles if haskey(rm, r)]   # skip absent roles
    if length(chosen) != length(it) || length(unique(ITensors.id.(chosen))) != length(chosen)
        error("reorder_to_roles: roles $roles resolve to $(length(chosen)) of $(length(it)) inds " *
              "(available roles: $(sort(collect(keys(rm)); by=string))) — target must cover T exactly once")
    end
    it == chosen && return T                      # already in target order → no-op
    return ITensors.permute(T, chosen...)
end
