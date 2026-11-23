# aliased/merge.jl
#
# Channel fusion for factorize: on the bond being factorized (the one key index of the site outside
# Linds), values whose columns are proportional are fused. The column of value c is every block with
# key[b] = c, read as (other key coordinates, template id, scalar); values a, c are one class when their
# (coordinates, template id) lists are equal and scalars_c = λ_c * scalars_a for one λ_c (to working
# precision, see `_merge_classes`). The site keeps
# one value per class, renumbered 1..n on a new Index nb. The channel map (rep, λ, nb, b) goes into the
# R that factorize returns (rep[c]: new value of old value c, column c = λ[c] * column of rep[c]), so
# `_absorb_bond(neighbour, R)` carries the dropped columns. The COO method (tensor_wrappers.jl) uses the
# same `_merge_classes`.

import ITensors

# Classes of the values of key axis `ax` (dims D). Returns rep[c] (new value of c, 0 for a
# value with no entries), λ[c] (column c = λ[c] * column of its representative), isrep[c]
# and the class count n. Proportionality is tested by cross products to working precision:
# |x - y| <= rtol * max(|x|, |y|) with x = vals_c[k] * u1, y = vals_a[k] * v1 and
# rtol = _MERGE_RTOL_ULPS * eps. Columns proportional in exact arithmetic come out of the
# kernels with scalars multiplied along different paths, so their cross products differ by
# a few ulp and an exact `==` keeps them as separate channels. 4096 eps (~9e-13): in the noisy d5
# cultivation chain (Rz 0.1 on all qubits) the roundoff mismatches reach 1e-13 and the smallest genuine
# difference is ~1e-6, with no mismatch in between. Genuine differences scale with the amplitudes of
# the state (e.g. powers of a small noise angle), so a much smaller one can fall below this tolerance.
const _MERGE_RTOL_ULPS = 4096
function _merge_classes(keys::AbstractVector{<:Tuple}, tids::AbstractVector,
                        vals::AbstractVector{T}, ax::Int, D::Int) where {T}
    rtol = _MERGE_RTOL_ULPS * eps(real(float(T)))
    near(x, y) = abs(x - y) <= rtol * max(abs(x), abs(y))
    cols = [Int[] for _ in 1:D]
    for i in eachindex(keys)
        push!(cols[Int(keys[i][ax])], i)
    end
    sig(i) = (Base.setindex(keys[i], zero(keys[i][ax]), ax), Int(tids[i]))
    rep   = zeros(Int, D)
    λ     = zeros(T, D)
    isrep = falses(D)
    reps  = Dict{Vector{Any},Vector{Int}}()     # column signature -> representative values
    n = 0
    for c in 1:D
        isempty(cols[c]) && continue
        sort!(cols[c]; by = sig)
        s  = Any[sig(i) for i in cols[c]]
        cc = cols[c]
        found = false
        for a in get(reps, s, Int[])
            ca = cols[a]
            u1, v1 = vals[ca[1]], vals[cc[1]]
            (iszero(u1) || iszero(v1)) && continue
            if all(k -> near(vals[cc[k]] * u1, vals[ca[k]] * v1), eachindex(ca))
                rep[c] = rep[a]; λ[c] = v1 / u1; found = true
                break
            end
        end
        if !found
            n += 1
            rep[c] = n; λ[c] = one(T); isrep[c] = true
            push!(get!(reps, s, Int[]), c)
        end
    end
    return rep, λ, isrep, n
end

# R = sum_c λ[c] |rep[c]><c| over (nb, b), as a COO ITensor.
function _merge_map(rep::Vector{Int}, λ::Vector{T}, n::Int,
                    nb::ITensors.Index, b::ITensors.Index) where {T}
    ks = NTuple{2,Int}[]
    vs = T[]
    for c in eachindex(rep)
        rep[c] == 0 && continue
        push!(ks, (rep[c], c)); push!(vs, λ[c])
    end
    coo = COOTensor{T,2}((n, length(rep)), ks, vs, true)
    sort!(coo)
    return ITensors._itensor_from_external_storage(WrappedCOOTensor{T,2}(coo, (nb, b)))
end

# Nothing merged: every value with entries is its own class and no value is empty.
_nothing_merged(rep, n) = n == length(rep) && all(>(0), rep)

# Fuse the channel of the bond outside Lis: (A', Aw', chan), chan = (rep, λ, nb, b) or nothing when
# nothing fuses (or the bond has no key index). More than one key index outside Lis errors.
function _fuse_channels(X::AliasedBlockSparse, Aw::WrappedAliasedBlockSparse, A::ITensors.ITensor, Lis)
    axs = [p for p in 1:_abs_head_len(Aw) if !(Aw.inds[p] in Lis)]
    isempty(axs) && return A, Aw, nothing
    length(axs) == 1 || error("channel fusion: expected one key index outside Linds, found $(length(axs))")
    return _merge_key_axis(X, Aw, A, axs[1])
end

function _merge_key_axis(X::AliasedBlockSparse{T,N,N2,P,K}, Mw::WrappedAliasedBlockSparse,
                         M::ITensors.ITensor, ax::Int) where {T,N,N2,P,K}
    b = Mw.inds[ax]
    rep, λ, isrep, n = _merge_classes(X.keys, X.alias_ids, X.scalars, ax, X.dims[ax])
    _nothing_merged(rep, n) && return M, Mw, nothing
    n > 0 || error("channel fusion: site has no entries on bond $b")

    keep = [isrep[Int(k[ax])] for k in X.keys]
    keys = [Base.setindex(k, K(rep[Int(k[ax])]), ax) for k in X.keys[keep]]
    dims = Base.setindex(X.dims, n, ax)
    pd   = ntuple(i -> dims[i], Val(P))
    p    = sortperm(keys; by = k -> _prefix_lin(k, pd))
    ids  = X.alias_ids[keep][p]
    # keep only the templates still referenced, renumbered in order
    used = sort!(unique(ids))
    new_of = Dict(t => i for (i, t) in enumerate(used))
    bs = X.blksize
    templates = Vector{T}(undef, length(used) * bs)
    for (i, t) in enumerate(used)
        templates[((i - 1) * bs + 1):(i * bs)] .= @view X.templates[((Int(t) - 1) * bs + 1):(Int(t) * bs)]
    end
    AI = eltype(ids)
    ali = _same_kind(X, AliasedBlockSparse{T,N,N2,P,K}(dims, bs, templates, length(used), keys[p],
                                                      AI[new_of[t] for t in ids], X.scalars[keep][p]))
    nb = ITensors.Index(n; tags = ITensors.tags(b))
    Lw = WrappedAliasedBlockSparse{T,N,N2,P}(ali, Base.setindex(Mw.inds, nb, ax))
    return ITensors._itensor_from_external_storage(Lw), Lw, (; rep, λ, nb, b)
end
