# aliased/conversions.jl
#
# Conversions to/from AliasedBlockSparse.
# Mirrors the pattern of blocksparse/conversions.jl and coo/conversions.jl.

# ─────────────────────────────────────────────────────────────────────────────
# to_blocksparse  —  expand AliasedBlockSparse into NewBlockSparseSorted
# ─────────────────────────────────────────────────────────────────────────────

"""
    to_blocksparse(A::AliasedBlockSparse) -> NewBlockSparseSorted

Expand the aliased format into a standard `NewBlockSparseSorted` by computing
`scalar * template` for each block.  Use this when downstream operations
(e.g. further contractions written for `NewBlockSparseSorted`) require the
fully materialised format.
"""
function to_blocksparse(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K}
    nblocks = length(A.keys)
    data    = Vector{T}(undef, nblocks * A.blksize)
    @inbounds for i in 1:nblocks
        α       = A.scalars[i]
        src_off = (A.alias_ids[i] - 1) * A.blksize
        dst_off = (i               - 1) * A.blksize
        @simd for j in 1:A.blksize
            data[dst_off + j] = α * A.templates[src_off + j]
        end
    end
    return NewBlockSparseSorted{T,N,N2,P,K}(
        A.dims, A.blksize, copy(A.keys), collect(1:nblocks), data,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# to_dense  —  materialise to a plain dense Array
# ─────────────────────────────────────────────────────────────────────────────

"""
    to_dense(A::AliasedBlockSparse{T}; init=zero(T)) -> Array{T,N}

Materialise `A` as a dense N-dimensional array.  Missing blocks are filled
with `init` (default: zero).  Equivalent to `to_dense(to_blocksparse(A))` but
avoids constructing the intermediate `NewBlockSparseSorted`.
"""
function to_dense(A::AliasedBlockSparse{T,N,N2,P,K}; init::T = zero(T)) where {T,N,N2,P,K}
    out         = fill(init, A.dims)
    suffix_dims = ntuple(i -> A.dims[P+i], Val(N2))
    suffix_CI   = CartesianIndices(suffix_dims)
    suffix_LI   = LinearIndices(suffix_dims)
    @inbounds for i in eachindex(A.keys)
        prefix   = A.keys[i]
        α        = A.scalars[i]
        tmpl_off = (A.alias_ids[i] - 1) * A.blksize
        for sCI in suffix_CI
            suffix = Tuple(sCI)::NTuple{N2,Int}
            full   = ntuple(k -> k <= P ? prefix[k] : suffix[k-P], Val(N))
            lin    = suffix_LI[sCI]
            out[full...] = α * A.templates[tmpl_off + lin]
        end
    end
    return out
end

Base.Array(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} = to_dense(A)

# ─────────────────────────────────────────────────────────────────────────────
# rekey — convert prefix key integer type
# ─────────────────────────────────────────────────────────────────────────────

"""
    rekey(A::AliasedBlockSparse{T,N,N2,P,K}, ::Type{K2}; check=true) → AliasedBlockSparse{T,N,N2,P,K2}

Convert the prefix key integer type.  With `check=true` (default), errors if
any prefix dimension size or coordinate value does not fit in K2.
"""
function rekey(A::AliasedBlockSparse{T,N,N2,P,K}, ::Type{K2}; check::Bool=true) where {T,N,N2,P,K,K2<:Integer}
    if check
        for i in 1:P
            A.dims[i] <= typemax(K2) ||
                error("Dimension $(A.dims[i]) at prefix axis $i cannot be indexed by $K2 " *
                      "(max=$(typemax(K2)))")
        end
        for key in A.keys, v in key
            (typemin(K2) <= v <= typemax(K2)) ||
                error("Key coordinate $v is out of range for $K2 " *
                      "($(typemin(K2))..$(typemax(K2)))")
        end
    end
    new_keys = [ntuple(j -> K2(key[j]), Val(P)) for key in A.keys]
    return AliasedBlockSparse{T,N,N2,P,K2}(
        A.dims, A.blksize, copy(A.templates), A.n_templates,
        new_keys, copy(A.alias_ids), copy(A.scalars),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# fuse_sparse_axes / fuse_dense_axes! — native aliasing-preserving fusion.
# Mirrors blocksparse/conversions.jl::fuse_sparse_axes / fuse_dense_axes!.
# Prefix fuse is a pure key rewrite (templates untouched).
# Dense-tail fuse permutes templates so the two tail axes become adjacent
# (cost = n_templates, not n_blocks), then collapses their dims.
# ─────────────────────────────────────────────────────────────────────────────

function fuse_sparse_axes(
    A::AliasedBlockSparse{T,N,N2,P,K},
    ax1::Int, ax2::Int,
) where {T,N,N2,P,K}
    (1 <= ax1 <= P) || throw(ArgumentError("ax1 must be in sparse head 1:$P"))
    (1 <= ax2 <= P) || throw(ArgumentError("ax2 must be in sparse head 1:$P"))
    ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
    a1, a2 = min(ax1, ax2), max(ax1, ax2)
    dims = A.dims
    d1, d2 = dims[a1], dims[a2]
    newdims = ntuple(i -> begin
        if i < a2
            i == a1 ? d1 * d2 : dims[i]
        else
            dims[i + 1]
        end
    end, Val(N - 1))
    nblocks = length(A.keys)
    newkeys = Vector{NTuple{P-1,K}}(undef, nblocks)
    @inbounds for i in 1:nblocks
        k = A.keys[i]
        fused = (k[a2] - 1) * d1 + k[a1]
        newkeys[i] = ntuple(j -> begin
            if j < a2
                j == a1 ? K(fused) : k[j]
            else
                k[j + 1]
            end
        end, Val(P - 1))
    end
    # Templates, alias_ids, scalars, blksize all untouched.
    return AliasedBlockSparse{T,N-1,N2,P-1,K}(
        newdims, A.blksize, copy(A.templates), A.n_templates,
        newkeys, copy(A.alias_ids), copy(A.scalars),
    )
end

function fuse_dense_axes!(
    A::AliasedBlockSparse{T,N,N2,P,K},
    ax1::Int, ax2::Int,
) where {T,N,N2,P,K}
    (P + 1 <= ax1 <= N) || throw(ArgumentError("ax1 must be in dense tail $(P+1):$N"))
    (P + 1 <= ax2 <= N) || throw(ArgumentError("ax2 must be in dense tail $(P+1):$N"))
    ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
    a1, a2 = min(ax1, ax2), max(ax1, ax2)
    p1, p2 = a1 - P, a2 - P
    # If the two tail axes are not adjacent, permute templates so they are.
    # Cost = n_templates template permutations (not n_blocks), via the
    # template-aware permutedims defined in aliased/storage.jl.
    if p2 != p1 + 1
        perm = collect(1:N)
        # Move global axis a2 to position a1+1 (within tail; no cross of P boundary).
        target_pos = a1 + 1
        ax_id = perm[a2]
        deleteat!(perm, a2)
        insert!(perm, target_pos, ax_id)
        A = permutedims(A, perm)
        a2 = a1 + 1
    end
    dims = A.dims
    newdims = ntuple(i -> begin
        if i < a2
            i == a1 ? dims[a1] * dims[a2] : dims[i]
        else
            dims[i + 1]
        end
    end, Val(N - 1))
    # blksize and templates layout collapse: the two adjacent tail dims merge,
    # so the flat template buffer is unchanged in memory (column-major), only
    # the logical shape changes.
    return AliasedBlockSparse{T,N-1,N2-1,P,K}(
        newdims, A.blksize, copy(A.templates), A.n_templates,
        copy(A.keys), copy(A.alias_ids), copy(A.scalars),
    )
end

function fuse_two_axes!(
    A::AliasedBlockSparse{T,N,N2,P,K},
    ax1::Int, ax2::Int,
) where {T,N,N2,P,K}
    (1 <= ax1 <= N && 1 <= ax2 <= N) || throw(ArgumentError("axes must be in 1:$N"))
    ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
    in_sparse1 = ax1 <= P
    in_sparse2 = ax2 <= P
    if in_sparse1 && in_sparse2
        return fuse_sparse_axes(A, ax1, ax2)
    elseif (!in_sparse1) && (!in_sparse2)
        return fuse_dense_axes!(A, ax1, ax2)
    else
        # Mixed prefix/tail: not handled natively. Caller (fuse_axes! on
        # WrappedAliasedBlockSparse) demotes to BlockSparse for this case.
        throw(ArgumentError("Cannot fuse across sparse-head (1:$P) and dense-tail ($(P+1):$N) axes; demote to BlockSparse first"))
    end
end
