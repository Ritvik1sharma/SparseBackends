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
