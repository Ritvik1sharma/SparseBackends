# aliased/storage.jl
#
# AliasedBlockSparse: a memory-efficient block-sparse format that eliminates
# data redundancy when multiple sparse blocks are scalar multiples of a shared
# dense template block.
#
# Storage layout
# ─────────────────────────────────────────────────────────────────────────────
#   templates[1 : n_templates * blksize]  — flat array of unique dense blocks.
#   Block i satisfies:  logical_block[i] = scalars[i] * templates[alias_ids[i]]
#
# Primary use case: result of a COO × Dense contraction where at most one
# nonzero per prefix key means every result block is  α * B[:,rv].
#
# Memory: O(n_templates × blksize  +  n_blocks)
#         vs O(n_blocks × blksize) for NewBlockSparseSorted.

using Base: OneTo

# ─────────────────────────────────────────────────────────────────────────────
# Struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    AliasedBlockSparse{T,N,N2,P}

A block-sparse tensor where each logical block is a scalar multiple of a stored
template block, eliminating redundancy when many blocks share the same
underlying dense data.

Type parameters
- `T`:  element type
- `N`:  total number of dimensions
- `N2`: number of dense (block-interior) dimensions
- `P`:  number of sparse (prefix) dimensions  (P = N − N2)

Storage layout
- `templates[1 : n_templates * blksize]` — flat array of unique dense blocks.
  Template `t` occupies `templates[(t-1)*blksize+1 : t*blksize]`.
- Block at index `i` (key `keys[i]`) satisfies
    logical_block[i] = scalars[i] * templates[alias_ids[i]]

`keys` are kept sorted in column-major order (same convention as
`NewBlockSparseSorted`), enabling O(log N) binary-search element access.
"""
mutable struct AliasedBlockSparse{T,N,N2,P,K<:Integer} <: SparseTensor{T,N}
    dims        :: NTuple{N,Int}
    blksize     :: Int                       # prod(dims[P+1 : N])
    templates   :: Vector{T}                 # length = n_templates * blksize
    n_templates :: Int
    keys        :: Vector{NTuple{P,K}}       # sorted sparse prefix keys (key type K)
    alias_ids   :: Vector{Int}               # alias_ids[i] → template index (1-based)
    scalars     :: Vector{T}                 # scalar multiplier per block
end

# Backwards-compatible outer constructor (K=Int default)
AliasedBlockSparse{T,N,N2,P}(dims, blksize, templates, n_templates,
                              keys::Vector{NTuple{P,Int}}, alias_ids, scalars) where {T,N,N2,P} =
    AliasedBlockSparse{T,N,N2,P,Int}(dims, blksize, templates, n_templates, keys, alias_ids, scalars)

# ─────────────────────────────────────────────────────────────────────────────
# AbstractArray interface
# ─────────────────────────────────────────────────────────────────────────────

_dims(x::AliasedBlockSparse) = x.dims

Base.eltype(::Type{AliasedBlockSparse{T,N,N2,P,K}}) where {T,N,N2,P,K} = T
Base.eltype(A::AliasedBlockSparse) = eltype(typeof(A))
Base.size(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} = A.dims
Base.axes(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} =
    ntuple(i -> OneTo(A.dims[i]), Val(N))
Base.IndexStyle(::Type{<:AliasedBlockSparse}) = IndexCartesian()

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# View of the t-th template block
@inline function _aliased_template_view(A::AliasedBlockSparse{T,N,N2,P,K}, t::Int) where {T,N,N2,P,K}
    off = (t - 1) * A.blksize
    return @view(A.templates[off+1 : off + A.blksize])
end

# Column-major linear index into the dense tail (axes P+1..N) of a full index tuple
@inline function _dense_tail_lin(I::NTuple{N,Int}, dims::NTuple{N,Int}, ::Val{P}, ::Val{N2}) where {N,P,N2}
    lin    = 1
    stride = 1
    @inbounds for j in 1:N2
        lin    += (I[P+j] - 1) * stride
        stride *= dims[P+j]
    end
    return lin
end

# Binary search for prefix key k in sorted A.keys using column-major linearisation.
# Returns (position, found::Bool).
@inline function _find_aliased_key(A::AliasedBlockSparse{T,N,N2,P,K}, k::NTuple{P,K}) where {T,N,N2,P,K}
    pdims  = ntuple(i -> A.dims[i], Val(P))
    target = _prefix_lin(k, pdims)
    lo, hi = 1, length(A.keys) + 1
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if _prefix_lin(A.keys[mid], pdims) < target
            lo = mid + 1
        else
            hi = mid
        end
    end
    found = lo <= length(A.keys) && A.keys[lo] == k
    return lo, found
end

# ─────────────────────────────────────────────────────────────────────────────
# Constructor
# ─────────────────────────────────────────────────────────────────────────────

"""
    AliasedBlockSparse{T,N,N2}(dims)

Allocate an empty AliasedBlockSparse tensor with the given shape and no blocks.
"""
function AliasedBlockSparse{T,N,N2}(dims::NTuple{N,Int}) where {T,N,N2}
    P = N - N2
    # N2=0 is allowed: every axis lives in the sparse prefix and each block
    # is a single scalar (blksize=1). Needed for aliased outputs where the
    # contraction reduces away every dense-tail axis.
    @assert 0 <= N2 <= N
    @assert all(dims .>= 1)
    blksize = N2 == 0 ? 1 : prod(ntuple(i -> dims[P+i], Val(N2)))
    return AliasedBlockSparse{T,N,N2,P,Int}(
        dims, blksize,
        Vector{T}(), 0,
        Vector{NTuple{P,Int}}(), Int[], Vector{T}(),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Element access
# ─────────────────────────────────────────────────────────────────────────────

function Base.getindex(A::AliasedBlockSparse{T,N,N2,P,K}, I::Vararg{Int,N}) where {T,N,N2,P,K}
    It = ntuple(i -> I[i], Val(N))
    _check_indexes!(A.dims, It)
    k = ntuple(i -> K(It[i]), Val(P))   # convert to key type K
    pos, found = _find_aliased_key(A, k)
    found || return zero(T)
    tid = A.alias_ids[pos]
    bi  = _dense_tail_lin(It, A.dims, Val(P), Val(N2))
    return A.scalars[pos] * A.templates[(tid-1)*A.blksize + bi]
end

# ─────────────────────────────────────────────────────────────────────────────
# Metadata / statistics
# ─────────────────────────────────────────────────────────────────────────────

n_blocks(A::AliasedBlockSparse)    = length(A.keys)
n_templates(A::AliasedBlockSparse) = A.n_templates

"""
    compression_ratio(A::AliasedBlockSparse) -> Float64

Ratio `n_blocks / n_templates`.  A value > 1 means multiple blocks share
templates; e.g. a ratio of 10 means on average 10 blocks reference each
template, giving ~10× savings in dense data storage.
"""
compression_ratio(A::AliasedBlockSparse) =
    A.n_templates == 0 ? 1.0 : length(A.keys) / A.n_templates

# ─────────────────────────────────────────────────────────────────────────────
# permutedims
# ─────────────────────────────────────────────────────────────────────────────
# Key efficiency advantage: permuting the dense tail only touches n_templates
# blocks (not n_blocks), which is much cheaper when templates are shared.

function Base.permutedims(A::AliasedBlockSparse{T,N,N2,P,K},
                          perm::AbstractVector{Int}) where {T,N,N2,P,K}
    _check_no_cross_perm(perm, N, P, N2)
    dimsB = ntuple(j -> A.dims[perm[j]], Val(N))

    perm_prefix = perm[1:P]
    perm_block  = ntuple(i -> perm[P+i] - P, Val(N2))  # local 1:N2 indices

    blockDimsA = ntuple(i -> A.dims[P+i], Val(N2))
    blockDimsB = ntuple(i -> dimsB[P+i], Val(N2))
    blksizeB   = prod(blockDimsB)
    @assert blksizeB == A.blksize  # permutation within tail preserves block size

    # ── Step 1: permute the dense tail of each TEMPLATE (only n_templates ops) ──
    is_identity_block = all(perm_block[i] == i for i in 1:N2)
    new_templates = if is_identity_block
        copy(A.templates)
    else
        # Build mapping  new_lin -> old_lin  for the dense tail (computed once)
        invp = invperm(collect(perm_block))
        strides_old = let s = 1, sv = Vector{Int}(undef, N2)
            for d in 1:N2; sv[d] = s; s *= blockDimsA[d]; end
            sv
        end
        map_old = Vector{Int}(undef, A.blksize)
        idx_new = ones(Int, N2)
        for new_lin in 1:A.blksize
            old_lin = 1
            for d in 1:N2
                old_lin += (idx_new[invp[d]] - 1) * strides_old[d]
            end
            map_old[new_lin] = old_lin
            for d in 1:N2
                idx_new[d] += 1
                idx_new[d] <= blockDimsB[d] && break
                idx_new[d] = 1
            end
        end

        # Permute each unique template (only n_templates, not n_blocks)
        tmpl_new = Vector{T}(undef, A.n_templates * blksizeB)
        for t in 1:A.n_templates
            src_off = (t - 1) * A.blksize
            dst_off = (t - 1) * blksizeB
            @inbounds @simd for new_lin in 1:blksizeB
                tmpl_new[dst_off + new_lin] = A.templates[src_off + map_old[new_lin]]
            end
        end
        tmpl_new
    end

    # ── Step 2: remap prefix keys and sort ──
    nblocks   = length(A.keys)
    newkeys   = Vector{NTuple{P,K}}(undef, nblocks)
    @inbounds for i in 1:nblocks
        newkeys[i] = ntuple(j -> A.keys[i][perm_prefix[j]], Val(P))
    end
    prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
    p = sortperm(newkeys; by = k -> _prefix_lin(k, prefix_dimsB))

    return AliasedBlockSparse{T,N,N2,P,K}(
        dimsB, blksizeB, new_templates, A.n_templates,
        newkeys[p], A.alias_ids[p], A.scalars[p],
    )
end

Base.permutedims(A::AliasedBlockSparse{T,N,N2,P,K}, perm::NTuple{N,Int}) where {T,N,N2,P,K} =
    permutedims(A, collect(perm))
