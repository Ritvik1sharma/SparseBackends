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

# Default element type for the alias-id vector (template indices). Settable per
# construction via the `AI` type parameter; this is just the default for freshly
# built aliased tensors. UInt8 caps at 255 templates — many call sites widen to
# UInt16 (65535) by constructing with an explicit `AI`.
const DEFAULT_ALIAS_ID_TYPE = UInt8

# Convert a (1-based) template index to the alias-id type, with a clear error
# when it exceeds the type's capacity (instead of a cryptic InexactError).
@inline function _alias_id(::Type{AI}, tid::Integer) where {AI<:Integer}
    tid <= typemax(AI) || error(
        "AliasedBlockSparse: template index ($tid) exceeds the alias-id type " *
        "$AI capacity ($(typemax(AI))); construct with a wider alias-id type " *
        "(e.g. UInt16 covers 65535, UInt32 covers ~4.3e9).")
    return tid % AI
end

# Build an identity alias mapping 1:nb as a Vector{AI}, capacity-checked.
@inline function _alias_id_range(::Type{AI}, nb::Integer) where {AI<:Integer}
    nb <= typemax(AI) || error(
        "AliasedBlockSparse: number of blocks ($nb) exceeds the alias-id type " *
        "$AI capacity ($(typemax(AI))); construct the input with a wider " *
        "alias-id type (e.g. UInt16 covers 65535).")
    return collect(AI, 1:nb)
end

# ─────────────────────────────────────────────────────────────────────────────
# Struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    AliasedBlockSparse{T,N,N2,P,K,AI}

Abstract parent of the two aliased storage types. Both hold the same fields and differ
only in how a bond is factorized:
- `SharedAliasedBlockSparse`: ψ = P·core, one multiplicity basis shared by all of a
  bond's channels (aliased/shared/: factorize, _absorb_bond). The single-core features
  (read_core, slice_to_template, window map, core_split; aliased/shared/factor_core.jl) are
  defined for it only.
- `SectorSplitAliasedBlockSparse`: the same fields; one multiplicity basis per channel value
  of the bond (one SVD per channel value; aliased/sector_split/: factorize, _absorb_bond).
  Meant for a P with orthogonal channels; nothing
  checks that, the type is the user's choice (`Cbackend = :aliased_sector_split`).
`AliasedBlockSparse{...}(...)` constructs a `SharedAliasedBlockSparse`.

A block-sparse tensor where each logical block is a scalar multiple of a stored
template block, eliminating redundancy when many blocks share the same
underlying dense data.

Type parameters
- `T`:  element type
- `N`:  total number of dimensions
- `N2`: number of dense (block-interior) dimensions
- `P`:  number of sparse (prefix) dimensions  (P = N − N2)
- `K`:  element type of the sparse prefix keys (`<: Integer`)
- `AI`: element type of the alias-id vector (`<: Integer`), i.e. the template
        indices. Independent of `K`; defaults to `$(DEFAULT_ALIAS_ID_TYPE)` for
        freshly built tensors (see `DEFAULT_ALIAS_ID_TYPE`). Widen per
        construction when a tensor needs more than `typemax(AI)` templates.

Storage layout
- `templates[1 : n_templates * blksize]` — flat array of unique dense blocks.
  Template `t` occupies `templates[(t-1)*blksize+1 : t*blksize]`.
- Block at index `i` (key `keys[i]`) satisfies
    logical_block[i] = scalars[i] * templates[alias_ids[i]]

`keys` are kept sorted in column-major order (same convention as
`NewBlockSparseSorted`), enabling O(log N) binary-search element access.
"""
# Shared empty default for the factor-core hint below: constructing an aliased
# tensor without the hint costs ZERO allocation (all empties alias this one). Safe
# because the hint is only ever *reassigned* (population builds a fresh vector),
# never mutated in place.
const _EMPTY_SLICE_MAP = Int[]
const _EMPTY_WINDOW_MAP = Dict{NTuple{2,Int},Int}()

abstract type AliasedBlockSparse{T,N,N2,P,K<:Integer,AI<:Integer} <: SparseTensor{T,N} end

mutable struct SharedAliasedBlockSparse{T,N,N2,P,K<:Integer,AI<:Integer} <: AliasedBlockSparse{T,N,N2,P,K,AI}
    dims        :: NTuple{N,Int}
    blksize     :: Int                       # prod(dims[P+1 : N])
    templates   :: Vector{T}                 # length = n_templates * blksize
    n_templates :: Int
    keys        :: Vector{NTuple{P,K}}       # sorted sparse prefix keys (key type K)
    alias_ids   :: Vector{AI}                # alias_ids[i] → template index (1-based)
    scalars     :: Vector{T}                 # scalar multiplier per block
    # factor-core hint (empty by default): slice_to_template[s] = template id holding
    # core's physical-slice s, for ψ built via COO-MPO × dense-MPS (ψ = P·core, diagonal
    # P). Lets read_core/write_core! recover/mutate the underlying core MPS in place.
    # Only populated by the contract_aliased_coo_dense kernel; every other constructor
    # / copy / permute leaves it EMPTY (read_core then re-derives it from `keys`).
    slice_to_template :: Vector{Int}
    # factor-core 2-site hint (empty by default): window_slice_map[(rv_b, rv_{b+1})] =
    # template id of the merged 2-site core block. Emitted ONLY by the AliasedBS×AliasedBS
    # contract_shared! (emit_window_map=true) when BOTH inputs carry slice_to_template —
    # i.e. the ψ[b]·ψ[b+1] window build. Holds the (rv_b,rv_{b+1})→tid routing that CANNOT
    # be recovered post-hoc for off-diagonal (flip) P, because φ sums over the internal
    # FSM bond; the pairing is known only while that bond is being contracted. Lets
    # write_core_window! re-attach a merged core. Every other constructor/copy leaves it
    # EMPTY (never mutated in place; always reassigned a fresh dict when populated).
    window_slice_map :: Dict{NTuple{2,Int},Int}
    # 7-arg inner constructor keeps every existing call site working (defaults both hints
    # empty); struct is mutable so the COO×dense / window kernels set them after construction.
    function SharedAliasedBlockSparse{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates,
                                                     keys, alias_ids, scalars) where {T,N,N2,P,K,AI}
        return new{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates,
                                  keys, alias_ids, scalars, _EMPTY_SLICE_MAP, _EMPTY_WINDOW_MAP)
    end
end

"""
    SectorSplitAliasedBlockSparse{T,N,N2,P,K,AI}

The same fields as `SharedAliasedBlockSparse` (no extra field); a distinct type because a bond
is factorized differently: one SVD per channel value of the bond's key index, so each channel
keeps its own multiplicity basis (aliased/sector_split/factorize.jl). The hint fields are carried like any
field but no single-core feature reads them for this type.
"""
mutable struct SectorSplitAliasedBlockSparse{T,N,N2,P,K<:Integer,AI<:Integer} <: AliasedBlockSparse{T,N,N2,P,K,AI}
    dims        :: NTuple{N,Int}
    blksize     :: Int
    templates   :: Vector{T}
    n_templates :: Int
    keys        :: Vector{NTuple{P,K}}
    alias_ids   :: Vector{AI}
    scalars     :: Vector{T}
    slice_to_template :: Vector{Int}
    window_slice_map  :: Dict{NTuple{2,Int},Int}
    function SectorSplitAliasedBlockSparse{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates,
                                                          keys, alias_ids, scalars) where {T,N,N2,P,K,AI}
        return new{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates, keys, alias_ids, scalars,
                                  _EMPTY_SLICE_MAP, _EMPTY_WINDOW_MAP)
    end
end

# The parent's constructors build a SharedAliasedBlockSparse, so every existing
# `AliasedBlockSparse{...}(...)` call site behaves as before.
AliasedBlockSparse{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates,
                                  keys, alias_ids, scalars) where {T,N,N2,P,K,AI} =
    SharedAliasedBlockSparse{T,N,N2,P,K,AI}(dims, blksize, templates, n_templates,
                                            keys, alias_ids, scalars)

# A SectorSplit tensor with `a`'s fields (the same vectors, not copies; the hints too).
function _as_sector_split(a::SharedAliasedBlockSparse{T,N,N2,P,K,AI}) where {T,N,N2,P,K,AI}
    s = SectorSplitAliasedBlockSparse{T,N,N2,P,K,AI}(a.dims, a.blksize, a.templates, a.n_templates,
                                                     a.keys, a.alias_ids, a.scalars)
    s.slice_to_template = a.slice_to_template
    s.window_slice_map  = a.window_slice_map
    return s
end
_as_sector_split(a::SectorSplitAliasedBlockSparse) = a

# `ali` (freshly built from `src`) as the child type of `src`: unchanged for Shared. Every
# helper that rebuilds a tensor from another (copy, permute, rekey, fuse, frozen-schema factor,
# the gate kernel) goes through this.
@inline _same_kind(::SharedAliasedBlockSparse, ali::SharedAliasedBlockSparse) = ali
_same_kind(::SectorSplitAliasedBlockSparse, ali::SharedAliasedBlockSparse) = _as_sector_split(ali)

# Storage class of the aliased product of operands A and B (storages): SectorSplit when an
# operand is SectorSplit (e.g. a SectorSplit site times its neighbour's R), else Shared. The
# contraction entry allocates the output as this class and the kernels fill it.
@inline _product_kind(A, B) =
    (A isa SectorSplitAliasedBlockSparse || B isa SectorSplitAliasedBlockSparse) ?
        SectorSplitAliasedBlockSparse : SharedAliasedBlockSparse

# Outer constructor inferring the alias-id type AI from the passed vector. Keeps
# every existing `AliasedBlockSparse{T,N,N2,P,K}(...)` call site working and
# preserves the source alias-id width when propagating (permutedims, slicing,
# copies, etc.).
AliasedBlockSparse{T,N,N2,P,K}(dims, blksize, templates, n_templates,
                               keys, alias_ids, scalars) where {T,N,N2,P,K} =
    AliasedBlockSparse{T,N,N2,P,K,eltype(alias_ids)}(
        dims, blksize, templates, n_templates, keys, alias_ids, scalars)

# Backwards-compatible outer constructor (K=Int default; AI inferred above)
AliasedBlockSparse{T,N,N2,P}(dims, blksize, templates, n_templates,
                              keys::Vector{NTuple{P,Int}}, alias_ids, scalars) where {T,N,N2,P} =
    AliasedBlockSparse{T,N,N2,P,Int}(dims, blksize, templates, n_templates, keys, alias_ids, scalars)

# ─────────────────────────────────────────────────────────────────────────────
# AbstractArray interface
# ─────────────────────────────────────────────────────────────────────────────

_dims(x::AliasedBlockSparse) = x.dims

Base.eltype(::Type{<:AliasedBlockSparse{T}}) where {T} = T
Base.eltype(A::AliasedBlockSparse) = eltype(typeof(A))
Base.size(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} = A.dims
Base.axes(A::AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} =
    ntuple(i -> OneTo(A.dims[i]), Val(N))
Base.IndexStyle(::Type{<:AliasedBlockSparse}) = IndexCartesian()

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# View of the t-th template block
@inline function _aliased_template_view(A::AliasedBlockSparse{T,N,N2,P,K}, t::Integer) where {T,N,N2,P,K}
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

# SharedAliasedBlockSparse{T,N,N2}(dims), SectorSplitAliasedBlockSparse{T,N,N2}(dims): an empty
# tensor of that class with the given shape and no blocks. The contraction entry allocates its
# aliased output this way (the requested class) and the kernels fill it in place.
for K in (:SharedAliasedBlockSparse, :SectorSplitAliasedBlockSparse)
    @eval function $K{T,N,N2}(dims::NTuple{N,Int}) where {T,N,N2}
        P = N - N2
        # N2=0 is allowed: every axis lives in the sparse prefix and each block
        # is a single scalar (blksize=1). Needed for aliased outputs where the
        # contraction reduces away every dense-tail axis.
        @assert 0 <= N2 <= N
        @assert all(dims .>= 1)
        blksize = N2 == 0 ? 1 : prod(ntuple(i -> dims[P+i], Val(N2)))
        return $K{T,N,N2,P,Int,DEFAULT_ALIAS_ID_TYPE}(
            dims, blksize,
            Vector{T}(), 0,
            Vector{NTuple{P,Int}}(), DEFAULT_ALIAS_ID_TYPE[], Vector{T}(),
        )
    end
end
"""
    AliasedBlockSparse{T,N,N2}(dims)

Allocate an empty AliasedBlockSparse tensor (a SharedAliasedBlockSparse) with the given
shape and no blocks.
"""
AliasedBlockSparse{T,N,N2}(dims::NTuple{N,Int}) where {T,N,N2} = SharedAliasedBlockSparse{T,N,N2}(dims)

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

# permutedims(A, perm) returns a tensor with its own vectors (templates copied even when only
# key axes move).
function Base.permutedims(A::AliasedBlockSparse{T,N,N2,P,K},
                          perm::AbstractVector{Int}) where {T,N,N2,P,K}
    B = _permutedims_sharing(A, perm)
    B.templates === A.templates && (B.templates = copy(A.templates))
    return B
end

# The same permutation, but when the dense block is not permuted the result's `templates` IS
# A's vector (keys, alias_ids and scalars are always new). For a read-only temporary (a kernel's
# permuted operand): the result must not be mutated in place.
function _permutedims_sharing(A::AliasedBlockSparse{T,N,N2,P,K},
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
        A.templates
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

    return _same_kind(A, AliasedBlockSparse{T,N,N2,P,K}(
        dimsB, blksizeB, new_templates, A.n_templates,
        newkeys[p], A.alias_ids[p], A.scalars[p],
    ))
end

Base.permutedims(A::AliasedBlockSparse{T,N,N2,P,K}, perm::NTuple{N,Int}) where {T,N,N2,P,K} =
    permutedims(A, collect(perm))

# footprint(storage) -> (blocks, templates, dense_elems, bs_elems, stored_elems): element
# counts of one tensor's storage (dense_elems: every entry; bs_elems: one dense block per
# key; stored_elems: what this storage actually holds). All are counts, so a tensor
# network's footprint is the sum over its tensors.
footprint(A::AliasedBlockSparse) =
    (blocks = length(A.keys), templates = A.n_templates, dense_elems = prod(A.dims),
     bs_elems = length(A.keys) * A.blksize,
     stored_elems = A.n_templates * A.blksize + length(A.keys))   # templates + one scalar per block
