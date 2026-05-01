# tensor_wrappers_aliased.jl
#
# WrappedAliasedBlockSparse: ITensor wrapper for the AliasedBlockSparse format.
# Mirrors WrappedBlockSparse (tensor_wrappers.jl / tensor_contraction.jl) but
# routes contractions through the aliased kernels
# (contract_aliased! / contract_shared!).
#
# ─ Contents ──────────────────────────────────────────────────────────────────
#   1.  WrappedAliasedBlockSparse struct + basic traits
#   2.  Index operations: replaceprime, prime, noprime, setprime, dag, norm
#   3.  fuse_axes! (delegates to BlockSparse via to_blocksparse)
#   4.  head_tail_regions / _link_region  (same logic as WrappedBlockSparse)
#   5.  Canonicalization helpers           (reuse _canonicalize_bond_blocksparse!)
#   6.  C-backend inference + dense_inds overloads
#   7.  wrap_itensor backend=:aliased path + WrappedAliasedBlockSparse constructors
#   8.  contract! overloads for AliasedBlockSparse output (COO × Dense → Aliased)
#   9.  wrapped_contract aliased output path
#  10.  contract_aliased_itensor — top-level function (ITensor × ITensor → ITensor)
#  11.  contract_and_fuse_links overloads that produce aliased output
#  12.  ITensors extension hooks (dag, scale!, etc.)

import ITensors
using LinearAlgebra: norm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Struct + basic traits
# ─────────────────────────────────────────────────────────────────────────────

"""
    WrappedAliasedBlockSparse{T,N,N2,P}

An ITensor-aware wrapper for `AliasedBlockSparse{T,N,N2,P}`.
Layout convention mirrors `WrappedBlockSparse`: the first P indices are the
sparse prefix and the last N2 indices are the dense tail.
"""
mutable struct WrappedAliasedBlockSparse{T,N,N2,P} <: WrappedTensorTypes{T,N}
    aliased :: AliasedBlockSparse{T,N,N2,P}
    inds    :: NTuple{N,ITensors.Index}
end

_dims(w::WrappedAliasedBlockSparse)    = w.aliased.dims
_backend(::WrappedAliasedBlockSparse)  = :aliased

@inline rep(A::WrappedAliasedBlockSparse) = A.aliased
@inline _abs_head_len(::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = P

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Index operations
# ─────────────────────────────────────────────────────────────────────────────

function replaceprime(w::WrappedAliasedBlockSparse{T,N,N2,P}, ps::Pair{Int,Int}...) where {T,N,N2,P}
    new_inds = _replaceprime_inds(w.inds, ps...)
    return WrappedAliasedBlockSparse{T,N,N2,P}(w.aliased, new_inds)
end

function prime(w::WrappedAliasedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
    new_inds = _prime_inds(w.inds, args...)
    return WrappedAliasedBlockSparse{T,N,N2,P}(w.aliased, new_inds)
end

function noprime(w::WrappedAliasedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
    new_inds = _noprime_inds(w.inds, args...)
    return WrappedAliasedBlockSparse{T,N,N2,P}(w.aliased, new_inds)
end

function setprime(w::WrappedAliasedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
    new_inds = _setprime_inds(w.inds, args...)
    return WrappedAliasedBlockSparse{T,N,N2,P}(w.aliased, new_inds)
end

function ITensors.dag(A::AliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    new_templates = conj.(A.templates)
    return AliasedBlockSparse{T,N,N2,P}(
        A.dims, A.blksize, new_templates, A.n_templates,
        copy(A.keys), copy(A.alias_ids), copy(A.scalars),
    )
end

function ITensors.dag(W::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    new_aliased = ITensors.dag(W.aliased)
    new_inds    = ntuple(i -> ITensors.dag(W.inds[i]), Val(N))
    return ITensors._itensor_from_external_storage(
        WrappedAliasedBlockSparse{T,N,N2,P}(new_aliased, new_inds))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  fuse_axes! — delegate to BlockSparse
#
# AliasedBlockSparse has no native fuse operation yet. When fusion is needed
# (bond canonicalization), we materialise to NewBlockSparseSorted first.
# ─────────────────────────────────────────────────────────────────────────────

function fuse_axes!(W::WrappedAliasedBlockSparse{T,N,N2,P}, ax1::Int, ax2::Int) where {T,N,N2,P}
    ax1 == ax2 && return W
    # Materialise → BlockSparse → fuse → return WrappedBlockSparse
    bs  = to_blocksparse(W.aliased)
    Wbs = WrappedBlockSparse(bs, W.inds)
    return fuse_axes!(Wbs, ax1, ax2)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4.  head_tail_regions / _link_region
# ─────────────────────────────────────────────────────────────────────────────

function head_tail_regions(W::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    head = Int[]; tail = Int[]
    @inbounds for ax in 1:N
        if _is_link(W.inds[ax])
            ax <= P ? push!(head, ax) : push!(tail, ax)
        end
    end
    return head, tail
end

@inline function _link_region(W::WrappedAliasedBlockSparse{T,N,N2,P}, ax::Int) where {T,N,N2,P}
    return ax <= P ? :head : :tail
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Bond canonicalization — reuse the BlockSparse helper
#
# _canonicalize_bond_blocksparse! already dispatches on WrappedBlockSparse.
# For WrappedAliasedBlockSparse we delegate by converting, then wrapping back
# (or we can just call the same logic directly — the helper only calls
# fuse_axes! which we've already overloaded above).
# ─────────────────────────────────────────────────────────────────────────────

# No extra overload needed: _canonicalize_bond_blocksparse! calls fuse_axes!
# which is now defined above and returns a WrappedBlockSparse.  The caller in
# contract_and_fuse_links_aliased! below handles the return type.

# ─────────────────────────────────────────────────────────────────────────────
# 6.  C-backend inference + dense_inds
# ─────────────────────────────────────────────────────────────────────────────

# Dense indices: last N2 indices are the block-interior (dense tail).
dense_inds(w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} =
    Set(ntuple(i -> w.inds[N - N2 + i], Val(N2)))

# When either input is already WrappedAliasedBlockSparse the output is aliased.
@inline infer_C_backend(::WrappedAliasedBlockSparse, ::WrappedAliasedBlockSparse) = :aliased
@inline infer_C_backend(::WrappedAliasedBlockSparse, ::WrappedTensor)             = :aliased
@inline infer_C_backend(::WrappedTensor,             ::WrappedAliasedBlockSparse) = :aliased
@inline infer_C_backend(::WrappedAliasedBlockSparse, ::WrappedBlockSparse)        = :aliased
@inline infer_C_backend(::WrappedBlockSparse,        ::WrappedAliasedBlockSparse) = :aliased
@inline infer_C_backend(::WrappedAliasedBlockSparse, ::WrappedCOOTensor)          = :aliased
@inline infer_C_backend(::WrappedCOOTensor,          ::WrappedAliasedBlockSparse) = :aliased

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Constructors + wrap_itensor backend=:aliased
# ─────────────────────────────────────────────────────────────────────────────

"""
    WrappedAliasedBlockSparse(T, dims, denseLinks, inds)

Allocate an empty `WrappedAliasedBlockSparse` with the given element type,
dimension tuple, number of dense (tail) axes, and ITensor index tuple.
"""
function WrappedAliasedBlockSparse(
    ::Type{T},
    dims     :: NTuple{N,Int},
    denseLinks :: Int,
    inds     :: NTuple{N,ITensors.Index},
) where {T,N}
    ali = AliasedBlockSparse{T,N,denseLinks}(dims)
    return WrappedAliasedBlockSparse(ali, inds)
end

"""
    WrappedAliasedBlockSparse(T::ITensors.ITensor, denseLinks; kwargs...)

Convert an `ITensor` to `WrappedAliasedBlockSparse` by first reading out the
dense array in the canonical (bra, ket, links) axis order and then wrapping
the dense array as a COO tensor contracted against itself — effectively
treating every nonzero as its own scalar×template block.

If the ITensor already has `WrappedAliasedBlockSparse` external storage, it
is returned unwrapped directly.
"""
function WrappedAliasedBlockSparse(
    T       :: ITensors.ITensor,
    denseLinks :: Int;
    bra_plev :: Union{Nothing,Int} = nothing,
    ket_plev :: Union{Nothing,Int} = nothing,
)
    bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
    links = sort_links(links)
    inds_full = if bra !== nothing && ket !== nothing
        Tuple(vcat(bra, ket, links))
    elseif bra !== nothing
        Tuple(vcat(bra, links))
    elseif ket !== nothing
        Tuple(vcat(ket, links))
    else
        Tuple(ITensors.Index[])
    end
    N         = length(inds_full)
    denseLinks_kept = count(I -> ITensors.hastags(I, "Link"), inds_full)
    @assert denseLinks == denseLinks_kept "denseLinks=$denseLinks but found $denseLinks_kept Link axes"

    array_full = Array(T, inds_full...)
    # Build AliasedBlockSparse via COO × Dense: treat the dense array as B and
    # construct a unit COO tensor as A to trigger the COO × Dense aliased kernel.
    # Simpler: use blocksparse_from_dense then wrap as AliasedBlockSparse with
    # one template per block (trivial aliasing; compression comes on contraction).
    N2  = denseLinks_kept
    dims = ntuple(i -> ITensors.dim(inds_full[i]), Val(N))
    ali  = AliasedBlockSparse{eltype(array_full),N,N2}(dims)
    bs   = blocksparse_from_dense(array_full, Val(N2))
    # Populate ali: one template per block, scalar = 1
    for (i, key) in enumerate(bs.keys)
        ali.n_templates += 1
        blk_off = (bs.ids[i] - 1) * bs.blksize
        append!(ali.templates, @view bs.data[blk_off+1 : blk_off+bs.blksize])
        push!(ali.keys,      key)
        push!(ali.alias_ids, ali.n_templates)
        push!(ali.scalars,   one(eltype(array_full)))
    end
    return WrappedAliasedBlockSparse(ali, inds_full)
end

# Extend wrap_itensor to support backend=:aliased
function wrap_itensor_aliased(
    T        :: ITensors.ITensor;
    denseLinks :: Union{Nothing,Int} = nothing,
)
    if ITensors.has_external_storage(T)
        tensor = ITensors.get_external_storage(T)
        if tensor isa WrappedAliasedBlockSparse
            return tensor
        elseif tensor isa WrappedBlockSparse
            # Promote BlockSparse → AliasedBlockSparse (trivial wrapping)
            ali = _blocksparse_to_aliased(tensor.aliased)
            return WrappedAliasedBlockSparse(ali, tensor.inds)
        else
            throw(ArgumentError("External storage is not a WrappedAliasedBlockSparse"))
        end
    end
    @assert denseLinks !== nothing "must pass denseLinks for :aliased backend"
    return WrappedAliasedBlockSparse(T, denseLinks)
end

# Helper: trivially wrap a NewBlockSparseSorted as AliasedBlockSparse
# (one template per block, scalar = 1).
function _blocksparse_to_aliased(bs::NewBlockSparseSorted{T,N,N2,P}) where {T,N,N2,P}
    nblocks  = length(bs.keys)
    ali      = AliasedBlockSparse{T,N,N2,P}(
        bs.dims, bs.blksize, copy(bs.data), nblocks,
        copy(bs.keys), collect(1:nblocks), ones(T, nblocks),
    )
    return ali
end

# ─────────────────────────────────────────────────────────────────────────────
# 8.  contract! overloads for AliasedBlockSparse C output
#
# These allow the standard top-level SparseBackends.contract! to write into an
# AliasedBlockSparse result, bridging cases not yet covered by
# contract_aliased_shared.jl (namely COO × Dense → AliasedBS).
# ─────────────────────────────────────────────────────────────────────────────

# COO × Dense → AliasedBlockSparse (single reduced label)
function contract!(
    C       :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC :: AbstractVector,
    A       :: COOTensor{TA,NA},
    labelsA :: AbstractVector,
    B       :: StridedArray{TB,NB},
    labelsB :: AbstractVector,
    mapA    :: Dict,
    mapB    :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,TB,NB}
    return contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9.  wrapped_contract — aliased output path
#
# Called from contract_aliased_wrapped / contract (below) when the inferred
# C backend is :aliased.  Mirrors the :blocksparse branch in wrapped_contract.
# ─────────────────────────────────────────────────────────────────────────────

function wrapped_contract_aliased(
    A :: WrappedTensorTypes{TA,NA},
    B :: WrappedTensorTypes{TB,NB},
) where {TA,TB,NA,NB}
    Arep   = rep(A)
    Brep   = rep(B)
    indsA  = A.inds
    indsB  = B.inds
    denseA = dense_inds(A)
    denseB = dense_inds(B)

    indsC, denseC = output_inds(indsA, indsB, denseA, denseB)
    dimsC  = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
    TC     = promote_type(eltype(Arep), eltype(Brep))

    denseLinksC = length(denseC)

    sc       = scratch()
    labelsA_vec = fill_labels!(sc.labelsA, indsA)
    labelsB_vec = fill_labels!(sc.labelsB, indsB)
    labelsC_vec = fill_labels!(sc.labelsC, indsC)

    if denseLinksC == length(indsC)
        # P_C = 0: all output indices are dense → plain ITensor
        # For aliased, materialise via BlockSparse path.
        C_bs = NewBlockSparseSorted{TC, length(indsC), length(indsC)}(dimsC)
        C_bs = contract!(C_bs, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec)
        C_dense = to_dense(C_bs)
        return length(indsC) == 0 ?
            ITensors.ITensor(C_dense[]) :
            ITensors.ITensor(C_dense, indsC...)
    end

    C = WrappedAliasedBlockSparse(TC, dimsC, denseLinksC, indsC)
    C.aliased = contract!(
        C.aliased, labelsC_vec,
        Arep, labelsA_vec,
        Brep, labelsB_vec,
    )
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Top-level contract functions for WrappedAliasedBlockSparse inputs/outputs
# ─────────────────────────────────────────────────────────────────────────────

# Dispatch: when C backend infers to :aliased, route to wrapped_contract_aliased.
# We add specialised contract() overloads for all input combinations involving
# WrappedAliasedBlockSparse so they always take the aliased path.

function contract(A::WrappedAliasedBlockSparse, B::WrappedTensorTypes; kwargs...)
    return wrapped_contract_aliased(A, B)
end

function contract(A::WrappedTensorTypes, B::WrappedAliasedBlockSparse; kwargs...)
    return wrapped_contract_aliased(A, B)
end

function contract(A::WrappedAliasedBlockSparse, B::WrappedAliasedBlockSparse; kwargs...)
    return wrapped_contract_aliased(A, B)
end

"""
    contract_aliased_itensor(A, B, Abackend, Bbackend; denseLinksA, denseLinksB)

Top-level function: contract ITensors `A` and `B`, returning the result as an
`ITensor` whose external storage is a `WrappedAliasedBlockSparse`.

- `Abackend`, `Bbackend` — one of `:coo`, `:blocksparse`, `:aliased`, `:dense`
- `denseLinksA`, `denseLinksB` — number of dense (Link) axes for each tensor
  (required when `Abackend` or `Bbackend` is `:blocksparse` / `:aliased`)

The contraction kernels are the aliasing-aware ones from contract_aliased.jl
and contract_aliased_shared.jl, so the result preserves (and can amplify) any
aliasing structure.  When all output indices happen to be dense (P_C = 0) a
plain ITensor is returned instead.
"""
function contract_aliased_itensor(
    A        :: ITensors.ITensor,
    B        :: ITensors.ITensor,
    Abackend :: Symbol,
    Bbackend :: Symbol;
    denseLinksA :: Union{Nothing,Int} = nothing,
    denseLinksB :: Union{Nothing,Int} = nothing,
)
    if Abackend === :dense && Bbackend === :dense
        return ITensors.contract(A, B)
    end

    # Wrap inputs
    Aw = if Abackend === :aliased
        wrap_itensor_aliased(A; denseLinks=denseLinksA)
    else
        wrap_itensor(A; backend=Abackend, denseLinks=denseLinksA)
    end
    Bw = if Bbackend === :aliased
        wrap_itensor_aliased(B; denseLinks=denseLinksB)
    else
        wrap_itensor(B; backend=Bbackend, denseLinks=denseLinksB)
    end

    Cw = wrapped_contract_aliased(Aw, Bw)

    Cw isa ITensors.ITensor && return Cw   # P_C = 0
    if Cw isa WrappedAliasedBlockSparse && _abs_head_len(Cw) == 0
        dense_data = to_dense(Cw.aliased)
        return ITensors.ITensor(dense_data, Cw.inds...)
    end
    return ITensors._itensor_from_external_storage(Cw)
end

# Convenience: COO × Dense → AliasedBlockSparse (most common use case)
"""
    contract_coo_dense_aliased(A, B; denseLinksB=0)

Shorthand for `contract_aliased_itensor(A, B, :coo, :dense)` — contracts a
COO-format ITensor `A` with a dense ITensor `B` and returns an
`AliasedBlockSparse`-backed ITensor.
"""
function contract_coo_dense_aliased(
    A :: ITensors.ITensor,
    B :: ITensors.ITensor;
    denseLinksB :: Int = 0,
)
    return contract_aliased_itensor(A, B, :coo, :dense; denseLinksA=nothing, denseLinksB=nothing)
end

# ─────────────────────────────────────────────────────────────────────────────
# 11.  contract_and_fuse_links_aliased — bond-canonicalization variant
#
# Mirrors contract_and_fuse_links (tensor_contraction.jl) but always writes
# into WrappedAliasedBlockSparse.
# ─────────────────────────────────────────────────────────────────────────────

function contract_and_fuse_links_aliased(
    A        :: ITensors.ITensor,
    B        :: ITensors.ITensor,
    Abackend :: Symbol,
    Bbackend :: Symbol,
    bondmap  :: BondMap;
    denseLinksA :: Union{Nothing,Int} = nothing,
    denseLinksB :: Union{Nothing,Int} = nothing,
    aLeft  :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    aRight :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    bLeft  :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    bRight :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
)
    if Abackend === :dense && Bbackend === :dense
        return ITensors.contract(A, B), bondmap
    end
    Aw = if Abackend === :aliased
        wrap_itensor_aliased(A; denseLinks=denseLinksA)
    else
        wrap_itensor(A; backend=Abackend, denseLinks=denseLinksA)
    end
    Bw = if Bbackend === :aliased
        wrap_itensor_aliased(B; denseLinks=denseLinksB)
    else
        wrap_itensor(B; backend=Bbackend, denseLinks=denseLinksB)
    end
    return contract_and_fuse_links_aliased(Aw, Bw, bondmap;
        aLeft=aLeft, aRight=aRight, bLeft=bLeft, bRight=bRight)
end

function contract_and_fuse_links_aliased(
    Aw      :: WrappedTensorTypes,
    Bw      :: WrappedTensorTypes,
    bondmap :: BondMap;
    aLeft  :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    aRight :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    bLeft  :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
    bRight :: Union{Nothing,AbstractVector{<:ITensors.Index}} = nothing,
)
    Cw = wrapped_contract_aliased(Aw, Bw)

    _nz(v) = (v !== nothing && !isempty(v))

    # Bond canonicalization: fuse_axes! on WrappedAliasedBlockSparse degrades
    # to WrappedBlockSparse (see §3 above), so after canonicalization Cw may
    # no longer be WrappedAliasedBlockSparse.  That is acceptable: the aliasing
    # benefit was in the contraction itself.
    if _nz(aLeft) && _nz(bLeft)
        aBondLogical = first(aLeft)
        bBondLogical = first(bLeft)
        if Cw isa WrappedAliasedBlockSparse
            # Convert to WrappedBlockSparse for canonicalization
            Cw_bs = WrappedBlockSparse(to_blocksparse(Cw.aliased), Cw.inds)
            Cw_bs, bondmap = _canonicalize_bond_blocksparse!(
                Cw_bs, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
            Cw = Cw_bs
        elseif Cw isa WrappedBlockSparse
            Cw, bondmap = _canonicalize_bond_blocksparse!(
                Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
        else
            Cw, bondmap = _canonicalize_bond_single!(
                Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
        end
    end
    if _nz(aRight) && _nz(bRight)
        aBondLogical = first(aRight)
        bBondLogical = first(bRight)
        if Cw isa WrappedAliasedBlockSparse
            Cw_bs = WrappedBlockSparse(to_blocksparse(Cw.aliased), Cw.inds)
            Cw_bs, bondmap = _canonicalize_bond_blocksparse!(
                Cw_bs, bondmap, aBondLogical, bBondLogical, aRight, bRight)
            Cw = Cw_bs
        elseif Cw isa WrappedBlockSparse
            Cw, bondmap = _canonicalize_bond_blocksparse!(
                Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
        else
            Cw, bondmap = _canonicalize_bond_single!(
                Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
        end
    end

    Cw isa ITensors.ITensor && return Cw, bondmap

    # Collapse P=0 results to plain ITensor
    if Cw isa WrappedAliasedBlockSparse && _abs_head_len(Cw) == 0
        dense_data = to_dense(Cw.aliased)
        return ITensors.ITensor(dense_data, Cw.inds...), bondmap
    end
    if Cw isa WrappedBlockSparse && _bs_head_len(Cw) == 0
        dense_data = to_dense(Cw.blocksparse)
        return ITensors.ITensor(dense_data, Cw.inds...), bondmap
    end

    return ITensors._itensor_from_external_storage(Cw), bondmap
end

# ─────────────────────────────────────────────────────────────────────────────
# 12.  ITensors extension hooks
# ─────────────────────────────────────────────────────────────────────────────

function ITensors.dag(es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse}; kwargs...)
    data    = es.data
    newdata = ITensors.dag(data)
    return ITensors._itensor_from_external_storage(newdata)
end

LinearAlgebra.norm(w::WrappedAliasedBlockSparse) = begin
    # norm = sqrt(sum_i scalars[i]^2 * ||template[alias_ids[i]]||^2)
    s = zero(real(eltype(w.aliased.templates)))
    A = w.aliased
    for i in eachindex(A.keys)
        α    = A.scalars[i]
        toff = (A.alias_ids[i] - 1) * A.blksize
        @inbounds for j in 1:A.blksize
            s += abs2(α * A.templates[toff + j])
        end
    end
    return sqrt(s)
end

function _apply_elementwise!(
    f  :: Function,
    R  :: WrappedAliasedBlockSparse{T},
    A  :: WrappedAliasedBlockSparse{T},
) where {T}
    # Element-wise ops (scale!, axpy!) are applied to the materialised data.
    # For in-place safety, fall back to operating on scalars only when possible.
    # Here we apply to templates (the canonical approach: scale the scalars).
    length(R.aliased.scalars) == length(A.aliased.scalars) ||
        error("Mismatched sparsity patterns in _apply_elementwise! (WrappedAliasedBlockSparse)")
    @inbounds @simd for i in eachindex(R.aliased.scalars)
        R.aliased.scalars[i] = f(R.aliased.scalars[i], A.aliased.scalars[i])
    end
end

function ITensors._external_map_storage!(
    f :: Function,
    R :: WrappedAliasedBlockSparse,
    A :: WrappedAliasedBlockSparse,
)
    _apply_elementwise!(f, R, A)
    return nothing
end
