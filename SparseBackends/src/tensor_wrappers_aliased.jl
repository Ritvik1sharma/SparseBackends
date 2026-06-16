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
const _ADD_DBG_COUNT1 = Ref(0)
@inline _add_dbg_enabled1() = get(ENV, "SB_ALIASED_DEBUG", "0") == "1"
@inline _add_dbg_max1()     = parse(Int, get(ENV, "SB_ALIASED_DEBUG_MAX", "12"))


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

# Diagnostic (SB_MATVEC_DIAG): size of a wrapped matvec intermediate →
#   (n_blocks, n_templates, blksize, logical_elems = n_blocks·blksize).
# logical_elems is the materialized size (what a direct/BS step processes); the
# stored size is n_templates·blksize. Lets us compare aliased vs BS intermediate
# sizes per matvec step.
matvec_size_info(w::WrappedAliasedBlockSparse) =
    (length(w.aliased.keys), w.aliased.n_templates, w.aliased.blksize,
     length(w.aliased.keys) * w.aliased.blksize)
matvec_size_info(w::WrappedBlockSparse) =
    (length(w.blocksparse.keys), length(w.blocksparse.keys), w.blocksparse.blksize,
     length(w.blocksparse.keys) * w.blocksparse.blksize)
matvec_size_info(::Any) = (0, 0, 0, 0)

# Re-permute axes so the output's `inds` tuple matches `Tw`'s exactly.
# Aliased analog of `recast_bs_to_template`. Both Cw and Tw must have the
# same Set(inds); we just permute Cw's axes to align with Tw's order. The
# alias schema (keys, alias_ids, scalars) is preserved by permutedims.
function recast_aliased_to_template(Cw::WrappedAliasedBlockSparse{TC,N,N2c,Pc},
                                     Tw::WrappedAliasedBlockSparse{TT,N,N2t,Pt}) where {TC,TT,N,N2c,Pc,N2t,Pt}
    c_inds = collect(Cw.inds)
    t_inds = collect(Tw.inds)
    if Set(c_inds) != Set(t_inds)
        return Cw   # Different Index identities; can't recast.
    end
    if c_inds == t_inds
        return Cw   # Already aligned.
    end
    # Build perm such that new_axis_i = old_axis_perm[i] (so Cw.inds[perm[i]] = Tw.inds[i]).
    perm = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        j = findfirst(==(t_inds[i]), c_inds)
        @assert j !== nothing
        perm[i] = j
    end
    # Only safe to permute if prefix and tail regions don't cross (alias storage
    # constraint). If they would cross, fall through without recast.
    @inbounds for i in 1:N
        if (i <= Pc) != (perm[i] <= Pc)
            return Cw   # Cross-region permutation not supported here.
        end
    end
    if get(ENV, "SB_RECAST_DBG", "0") == "1" && _RECAST_DBG_N[] < 6
        _RECAST_DBG_N[] += 1
        _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
        println("[RECAST #", _RECAST_DBG_N[], "] Pc=", Pc, " perm=", perm,
                "\n   c_inds=", [_tg(I) for I in c_inds],
                "\n   t_inds=", [_tg(I) for I in t_inds])
    end
    _RECAST_PERMUTE_HITS[] += 1
    new_ali = permutedims(Cw.aliased, perm)
    return WrappedAliasedBlockSparse{TC,N,N2c,Pc}(new_ali, Tuple(t_inds))
end
Base.eltype(::WrappedAliasedBlockSparse{T}) where {T} = T
Base.eltype(::Type{<:WrappedAliasedBlockSparse{T}}) where {T} = T
Base.eltype(es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse}) = eltype(es.data)
Base.eltype(::Type{ITensors.ExternalStorage{W}}) where {W<:WrappedAliasedBlockSparse} = eltype(W)

# Snap an aliased ITensor `Hv` to a target schema (from `v`): produce a new
# aliased tensor with v's exact (keys, alias_ids, scalars, n_templates)
# while preserving Hv's NUMERIC values at each block. Templates are derived
# such that scalars_v[i] * templates_new[alias_ids_v[i]] = Hv's block value
# at v's key i. For blocks where Hv has no value, the contribution is zero.
#
# This is the key trick to keep Lanczos arithmetic stable: every matvec
# output has the SAME (keys, alias_ids, scalars) as v, so subsequent
# Aliased + Aliased adds fire the same-schema fast path (template+template)
# instead of falling through to cross-schema merge / dense.
function _snap_to_schema(Hv::WrappedAliasedBlockSparse{T,N,N2,P},
                          v_schema::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    @assert Hv.aliased.dims == v_schema.aliased.dims "dims mismatch in _snap_to_schema"
    @assert Hv.aliased.blksize == v_schema.aliased.blksize "blksize mismatch"
    V = v_schema.aliased
    H = Hv.aliased
    Kt = eltype(eltype(V.keys))
    blksize = V.blksize
    n_t = V.n_templates
    new_templates = zeros(T, n_t * blksize)
    # Index Hv's blocks by key.
    h_lookup = Dict{NTuple{P,Kt}, Int}()
    @inbounds for (i, k) in enumerate(H.keys)
        h_lookup[k] = i
    end
    # ── ERROR TRACKING (SB_ALIASED_SNAP_DBG=1) ────────────────────────────
    # Track norm of Hv data that's dropped (keys in H but not in V).
    dbg = get(ENV, "SB_ALIASED_SNAP_DBG", "0") == "1"
    if dbg
        v_keys_set = Set(V.keys)
        total_norm2 = 0.0
        kept_norm2  = 0.0
        dropped_count = 0
        for i_h in eachindex(H.keys)
            αH = H.scalars[i_h]
            h_off = (H.alias_ids[i_h] - 1) * H.blksize
            block_norm2 = 0.0
            for j in 1:H.blksize
                block_norm2 += abs2(αH * H.templates[h_off + j])
            end
            total_norm2 += block_norm2
            if H.keys[i_h] in v_keys_set
                kept_norm2 += block_norm2
            else
                dropped_count += 1
            end
        end
        if total_norm2 > 0
            dropped_frac = 1 - kept_norm2 / total_norm2
            if _SNAP_DBG_COUNT[] < 5
                _SNAP_DBG_COUNT[] += 1
                println("[SNAP_DBG #$(_SNAP_DBG_COUNT[])] Hv n_blocks=$(length(H.keys))  V n_blocks=$(length(V.keys))  dropped_keys=$dropped_count  dropped_norm_frac=$(round(dropped_frac, sigdigits=4))  total_norm=$(round(sqrt(total_norm2), sigdigits=4))")
            end
        end
    end
    # For each unique alias_id, pick a representative v-block with that id and
    # divide Hv's block value at that key by v.scalars[i_v] to get the new template.
    # Track which templates we've set so we don't overwrite.
    seen = falses(n_t)
    @inbounds for i_v in eachindex(V.keys)
        t = V.alias_ids[i_v]
        if seen[t]; continue; end
        k = V.keys[i_v]
        i_h = get(h_lookup, k, 0)
        sv = V.scalars[i_v]
        # If Hv has the key, get its block value; else zero.
        if i_h > 0 && sv != zero(T)
            αH = H.scalars[i_h]
            h_off = (H.alias_ids[i_h] - 1) * H.blksize
            t_off = (t - 1) * blksize
            inv_sv = one(T) / sv
            @inbounds @simd for j in 1:blksize
                new_templates[t_off + j] = αH * H.templates[h_off + j] * inv_sv
            end
        end
        seen[t] = true
    end
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        V.dims, blksize,
        new_templates, n_t,
        copy(V.keys), copy(V.alias_ids), copy(V.scalars),
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, v_schema.inds)
end
const _SNAP_DBG_COUNT = Ref(0)

# zero-similar for VectorInterface (used by KrylovKit when building zero
# Krylov vectors with potentially different element type).
function _zero_similar(w::WrappedAliasedBlockSparse{_T,N,N2,P}, ::Type{ElT}) where {_T,N,N2,P,ElT}
    A = w.aliased
    Kt = eltype(eltype(A.keys))
    new_ali = AliasedBlockSparse{ElT,N,N2,P,Kt}(
        A.dims, A.blksize,
        zeros(ElT, length(A.templates)), A.n_templates,
        copy(A.keys), copy(A.alias_ids), zeros(ElT, length(A.scalars)),
    )
    return WrappedAliasedBlockSparse{ElT,N,N2,P}(new_ali, w.inds)
end

@inline rep(A::WrappedAliasedBlockSparse) = A.aliased
@inline _abs_head_len(::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = P
@inline _abs_tail_len(::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = N2

# Convenience: dense-tail count for any ITensor (returns nothing if not aliased).
function _dense_tail_count(T::ITensors.ITensor)
    if ITensors.has_external_storage(T) && T.tensor.data isa WrappedAliasedBlockSparse
        return _abs_tail_len(T.tensor.data)
    end
    return nothing
end

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
# 3.  fuse_axes! — native aliasing-preserving fusion (prefix-only or tail-only)
#
# Mixed (one prefix + one tail) axes are not handled natively and demote to
# BlockSparse as before. The fall-through cases are the only ones that arise
# in aliased-psi DMRG bond canonicalization.
# ─────────────────────────────────────────────────────────────────────────────

function fuse_axes!(W::WrappedAliasedBlockSparse{T,N,N2,P}, ax1::Int, ax2::Int) where {T,N,N2,P}
    ax1 == ax2 && return W
    (1 ≤ ax1 ≤ N) || error("ax1 out of range")
    (1 ≤ ax2 ≤ N) || error("ax2 out of range")
    in_sparse1 = ax1 <= P
    in_sparse2 = ax2 <= P
    if in_sparse1 == in_sparse2
        # Both in prefix or both in tail — native path (templates untouched
        # for prefix case; cost = n_templates for tail case via permutedims).
        new_ali = fuse_two_axes!(W.aliased, ax1, ax2)
        ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)
        newinds = _fused_inds_tuple(W.inds, ax1, ax2; ax1_out=ax1_out)
        return WrappedAliasedBlockSparse{T,N-1,
            in_sparse1 ? N2 : (N2-1),
            in_sparse1 ? (P-1) : P}(new_ali, newinds)
    end
    # Mixed (one prefix, one tail): demote to BlockSparse (rare; logs when
    # SB_ALIASED_DEBUG=1 so we can confirm DMRG doesn't hit this path).
    if _add_dbg_enabled1()
        @info "fuse_axes!(WrappedAliasedBlockSparse): mixed prefix+tail axes ($ax1, $ax2) — demoting to BlockSparse"
    end
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

# #1 instrumentation: count how often the both-aliased contraction takes the
# native kernel vs the dense-materialise fallback. Reset/read from test code via
# SparseBackends._NATIVE_DOT_HITS[] / ._DENSIFY_DOT_HITS[]. Cheap Ref bumps.
const _NATIVE_DOT_HITS  = Ref(0)
const _DENSIFY_DOT_HITS = Ref(0)
const _RECAST_DBG_N     = Ref(0)     # SB_RECAST_DBG: one-shot recast perm dump counter
const _RECAST_PERMUTE_HITS = Ref(0)  # # of actual (non-identity) recast permutedims calls
const _ALIGN_OK         = Ref(0)     # SB_ALIASED_ALIGN_OUTPUT: native emitted aligned output
const _ALIGN_FALLBACK   = Ref(0)     # align attempted but kernel couldn't honor it → default order
const _ALIGN_DBG_N      = Ref(0)     # SB_ALIASED_ALIGN_DBG: one-shot fallback-exception dump
const _DOT_CHECK_MAXABS = Ref(0.0)   # SB_ALIASED_DOT_CHECK: max ABSOLUTE |native−dense| per dot
const _DOT_CHECK_MAXMAG = Ref(0.0)   # |dense| at the worst-abs dot (context for the abs error)
reset_dot_hits!() = (_NATIVE_DOT_HITS[] = 0; _DENSIFY_DOT_HITS[] = 0;
                     _DOT_CHECK_MAXABS[] = 0.0; _DOT_CHECK_MAXMAG[] = 0.0;
                     _RECAST_PERMUTE_HITS[] = 0; _ALIGN_OK[] = 0; _ALIGN_FALLBACK[] = 0)

# SB_SCHEMA_DBG: print the aliased P-classification (sparse keys vs dense tail) of
# a tensor at a labelled point, to trace WHERE the constructed schema (e.g. P=3:
# site+2 link-channels) collapses (e.g. P=2: channels pushed to dense). Budgeted.
const _SCHEMA_DBG_BUDGET = Ref(-1)
function schema_dbg(label, T)
    get(ENV, "SB_SCHEMA_DBG", "0") == "1" || return nothing
    if _SCHEMA_DBG_BUDGET[] < 0
        _SCHEMA_DBG_BUDGET[] = parse(Int, get(ENV, "SB_SCHEMA_DBG_BUDGET", "40"))
    end
    _SCHEMA_DBG_BUDGET[] <= 0 && return nothing
    _SCHEMA_DBG_BUDGET[] -= 1
    _fmt(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
    if T isa ITensors.ITensor && ITensors.has_external_storage(T) &&
       (w = ITensors.get_external_storage(T)) isa WrappedAliasedBlockSparse
        P = _abs_head_len(w); N = ndims(w.aliased); den = dense_inds(w)
        a = w.aliased
        spk = [_fmt(I) for I in ITensors.inds(T) if !(I in den)]
        dns = [_fmt(I) for I in ITensors.inds(T) if I in den]
        # alias-map summary: n_keys (blocks), n_templates, dedup ratio, and a cheap
        # hash of the (keys, scalars) so a change in the real alias map is visible.
        kh = hash(a.keys); sh = hash(a.scalars)
        println("[SCHEMA ", label, "]  P=", P, " N=", N,
                "  n_keys=", length(a.keys), " n_tmpl=", a.n_templates,
                " dedup=", round(length(a.keys)/max(a.n_templates,1), digits=2),
                "  keyhash=", string(kh % 0x10000, base=16), " scalhash=", string(sh % 0x10000, base=16),
                "\n           SPARSE=", spk, "  DENSE=", dns)
    else
        cls = T isa ITensors.ITensor ?
            (ITensors.has_external_storage(T) ? string(typeof(ITensors.get_external_storage(T))) : "dense ITensor") :
            string(typeof(T))
        println("[SCHEMA ", label, "]  (", cls, ")")
    end
    return nothing
end

# #1 helper: do all REDUCED shared labels sit on the SAME side (sparse prefix
# vs dense tail) in A and B?  The native AliasedBS×AliasedBS contract_shared!
# errors on a shared label that is prefix in one operand and dense-tail in the
# other; when this holds we can contract natively (merge-join + GEMM) instead of
# materialising both md-sized operands to dense. Used by the SB_ALIASED_NATIVE_DOT
# path to skip the densify fallback for matching-schema inner products / dots.
function _aliased_reduced_class_match(labelsA::AbstractVector, PA::Int,
                                      labelsB::AbstractVector, PB::Int,
                                      labelsC::AbstractVector)
    setC = Set(labelsC)
    posB = Dict(l => i for (i, l) in enumerate(labelsB))
    @inbounds for (ia, l) in enumerate(labelsA)
        (l in setC) && continue            # kept (appears in C) — not reduced
        ib = get(posB, l, 0)
        ib == 0 && continue                # not shared with B
        ((ia <= PA) == (ib <= PB)) || return false
    end
    return true
end

# #1 LEAN scalar dot: full reduction (scalar output) of two same-schema aliased
# operands is ⟨A,B⟩ = Σ_k α_A[k]·α_B[k]·⟨tmplA[aA(k)], tmplB[aB(k)]⟩ — a plain
# dotu per matching block, with a (aA,aB) template-dot cache so each distinct
# template pair is summed once (alias dedup). No permutedims, no position dicts,
# no wrapper, no densify of the md-sized operands. Returns the scalar, or
# `nothing` if the fast preconditions don't hold (caller falls back to densify).
# NOTE: the contraction is dotu (no conj) — `dag` is applied upstream by the
# caller, so Arep already carries the conjugated values (verified FP-equal to the
# densified contract, |Δ|/|val| ≈ eps).
function _aliased_lean_scalar_dot(A::AliasedBlockSparse{TA,NA,N2A,PA},
                                  labelsA::AbstractVector,
                                  B::AliasedBlockSparse{TB,NB,N2B,PB},
                                  labelsB::AbstractVector) where {TA,NA,N2A,PA,TB,NB,N2B,PB}
    TC = promote_type(TA, TB)
    # preconditions for a pure inner product: identical label correspondence (so
    # prefix keys compare directly and dense tails align element-wise), equal
    # prefix length, matching block size and key type.
    (NA == NB && PA == PB && length(labelsA) == length(labelsB)) || return nothing
    A.blksize == B.blksize || return nothing
    eltype(A.keys) == eltype(B.keys) || return nothing
    @inbounds for i in eachindex(labelsA)
        labelsA[i] == labelsB[i] || return nothing
    end
    bs = A.blksize
    posB = Dict{eltype(B.keys), Int}()
    @inbounds for j in eachindex(B.keys); posB[B.keys[j]] = j; end
    cache = Dict{Tuple{Int,Int}, TC}()
    tA = A.templates; tB = B.templates
    s = zero(TC)
    @inbounds for i in eachindex(A.keys)
        j = get(posB, A.keys[i], 0); j == 0 && continue
        aA = A.alias_ids[i]; aB = B.alias_ids[j]
        td = get(cache, (aA, aB), nothing)
        if td === nothing
            offA = (aA-1)*bs; offB = (aB-1)*bs
            acc = zero(TC)
            @simd for t in 1:bs
                acc += TC(tA[offA+t]) * TC(tB[offB+t])
            end
            td = acc; cache[(aA, aB)] = td
        end
        s += TC(A.scalars[i]) * TC(B.scalars[j]) * td
    end
    return s
end

# #recast-kill: reorder output indices to a desired order (the template's ordered
# inds) while PRESERVING the sparse-prefix / dense-tail split. Returns a perm
# `ord` (ord[i] = position in `inds` of the axis that belongs at output position
# i) or `nothing` if `preferred` doesn't cover `inds` exactly or would cross the
# prefix/dense boundary. Used to make the native contract emit aligned output so
# the downstream recast_aliased_to_template becomes a no-op.
# Match by id+dim, IGNORING plev: the contract output is aligned pre-replaceprime,
# where Link axes are still primed (plev=1) relative to the unprimed template;
# the subsequent replaceprime(1=>0) only flips plev (preserving id), so id+dim
# matching maps the pre-replaceprime output order onto the template order.
_ind_match_pref(I::ITensors.Index, p::ITensors.Index) =
    (ITensors.id(I) == ITensors.id(p) && ITensors.dim(I) == ITensors.dim(p))
_ind_match_pref(I::ITensors.Index, p) = (label_key_for_ind(I) == p)
function _align_output_order(inds, denseLinks::Int, preferred)
    N = length(inds)
    length(preferred) == N || return nothing
    Pc = N - denseLinks
    ord = Vector{Int}(undef, N)
    used = fill(false, N)
    @inbounds for i in 1:N
        p = preferred[i]; j = 0
        for k in 1:N
            if !used[k] && _ind_match_pref(inds[k], p); j = k; break; end
        end
        j == 0 && return nothing
        ((i <= Pc) == (j <= Pc)) || return nothing   # keep prefix/dense split
        ord[i] = j; used[j] = true
    end
    all(i -> ord[i] == i, 1:N) && return nothing       # already aligned; no-op
    return ord
end

function wrapped_contract_aliased(
    A :: WrappedTensorTypes{TA,NA},
    B :: WrappedTensorTypes{TB,NB};
    preserve_bs_output::Bool=false,
    preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
) where {TA,TB,NA,NB}
    @timeit TIMER "get_data_info" begin
        Arep   = rep(A)
        Brep   = rep(B)
        indsA  = A.inds
        indsB  = B.inds
        denseA = dense_inds(A)
        denseB = dense_inds(B)
    end

    @timeit TIMER "output_inds" begin
        # Forward the hint to output_inds: it knows how to reorder indsC so
        # sparse-prefix axes come first and dense-tail axes come last, and
        # how to classify hint-marked axes as dense.
        indsC, denseC = output_inds(indsA, indsB, denseA, denseB;
            output_inds_hint=output_inds_hint)
        dimsC  = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
        TC     = promote_type(eltype(Arep), eltype(Brep))
    end

    @timeit TIMER "classify_contraction" begin
        denseLinksC = length(denseC)

        sc       = scratch()
        labelsA_vec = fill_labels!(sc.labelsA, indsA)
        labelsB_vec = fill_labels!(sc.labelsB, indsB)
        labelsC_vec = fill_labels!(sc.labelsC, indsC)
    end

    # Default (preserve_bs_output=false) — write dense output directly.
    # For the Aliased×Dense → Dense path, allocate C in the kernel's CANONICAL
    # layout (keepA, keepB, c_prefix). The kernel then writes directly into C
    # with perm_C = identity and skips its `add.permute_back` pass.  The
    # returned ITensor carries the canonical-order indices; downstream
    # ITensors operations transparently handle the index reorder.
    if !preserve_bs_output
        ali_dense_path = (Arep isa AliasedBlockSparse && Brep isa AbstractArray &&
                          !(Brep isa AliasedBlockSparse) && !(Brep isa NewBlockSparseSorted) &&
                          !(Brep isa COOTensor)) ||
                         (Brep isa AliasedBlockSparse && Arep isa AbstractArray &&
                          !(Arep isa AliasedBlockSparse) && !(Arep isa NewBlockSparseSorted) &&
                          !(Arep isa COOTensor))

        if ali_dense_path
            # Compute canonical (keepA, keepB, c_prefix) layout to allocate C in it.
            @timeit TIMER "ali_dense_prep" begin
                swap = !(Arep isa AliasedBlockSparse)
                Arep_a    = swap ? Brep         : Arep
                Brep_a    = swap ? Arep         : Brep
                labelsA_a = swap ? labelsB_vec  : labelsA_vec
                labelsB_a = swap ? labelsA_vec  : labelsB_vec
                indsA_a   = swap ? indsB        : indsA
                indsB_a   = swap ? indsA        : indsB
                PA        = _abs_head_len(swap ? B : A)
            end

            mapA_local = Dict(l => i for (i, l) in enumerate(labelsA_a))
            mapB_local = Dict(l => i for (i, l) in enumerate(labelsB_a))
            mapC_local = Dict(l => i for (i, l) in enumerate(labelsC_vec))

            keepA_labs    = [l for l in labelsA_a[PA+1:end] if  haskey(mapC_local, l)]
            red_dense     = [l for l in labelsA_a[PA+1:end] if !haskey(mapC_local, l)]
            keepB_labs    = [l for l in labelsB_a            if  haskey(mapC_local, l)]
            c_prefix_labs = [l for l in labelsA_a[1:PA]      if  haskey(mapC_local, l)]
            shared_prefix = [l for l in labelsA_a[1:PA]      if  haskey(mapB_local, l) && !haskey(mapC_local, l)]

            @timeit TIMER "ali_dense_prep_labels" begin
                label_to_ind = Dict(l => indsC[mapC_local[l]] for l in labelsC_vec)
                # Canonical-C ordering selection (priority order):
                #   1. Caller-provided `preferred_output_labels` — explicit hint.
                #   2. SB_FUSE_LINKS heuristic — data-driven order so the NEXT
                #      sparse-H × dense kernel sees its B input as
                #      [red_dense, keepB, shared_prefix] with permB = identity.
                #      Mapping (from SB_PERMB_DBG diagnosis):
                #        upstream's b_other   → next call's red_dense
                #        upstream's keepA     → next call's keepB
                #        upstream's b_to_next → next call's keepB
                #        upstream's c_prefix  → next call's shared_prefix
                #      So the right canonical order is
                #        [b_other, keepA_labs, b_to_next, c_prefix_labs].
                #      (The old order [keepA, b_other, c_prefix, b_to_next] was
                #      based on a tag heuristic that misidentified which axis
                #      becomes the next call's shared_prefix.)
                #   3. Plain default — [keepA, keepB, c_prefix].
                if preferred_output_labels !== nothing
                    # Accept either Vector{Index} (natural for callers) or the
                    # internal label tuple format. Convert as needed.
                    canon_labels = if !isempty(preferred_output_labels) &&
                                       first(preferred_output_labels) isa ITensors.Index
                        [label_key_for_ind(I) for I in preferred_output_labels]
                    else
                        collect(preferred_output_labels)
                    end
                elseif get(ENV, "SB_FUSE_LINKS", "0") == "1"
                    is_b_site_to_next(l) = begin
                        I = label_to_ind[l]
                        ts = string(ITensors.tags(I))
                        !contains(ts, "Link") && ITensors.plev(I) == 0
                    end
                    b_to_next = [l for l in keepB_labs if  is_b_site_to_next(l)]
                    b_other   = [l for l in keepB_labs if !is_b_site_to_next(l)]
                    canon_labels = vcat(b_other, keepA_labs, b_to_next, c_prefix_labs)
                else
                    canon_labels = vcat(keepA_labs, keepB_labs, c_prefix_labs)
                end
                canon_inds = ITensors.Index[label_to_ind[l] for l in canon_labels]
                canon_dims = ntuple(i -> ITensors.dim(canon_inds[i]), length(canon_inds))
            end
            @timeit TIMER "ali_dense_alloc" begin
                C_canon = zeros(TC, canon_dims...)
            end

            if _add_dbg_enabled1() && _ADD_DBG_COUNT1[] < _add_dbg_max1()
                _ADD_DBG_COUNT1[] += 1
                idx = _ADD_DBG_COUNT1[]
                println("\n[wrapper #", idx, "] Aliased×Dense → Dense path")
                println("  swap = ", swap)
                println("  A.inds (aliased) = ", indsA_a)
                println("  B.inds (env)     = ", indsB_a)
                println("  canon_labels (C order) → canon_inds = ", canon_inds)
                println("  keepA = ", keepA_labs)
                println("  red_dense = ", red_dense)
                println("  keepB = ", keepB_labs)
                println("  c_prefix = ", c_prefix_labs)
                println("  shared_prefix = ", shared_prefix)
            end

            @timeit TIMER "contract_aliased_dense_to_dense" begin
                contract_aliased_dense_to_dense!(C_canon, canon_labels, Arep_a, labelsA_a, Brep_a, labelsB_a)
            end

            @timeit TIMER "wrap_output" begin
                output = length(canon_inds) == 0 ? ITensors.ITensor(C_canon[]) :
                                                   ITensors.itensor(C_canon, canon_inds...)
            end
            return output
        end

        # Fallback: not the aliased×dense path → produce dense output.
        #
        # When BOTH inputs are aliased, demoting them to BlockSparse and
        # calling contract_bs_bs_to_dense! triggers the BS kernel's
        # cross-region assertion when the two aliased classifications don't
        # match (a common case for matvec / inner product results in DMRG,
        # where one side has P>0 site/link structure and the other ended up
        # with all axes in dense tail). Sidestep that by materialising both
        # aliased inputs to plain dense Arrays and using a generic
        # dense×dense contract via ITensors.
        C_data = zeros(TC, dimsC...)
        if Arep isa AliasedBlockSparse && Brep isa AliasedBlockSparse
            # #1 lean native scalar dot: the dominant both-aliased case in Path-B
            # is the Lanczos inner product (scalar output). Compute it directly as
            # scaled template dots (see _aliased_lean_scalar_dot) — NO permutedims,
            # dicts-of-positions, wrapper, or densify of the md-sized operands.
            # Gated by SB_ALIASED_NATIVE_DOT; falls back to dense materialisation
            # for partial reductions / non-aligned labels / gate off.
            if get(ENV, "SB_ALIASED_NATIVE_DOT", "0") == "1" && length(indsC) == 0
                s = _aliased_lean_scalar_dot(Arep, labelsA_vec, Brep, labelsB_vec)
                if s !== nothing
                    _NATIVE_DOT_HITS[] += 1
                    sc = convert(TC, s)
                    if get(ENV, "SB_ALIASED_DOT_CHECK", "0") == "1"
                        refC = ITensors.contract(ITensors.ITensor(to_dense(Arep), indsA...),
                                                 ITensors.ITensor(to_dense(Brep), indsB...))
                        aerr = abs(sc - refC[])
                        if aerr > _DOT_CHECK_MAXABS[]
                            _DOT_CHECK_MAXABS[] = aerr; _DOT_CHECK_MAXMAG[] = abs(refC[])
                        end
                    end
                    return ITensors.ITensor(sc)
                end
            end
            _DENSIFY_DOT_HITS[] += 1
            tmpA = ITensors.ITensor(to_dense(Arep), indsA...)
            tmpB = ITensors.ITensor(to_dense(Brep), indsB...)
            return ITensors.contract(tmpA, tmpB)
        end
        Arep_bs = Arep isa AliasedBlockSparse ? to_blocksparse(Arep) : Arep
        Brep_bs = Brep isa AliasedBlockSparse ? to_blocksparse(Brep) : Brep
        if Arep_bs isa NewBlockSparseSorted && Brep_bs isa NewBlockSparseSorted
            contract_bs_bs_to_dense!(C_data, labelsC_vec, Arep_bs, labelsA_vec, Brep_bs, labelsB_vec)
        elseif Arep_bs isa NewBlockSparseSorted && Brep_bs isa AbstractArray
            contract_bs_dense_to_dense!(C_data, labelsC_vec, Arep_bs, labelsA_vec, Brep_bs, labelsB_vec)
        elseif Arep_bs isa AbstractArray && Brep_bs isa NewBlockSparseSorted
            contract_bs_dense_to_dense!(C_data, labelsC_vec, Brep_bs, labelsB_vec, Arep_bs, labelsA_vec)
        else
            C_bs = NewBlockSparseSorted{TC, length(indsC), length(indsC)}(dimsC)
            contract!(C_bs, labelsC_vec, Arep_bs, labelsA_vec, Brep_bs, labelsB_vec)
            C_data .= to_dense(C_bs)
        end
        @timeit TIMER "wrap_output" begin
            output = length(indsC) == 0 ? ITensors.ITensor(C_data[]) :
                                        ITensors.itensor(C_data, indsC...)
        end
        return output
    end

    @timeit TIMER "contract_aliased_densify" begin
        # P_C=0 densify path: only fires when the caller didn't ask to
        # preserve aliased storage. This mirrors the BS path's
        # `denseLinksC == length(indsC) && !preserve_bs_output` guard.
        # Without this guard, every matvec step with all-dense-tail output
        # densifies even when the caller (Hv * H_site through
        # contract_preserve_bs) explicitly requested aliased preservation,
        # which collapses subsequent matvec steps to denseH_denseV.
        if denseLinksC == length(indsC) && !preserve_bs_output
            A_dense = Arep isa AliasedBlockSparse ? to_dense(Arep) : Arep
            B_dense = Brep isa AliasedBlockSparse ? to_dense(Brep) : Brep
            C_data = zeros(TC, dimsC...)
            if A_dense isa AbstractArray && B_dense isa AbstractArray
                tmpA = ITensors.ITensor(A_dense, indsA...)
                tmpB = ITensors.ITensor(B_dense, indsB...)
                return ITensors.contract(tmpA, tmpB)
            end
            return length(indsC) == 0 ?
                ITensors.ITensor(C_data[]) :
                ITensors.ITensor(C_data, indsC...)
        end
    end

    @timeit TIMER "contract_aliased" begin
        if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
            A_class = if Arep isa AliasedBlockSparse
                "Aliased{N=$(ndims(Arep)),P=$(length(Arep.dims)-prod(size(Arep))÷Arep.blksize == 0 ? 0 : 0)}"
            else
                string(typeof(Arep))
            end
            sA = Arep isa AliasedBlockSparse ? "Aliased{N=$(ndims(Arep)),P=$(length(_dims(Arep))-(ndims(Arep)-length(_dims(Arep))))}" : string(typeof(Arep))
            println("[SB_ALIASED_TRACE wrapped_contract_aliased] preserve_bs_output=$preserve_bs_output  A=", string(typeof(Arep)), "  B=", string(typeof(Brep)),
                "  N_C=$(length(indsC))  N2_C=$denseLinksC  P_C=$(length(indsC)-denseLinksC)")
        end
        # Forward output_inds_hint so the inner kernel can trigger its
        # fission path (for aliased kernels: my _aliased_shared_via_bs_fission!;
        # for BS kernels: _contract_shared_hint!). Convert from Set{Index} to
        # the Set{Label} expected by inner contract! (label = (id, plev) tuple).
        hint_labels = output_inds_hint === nothing ? nothing :
            Set(label_key_for_ind(I) for I in output_inds_hint)
        # #recast-kill (gated SB_ALIASED_ALIGN_OUTPUT): emit the output already in
        # the template's axis order so the downstream recast_aliased_to_template is
        # a no-op. Reorders indsC/labelsC_vec within the prefix and dense blocks,
        # then lets contract! write in that order. try/catch: if the kernel can't
        # produce that order (e.g. interleaved A/B kept dims), fall back to the
        # default order (recast still fixes it ⇒ correctness preserved).
        if preferred_output_labels !== nothing &&
           get(ENV, "SB_ALIASED_ALIGN_OUTPUT", "0") == "1"
            ord = _align_output_order(indsC, denseLinksC, preferred_output_labels)
            if ord !== nothing
                try
                    indsC2  = ntuple(i -> indsC[ord[i]], length(indsC))
                    dimsC2  = ntuple(i -> ITensors.dim(indsC2[i]), length(indsC2))
                    labelsC2 = [labelsC_vec[ord[i]] for i in 1:length(labelsC_vec)]
                    Cc = WrappedAliasedBlockSparse(TC, dimsC2, denseLinksC, indsC2)
                    Cc.aliased = contract!(Cc.aliased, labelsC2, Arep, labelsA_vec,
                                           Brep, labelsB_vec; output_inds_hint=hint_labels)
                    _ALIGN_OK[] += 1
                    return Cc
                catch e
                    _ALIGN_FALLBACK[] += 1   # kernel couldn't honor order; default below
                    if get(ENV, "SB_ALIASED_ALIGN_DBG", "0") == "1" && _ALIGN_DBG_N[] < 4
                        _ALIGN_DBG_N[] += 1
                        _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
                        println("[ALIGN FALLBACK #", _ALIGN_DBG_N[], "] Pc=", length(indsC)-denseLinksC,
                                " ord=", ord, "\n   ERROR: ", sprint(showerror, e),
                                "\n   indsC(default)=", [_tg(I) for I in indsC],
                                "\n   preferred     =", [_tg(I) for I in preferred_output_labels])
                    end
                end
            end
        end
        C = WrappedAliasedBlockSparse(TC, dimsC, denseLinksC, indsC)
        C.aliased = contract!(
            C.aliased, labelsC_vec,
            Arep, labelsA_vec,
            Brep, labelsB_vec;
            output_inds_hint=hint_labels,
        )
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Top-level contract functions for WrappedAliasedBlockSparse inputs/outputs
# ─────────────────────────────────────────────────────────────────────────────

# Dispatch: when C backend infers to :aliased, route to wrapped_contract_aliased.
# We add specialised contract() overloads for all input combinations involving
# WrappedAliasedBlockSparse so they always take the aliased path.

function contract(A::WrappedAliasedBlockSparse, B::WrappedTensorTypes;
                  preserve_bs_output::Bool=false,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint)
end

function contract(A::WrappedTensorTypes, B::WrappedAliasedBlockSparse;
                  preserve_bs_output::Bool=false,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint)
end

function contract(A::WrappedAliasedBlockSparse, B::WrappedAliasedBlockSparse;
                  preserve_bs_output::Bool=false,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint)
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
    preserve_bs_output :: Bool = true,
    preferred_output_labels :: Union{Nothing,AbstractVector} = nothing,
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

    Cw = wrapped_contract_aliased(Aw, Bw; preserve_bs_output, preferred_output_labels)

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
    # :aliased on a plain dense ITensor (no external storage) means "this
    # input is dense; route through the aliased-output kernel". The COO ×
    # Dense → Aliased and Aliased × Dense → Aliased kernels are what produce
    # the alias-structured output, so the dense input just needs a :dense
    # wrap.
    Aw = if Abackend === :aliased
        if ITensors.has_external_storage(A)
            wrap_itensor_aliased(A; denseLinks=denseLinksA)
        else
            wrap_itensor(A; backend=:dense, denseLinks=denseLinksA)
        end
    else
        wrap_itensor(A; backend=Abackend, denseLinks=denseLinksA)
    end
    Bw = if Bbackend === :aliased
        if ITensors.has_external_storage(B)
            wrap_itensor_aliased(B; denseLinks=denseLinksB)
        else
            wrap_itensor(B; backend=:dense, denseLinks=denseLinksB)
        end
    else
        wrap_itensor(B; backend=Bbackend, denseLinks=denseLinksB)
    end
    return contract_and_fuse_links_aliased(Aw, Bw, bondmap;
        aLeft=aLeft, aRight=aRight, bLeft=bLeft, bRight=bRight)
end

# Aliased-aware bond canonicalization. Mirror of _canonicalize_bond_blocksparse!
# (tensor_contraction.jl) but operates on WrappedAliasedBlockSparse. Each
# fuse_axes! call is native (prefix-only or tail-only) so the result stays
# WrappedAliasedBlockSparse; mixed fuses inside fuse_axes! itself decide
# whether to demote.
function _canonicalize_bond_aliased!(
    Cw::WrappedAliasedBlockSparse{T,N,N2,P},
    bondmap::BondMap,
    aBondLogical::ITensors.Index,
    bBondLogical::ITensors.Index,
    aLegs::AbstractVector{<:ITensors.Index},
    bLegs::AbstractVector{<:ITensors.Index},
) where {T,N,N2,P}
    function collect_head_tail_axes(legs)
        head = Int[]
        tail = Int[]
        for ind in legs
            for ax in _find_axes_of_ind(Cw.inds, ind)
                if ax <= Int(P)
                    push!(head, ax)
                else
                    push!(tail, ax)
                end
            end
        end
        unique!(head); sort!(head)
        unique!(tail); sort!(tail)
        return head, tail
    end
    headA, tailA = collect_head_tail_axes(aLegs)
    headB, tailB = collect_head_tail_axes(bLegs)
    head_axes = sort!(unique!(vcat(headA, headB)))
    tail_axes = sort!(unique!(vcat(tailA, tailB)))
    function fuse_bucket!(axes::Vector{Int}, region::Symbol)
        length(axes) >= 2 || return Cw, bondmap
        base = first(axes)
        for ax in reverse(axes[2:end])
            Cw = fuse_axes!(Cw, base, ax)
        end
        raw = Cw.inds[base]
        can = get_or_create_cbond!(bondmap, aBondLogical, bBondLogical, region, raw)
        relabel_ind!(Cw, raw, can)
        return Cw, bondmap
    end
    Cw, bondmap = fuse_bucket!(tail_axes, :dense)
    Cw, bondmap = fuse_bucket!(head_axes, :sparse)
    return Cw, bondmap
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
    # preserve_bs_output=true is required to actually produce aliased output;
    # the default false path collapses to a dense ITensor.
    Cw = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)

    _nz(v) = (v !== nothing && !isempty(v))

    # Bond canonicalization: with native fuse_axes! (prefix-only / tail-only)
    # aliasing is preserved.  Mixed-region fuses still demote to BlockSparse.
    if _nz(aLeft) && _nz(bLeft)
        aBondLogical = first(aLeft)
        bBondLogical = first(bLeft)
        if Cw isa WrappedAliasedBlockSparse
            # Native canonicalization preserves aliasing (fuse_axes! is now
            # native for prefix-only / tail-only cases). If a mixed fuse is
            # forced, fuse_axes! demotes locally and the result becomes BS.
            Cw, bondmap = _canonicalize_bond_aliased!(
                Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
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
            Cw, bondmap = _canonicalize_bond_aliased!(
                Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
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
    # Element-wise (scale!, axpy!). Requires identical schema: keys, alias_ids,
    # and n_templates must match (so positions in `templates` align). If they
    # do, applying f to the templates buffers gives the correct value at every
    # block. If not, we don't have a path that preserves aliasing — error per
    # user-stated policy (no BS demotion fallback at the elementwise layer).
    Rali = R.aliased; Aali = A.aliased
    (Rali.dims == Aali.dims && Rali.blksize == Aali.blksize &&
     Rali.n_templates == Aali.n_templates &&
     Rali.keys == Aali.keys && Rali.alias_ids == Aali.alias_ids &&
     Rali.scalars == Aali.scalars) ||
        error("Mismatched aliased schema in _apply_elementwise! (WrappedAliasedBlockSparse) — implement BS demotion when this case arises")
    _ADD_AXPY_MATCH[] += 1
    @inbounds @simd for i in eachindex(Rali.templates)
        Rali.templates[i] = f(Rali.templates[i], Aali.templates[i])
    end
end

# ── Scalar multiply (preserves schema exactly by scaling `scalars`) ─────────
function Base.:*(α::Number, w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    A = w.aliased
    new_scalars = T(α) .* A.scalars
    Kt = eltype(eltype(A.keys))
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        A.dims, A.blksize,
        copy(A.templates), A.n_templates,
        copy(A.keys), copy(A.alias_ids), new_scalars,
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, w.inds)
end
Base.:*(w::WrappedAliasedBlockSparse, α::Number) = α * w

Base.:*(α::Number, es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse}) =
    ITensors._itensor_from_external_storage(α * es.data)
Base.:*(es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse}, α::Number) = α * es

Base.:-(w::WrappedAliasedBlockSparse) = (-1) * w
Base.:/(w::WrappedAliasedBlockSparse, α::Number) = (one(eltype(w.aliased.templates))/α) * w

# ── Deep copy ────────────────────────────────────────────────────────────────
function Base.copy(w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    A = w.aliased
    Kt = eltype(eltype(A.keys))
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        A.dims, A.blksize,
        copy(A.templates), A.n_templates,
        copy(A.keys), copy(A.alias_ids), copy(A.scalars),
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, w.inds)
end

# ── Element-wise addition ────────────────────────────────────────────────────
# Two aliased tensors can be added in-place into a fresh aliased only when
# their schemas are identical (same keys, alias_ids, scalars, n_templates).
# In that case C[i] = scalars[i] * (templates_A[a[i]] + templates_B[a[i]]), so
# the sum's schema is the same and templates_C = templates_A + templates_B.
# Otherwise we can't stay aliased — error per user policy.
function Base.:+(wA::WrappedAliasedBlockSparse{T,N,N2,P},
                 wB::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    # Align B's axis order to A's via Index identity (handles cases where the
    # same Indices appear in different positions in wA.inds vs wB.inds).
    # Without this, two tensors of the same shape but permuted axes would
    # fail the strict dims-equality check below and fall to dense.
    if wA.inds != wB.inds
        perm = Vector{Int}(undef, N)
        @inbounds for i in 1:N
            j = findfirst(==(wA.inds[i]), wB.inds)
            if j === nothing
                return _add_aliased_via_dense(wA, wB)   # truly different inds
            end
            perm[i] = j
        end
        if perm != collect(1:N)
            B_aligned = SparseBackends.permutedims(wB.aliased, perm)
            wB = WrappedAliasedBlockSparse{T,N,N2,P}(B_aligned, wA.inds)
        end
    end
    A = wA.aliased; B = wB.aliased
    # Best case: schemas identical → add templates element-wise, keep schema.
    schemas_match = A.dims == B.dims && A.blksize == B.blksize &&
                    A.n_templates == B.n_templates &&
                    A.keys == B.keys && A.alias_ids == B.alias_ids &&
                    A.scalars == B.scalars
    _dump_add_operands(
        schemas_match ? "plus_match" : (A.dims == B.dims ? "plus_merge" : "plus_dense"),
        A, B)
    if schemas_match
        _ADD_PLUS_MATCH[] += 1
        Kt = eltype(eltype(A.keys))
        new_templates = A.templates .+ B.templates
        new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
            A.dims, A.blksize,
            new_templates, A.n_templates,
            copy(A.keys), copy(A.alias_ids), copy(A.scalars),
        )
        return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
    end
    # Same dims but mismatched schemas (different keys, alias_ids, or scalars).
    # Merge via key union with one template per resulting block (trivial dedup,
    # no template-level compression but storage stays aliased so the result
    # can keep flowing through aliased kernels).
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1" && _CROSS_SCHEMA_TRACE_COUNT[] < 10
        _CROSS_SCHEMA_TRACE_COUNT[] += 1
        println("[DEEP cross-schema merge #$(_CROSS_SCHEMA_TRACE_COUNT[])]  A=Aliased{N=$(ndims(A)),nb=$(length(A.keys)),nt=$(A.n_templates)}  B=Aliased{nb=$(length(B.keys)),nt=$(B.n_templates)}")
    end
    A.dims != B.dims && return _add_aliased_via_dense(wA, wB)
    _ADD_PLUS_MERGE[] += 1
    Kt = eltype(eltype(A.keys))
    blksize = A.blksize
    # Index A's and B's blocks by key for lookup.
    a_lookup = Dict{NTuple{P,Kt}, Int}()
    for (i, k) in enumerate(A.keys); a_lookup[k] = i; end
    b_lookup = Dict{NTuple{P,Kt}, Int}()
    for (i, k) in enumerate(B.keys); b_lookup[k] = i; end
    all_keys = collect(NTuple{P,Kt}, union(keys(a_lookup), keys(b_lookup)))
    sort!(all_keys; by = k -> _prefix_lin(k, ntuple(i -> A.dims[i], Val(P))))
    nb = length(all_keys)
    new_templates = Vector{T}(undef, nb * blksize)
    new_alias_ids = collect(1:nb)
    new_scalars   = ones(T, nb)
    @inbounds for (idx, k) in enumerate(all_keys)
        off = (idx - 1) * blksize
        i_a = get(a_lookup, k, 0)
        i_b = get(b_lookup, k, 0)
        if i_a > 0
            αa = A.scalars[i_a]
            ta = (A.alias_ids[i_a] - 1) * A.blksize
            for j in 1:blksize
                new_templates[off + j] = αa * A.templates[ta + j]
            end
        else
            for j in 1:blksize
                new_templates[off + j] = zero(T)
            end
        end
        if i_b > 0
            αb = B.scalars[i_b]
            tb = (B.alias_ids[i_b] - 1) * B.blksize
            for j in 1:blksize
                new_templates[off + j] += αb * B.templates[tb + j]
            end
        end
    end
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        A.dims, blksize,
        new_templates, nb,
        all_keys, new_alias_ids, new_scalars,
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
end

const _CROSS_SCHEMA_TRACE_COUNT = Ref(0)
const _ADD_DENSE_TRACE_COUNT = Ref(0)

# ── Add-operand dump (gated SB_ADD_OPERAND_DUMP=1) ────────────────────────────
# Dumps the schema (key → alias_id [scalar], + template value-hash) of BOTH
# operands of an aliased Base.:+ — i.e. the actual vectors the Lanczos eigsolve
# is adding (the M^{±1/2}-dressed Krylov vectors). Then classifies why they did
# (or didn't) match: same key-SET? same key-ORDER? same key→template-VALUE map?
# This is the decisive datum for "can the add keep dedup". Latched to the first
# SB_ADD_OPERAND_DUMP_MAX (default 4) calls.
const _ADD_OPERAND_DUMP_COUNT = Ref(0)
function _dump_add_operands(tier::String, A, B; max_blocks::Int=24)
    get(ENV, "SB_ADD_OPERAND_DUMP", "0") == "1" || return
    _ADD_OPERAND_DUMP_COUNT[] < parse(Int, get(ENV, "SB_ADD_OPERAND_DUMP_MAX", "4")) || return
    _ADD_OPERAND_DUMP_COUNT[] += 1
    n = _ADD_OPERAND_DUMP_COUNT[]
    _vh(X, tid) = (off = (tid - 1) * X.blksize; string(hash(@view X.templates[(off+1):(off+X.blksize)]), base=16)[1:6])
    println("\n========== [SB_ADD_OPERAND_DUMP #$n] aliased Base.:+  tier=$tier ==========")
    for (tag, X) in (("A", A), ("B", B))
        nb = length(X.keys); nt = X.n_templates
        println("  [operand $tag] dims=$(collect(X.dims)) prefix=$(nb==0 ? 0 : length(first(X.keys))) blksize=$(X.blksize) nb=$nb nt=$nt dedup=$(round(nb/max(nt,1),digits=2))x")
        for i in 1:min(nb, max_blocks)
            println("     $(lpad(i,3))  $(X.keys[i]) → tid=$(X.alias_ids[i]) [s=$(round(X.scalars[i],digits=4))] vhash=$(_vh(X,X.alias_ids[i]))")
        end
        nb > max_blocks && println("     … ($(nb-max_blocks) more)")
    end
    if A.dims == B.dims
        ka = Set(A.keys); kb = Set(B.keys)
        same_set   = ka == kb
        same_order = A.keys == B.keys
        # Per-shared-key: does each operand resolve to the SAME template VALUE?
        # If yes everywhere, the two are equal up to (order + dedup-map) and the
        # merge is throwing away dedup that a smarter add could keep.
        shared = intersect(ka, kb)
        bpos = Dict(k => i for (i, k) in enumerate(B.keys))
        val_match = 0; val_diff = 0
        for (ia, k) in enumerate(A.keys)
            haskey(bpos, k) || continue
            ib = bpos[k]
            (_vh(A, A.alias_ids[ia]) == _vh(B, B.alias_ids[ib])) ? (val_match += 1) : (val_diff += 1)
        end
        println("  >> same key-SET=$same_set  same key-ORDER=$same_order  |A∩B|=$(length(shared))  shared-keys-with-EQUAL-template-value=$val_match  DIFFER=$val_diff")
        println("     (nb_A=$(length(A.keys)) nb_B=$(length(B.keys)) nt_A=$(A.n_templates) nt_B=$(B.n_templates))")
    else
        println("  >> dims differ: A=$(collect(A.dims)) B=$(collect(B.dims))")
    end
end

# SB_MRED_DIAG add-path counters: classify every aliased add during the eigsolve.
# axpy_match  — KrylovKit axpy!/scale! on identical schema (exact, compression kept)
# plus_match  — Base.:+ identical schema (exact, compression kept)
# plus_merge  — Base.:+ same dims, different schema → key-union, 1 template/block
#               (EXACT but compression destroyed)
# plus_dense  — Base.:+ mismatched classification → dense ITensor (EXACT, dense)
# All paths are mathematically exact; merge/dense only lose COMPRESSION, not value.
const _ADD_AXPY_MATCH = Ref(0)
const _ADD_PLUS_MATCH = Ref(0)
const _ADD_PLUS_MERGE = Ref(0)
const _ADD_PLUS_DENSE = Ref(0)

# Fallback: + on two aliased tensors with mismatched classification (e.g.
# different (P, N2) splits or different keys). Returns a dense ITensor.
function _add_aliased_via_dense(wA::WrappedAliasedBlockSparse,
                                 wB::WrappedAliasedBlockSparse)
    _ADD_PLUS_DENSE[] += 1
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1" && _ADD_DENSE_TRACE_COUNT[] < 10
        _ADD_DENSE_TRACE_COUNT[] += 1
        println("[DEEP _add_aliased_via_dense #$(_ADD_DENSE_TRACE_COUNT[])]  A=$(typeof(wA.aliased))  B=$(typeof(wB.aliased))  A.dims=$(wA.aliased.dims)  B.dims=$(wB.aliased.dims)  → DENSE")
    end
    A_dense = to_dense(wA.aliased)
    B_dense = to_dense(wB.aliased)
    tmpA = ITensors.ITensor(A_dense, wA.inds...)
    tmpB = ITensors.ITensor(B_dense, wB.inds...)
    return tmpA + tmpB
end

function Base.:+(wA::WrappedAliasedBlockSparse{TA,N,N2A,PA},
                 wB::WrappedAliasedBlockSparse{TB,N,N2B,PB}) where {TA,TB,N,N2A,PA,N2B,PB}
    return _add_aliased_via_dense(wA, wB)
end

# ── Element-wise subtraction ──────────────────────────────────────────────────
# Mirrors Base.:+ exactly (same axis-alignment + three tiers: match / merge /
# dense), with the templates combined by `-` instead of `+`. Stays aliased on
# the match and merge tiers (footprint preserved); densifies only when the
# (P,N2) split or dims differ — identical policy to addition. Reuses the shared
# _ADD_PLUS_* counters so SB_MRED_DIAG still reports densification regardless of
# whether the op was + or -.
function Base.:-(wA::WrappedAliasedBlockSparse{T,N,N2,P},
                 wB::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    if wA.inds != wB.inds
        perm = Vector{Int}(undef, N)
        @inbounds for i in 1:N
            j = findfirst(==(wA.inds[i]), wB.inds)
            if j === nothing
                return _subtract_aliased_via_dense(wA, wB)   # truly different inds
            end
            perm[i] = j
        end
        if perm != collect(1:N)
            B_aligned = SparseBackends.permutedims(wB.aliased, perm)
            wB = WrappedAliasedBlockSparse{T,N,N2,P}(B_aligned, wA.inds)
        end
    end
    A = wA.aliased; B = wB.aliased
    # Best case: schemas identical → subtract templates element-wise, keep schema.
    schemas_match = A.dims == B.dims && A.blksize == B.blksize &&
                    A.n_templates == B.n_templates &&
                    A.keys == B.keys && A.alias_ids == B.alias_ids &&
                    A.scalars == B.scalars
    if schemas_match
        _ADD_PLUS_MATCH[] += 1
        Kt = eltype(eltype(A.keys))
        new_templates = A.templates .- B.templates
        new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
            A.dims, A.blksize,
            new_templates, A.n_templates,
            copy(A.keys), copy(A.alias_ids), copy(A.scalars),
        )
        return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
    end
    # Same dims, mismatched schema → key-union merge (one template per block).
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1" && _CROSS_SCHEMA_TRACE_COUNT[] < 10
        _CROSS_SCHEMA_TRACE_COUNT[] += 1
        println("[DEEP cross-schema merge (sub) #$(_CROSS_SCHEMA_TRACE_COUNT[])]  A=Aliased{N=$(ndims(A)),nb=$(length(A.keys)),nt=$(A.n_templates)}  B=Aliased{nb=$(length(B.keys)),nt=$(B.n_templates)}")
    end
    A.dims != B.dims && return _subtract_aliased_via_dense(wA, wB)
    _ADD_PLUS_MERGE[] += 1
    Kt = eltype(eltype(A.keys))
    blksize = A.blksize
    a_lookup = Dict{NTuple{P,Kt}, Int}()
    for (i, k) in enumerate(A.keys); a_lookup[k] = i; end
    b_lookup = Dict{NTuple{P,Kt}, Int}()
    for (i, k) in enumerate(B.keys); b_lookup[k] = i; end
    all_keys = collect(NTuple{P,Kt}, union(keys(a_lookup), keys(b_lookup)))
    sort!(all_keys; by = k -> _prefix_lin(k, ntuple(i -> A.dims[i], Val(P))))
    nb = length(all_keys)
    new_templates = Vector{T}(undef, nb * blksize)
    new_alias_ids = collect(1:nb)
    new_scalars   = ones(T, nb)
    @inbounds for (idx, k) in enumerate(all_keys)
        off = (idx - 1) * blksize
        i_a = get(a_lookup, k, 0)
        i_b = get(b_lookup, k, 0)
        if i_a > 0
            αa = A.scalars[i_a]
            ta = (A.alias_ids[i_a] - 1) * A.blksize
            for j in 1:blksize
                new_templates[off + j] = αa * A.templates[ta + j]
            end
        else
            for j in 1:blksize
                new_templates[off + j] = zero(T)
            end
        end
        if i_b > 0
            αb = B.scalars[i_b]
            tb = (B.alias_ids[i_b] - 1) * B.blksize
            for j in 1:blksize
                new_templates[off + j] -= αb * B.templates[tb + j]
            end
        end
    end
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        A.dims, blksize,
        new_templates, nb,
        all_keys, new_alias_ids, new_scalars,
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
end

# Fallback: - on two aliased tensors with mismatched classification → dense.
function _subtract_aliased_via_dense(wA::WrappedAliasedBlockSparse,
                                     wB::WrappedAliasedBlockSparse)
    _ADD_PLUS_DENSE[] += 1
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1" && _ADD_DENSE_TRACE_COUNT[] < 10
        _ADD_DENSE_TRACE_COUNT[] += 1
        println("[DEEP _subtract_aliased_via_dense #$(_ADD_DENSE_TRACE_COUNT[])]  A=$(typeof(wA.aliased))  B=$(typeof(wB.aliased))  → DENSE")
    end
    A_dense = to_dense(wA.aliased)
    B_dense = to_dense(wB.aliased)
    tmpA = ITensors.ITensor(A_dense, wA.inds...)
    tmpB = ITensors.ITensor(B_dense, wB.inds...)
    return tmpA - tmpB
end

# Mismatched (P,N2) split → dense.
function Base.:-(wA::WrappedAliasedBlockSparse{TA,N,N2A,PA},
                 wB::WrappedAliasedBlockSparse{TB,N,N2B,PB}) where {TA,TB,N,N2A,PA,N2B,PB}
    return _subtract_aliased_via_dense(wA, wB)
end

# ITensors addition hook for ExternalStorage{<:WrappedAliasedBlockSparse}.
function ITensors._add(
    es_A::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
    es_B::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
)
    result = es_A.data + es_B.data
    # `+` may return either a WrappedAliasedBlockSparse (same-schema case) or
    # an ITensor (mismatched-schema → densified path).
    return result isa ITensors.ITensor ? result :
           ITensors._itensor_from_external_storage(result)
end

# Mixed add: Dense + Aliased (and the reverse). Lanczos / KrylovKit's add!! can
# call into here with one side already densified by an earlier matvec step.
# We materialise the aliased operand to a dense ITensor with matching index
# order, then forward to the standard Dense+Dense addition.
function _aliased_to_dense_itensor(es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse})
    w = es.data
    return ITensors.itensor(to_dense(w.aliased), w.inds...)
end

function ITensors._add(
    es_A::ITensors.NDTensors.Tensor,
    es_B::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
)
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        println("[DEEP _add(Dense, Aliased)] DENSIFY — Dense+Aliased mixed")
    end
    B_dense_T = _aliased_to_dense_itensor(es_B)
    A_T = ITensors.itensor(es_A)
    return A_T + B_dense_T
end

function ITensors._add(
    es_A::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
    es_B::ITensors.NDTensors.Tensor,
)
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        println("[DEEP _add(Aliased, Dense)] DENSIFY — Aliased+Dense mixed")
    end
    A_dense_T = _aliased_to_dense_itensor(es_A)
    B_T = ITensors.itensor(es_B)
    return A_dense_T + B_T
end

# ITensors subtraction hooks for ExternalStorage{<:WrappedAliasedBlockSparse},
# symmetric to the _add hooks above (routed via the core _subtract hook).
function ITensors._subtract(
    es_A::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
    es_B::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
)
    result = es_A.data - es_B.data
    # `-` may return either a WrappedAliasedBlockSparse (match/merge tiers) or an
    # ITensor (mismatched-schema → densified path).
    return result isa ITensors.ITensor ? result :
           ITensors._itensor_from_external_storage(result)
end

function ITensors._subtract(
    es_A::ITensors.NDTensors.Tensor,
    es_B::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
)
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        println("[DEEP _subtract(Dense, Aliased)] DENSIFY — Dense-Aliased mixed")
    end
    B_dense_T = _aliased_to_dense_itensor(es_B)
    A_T = ITensors.itensor(es_A)
    return A_T - B_dense_T
end

function ITensors._subtract(
    es_A::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse},
    es_B::ITensors.NDTensors.Tensor,
)
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        println("[DEEP _subtract(Aliased, Dense)] DENSIFY — Aliased-Dense mixed")
    end
    A_dense_T = _aliased_to_dense_itensor(es_A)
    B_T = ITensors.itensor(es_B)
    return A_dense_T - B_T
end

# fill! hook for zerovector!/broadcast scalar fill in KrylovKit.
function ITensors._external_fill!(T::ITensors.ITensor, storage::WrappedAliasedBlockSparse, x::Number)
    fill!(storage.aliased.templates, x)
    # When filling with 0, the per-block scalars don't matter; leave them alone.
    # When filling with a non-zero, value[i] = scalars[i] * x — which we can't
    # globally set to a uniform x unless all scalars are 1. Accept the
    # approximation (Krylov uses fill!(., 0) primarily).
    return T
end

function ITensors._external_map_storage!(
    f :: Function,
    R :: WrappedAliasedBlockSparse,
    A :: WrappedAliasedBlockSparse,
)
    _apply_elementwise!(f, R, A)
    return nothing
end
