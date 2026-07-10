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
@inline _add_dbg_enabled1() = false  # SB_ALIASED_DEBUG removed 2026-06 — flip to true here for debug output
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

# Pure axis PERMUTE: re-order Cw's axes so its `inds` tuple matches `Tw`'s
# exactly. Aliased analog of `recast_bs_to_template`. Both Cw and Tw must have
# the same Set(inds); the alias schema (keys, alias_ids, scalars, templates) is
# preserved unchanged by permutedims — this does NOT drop keys or re-dedup.
# Contrast `_snap_to_schema`, which rebuilds onto Tw's key-set (a lossy
# projection). Renamed from `recast_aliased_to_template` (the old name read like
# a projection); it is only an axis alignment.
function align_aliased_axes(Cw::WrappedAliasedBlockSparse{TC,N,N2c,Pc},
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
    if _RECAST_DBG && _RECAST_DBG_N[] < 6
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
# Public opt-in: value-dedup an aliased ITensor's templates (e.g. a ψ†ψ gram). Collapses
# templates with equal block-values (the AA kernel only dedups by input-pair, so a gram's
# group-difference duplicates persist). No-op on dense / non-aliased tensors. Mutates+returns.
function compress_aliased_templates!(t::ITensors.ITensor; atol::Real=1e-12)
    if ITensors.has_external_storage(t) && t.tensor.data isa WrappedAliasedBlockSparse
        _dedup_templates_by_value!(t.tensor.data.aliased; atol=atol)
    end
    return t
end

function _snap_to_schema(Hv::WrappedAliasedBlockSparse{T,N,N2,P},
                          v_schema::WrappedAliasedBlockSparse{T,N,N2,P};
                          dbg::Bool=false) where {T,N,N2,P}
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
    # ── ERROR TRACKING (pass dbg=true at a call site to enable; was
    # SB_ALIASED_SNAP_DBG=1) ────────────────────────────────────────────────
    # Track norm of Hv data that's dropped (keys in H but not in V).
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
            if _SNAP_DBG_COUNT[] < 5   # was SB_ALIASED_SNAP_DBG_MAX, default 5
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

# Snap a *dense* ITensor onto an aliased template's schema — the dense→aliased
# analogue of `_snap_to_schema` (which requires an already-aliased source).
# Needed by the `rr_dense_iter` diagnostic: the RR local eigensolve runs fully
# dense (dense φ, grams, H·v), so the Ritz vector comes out dense; there is no
# aliased tensor for `recast_to_phi`/`_snap_to_schema` to act on. This rebuilds
# the dense vector on `phi_ali`'s EXACT (keys, alias_ids, scalars, n_templates),
# dropping any support outside φ's schema (the projection back onto P's frame).
#
# Reuses existing machinery rather than hand-extracting blocks:
#   dense → blocksparse_from_dense (first P axes = keys, last N2 = dense blocks,
#           the same convention as φ's storage) → trivially-aliased (one template
#           per block) → `_snap_to_schema` against φ's schema.
# The dense array is read in φ's storage axis order (`dw.inds`) so keys and the
# per-block flatten match φ's exactly. No-op passthrough if φ is not aliased.
function snap_dense_to_aliased(d::ITensors.ITensor, phi_ali::ITensors.ITensor; dbg::Bool=false)
    ITensors.has_external_storage(phi_ali) || return d
    dw = ITensors.get_external_storage(phi_ali)
    dw isa WrappedAliasedBlockSparse || return d
    return _snap_dense_to_aliased_impl(d, dw; dbg=dbg)
end

function _snap_dense_to_aliased_impl(d::ITensors.ITensor,
                                     dw::WrappedAliasedBlockSparse{T,N,N2,P};
                                     dbg::Bool=false) where {T,N,N2,P}
    # Materialize d densely in φ's storage axis order (prefix axes first, dense
    # tail last), so blocksparse_from_dense produces φ-consistent keys/blocks.
    arr = Array(d, dw.inds...)
    bs  = blocksparse_from_dense(arr, Val(N2))          # NewBlockSparseSorted{T,N,N2,P}
    d_ali = _blocksparse_to_aliased(bs)                 # one template per block, scalar 1
    d_ali_w = WrappedAliasedBlockSparse(d_ali, dw.inds)
    snapped = _snap_to_schema(d_ali_w, dw; dbg=dbg)     # project onto φ's exact schema
    return ITensors._itensor_from_external_storage(snapped)
end

# Drop blocks whose prefix key is NOT in `allowed`, keeping the tensor's OWN
# dedup/template structure intact (templates + n_templates unchanged; kept
# alias_ids still index validly). Unlike _snap_to_schema this does NOT rebuild
# against another schema, so a dedup-1 input stays dedup-1 → the downstream
# KrylovKit add!! stays on the SAME (in-place) path. Discriminator for whether a
# Krylov-vector key-drop is exact when the storage schema is left unchanged
# (isolates the merge-add path from any genuine M^½-frame matrix-element loss).
# `allowed` is any container supporting `in` over the key NTuples.
function filter_keys_keepdedup(w::WrappedAliasedBlockSparse{T,N,N2,P}, allowed) where {T,N,N2,P}
    a = w.aliased
    keep = Int[]
    @inbounds for i in eachindex(a.keys)
        (a.keys[i] in allowed) && push!(keep, i)
    end
    length(keep) == length(a.keys) && return w   # nothing dropped — identity
    Kt = eltype(eltype(a.keys))
    # Compact templates: keep only those referenced by surviving blocks, remap
    # alias_ids → 1:n_kept (avoids orphaned templates inflating n_templates).
    old_aids = a.alias_ids[keep]
    used = sort!(unique(old_aids))
    remap = Dict{eltype(old_aids),eltype(old_aids)}()
    for (newt, oldt) in enumerate(used); remap[oldt] = eltype(old_aids)(newt); end
    n_kept = length(used)
    new_templates = Vector{T}(undef, n_kept * a.blksize)
    @inbounds for (newt, oldt) in enumerate(used)
        o = (Int(oldt) - 1) * a.blksize; nn = (newt - 1) * a.blksize
        @simd for j in 1:a.blksize; new_templates[nn + j] = a.templates[o + j]; end
    end
    new_aids = [remap[t] for t in old_aids]
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        a.dims, a.blksize, new_templates, n_kept,
        a.keys[keep], new_aids, a.scalars[keep],
    )
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, w.inds)
end

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
        push!(ali.alias_ids, _alias_id(eltype(ali.alias_ids), ali.n_templates))
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
# Was SB_ALIASED_TRACE — set from the top by dmrg(...; debug=true) (dmrg.jl),
# NOT an ENV var. Every trace print site below/elsewhere reads this Ref; each
# site's own fire-count budget (e.g. `< 10`, `< 30`) stays a hardcoded literal,
# not separately configurable.
const ALIASED_TRACE = Ref(false)
# DEBUGGING VAR — was SB_STEP env var. Current matvec-step index, set by
# abstractprojmpo.jl (SparseBackends.CURRENT_STEP[] = idx) before each operator
# in the matvec chain; consulted by trace prints below AND by
# _static_output_pref's (bondtype, step) STATIC_OUTPUT_PERM lookup. NOT an
# ENV var — a plain runtime Ref maintained across the process.
const CURRENT_STEP = Ref{Union{Nothing,Int}}(nothing)
# Was SB_RUN_LABEL env var — now threaded as a real `run_label::String` kwarg
# from dmrg(...) down through position!/product → makeL!/makeR!/contract,
# same as `debug`/`roofline`. This Ref is kept ONLY for the one remaining
# writer in contract_bs_dense.jl's shared BS dense-dense contraction kernel,
# which is reached via generic dispatch (no argument-carrying call chain from
# dmrg reaches it) — the same structural reason GEMM_DIMS_HIST/permute-profile
# needed a Ref instead of threading. Set alongside the threaded argument at
# each site that has one; not itself a user-facing knob.
const CURRENT_RUN_LABEL = Ref{String}("?")
const _NATIVE_DOT_HITS  = Ref(0)
const _DENSIFY_DOT_HITS = Ref(0)
const _RECAST_DBG       = false      # flip to true to enable recast perm dumps (was SB_RECAST_DBG)
const _RECAST_DBG_N     = Ref(0)     # one-shot recast perm dump counter, budgeted to 6
const _RECAST_PERMUTE_HITS = Ref(0)  # # of actual (non-identity) recast permutedims calls
const _ALIGN_OK         = Ref(0)     # SB_ALIASED_ALIGN_OUTPUT: native emitted aligned output
const _ALIGN_FALLBACK   = Ref(0)     # align attempted but kernel couldn't honor it → default order
const _ALIGN_DBG_N      = Ref(0)     # SB_ALIASED_ALIGN_DBG: one-shot fallback-exception dump
const _DOT_CHECK_MAXABS = Ref(0.0)   # SB_ALIASED_DOT_CHECK: max ABSOLUTE |native−dense| per dot
const _DOT_CHECK_MAXMAG = Ref(0.0)   # |dense| at the worst-abs dot (context for the abs error)
reset_dot_hits!() = (_NATIVE_DOT_HITS[] = 0; _DENSIFY_DOT_HITS[] = 0;
                     _DOT_CHECK_MAXABS[] = 0.0; _DOT_CHECK_MAXMAG[] = 0.0;
                     _RECAST_PERMUTE_HITS[] = 0; _ALIGN_OK[] = 0; _ALIGN_FALLBACK[] = 0)

# schema_dbg: print the aliased P-classification (sparse keys vs dense tail) of
# a tensor at a labelled point, to trace WHERE the constructed schema (e.g. P=3:
# site+2 link-channels) collapses (e.g. P=2: channels pushed to dense). Budgeted
# to 40 calls. Flip _SCHEMA_DBG to true here to enable (includes key/template detail).
const _SCHEMA_DBG = false
const _SCHEMA_DBG_BUDGET = Ref(40)
function schema_dbg(label, T)
    _SCHEMA_DBG || return nothing
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
        # fill = n_keys / prod(sparse-index dims): the genuine sparsity of the prefix
        # (fraction of the sparse key-space that is populated). Independent of dedup
        # (template sharing); should stay < 1 and the SPARSE index set should be stable
        # across M^{±1/2}/eigsolve even though dedup collapses to 1.
        psd = prod(Int[ITensors.dim(I) for I in ITensors.inds(T) if !(I in den)]; init=1)
        fill = length(a.keys) / psd
        println("[SCHEMA ", label, "]  P=", P, " N=", N,
                "  n_keys=", length(a.keys), " n_tmpl=", a.n_templates,
                " dedup=", round(length(a.keys)/max(a.n_templates,1), digits=2),
                "  prod(sparse_dims)=", psd, " fill=", round(fill, sigdigits=3),
                "  keyhash=", string(kh % 0x10000, base=16), " scalhash=", string(sh % 0x10000, base=16),
                "\n           SPARSE=", spk, "  DENSE=", dns)
        ks = a.keys
        println("           blksize=", a.blksize, "  KEYS(", length(ks), ") = ",
                length(ks) <= 64 ? ks : ks[1:64])
        nz = count(!iszero, a.templates); tot = length(a.templates)
        println("           TEMPLATES: n_tmpl=", a.n_templates, " blksize=", a.blksize,
                "  template_nonzeros=", nz, "/", tot, " (", round(100*nz/max(tot,1),digits=1), "% filled)")
        println("           alias_ids(key→tmpl) = ", length(a.alias_ids) <= 64 ? a.alias_ids : a.alias_ids[1:64])
        println("           scalars             = ", length(a.scalars) <= 64 ? round.(a.scalars, sigdigits=3) : round.(a.scalars[1:64], sigdigits=3))
    else
        cls = T isa ITensors.ITensor ?
            (ITensors.has_external_storage(T) ? string(typeof(ITensors.get_external_storage(T))) : "dense ITensor") :
            string(typeof(T))
        println("[SCHEMA ", label, "]  (", cls, ")")
    end
    return nothing
end

# TRACE_IMAGE: capture the constraint image of the TRUE φ (the 2-site wavefunction,
# before M^{±1/2}), so per-step matvec intermediates can be checked for keys that
# cannot belong to image(φ). Stores φ's sparse-leg ids (id→axis) and its key set.
const _PHI_IMAGE = Ref{Any}(nothing)
function capture_phi_image!(phi)
    # TRACE_IMAGE removed 2026-06 — this debug capture is now a permanent no-op.
    return nothing
    (phi isa ITensors.ITensor && ITensors.has_external_storage(phi)) || return nothing
    w = ITensors.get_external_storage(phi)
    w isa WrappedAliasedBlockSparse || return nothing
    den = dense_inds(w)
    sparse_legs = [I for I in ITensors.inds(phi) if !(I in den)]
    legid2axis = Dict{UInt64,Int}()
    for (ax, I) in enumerate(sparse_legs); legid2axis[ITensors.id(I)] = ax; end
    _PHI_IMAGE[] = (legid2axis = legid2axis,
                    keys = collect(w.aliased.keys),
                    tags = [string(ITensors.tags(I)) for I in sparse_legs])
    return nothing
end

# TRACE_IMAGE: print Hv's sparse legs + keys, and check each key against φ's image —
# project both onto the legs common (by Index id) to φ and report keys whose
# projection has NO matching φ key (i.e. cannot extend to image(φ)).
function check_image(label, Hv)
    # TRACE_IMAGE removed 2026-06 — this debug check is now a permanent no-op.
    return nothing
    img = _PHI_IMAGE[]
    (img === nothing || !(Hv isa ITensors.ITensor) || !ITensors.has_external_storage(Hv)) && return nothing
    w = ITensors.get_external_storage(Hv)
    w isa WrappedAliasedBlockSparse || return nothing
    den = dense_inds(w)
    hv_sparse = [I for I in ITensors.inds(Hv) if !(I in den)]
    hv_keys = collect(w.aliased.keys)
    common = Tuple{Int,Int}[]; common_tags = String[]   # (hv_axis, phi_axis)
    for (hax, I) in enumerate(hv_sparse)
        pid = ITensors.id(I)
        if haskey(img.legid2axis, pid)
            push!(common, (hax, img.legid2axis[pid])); push!(common_tags, string(ITensors.tags(I)))
        end
    end
    phi_proj = Set{Vector{Int}}()
    for k in img.keys; push!(phi_proj, Int[k[paxis] for (_, paxis) in common]); end
    n_out = 0; offenders = Vector{Int}[]
    for k in hv_keys
        sub = Int[k[hax] for (hax, _) in common]
        if !(sub in phi_proj)
            n_out += 1; length(offenders) < 6 && push!(offenders, sub)
        end
    end
    println("[IMAGE ", label, "]  sparse_legs=", [string(ITensors.tags(I)) for I in hv_sparse],
            "  n_keys=", length(hv_keys),
            "\n         keys=", length(hv_keys) <= 48 ? hv_keys : hv_keys[1:48],
            "\n         common_w/φ=", common_tags, "  out_of_image=", n_out, "/", length(hv_keys),
            n_out > 0 ? string("  ✗ offenders(proj)=", offenders) : "  ✓ all keys ∈ image(φ)")
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
# the downstream align_aliased_axes becomes a no-op.
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

# Build the canonical output-label ordering that makes the NEXT contraction's
# `permB` identity, given the operator `next_op` this result feeds into.
#   - output axis in next_op's dense TAIL    → red_dense     (FIRST, next-tail order)
#   - output axis in next_op's sparse PREFIX → shared_prefix (LAST,  next-prefix order)
#   - else (passes through next contraction) → keepB         (MIDDLE, indsA-then-indsB order)
# `labelsC_vec` is the kernel's already-computed output-label set, so the
# contraction's shared/output indices are NOT recomputed here. `indsA`/`indsB`
# are the original (pre-swap) operand inds, giving `keepB` its incoming order.
# Returns the ordered `Vector{Label}`, or `nothing` if next_op isn't aliased.
function _canon_labels_for_next(next_op, indsA, indsB, labelsC_vec::AbstractVector)
    (next_op isa ITensors.ITensor) || return nothing
    ITensors.has_external_storage(next_op) || return nothing
    st = ITensors.get_external_storage(next_op)
    (st isa WrappedAliasedBlockSparse) || return nothing
    PB = _abs_head_len(st)
    next_inds = ITensors.inds(next_op)
    outset = Set(labelsC_vec)
    red_dense     = Label[]
    shared_prefix = Label[]
    @inbounds for i in (PB + 1):length(next_inds)
        l = label_key_for_ind(next_inds[i]); (l in outset) && push!(red_dense, l)
    end
    @inbounds for i in 1:PB
        l = label_key_for_ind(next_inds[i]); (l in outset) && push!(shared_prefix, l)
    end
    placed = Set{Label}(red_dense); union!(placed, shared_prefix)
    keepB = Label[]
    @inbounds for I in indsA
        l = label_key_for_ind(I); (l in outset && !(l in placed)) && push!(keepB, l)
    end
    @inbounds for I in indsB
        l = label_key_for_ind(I); (l in outset && !(l in placed)) && push!(keepB, l)
    end
    return vcat(red_dense, keepB, shared_prefix)
end

# A-canonical sibling of `_canon_labels_for_next`, for the preserve_bs_output=true
# (Aliased×Dense → Aliased) matvec path. There this step's output becomes the
# NEXT step's *A* operand (the aliased accumulator Hv), so the permute to zero is
# the next kernel's `permA`, which wants A already laid out as
#   prefix: [keep_pref..., shared_pref...]   dense: [keepA..., red_dense...]
# i.e. the axes the next step CONTRACTS (shared with next_op) placed LAST within
# each region. We build that order as a reordering of THIS output's `indsC`,
# split-respecting (never crossing the prefix/dense boundary) so that
# `_align_output_order` accepts it. The dense tail additionally honours THIS
# kernel's requirement that A-origin and B-origin kept dims each be CONTIGUOUS
# (`_cdense_grouping_and_orders`). Returns a Vector{Index} (preferred order), or
# `nothing` if next_op carries no inds OR the next-reduced axes span both origins
# (the two constraints then conflict → genuine Phase-4 / strided-GEMM residual).
function _canon_inds_for_next_A(next_op, indsA, indsB, indsC, denseLinksC::Int)
    (next_op isa ITensors.ITensor) || return nothing
    nxt_ids = Set(ITensors.id(I) for I in ITensors.inds(next_op))
    isempty(nxt_ids) && return nothing
    aids = Set(ITensors.id(I) for I in indsA)   # this-step A (accumulator) origin
    N  = length(indsC)
    Pc = N - denseLinksC
    in_next(I) = ITensors.id(I) in nxt_ids       # contracted at the NEXT step
    from_A(I)  = ITensors.id(I) in aids          # output leg's THIS-step operand origin

    # Prefix (sparse): [keep_pref..., shared_pref...] — next-contracted LAST.
    # No A/B-contiguity constraint applies to the prefix.
    keep_pre = ITensors.Index[indsC[i] for i in 1:Pc if !in_next(indsC[i])]
    red_pre  = ITensors.Index[indsC[i] for i in 1:Pc if  in_next(indsC[i])]

    # Dense tail — two simultaneous constraints: (a) THIS kernel needs A-origin and
    # B-origin kept dims each contiguous; (b) NEXT permA wants next-contracted axes
    # last. Both hold iff the next-contracted axes live in ONE origin block, as its
    # suffix, and that block is placed last. Order each origin block
    # [not-next..., next-reduced...], then put the next-reduced-owning block last.
    den = (Pc+1):N
    A_keep = ITensors.Index[indsC[i] for i in den if  from_A(indsC[i]) && !in_next(indsC[i])]
    A_red  = ITensors.Index[indsC[i] for i in den if  from_A(indsC[i]) &&  in_next(indsC[i])]
    B_keep = ITensors.Index[indsC[i] for i in den if !from_A(indsC[i]) && !in_next(indsC[i])]
    B_red  = ITensors.Index[indsC[i] for i in den if !from_A(indsC[i]) &&  in_next(indsC[i])]
    dense_order = if isempty(A_red) && isempty(B_red)
        vcat(A_keep, B_keep)                  # nothing reduced next; order irrelevant
    elseif isempty(A_red)
        vcat(A_keep, B_keep, B_red)           # next-reduced only in B → B-block last
    elseif isempty(B_red)
        vcat(B_keep, A_keep, A_red)           # next-reduced only in A → A-block last
    else
        # next-reduced spans BOTH origins: can't satisfy A/B-contiguity AND
        # reduced-last with contiguous blocks. Emit the INTERLEAVED order (keeps
        # first, both reduction groups LAST) so the next step's permA is still the
        # identity; the kernel realizes this layout via a strided scatter (lmap).
        # Gated to the serial BLAS path (the only one wired for the strided write);
        # elsewhere fall back to the kernel's permA (return nothing). Interleaving
        # itself hardened 2026-06 — always on (was SB_ALIASED_INTERLEAVE,
        # default-on knob); SB_ALIASED_NTHREADS stays a live, genuine knob.
        if get(ENV, "SB_ALIASED_NTHREADS", "1") == "1"
            vcat(A_keep, B_keep, A_red, B_red)
        else
            return nothing
        end
    end
    return vcat(keep_pre, red_pre, dense_order)
end

# Full-chain reduction-rank ordering (the GENERATOR used to derive the static
# STATIC_OUTPUT_PERM tables). Sort each region (sparse prefix [1:Pc], dense tail
# [Pc+1:end]) so that legs contracted SOONEST in the remaining chain land LAST and
# never-contracted (output) legs come first. Stable sort ⇒ equal-rank legs keep
# their incoming order, giving a consistent relative order at every node (so each
# step's permA is the identity). Sorting within each region never crosses the
# prefix/dense split, so `_align_output_order` accepts it; any A/B interleave in
# the dense tail is realized by the kernel's `lmap` strided write. `remaining_ops`
# = the chain operators AFTER the current step (OneITensors ignored).
function _canon_by_rank(indsC, denseLinksC::Int, remaining_ops)
    idsets = [Set(ITensors.id(I) for I in ITensors.inds(op))
              for op in remaining_ops if op isa ITensors.ITensor]
    function rkey(I)
        d = ITensors.id(I)
        @inbounds for k in 1:length(idsets)
            (d in idsets[k]) && return -k     # contracted at remaining step k → -k (soonest = last)
        end
        return typemin(Int)                   # never contracted → first
    end
    N = length(indsC); Pc = N - denseLinksC
    pre = sort(collect(indsC[1:Pc]);   by=rkey, alg=Base.Sort.MergeSort)
    den = sort(collect(indsC[Pc+1:N]); by=rkey, alg=Base.Sort.MergeSort)
    return vcat(pre, den)
end

# Permute-minimizing output order (NRED; hardened always-on). Unlike _canon_by_rank (which
# sorts ALL dense by reduction rank → interleaves A/B origins → forces the 9.6 GiB
# output permute), this PRESERVES the [keepA; keepB] dense grouping (⇒ mode AthenB,
# out_dense_perm = identity) and sorts ONLY:
#   • the sparse PREFIX axes by is_reduced_next (reduced-next last)  — free key-walk reorder
#   • the keepB dense legs by is_reduced_next (reduced-next last)    — realized by the
#     existing cheap permB on the small operator (no output permute)
# keepA dense legs keep their incoming order (so THIS step's permA stays minimal).
# Result: next step's permA is identity for every reduced leg EXCEPT a carried
# keepA-dense leg (the terminal ×Renv residual). `next_op` = the next chain operator.
function _canon_keepB_red(indsA, indsC, denseLinksC::Int, next_op)
    nset = (next_op isa ITensors.ITensor) ?
           Set(ITensors.id(I) for I in ITensors.inds(next_op)) : Set{UInt64}()
    _isred(I) = ITensors.id(I) in nset      # reduced at the NEXT step
    N = length(indsC); Pc = N - denseLinksC
    pre   = sort(collect(indsC[1:Pc]); by=_isred, alg=Base.Sort.MergeSort)  # reduced-next last
    dense = collect(indsC[Pc+1:N])
    Aset  = Set(ITensors.id(I) for I in indsA)
    keepA = [I for I in dense if  (ITensors.id(I) in Aset)]                 # A-order preserved
    keepB = sort([I for I in dense if !(ITensors.id(I) in Aset)];           # keepB reduced-next last
                 by=_isred, alg=Base.Sort.MergeSort)
    return vcat(pre, keepA, keepB)
end

# Hard-coded per-(bond-type, step) output-order permutation table. Each value is a
# permutation applied to the kernel's default `output_inds` order: the desired
# (reduction-last) output is `indsC[perm]`. Derived once via the `_canon_by_rank`
# generator (SB_PERM_CAPTURE) and baked here. Empty ⇒ the generator/legacy path
# runs (so capture works before the table is filled). bond-type ∈ {:left,:bulk,:right}.
const STATIC_OUTPUT_PERM = Dict{Tuple{Symbol,Int}, Vector{Int}}(
    # Baked from the SB_PERM_CAPTURE generator (N=2, KL plaquette). Uniform across
    # all bulk bonds; the length-guard in _static_output_pref falls back for any
    # step whose output leg-count differs (e.g. sweep-1 warmup, before maxdim fills).
    (:bulk, 1) => [3, 1, 2, 6, 7, 4, 5],
    (:bulk, 2) => [2, 1, 3, 4, 7, 5, 6],
    (:bulk, 3) => [1, 2, 3, 4, 7, 5, 6],
    (:bulk, 4) => [1, 2, 4, 3, 6, 5],
    (:left, 1) => [1, 2, 4, 5, 3],
    (:left, 2) => [1, 2, 3, 5, 4],
    (:left, 3) => [1, 2, 3, 4],
    (:right, 1) => [1, 2, 4, 5, 3],
    (:right, 2) => [1, 2, 3, 5, 4],
    (:right, 3) => [1, 2, 3, 4],
)
# Stateless lookup of the static output permutation for a given (bond-type,
# step) — was _static_output_pref, gated on SB_IN_MATVEC (an ENV bracket set
# around the matvec loop, since this used to read the coordinate from ENV and
# had no other way to know whether it was stale). Now the matvec loop (the
# only caller with a real answer, in abstractprojmpo.jl) passes bondtype/step
# directly and threads the *result* down as `output_perm`; no other caller
# can reach this by accident, so no gate is needed at all.
function static_output_perm(bondtype::Symbol, step::Int)::Union{Nothing,Vector{Int}}
    isempty(STATIC_OUTPUT_PERM) && return nothing
    return get(STATIC_OUTPUT_PERM, (bondtype, step), nothing)
end

# Reorder an aliased φ so the legs it shares with `op` sit LAST within its sparse
# prefix and within its dense tail — i.e. φ laid out A-canonically for the
# contraction φ·op. Used (in dmrg `position!`) to put the eigensolver seed /
# recast template into the order the matvec's FIRST step (×Lenv) wants, so that
# step's permA is the identity across every Krylov iteration. Layout-only (a
# permute of φ's storage); the contraction math is unchanged. Returns φ unchanged
# if it isn't aliased, op carries no inds, or φ is already in that order.
function reorder_aliased_for_op(phi::ITensors.ITensor, op)
    ITensors.has_external_storage(phi) || return phi
    w = ITensors.get_external_storage(phi)
    (w isa WrappedAliasedBlockSparse{T,N,N2,P} where {T,N,N2,P}) || return phi
    (op isa ITensors.ITensor) || return phi
    op_ids = Set(ITensors.id(I) for I in ITensors.inds(op))
    isempty(op_ids) && return phi
    Pc = _abs_head_len(w)
    ph = collect(ITensors.inds(phi))
    Nc = length(ph)
    shared(I) = ITensors.id(I) in op_ids
    pre_keep = ITensors.Index[ph[i] for i in 1:Pc     if !shared(ph[i])]
    pre_red  = ITensors.Index[ph[i] for i in 1:Pc     if  shared(ph[i])]
    den_keep = ITensors.Index[ph[i] for i in (Pc+1):Nc if !shared(ph[i])]
    den_red  = ITensors.Index[ph[i] for i in (Pc+1):Nc if  shared(ph[i])]
    target = vcat(pre_keep, pre_red, den_keep, den_red)
    ph == target && return phi
    # perm[i] = old position of the axis that belongs at new position i. By
    # construction prefix↔dense never cross, so the alias schema is preserved.
    perm = Int[findfirst(==(target[i]), ph) for i in 1:Nc]
    new_ali = permutedims(w.aliased, perm)
    new_w = WrappedAliasedBlockSparse{eltype(w), Nc, Nc - Pc, Pc}(new_ali, Tuple(target))
    return ITensors._itensor_from_external_storage(new_w)
end

# φ reorder: lay out φ's prefix and dense blocks by full-chain reduction rank
# (the operators `ops` are the matvec chain; a leg contracted soonest → last),
# matching the per-step output ordering so step-1's permA is the identity. Called
# once per bond from dmrg `position!` — a layout-only permute of φ's storage.
function reorder_aliased_by_rank(phi::ITensors.ITensor, ops)
    ITensors.has_external_storage(phi) || return phi
    w = ITensors.get_external_storage(phi)
    (w isa WrappedAliasedBlockSparse{T,N,N2,P} where {T,N,N2,P}) || return phi
    idsets = [Set(ITensors.id(I) for I in ITensors.inds(op)) for op in ops if op isa ITensors.ITensor]
    isempty(idsets) && return phi
    rkey(I) = begin
        d = ITensors.id(I)
        @inbounds for k in 1:length(idsets); (d in idsets[k]) && return -k; end
        typemin(Int)
    end
    Pc = _abs_head_len(w); ph = collect(ITensors.inds(phi)); Nc = length(ph)
    pre = sort(ph[1:Pc];     by=rkey, alg=Base.Sort.MergeSort)
    den = sort(ph[Pc+1:Nc];  by=rkey, alg=Base.Sort.MergeSort)
    target = vcat(pre, den)
    perm = Int[findfirst(==(target[i]), ph) for i in 1:Nc]
    if _roofline_on()   # was SB_PERM_CAPTURE; folded into the roofline switch
        println("[PERMCAP-PHI] bondtype=", get(ENV, "SB_BONDTYPE", "?"),
                " ndims=", Nc, " Pc=", Pc, " perm=", perm)
    end
    ph == target && return phi
    new_ali = permutedims(w.aliased, perm)
    new_w = WrappedAliasedBlockSparse{eltype(w), Nc, Nc - Pc, Pc}(new_ali, Tuple(target))
    return ITensors._itensor_from_external_storage(new_w)
end

function wrapped_contract_aliased(
    A :: WrappedTensorTypes{TA,NA},
    B :: WrappedTensorTypes{TB,NB};
    preserve_bs_output::Bool=false,
    preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    next_op=nothing,
    remaining_ops=nothing,
    output_perm::Union{Nothing,Vector{Int}}=nothing,
    in_position::Bool=false,
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

    if false  # SB_IDX_DBG removed 2026-06 — flip to true here for debug output
        _dA = Set(denseA); _dB = Set(denseB); _dC = Set(denseC)
        # tag each index: S=sparse-prefix, D=dense-tail (per its own operand)
        _tg(I, dset) = string(I in dset ? "D:" : "S:", ITensors.dim(I), "|",
                              replace(string(ITensors.tags(I)),"\""=>""), "|p", ITensors.plev(I))
        _sA = Set(indsA); _sB = Set(indsB); _sC = Set(indsC)
        _red   = [I for I in indsA if I in _sB && !(I in _sC)]
        _keepA = [I for I in indsA if I in _sC]
        _keepB = [I for I in indsB if I in _sC && !(I in _sA)]
        println("\n[IDX bond=", get(ENV,"SB_BONDTYPE","?"), " step=", CURRENT_STEP[],
                "]  (S=sparse-prefix, D=dense-tail)",
                "\n   A     =", [_tg(I,_dA) for I in indsA],
                "\n   B     =", [_tg(I,_dB) for I in indsB],
                "\n   C     =", [_tg(I,_dC) for I in indsC],
                "\n   reduced=", [string(_tg(I,_dA),"/",I in _dB ? "D" : "S","B") for I in _red],
                "\n   keepA =", [_tg(I,_dC) for I in _keepA], "  keepB=", [_tg(I,_dC) for I in _keepB])
        flush(stdout)
    end

    @timeit TIMER "classify_contraction" begin
        denseLinksC = length(denseC)

        sc       = scratch()
        labelsA_vec = fill_labels!(sc.labelsA, indsA)
        labelsB_vec = fill_labels!(sc.labelsB, indsB)
        labelsC_vec = fill_labels!(sc.labelsC, indsC)
    end

    @timeit TIMER "!preserve_bs_output_path" begin
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
                    #   2. Link-fusion heuristic (the permanent fallback) — data-driven
                    #      order so the NEXT sparse-H × dense kernel sees its B input as
                    #      [red_dense, keepB, shared_prefix] with permB = identity.
                    #      Mapping (from SB_PERMB_DBG diagnosis):
                    #        upstream's b_other   → next call's red_dense
                    #        upstream's keepA     → next call's keepB
                    #        upstream's b_to_next → next call's keepB
                    #        upstream's c_prefix  → next call's shared_prefix
                    #      So the right canonical order is
                    #        [b_other, keepA_labs, b_to_next, c_prefix_labs].
                    # Priority 0: explicit `next_op` — derive the order from this
                    # step's own output labels (labelsC_vec) + next_op's schema, so
                    # the caller needn't recompute the shared/output index set.
                    _next_canon = next_op === nothing ? nothing :
                        _canon_labels_for_next(next_op, indsA, indsB, labelsC_vec)
                    if _next_canon !== nothing
                        canon_labels = _next_canon
                    elseif preferred_output_labels !== nothing
                        # Accept either Vector{Index} (natural for callers) or the
                        # internal label tuple format. Convert as needed.
                        canon_labels = if !isempty(preferred_output_labels) &&
                                        first(preferred_output_labels) isa ITensors.Index
                            [label_key_for_ind(I) for I in preferred_output_labels]
                        else
                            collect(preferred_output_labels)
                        end
                    else
                        # Fallback when no explicit next_op / preferred order is given
                        # (e.g. the last step in a chain, or a non-aliased next op):
                        # link-fusion-aware heuristic that puts the axis the NEXT
                        # sparse-H × dense kernel will keep on its B-site (non-Link,
                        # plev 0) in the keepB slot, so its permB stays identity.
                        # Formerly gated by SB_FUSE_LINKS; now the permanent default
                        # (the precise `next_op` branch above supersedes it whenever
                        # the next operator is known).
                        is_b_site_to_next(l) = begin
                            I = label_to_ind[l]
                            ts = string(ITensors.tags(I))
                            !contains(ts, "Link") && ITensors.plev(I) == 0
                        end
                        b_to_next = [l for l in keepB_labs if  is_b_site_to_next(l)]
                        b_other   = [l for l in keepB_labs if !is_b_site_to_next(l)]
                        canon_labels = vcat(b_other, keepA_labs, b_to_next, c_prefix_labs)
                    end
                    canon_inds = ITensors.Index[label_to_ind[l] for l in canon_labels]
                    canon_dims = ntuple(i -> ITensors.dim(canon_inds[i]), length(canon_inds))
                    if false  # DEBUG_HINT removed 2026-06 — flip to true here for debug output
                        println(" HINT (kernel) canon output order: ", canon_inds)
                    end
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
                    contract_aliased_dense_to_dense!(C_canon, canon_labels, Arep_a, labelsA_a, Brep_a, labelsB_a; in_position=in_position)
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
                # Retired 2026-06 — SB_ALIASED_NATIVE_DOT/SB_ALIASED_DOT_CHECK,
                # both default OFF, never exercised. Native scalar-dot fast path
                # (_aliased_lean_scalar_dot call) commented out below; densify
                # fallback right after this block is the live path.
                # if get(ENV, "SB_ALIASED_NATIVE_DOT", "0") == "1" && length(indsC) == 0
                #     s = _aliased_lean_scalar_dot(Arep, labelsA_vec, Brep, labelsB_vec)
                #     if s !== nothing
                #         _NATIVE_DOT_HITS[] += 1
                #         sc = convert(TC, s)
                #         if get(ENV, "SB_ALIASED_DOT_CHECK", "0") == "1"
                #             refC = ITensors.contract(ITensors.ITensor(to_dense(Arep), indsA...),
                #                                     ITensors.ITensor(to_dense(Brep), indsB...))
                #             aerr = abs(sc - refC[])
                #             if aerr > _DOT_CHECK_MAXABS[]
                #                 _DOT_CHECK_MAXABS[] = aerr; _DOT_CHECK_MAXMAG[] = abs(refC[])
                #             end
                #         end
                #         return ITensors.ITensor(sc)
                #     end
                # end
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
                contract_bs_dense_to_dense!(C_data, labelsC_vec, Arep_bs, labelsA_vec, Brep_bs, labelsB_vec; in_position=in_position)
            elseif Arep_bs isa AbstractArray && Brep_bs isa NewBlockSparseSorted
                contract_bs_dense_to_dense!(C_data, labelsC_vec, Brep_bs, labelsB_vec, Arep_bs, labelsA_vec; in_position=in_position)
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
        if ALIASED_TRACE[]
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
        # for BS kernels: _contract_shared_hint!).
        hint_labels = output_inds_hint === nothing ? nothing :
            Set(label_key_for_ind(I) for I in output_inds_hint)
        # #recast-kill (gated SB_ALIASED_ALIGN_OUTPUT): emit the output already in
        # the template's axis order so the downstream align_aliased_axes is
        # a no-op. Reorders indsC/labelsC_vec within the prefix and dense blocks,
        # then lets contract! write in that order.
        # try/catch: if the kernel can't
        # produce that order (e.g. interleaved A/B kept dims), fall back to the
        # default order (recast still fixes it ⇒ correctness preserved).
        # Output ordering, in priority:
        #   1. explicit caller preferred_output_labels (recast-align),
        #   2. the hard-coded static table by (bond-type, step) — the production path,
        #   3. the full-chain-rank generator (SB_PERM_CAPTURE only) used to (re)derive
        #      the table; prints the permutation per (bond-type, step).
        # The chosen order is emitted by the kernel via _align_output_order (+ lmap for
        # interleaved layouts), making the NEXT step's permA the identity.
        _pref_align = preferred_output_labels
        # SB_OUTSTAT_NATURAL=1 (experiment): skip the interleaving static schedule so
        # the kernel emits its NATURAL [keepA;keepB] order (out_dense_perm → identity,
        # no 9.6GiB output permute). The cost may migrate to the NEXT step's permA;
        # this measures whether the relayout is reducible or just moves. recast_to_phi
        # realigns the final output to φ ⇒ correctness preserved regardless.
        _os_natural = get(ENV, "SB_OUTSTAT_NATURAL", "0") == "1"
        # NRED permute-minimizing output order (HARDENED 2026-07, was the SB_OUTSTAT_NRED
        # env gate — now unconditional whenever a next_op is available): sparse prefix +
        # keepB sorted reduced-next-last, keepA grouping preserved. keepB order is realized
        # by the cheap permB (out_dense_perm stays identity); next permA is identity except
        # the terminal carried-keepA residual. Only fires for aliased contractions (this is
        # the aliased kernel); bit-identical E, ~2.2x faster matvec vs the old _canon_by_rank
        # order (kills the reduction-stationary scatter/accum path). See memory
        # project_nred_output_ordering.
        if _pref_align === nothing && next_op !== nothing
            _pref_align = _canon_keepB_red(indsA, indsC, denseLinksC, next_op)
        end
        if _pref_align === nothing && !_os_natural && output_perm !== nothing &&
           length(output_perm) == length(indsC)
            _pref_align = indsC[output_perm]
        end
        if _pref_align === nothing && !_os_natural && remaining_ops !== nothing
            _pref_align = _canon_by_rank(indsC, denseLinksC, remaining_ops)
            if _roofline_on()   # was SB_PERM_CAPTURE; folded into the roofline switch
                _ic = collect(indsC)
                _perm = Int[findfirst(==(_pref_align[i]), _ic) for i in 1:length(_ic)]
                println("[PERMCAP] bondtype=", get(ENV, "SB_BONDTYPE", "?"),
                        " step=", CURRENT_STEP[],
                        " ndims=", length(_ic), " Pc=", length(_ic) - denseLinksC,
                        " perm=", _perm)
            end
        end
        if _pref_align !== nothing
            ord = _align_output_order(indsC, denseLinksC, _pref_align)
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
                    if false && _ALIGN_DBG_N[] < 4  # SB_ALIASED_ALIGN_DBG removed 2026-06 — flip to true here for debug output
                        _ALIGN_DBG_N[] += 1
                        _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
                        println("[ALIGN FALLBACK #", _ALIGN_DBG_N[], "] Pc=", length(indsC)-denseLinksC,
                                " ord=", ord, "\n   ERROR: ", sprint(showerror, e),
                                "\n   indsC(default)=", [_tg(I) for I in indsC],
                                "\n   preferred     =", [_tg(I) for I in _pref_align])
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
                  next_op=nothing,
                  remaining_ops=nothing,
                  output_perm::Union{Nothing,Vector{Int}}=nothing,
                  in_position::Bool=false,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint, next_op=next_op,
                                    remaining_ops=remaining_ops, output_perm=output_perm,
                                    in_position=in_position)
end

function contract(A::WrappedTensorTypes, B::WrappedAliasedBlockSparse;
                  preserve_bs_output::Bool=false,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                  next_op=nothing,
                  remaining_ops=nothing,
                  output_perm::Union{Nothing,Vector{Int}}=nothing,
                  in_position::Bool=false,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint, next_op=next_op,
                                    remaining_ops=remaining_ops, output_perm=output_perm,
                                    in_position=in_position)
end

function contract(A::WrappedAliasedBlockSparse, B::WrappedAliasedBlockSparse;
                  preserve_bs_output::Bool=false,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                  next_op=nothing,
                  remaining_ops=nothing,
                  output_perm::Union{Nothing,Vector{Int}}=nothing,
                  in_position::Bool=false,
                  kwargs...)
    return wrapped_contract_aliased(A, B; preserve_bs_output, preferred_output_labels,
                                    output_inds_hint=output_inds_hint, next_op=next_op,
                                    remaining_ops=remaining_ops, output_perm=output_perm,
                                    in_position=in_position)
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
    Abackend :: Union{Symbol,Backend},
    Bbackend :: Union{Symbol,Backend};
    denseLinksA :: Union{Nothing,Int} = nothing,
    denseLinksB :: Union{Nothing,Int} = nothing,
    preserve_bs_output :: Bool = true,
    preferred_output_labels :: Union{Nothing,AbstractVector} = nothing,
    next_op = nothing,
)
    Ab = to_backend(Abackend)
    Bb = to_backend(Bbackend)
    if Ab === DENSE && Bb === DENSE
        return ITensors.contract(A, B)
    end

    # Wrap inputs
    @timeit TIMER "wrap_inputs" begin
        Aw = if Ab === ALIASED
            wrap_itensor_aliased(A; denseLinks=denseLinksA)
        else
            wrap_itensor(A; backend=Ab, denseLinks=denseLinksA)
        end
        Bw = if Bb === ALIASED
            wrap_itensor_aliased(B; denseLinks=denseLinksB)
        else
            wrap_itensor(B; backend=Bb, denseLinks=denseLinksB)
        end
    end

    @timeit TIMER "wrapped_contract_aliased_call" begin
        Cw = wrapped_contract_aliased(Aw, Bw; preserve_bs_output, preferred_output_labels, next_op)
    end
    @timeit TIMER "wrap_output" begin
        Cw isa ITensors.ITensor && return Cw   # P_C = 0
        if Cw isa WrappedAliasedBlockSparse && _abs_head_len(Cw) == 0
            dense_data = to_dense(Cw.aliased)
            result = ITensors.ITensor(dense_data, Cw.inds...)
        end
        result = ITensors._itensor_from_external_storage(Cw)
    end
    return result
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


# ─────────────────────────────────────────────────────────────────────────────
# 12.  ITensors extension hooks
# ─────────────────────────────────────────────────────────────────────────────

function ITensors.dag(es::ITensors.ExternalStorage{<:WrappedAliasedBlockSparse}; kwargs...)
    data    = es.data
    newdata = ITensors.dag(data)
    return ITensors._itensor_from_external_storage(newdata)
end

# norm = sqrt(sum_i scalars[i]^2 * ||template[alias_ids[i]]||^2)
# FUNCTION BARRIER (2026-07-05): the loop MUST run in a function dispatched on the
# CONCRETE AliasedBlockSparse type. The wrapper WrappedAliasedBlockSparse{T,N,N2,P}
# drops the key/alias-id type params {K,AI}, so `w.aliased` is only partially
# inferred → `alias_ids[i]` is an abstract Integer → `toff = (alias_ids[i]-1)*blksize`
# is abstractly typed → every `toff + j` in the inner loop BOXES. Measured: the
# inlined version allocated 8.76 GiB / 9.44 s across a 3-sweep N=12 md40 run. Passing
# `w.aliased` through this barrier makes the loop concrete + allocation-free (mirrors
# the _alias_inner fastpath; see project_inplace_add_redherring for the same class of bug).
@inline function _alias_norm(A::AliasedBlockSparse{T}) where {T}
    s = zero(real(T))
    tpl = A.templates; sc = A.scalars; aid = A.alias_ids; bs = A.blksize
    @inbounds for i in eachindex(A.keys)
        α    = sc[i]
        toff = (Int(aid[i]) - 1) * bs
        @simd for j in 1:bs
            s += abs2(α * tpl[toff + j])
        end
    end
    return sqrt(s)
end
LinearAlgebra.norm(w::WrappedAliasedBlockSparse) = @timeit TIMER "vecop.norm" _alias_norm(w.aliased)

# Recycle a DEAD aliased ITensor's template buffer back into the `pending` pool
# (:aliased_ws_pending, keyed by element type — same slot the aliased×dense kernel
# pops for its output) so the next contract reuses its capacity instead of growing a
# fresh result-sized buffer from empty. Used ONLY for consumed matvec intermediates
# in the dense-H aliased-ψ chain (caller guards `_Hv_prev !== v` so the input Krylov
# vector, owned by KrylovKit, is never recycled). Numerics unchanged: the next
# contract resize!s + overwrites (β=0) this buffer exactly as it would a fresh one.
# SAFETY: caller must guarantee `w` is dead (no live refs); the kernel's zero-copy
# finalize hands each output a DISTINCT buffer, so a consumed operand's buffer never
# aliases the live result.
function recycle_aliased_pending!(w::ITensors.ITensor)
    ITensors.has_external_storage(w) || return nothing
    ext = ITensors.get_external_storage(w)
    ext isa WrappedAliasedBlockSparse || return nothing
    buf = ext.aliased.templates
    isempty(buf) && return nothing
    task_local_storage((:aliased_ws_pending, eltype(buf)), buf)
    return nothing
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
  @timeit TIMER "vecop.scale" begin
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
# FUNCTION BARRIER for the Base.:+ / Base.:- MERGE tiers (2026-07-05 audit, same
# boxing class as norm/inner/axpby): those methods dispatch on the wrapper
# {T,N,N2,P} which drops {K,AI}, so `wA.aliased` is UnionAll-abstract and the inline
# merge loop's `toff = (alias_ids[i]-1)*blksize` boxes every element. This kernel
# takes the concrete field Vectors so the loop specializes (allocation-free).
# sgnB = +1 for `+`, -1 for `-`. Cold in dense-H aliased-ψ (all Krylov adds go
# through the in-place axpby barrier), but hot in both-aliased runs where
# cross-schema merges dominate (see project_aliased_lanczos_cross_schema_merge).
# ⚠ TODO: mechanical extraction of the former inline Base.:+/:- merge loops
# (equivalent by construction; compiles + dense-H bit-identical). NOT yet exercised
# end-to-end — the merge tier is cold in dense-H, and the both-aliased run that
# hits it dies first on a pre-existing UInt8 alias-id overflow. Validate in the
# separate aliased-H / aliased-ψ runs.
@inline function _merge_add!(new_templates::Vector{T}, all_keys::AbstractVector,
        a_lookup, b_lookup,
        tA::Vector{T}, aidA::AbstractVector, sA::Vector{T},
        tB::Vector{T}, aidB::AbstractVector, sB::Vector{T},
        bs::Int, sgnB::T) where {T}
    @inbounds for (idx, k) in enumerate(all_keys)
        off = (idx - 1) * bs
        i_a = get(a_lookup, k, 0)
        i_b = get(b_lookup, k, 0)
        if i_a > 0
            αa = sA[i_a]; ta = (Int(aidA[i_a]) - 1) * bs
            @simd for j in 1:bs
                new_templates[off + j] = αa * tA[ta + j]
            end
        else
            @simd for j in 1:bs
                new_templates[off + j] = zero(T)
            end
        end
        if i_b > 0
            αb = sB[i_b]; tb = (Int(aidB[i_b]) - 1) * bs
            @simd for j in 1:bs
                new_templates[off + j] += sgnB * αb * tB[tb + j]
            end
        end
    end
    return nothing
end

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
            B_aligned = @timeit TIMER "aliasadd.align_permute" SparseBackends.permutedims(wB.aliased, perm)
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
        return @timeit TIMER "aliasadd.match" begin
            Kt = eltype(eltype(A.keys))
            new_templates = A.templates .+ B.templates
            new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
                A.dims, A.blksize,
                new_templates, A.n_templates,
                copy(A.keys), copy(A.alias_ids), copy(A.scalars),
            )
            WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
        end
    end
    # Same dims but mismatched schemas (different keys, alias_ids, or scalars).
    # Merge via key union with one template per resulting block (trivial dedup,
    # no template-level compression but storage stays aliased so the result
    # can keep flowing through aliased kernels).
    if ALIASED_TRACE[] && _CROSS_SCHEMA_TRACE_COUNT[] < 10
        _CROSS_SCHEMA_TRACE_COUNT[] += 1
        println("[DEEP cross-schema merge #$(_CROSS_SCHEMA_TRACE_COUNT[])]  A=Aliased{N=$(ndims(A)),nb=$(length(A.keys)),nt=$(A.n_templates)}  B=Aliased{nb=$(length(B.keys)),nt=$(B.n_templates)}")
    end
    A.dims != B.dims && return _add_aliased_via_dense(wA, wB)
    if false && _ADD_PLUS_MERGE[] < 2  # SB_ADD_STACK removed 2026-06 — flip to true here for debug output
        println("\n[ADD_STACK] Base.:+ aliased merge caller stack:")
        for fr in stacktrace()[1:min(end,22)]; println("    ", fr); end
        flush(stdout)
    end
    _ADD_PLUS_MERGE[] += 1
    return @timeit TIMER "aliasadd.merge" begin
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
    # One template per block: alias-id range is 1:nb. Inherit the wider of the
    # two inputs' alias-id types (capacity-checked) so a widened input keeps its
    # width through the merge.
    new_alias_ids = _alias_id_range(promote_type(eltype(A.alias_ids), eltype(B.alias_ids)), nb)
    new_scalars   = ones(T, nb)
    _merge_add!(new_templates, all_keys, a_lookup, b_lookup,
        A.templates, A.alias_ids, A.scalars, B.templates, B.alias_ids, B.scalars,
        blksize, one(T))
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(
        A.dims, blksize,
        new_templates, nb,
        all_keys, new_alias_ids, new_scalars,
    )
    WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, wA.inds)
    end   # @timeit "aliasadd.merge" begin
end

const _CROSS_SCHEMA_TRACE_COUNT = Ref(0)
const _ADD_DENSE_TRACE_COUNT = Ref(0)

# ── Add-operand dump (was gated SB_ADD_OPERAND_DUMP=1, flip _ADD_OPERAND_DUMP
# below to true to re-enable) ─────────────────────────────────────────────────
# Dumps the schema (key → alias_id [scalar], + template value-hash) of BOTH
# operands of an aliased Base.:+ — i.e. the actual vectors the Lanczos eigsolve
# is adding (the M^{±1/2}-dressed Krylov vectors). Then classifies why they did
# (or didn't) match: same key-SET? same key-ORDER? same key→template-VALUE map?
# This is the decisive datum for "can the add keep dedup". Latched to the first
# 4 calls (was SB_ADD_OPERAND_DUMP_MAX, default 4).
const _ADD_OPERAND_DUMP = false
const _ADD_OPERAND_DUMP_COUNT = Ref(0)
function _dump_add_operands(tier::String, A, B; max_blocks::Int=24)
    _ADD_OPERAND_DUMP || return
    _ADD_OPERAND_DUMP_COUNT[] < 4 || return
    _ADD_OPERAND_DUMP_COUNT[] += 1
    n = _ADD_OPERAND_DUMP_COUNT[]
    _vh(X, tid) = (off = (tid - 1) * X.blksize; string(hash(@view X.templates[(off+1):(off+X.blksize)]), base=16)[1:6])
    println("\n========== [SB_ADD_OPERAND_DUMP #$n] aliased Base.:+  tier=$tier ==========")
    # EMPTY-KEY CHECK: 2-norm of each present block (scalar*template). A "present"
    # key whose block is ~0 carries no value — it's a key we instantiated/merged for
    # nothing. Threshold is relative to the operand's largest block.
    _bn(X, i) = (off=(X.alias_ids[i]-1)*X.blksize; abs(X.scalars[i]) * sqrt(sum(abs2, @view X.templates[(off+1):(off+X.blksize)])))
    for (tag, X) in (("A", A), ("B", B))
        nb = length(X.keys); nt = X.n_templates
        bn = [_bn(X, i) for i in 1:nb]
        bmax = isempty(bn) ? 0.0 : maximum(bn)
        nz = count(v -> v <= 1e-12 * max(bmax, eps()), bn)
        println("  [operand $tag] dims=$(collect(X.dims)) prefix=$(nb==0 ? 0 : length(first(X.keys))) blksize=$(X.blksize) nb=$nb nt=$nt dedup=$(round(nb/max(nt,1),digits=2))x  EMPTY(≈0-norm)=$nz/$nb")
        for i in 1:min(nb, max_blocks)
            println("     $(lpad(i,3))  $(X.keys[i]) → tid=$(X.alias_ids[i]) [s=$(round(X.scalars[i],digits=4))] vhash=$(_vh(X,X.alias_ids[i])) bnorm=$(round(bn[i],sigdigits=3))")
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
# in-place aliased add!/axpby! (VectorInterface, SB_ALIASED_INPLACE_ADD) — these
# bypass Base.:+ entirely, so if this rises and plus_merge falls, the fix fired.
const _ADD_INPLACE = Ref(0)
const _ADD_INPLACE_TRY = Ref(0)   # entries into our more-specific add!!(::Real)
const _ADD_INPLACE_FAIL = Ref(0)  # entered but _alias_inplace_axpby! returned false

# Return the WrappedAliasedBlockSparse storage of an ITensor, or nothing.
@inline function _alias_storage(t::ITensors.ITensor)
    ITensors.has_external_storage(t) || return nothing
    w = ITensors.get_external_storage(t)
    return w isa WrappedAliasedBlockSparse ? w : nothing
end

# In-place a = β*a + α*b for two aliased operands that share keys (set + order)
# where `a` is dedup-1 (n_templates == nb ⇒ each block owns a DISTINCT template
# slot ⇒ writing block i can never corrupt another block). Both operands'
# (scalar, alias_id) indirection is resolved on the fly; a's scalars fold to 1.
# This is the truly-in-place axpby used to short-circuit KrylovKit's allocating
# Base.:+ → cross-schema `plus_merge` for the (key-aligned) Lanczos adds.
# Returns true iff it mutated `a` in place.
function _alias_inplace_axpby!(aw::WrappedAliasedBlockSparse{T},
                               bw::WrappedAliasedBlockSparse{T},
                               α::Number, β::Number) where {T}
    A = aw.aliased; B = bw.aliased
    nb = length(A.keys)
    _dbg = false && _ADD_INPLACE_FAIL[] < 3  # SB_INPLACE_DBG removed 2026-06 — flip to true here for debug output
    if !(A.dims == B.dims && A.blksize == B.blksize && length(B.keys) == nb)
        _dbg && println("[INPLACE_FAIL] dims/blksize/nb: A.dims=$(A.dims) B.dims=$(B.dims) A.bs=$(A.blksize) B.bs=$(B.blksize) nbA=$nb nbB=$(length(B.keys))")
        return false
    end
    if A.n_templates != nb                          # `a` must be dedup-1 (distinct slots)
        _dbg && println("[INPLACE_FAIL] a not dedup-1: n_templates=$(A.n_templates) nb=$nb")
        return false
    end
    @inbounds for i in 1:nb
        if A.keys[i] != B.keys[i]                    # identical key set AND order
            _dbg && println("[INPLACE_FAIL] key order differs at i=$i: $(A.keys[i]) vs $(B.keys[i])")
            return false
        end
    end
    bs = A.blksize
    # Function barrier: `aw.aliased` is typed AliasedBlockSparse{T,N,N2,P} (K,AI
    # free ⇒ UnionAll), so accessing its fields here infers abstractly and the
    # arithmetic boxes every element. Passing the concrete Vectors into a typed
    # kernel lets Julia specialize ⇒ truly allocation-free in-place axpby.
    @timeit TIMER "aliasadd.inplace" _inplace_axpby_kernel!(
        A.templates, B.templates, A.alias_ids, B.alias_ids,
        A.scalars, B.scalars, nb, bs, T(α), T(β))
    _ADD_INPLACE[] += 1
    return true
end

# Typed inner kernel for _alias_inplace_axpby! (a = β*a + α*b, block-aligned,
# resolving each operand's alias_id indirection; a's scalars fold to 1).
@inline function _inplace_axpby_kernel!(tA::Vector{T}, tB::Vector{T},
                                        aidA::AbstractVector, aidB::AbstractVector,
                                        sA::Vector{T}, sB::Vector{T},
                                        nb::Int, bs::Int, αT::T, βT::T) where {T}
    @inbounds for i in 1:nb
        ta = (Int(aidA[i]) - 1) * bs
        tb = (Int(aidB[i]) - 1) * bs
        c  = αT * sB[i]
        d  = βT * sA[i]
        @simd for j in 1:bs
            tA[ta + j] = d * tA[ta + j] + c * tB[tb + j]
        end
        sA[i] = one(T)
    end
    return nothing
end

const _INNER_INPLACE = Ref(0)   # aliased inner ⟨a|b⟩ taken position-wise (no contraction)

# Position-wise ⟨a|b⟩ = Σ conj(a)·b for two aliased operands sharing keys (set +
# order). Matches ITensors.inner(a,b) = (dag(a)*b)[] (conj on FIRST arg) but skips
# the full aliased×aliased contraction KrylovKit otherwise falls into (the
# `wrapped×wrapped` dots). Works for ANY dedup (read-only). Returns the scalar, or
# nothing if the operands aren't key-aligned (caller falls back to ITensors.inner).
function _alias_inner(aw::WrappedAliasedBlockSparse{T},
                      bw::WrappedAliasedBlockSparse{T}) where {T}
    A = aw.aliased; B = bw.aliased
    nb = length(A.keys)
    (A.dims == B.dims && A.blksize == B.blksize && length(B.keys) == nb) || return nothing
    @inbounds for i in 1:nb
        A.keys[i] == B.keys[i] || return nothing
    end
    s = _alias_inner_kernel(A.templates, B.templates, A.alias_ids, B.alias_ids,
                            A.scalars, B.scalars, nb, A.blksize)
    _INNER_INPLACE[] += 1
    return s
end

# Typed inner kernel (function barrier — `aw.aliased` is UnionAll-abstract).
@inline function _alias_inner_kernel(tA::Vector{T}, tB::Vector{T},
                                     aidA::AbstractVector, aidB::AbstractVector,
                                     sA::Vector{T}, sB::Vector{T},
                                     nb::Int, bs::Int) where {T}
    s = zero(T)
    @inbounds for i in 1:nb
        ta = (Int(aidA[i]) - 1) * bs
        tb = (Int(aidB[i]) - 1) * bs
        acc = zero(T)
        @simd for j in 1:bs
            acc += conj(tA[ta + j]) * tB[tb + j]
        end
        s += conj(sA[i]) * sB[i] * acc
    end
    return s
end

# Fallback: + on two aliased tensors with mismatched classification (e.g.
# different (P, N2) splits or different keys). Returns a dense ITensor.
function _add_aliased_via_dense(wA::WrappedAliasedBlockSparse,
                                 wB::WrappedAliasedBlockSparse)
    _ADD_PLUS_DENSE[] += 1
    if ALIASED_TRACE[] && _ADD_DENSE_TRACE_COUNT[] < 10
        _ADD_DENSE_TRACE_COUNT[] += 1
        println("[DEEP _add_aliased_via_dense #$(_ADD_DENSE_TRACE_COUNT[])]  A=$(typeof(wA.aliased))  B=$(typeof(wB.aliased))  A.dims=$(wA.aliased.dims)  B.dims=$(wB.aliased.dims)  → DENSE")
    end
    return @timeit TIMER "aliasadd.dense" begin
        A_dense = to_dense(wA.aliased)
        B_dense = to_dense(wB.aliased)
        tmpA = ITensors.ITensor(A_dense, wA.inds...)
        tmpB = ITensors.ITensor(B_dense, wB.inds...)
        tmpA + tmpB
    end
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
            B_aligned = @timeit TIMER "aliasadd.align_permute" SparseBackends.permutedims(wB.aliased, perm)
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
    if ALIASED_TRACE[] && _CROSS_SCHEMA_TRACE_COUNT[] < 10
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
    # One template per block: alias-id range is 1:nb. Inherit the wider of the
    # two inputs' alias-id types (capacity-checked) so a widened input keeps its
    # width through the merge.
    new_alias_ids = _alias_id_range(promote_type(eltype(A.alias_ids), eltype(B.alias_ids)), nb)
    new_scalars   = ones(T, nb)
    _merge_add!(new_templates, all_keys, a_lookup, b_lookup,
        A.templates, A.alias_ids, A.scalars, B.templates, B.alias_ids, B.scalars,
        blksize, -one(T))
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
    if ALIASED_TRACE[] && _ADD_DENSE_TRACE_COUNT[] < 10
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
    if ALIASED_TRACE[]
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
    if ALIASED_TRACE[]
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
    if ALIASED_TRACE[]
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
    if ALIASED_TRACE[]
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
