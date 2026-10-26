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
#   6.  dense_inds overload
#   7.  wrap_itensor backend=:aliased path + WrappedAliasedBlockSparse constructors
#   8.  contract! overloads for AliasedBlockSparse output (COO × Dense → Aliased)
#   9.  wrapped_contract methods with an aliased operand or output (the entry is
#       contract(A, B; ...) in tensor_wrappers.jl)
#  10.  contract_aliased_itensor — top-level function (ITensor × ITensor → ITensor)
#  11.  _canonicalize_bond_aliased! (bond-link fusion of an aliased product)
#  12.  ITensors extension hooks (factorize, dag, scale!, etc.)

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
#
# `projective=true` merges templates equal only UP TO A SCALAR, folding the ratio into the
# per-block scalars — the format's own equivalence class, and a strict superset of the
# default exact-value merge. Off by default so the existing gram call site is unchanged.
function compress_aliased_templates!(t::ITensors.ITensor; atol::Real=1e-12,
                                     projective::Bool=false)
    if ITensors.has_external_storage(t) && t.tensor.data isa WrappedAliasedBlockSparse
        _dedup_templates_by_value!(t.tensor.data.aliased; atol=atol, projective=projective)
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
    new_ali = _same_kind(V, new_ali)
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
    new_ali = _same_kind(a, new_ali)
    # Carry the factor-core hint through the template RENUMBERING. `remap` above is
    # the old->new template id map, so the routing is still fully known here. A
    # template that was dropped maps to 0, which read_core already handles ("pre-P
    # slice absent"). Basis and values are untouched, so the rv -> tid routing is as
    # valid after the filter as before it.
    #
    # Without this the field defaulted to EMPTY and read_core fell back to deriving
    # rv from the KEY's Site coordinate -- i.e. the POST-P index s'. That is exact
    # for diagonal P (PXP) and a silent PERMUTATION for a flip P (D4/KL), which is
    # how route C in run_fidelity_sweep.jl read exactly the sector weight 1/8 at
    # every cycle with no drift.
    if !isempty(a.slice_to_template)
        new_ali.slice_to_template =
            [t == 0 ? 0 : Int(get(remap, eltype(old_aids)(t), zero(eltype(old_aids))))
             for t in a.slice_to_template]
    end
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
    new_ali = _same_kind(A, new_ali)
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

# The other two wrappers get `replaceinds` in tensor_index.jl:118,123; the aliased
# one was missing, so any generic ITensor path that renames indices on an aliased
# tensor died with `MethodError: no method matching replaceinds(
# ::WrappedAliasedBlockSparse, ...)` -- the ExternalStorage dispatcher
# (tensor_index.jl:128) forwards to the inner wrapper and found nothing.
#
# It lives HERE, not with the other two: tensor_index.jl is included at
# SparseBackends.jl:197, before this file defines the type, so a method written
# there fails to precompile with `UndefVarError: WrappedAliasedBlockSparse`.
#
# Index metadata only -- `aliased` is passed through untouched, so templates,
# keys, alias_ids and scalars (and therefore the dedup ratio) are unaffected.
function replaceinds(w::WrappedAliasedBlockSparse{T,N,N2,P}, inds1, inds2) where {T,N,N2,P}
    new_inds = ITensors.replaceinds(w.inds, inds1, inds2)
    return WrappedAliasedBlockSparse{T,N,N2,P}(w.aliased, new_inds)
end

function ITensors.dag(A::AliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    new_templates = conj.(A.templates)
    return _same_kind(A, AliasedBlockSparse{T,N,N2,P}(
        A.dims, A.blksize, new_templates, A.n_templates,
        copy(A.keys), copy(A.alias_ids), copy(A.scalars),
    ))
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
# which is now defined above and returns a WrappedBlockSparse. top_level_contract
# (tensor_contraction.jl) handles the return type.

# ─────────────────────────────────────────────────────────────────────────────
# 6.  dense_inds
# ─────────────────────────────────────────────────────────────────────────────

# Dense indices: last N2 indices are the block-interior (dense tail).
dense_inds(w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} =
    Set(ntuple(i -> w.inds[N - N2 + i], Val(N2)))

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
    # Populate ali: one template per block, scalar = 1 (templates sized exactly)
    ali.templates = Vector{eltype(array_full)}(undef, length(bs.keys) * bs.blksize)
    for (i, key) in enumerate(bs.keys)
        ali.n_templates += 1
        blk_off = (bs.ids[i] - 1) * bs.blksize
        copyto!(ali.templates, (i - 1) * bs.blksize + 1, bs.data, blk_off + 1, bs.blksize)
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
# 9.  wrapped_contract with an aliased operand or an aliased output class
#
# Methods of the one wrapped entry wrapped_contract(A, B, C) (see the overview in
# tensor_wrappers.jl): C = SharedAliasedBlockSparse / SectorSplitAliasedBlockSparse, and
# C = Array (dense) with an aliased operand.
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
const _DOT_CHECK_MAXABS = Ref(0.0)   # SB_ALIASED_DOT_CHECK: max ABSOLUTE |native−dense| per dot
const _DOT_CHECK_MAXMAG = Ref(0.0)   # |dense| at the worst-abs dot (context for the abs error)
reset_dot_hits!() = (_NATIVE_DOT_HITS[] = 0; _DENSIFY_DOT_HITS[] = 0;
                     _DOT_CHECK_MAXABS[] = 0.0; _DOT_CHECK_MAXMAG[] = 0.0;
                     _RECAST_PERMUTE_HITS[] = 0)

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

# Dense-ψ × aliased-H matvec output-order table. Value = perm of the kernel's
# NATURAL output labels (labelsC_vec) giving the chosen output order:
# `canon_labels = labelsC_vec[perm]`. Only the two aliased matvec steps (2,3) hit
# the ali_dense_path; env steps (1,4) are dense. Deterministic across ALL sweeps
# because φ is pinned to [link1, s(b+1), link2, s(b)] at dmrg phi_build AND
# step 1 runs `env * φ` (swapped) with env pre-ordered [ket,dense-H,fused,bra] by
# _permute_env_to_canonical!, so step-1 emits T1 in "Layout C". `ndims` is fixed by
# bondtype (leg count is bd-independent), so (bondtype, step) is a sufficient key.
#
# NOTATION (see MATVEC_INDEX_ORDER_ANALYSIS): links l1(left) l2(mid) l3(right),
# strands per link ⁰=ket, ⁴¹=dense-H channel, ³¹/²¹=bra, F1/F2/F3=fused-sparse
# channel; sites s2(first) s3(second), ⁰=ket ¹=bra. Each aliased step is the kernel
# einsum `H(...) * B(...) -> out(...)`; the kernel reads B STRIDED (no permute_B)
# iff B = [red | gapBefore(shared) | keepB(CONTIG) | gapAfter(shared)]. Maps below
# are the GROUND (pinned-φ) path; EXCITED reuses these entries but φ is NOT pinned
# there, so its T1/T2 differ and the strided read generally does not fire.
#
# Steps 2,3 einsums and where the transpose lands (verified from do_trace, N=8):
#   (bulk,2)  H_b   (s2⁰,s2¹,F1,F2,l2⁴¹,l1⁴¹) * T1(l1⁴¹,F1,l1³¹,s3⁰,l3⁰,s2⁰)
#                 -> T2(l2⁴¹,s2¹,s3⁰,l3⁰,l1³¹,F2)         [STRIDED: keepB{l1³¹,s3⁰,l3⁰} contig]
#   (bulk,3)  H_b+1 (s3⁰,s3¹,F2,F3,l3⁴¹,l2⁴¹) * T2(l2⁴¹,s2¹,s3⁰,l3⁰,l1³¹,F2)
#                 -> T3(s2¹,l3⁰,l1³¹,l3⁴¹,s3¹,F3)         [permB: keepB{s2¹,l3⁰,l1³¹} split by s3⁰ — the forced 2→3 transpose]
#   (left,2)  H_s2  (s2⁰,s2¹,F1,F2,l2⁴¹,l1⁴¹) * T1(l2⁴¹,F2,l2²¹,s2⁰,s1⁰)
#                 -> T2(l1⁴¹,s2¹,s1⁰,l2²¹,F1)             [permB: keepB{l2²¹,s1⁰} split by s2⁰ — chain reversed, contracts right link first]
#   (left,3)  H_s1  (s1⁰,s1¹,F1,l1⁴¹) * T2(l1⁴¹,s2¹,s1⁰,l2²¹,F1)
#                 -> T3(s2¹,l2²¹,s1¹)                     [permB: keepB{s2¹,l2²¹} split by s1⁰]
#   (right,2) H_s   (sa⁰,sa¹,F1,F2,l2⁴¹,l1⁴¹) * T1(l1⁴¹,F1,l1²¹,sb⁰,sa⁰)
#                 -> T2(l2⁴¹,sa¹,sb⁰,l1²¹,F2)             [STRIDED: keepB{l1²¹,sb⁰} contig]
#   (right,3) H_s+1 (sb⁰,sb¹,F2,l2⁴¹) * T2(l2⁴¹,sa¹,sb⁰,l1²¹,F2)
#                 -> T3(sa¹,l1²¹,sb¹)                     [permB: keepB{sa¹,l1²¹} split by sb⁰]
# So the ground strided read fires at (bulk,2) and (right,2); the rest permute
# (the (·,3) fires are the intrinsic forced 2→3 transpose; (left,2) is edge geometry).
const _STATIC_OUTPUT_PERM_DENSE = Dict{Tuple{Symbol,Int}, Vector{Int}}(
    (:bulk,  2) => [3, 1, 5, 6, 4, 2],   # -> T2(l2⁴¹,s2¹,s3⁰,l3⁰,l1³¹,F2)
    (:bulk,  3) => [4, 5, 6, 3, 1, 2],   # -> T3(s2¹,l3⁰,l1³¹,l3⁴¹,s3¹,F3)
    (:left,  2) => [3, 1, 5, 4, 2],      # -> T2(l1⁴¹,s2¹,s1⁰,l2²¹,F1)
    (:left,  3) => [2, 3, 1],            # -> T3(s2¹,l2²¹,s1¹)
    (:right, 2) => [3, 1, 5, 4, 2],      # -> T2(l2⁴¹,sa¹,sb⁰,l1²¹,F2)
    (:right, 3) => [2, 3, 1],            # -> T3(sa¹,l1²¹,sb¹)
)
# Private: read only by matvec_dense_layout (output_layout_builders.jl).
_static_output_perm_dense(bondtype::Symbol, step::Int)::Union{Nothing,Vector{Int}} =
    get(_STATIC_OUTPUT_PERM_DENSE, (bondtype, step), nothing)

# wrapped_contract with an aliased output class C (SharedAliasedBlockSparse or
# SectorSplitAliasedBlockSparse; any operands, e.g. COO × dense -> aliased). The output is
# allocated as C and the kernels (contract! -> contract_aliased! / contract_shared! / the COO
# kernels) fill it in place, so the product has the requested class directly.
function wrapped_contract(A::WrappedTensorTypes, B::WrappedTensorTypes, ::Type{C};
                          in_position::Bool=false, emit_window_map::Bool=false,
                          layout::Union{Nothing,OutputLayout}=nothing) where {C<:AliasedBlockSparse}
    if layout !== nothing
        # Roles name the operands as passed (:A = first). Resolve them now; then, for
        # dense × aliased, put the aliased operand first: the aliased × dense kernel
        # moves indices into the key only from its second operand. A resolved layout
        # lists Index objects, so the swap does not change what it means.
        layout = resolve_layout(layout, A, B)
        if B isa WrappedAliasedBlockSparse && A isa WrappedTensor
            return wrapped_contract(B, A, C; in_position, emit_window_map, layout)
        end
    end
    Arep, Brep = rep(A), rep(B)
    # Requested layout: it IS the output's index order and key/dense split. Stock: the split
    # is inherited from the operands, the order is output_inds' fixed grouping.
    indsC, denseC, labelsA_vec, labelsB_vec, labelsC_vec = _contract_setup(A, B, layout)
    dimsC = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
    TC    = promote_type(eltype(Arep), eltype(Brep))
    Cw    = WrappedAliasedBlockSparse(C{TC,length(indsC),length(denseC)}(dimsC), indsC)

    # ── factor-core window-map path (off by default) ──────────────────────────
    # ψ[b]·ψ[b+1] with BOTH inputs aliased P·core (slice_to_template populated). Route
    # straight to the AliasedBS×AliasedBS contract_shared! with emit_window_map=true so
    # it emits C.window_slice_map (the (rv_b,rv_{b+1})→tid routing). Bypasses the output-
    # order alignment (template ids are order-independent; φ is read by explicit index
    # lists downstream, so order is irrelevant here). This build is once-per-bond, not
    # the hot matvec, so the align optimization is not needed.
    if emit_window_map && Arep isa AliasedBlockSparse && Brep isa AliasedBlockSparse &&
       !isempty(Arep.slice_to_template) && !isempty(Brep.slice_to_template)
        Cw.aliased = contract_shared_emit_window!(Cw.aliased, labelsC_vec,
                                                  Arep, labelsA_vec, Brep, labelsB_vec)
        return Cw
    end

    @timeit TIMER "contract_aliased" begin
        if ALIASED_TRACE[]
            println("[SB_ALIASED_TRACE wrapped_contract] C=", C, "  A=", string(typeof(Arep)),
                    "  B=", string(typeof(Brep)), "  N_C=$(length(indsC))  N2_C=$(length(denseC))",
                    "  P_C=$(length(indsC)-length(denseC))")
        end
        # The kernel reads labelsC: indices of an operand's dense block that are listed
        # in C's key are moved into it ("fission"); a request it cannot honour throws.
        Cw.aliased = contract!(Cw.aliased, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec)
    end
    return Cw
end

# wrapped_contract with a dense output (C = Array) and an aliased operand; operand pairs without
# aliasing take the more specific _BSFamily method (tensor_wrappers.jl). The kernel by storage:
#   aliased × dense array    contract_aliased_dense_to_dense!, written straight into its
#                            canonical order (or the layout's)
#   aliased × aliased        both densified, ITensors.contract
#   aliased × block-sparse / COO   the aliased operand(s) demoted to block-sparse, then the
#                            block-sparse dense-output kernels
function wrapped_contract(A::WrappedTensorTypes, B::WrappedTensorTypes, ::Type{Array};
                          in_position::Bool=false, layout::Union{Nothing,OutputLayout}=nothing)
    if layout !== nothing
        # as in the aliased-output method: resolve, aliased operand first
        layout = resolve_layout(layout, A, B)
        if B isa WrappedAliasedBlockSparse && A isa WrappedTensor
            return wrapped_contract(B, A, Array; in_position, layout)
        end
    end
    Arep, Brep = rep(A), rep(B)
    indsA, indsB = A.inds, B.inds
    indsC, denseC, labelsA_vec, labelsB_vec, labelsC_vec = _contract_setup(A, B, layout)
    dimsC = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
    TC    = promote_type(eltype(Arep), eltype(Brep))

    @timeit TIMER "!preserve_bs_output_path" begin
        # Aliased×Dense → Dense: allocate C in the kernel's CANONICAL layout (keepA, keepB,
        # c_prefix). The kernel then writes directly into C with perm_C = identity and skips
        # its `add.permute_back` pass. The returned ITensor carries the canonical-order
        # indices; downstream ITensors operations transparently handle the index reorder.
        if layout !== nothing && !isempty(layout.sparse)
            throw(OutputLayoutError("this contraction returns a dense tensor, which has " *
                "no key: pass OutputLayout(sparse=Index[], dense=[full output order]) " *
                "(see as_dense)"))
        end
        ali_dense_path =(Arep isa AliasedBlockSparse && Brep isa AbstractArray &&
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
                # Output order: the requested layout if given; otherwise this path's
                # stock order, the link-fusion heuristic: the order the NEXT sparse-H ×
                # dense kernel wants for its B input ([red_dense, keepB, shared_prefix]
                # with permB = identity):
                #   upstream's b_other   → next call's red_dense
                #   upstream's keepA     → next call's keepB
                #   upstream's b_to_next → next call's keepB
                #   upstream's c_prefix  → next call's shared_prefix
                # i.e. [b_other, keepA_labs, b_to_next, c_prefix_labs].
                if layout !== nothing
                    canon_labels = collect(labelsC_vec)   # the requested order
                else
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
            end
            @timeit TIMER "ali_dense_alloc" begin
                C_canon = _ali_dense_ctgt(TC, canon_dims)
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

        # Not the aliased×dense path → dense output.
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
            # (A native scalar-dot fast path, _aliased_lean_scalar_dot, was retired 2026-06;
            # the densify below is the live path.)
            _DENSIFY_DOT_HITS[] += 1
            tmpA = ITensors.ITensor(to_dense(Arep), indsA...)
            tmpB = ITensors.ITensor(to_dense(Brep), indsB...)
            res = ITensors.contract(tmpA, tmpB)
            # honour a requested order (ITensors.contract picks its own)
            return layout === nothing ? res : ITensors.permute(res, indsC...)
        end
        Arep_bs =Arep isa AliasedBlockSparse ? to_blocksparse(Arep) : Arep
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

# ─────────────────────────────────────────────────────────────────────────────
# 10.  contract_aliased_itensor — ITensor × ITensor → aliased (or dense) ITensor
# ─────────────────────────────────────────────────────────────────────────────

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
    emit_window_map :: Bool = false,
    layout :: Union{Nothing,OutputLayout} = nothing,
)
    Ab = to_backend(Abackend)
    Bb = to_backend(Bbackend)
    if Ab === DENSE && Bb === DENSE
        layout === nothing || throw(OutputLayoutError(
            "contract_aliased_itensor: both operands are dense; a layout cannot be honoured"))
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

    # Output class: the aliased product (_product_kind), or dense without preserve_bs_output.
    emit_window_map && !preserve_bs_output && throw(ArgumentError(
        "contract_aliased_itensor: emit_window_map needs preserve_bs_output = true (an aliased output)"))
    @timeit TIMER "wrapped_contract_call" begin
        Cw = preserve_bs_output ?
             wrapped_contract(Aw, Bw, _product_kind(rep(Aw), rep(Bw)); emit_window_map, layout) :
             wrapped_contract(Aw, Bw, Array; layout)
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

# The `factorize` hook for an aliased site (orthogonalize!, truncate!, TEBD). It checks the
# options and dispatches on the storage's child type: SharedAliasedBlockSparse -> the shared
# SVD (aliased/shared/factorize.jl); SectorSplitAliasedBlockSparse -> one weighted SVD per
# channel value (aliased/sector_split/factorize.jl).
function ITensors._external_factorize_storage(
    Aw :: WrappedAliasedBlockSparse,
    A  :: ITensors.ITensor,
    Linds...;
    mindim = nothing, maxdim = nothing, cutoff = nothing, ortho = nothing,
    eigen_perturbation = nothing, kwargs...,
)
    something(ortho, "left") == "left" ||
        error("aliased factorize: only ortho=\"left\" is supported (got $ortho)")
    eigen_perturbation === nothing ||
        error("aliased factorize: eigen_perturbation is not supported")
    return _aliased_factorize(Aw.aliased, Aw, A, Linds...; mindim, maxdim, cutoff)
end

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

# (Removed recycle_aliased_pending! — recycled consumed intermediates for the
# dense-H aliased-ψ matvec, which was the Path-B path now pruned.)

# ── DENSE-output buffer pool for the aliased×dense→dense path (ali_dense_alloc) ──
# The kernel's output C_canon is a fresh `zeros(TC, dims)` each call and becomes the
# returned ITensor's storage — at N=100 that is ~12 GiB churned through GC per run
# (the single largest non-GEMM cost). We pool it: a CONSUMED (dead) dense matvec
# intermediate's buffer is recycled into a task-local single slot, and the next
# ali_dense_alloc pops+zeros it instead of allocating. Pop REMOVES from the slot
# (defensive — two pops with no intervening recycle can never share a live buffer),
# so at worst we fall back to a fresh alloc. Zeroing still happens (the kernel
# accumulates with β=1); only the allocation/GC churn is removed ⇒ numerics identical.
# SAFETY: the caller recycles ONLY consumed operands (never the produced Hv, never the
# input v), so the returned final output — which KrylovKit keeps live — is never pooled.
function _ali_dense_ctgt(::Type{TC}, dims::NTuple{N,Int}) where {TC,N}
    n = prod(dims)
    key = (:ali_dense_ctgt_pending, TC)
    tls = task_local_storage()
    buf = get(tls, key, nothing)
    if buf isa Vector{TC} && length(buf) >= n
        delete!(tls, key)              # remove-on-pop
        resize!(buf, n)
        fill!(buf, zero(TC))
        return reshape(buf, dims)
    end
    return zeros(TC, dims)
end

function recycle_dense_ctgt!(w::ITensors.ITensor)
    ITensors.has_external_storage(w) && return nothing   # plain-dense intermediates only
    buf = ITensors.NDTensors.data(ITensors.tensor(w))
    (buf isa Vector && !isempty(buf)) || return nothing
    task_local_storage((:ali_dense_ctgt_pending, eltype(buf)), buf)
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
    new_ali = _same_kind(A, new_ali)
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
    new_ali = _same_kind(A, new_ali)
    # A copy changes nothing the hint refers to -- same keys, same alias_ids, same
    # n_templates -- so dropping it here was pure loss. Fresh vector, never shared:
    # the field is only ever reassigned, but a copy that aliased the source's vector
    # would tie the two tensors together if that ever changed.
    isempty(A.slice_to_template) || (new_ali.slice_to_template = copy(A.slice_to_template))
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
            new_ali = _same_kind(A, new_ali)
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
    new_ali = _same_kind(A, new_ali)
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
        new_ali = _same_kind(A, new_ali)
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
    new_ali = _same_kind(A, new_ali)
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

# Predicate: does this MPS carry sparse/aliased storage? (Gate used by dmrg to route an
# aliased ψ to the factor-core method.) Relocated here from the removed path_b_helpers.jl;
# depends only on the wrapper storage types defined above.
function is_sparse_mps(psi)::Bool
    length(psi) == 0 && return false
    T = psi[1]
    ITensors.has_external_storage(T) || return false
    s = ITensors.get_external_storage(T)
    return s isa WrappedBlockSparse || s isa WrappedAliasedBlockSparse
end

# True only for a PLAIN block-sparse ψ (WrappedBlockSparse), NOT the aliased variant.
# Lets the dmrg front-end distinguish it from an aliased ψ (which routes to factor-core).
function is_blocksparse_mps(psi)::Bool
    length(psi) == 0 && return false
    T = psi[1]
    ITensors.has_external_storage(T) || return false
    return ITensors.get_external_storage(T) isa WrappedBlockSparse
end
