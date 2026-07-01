# contract_aliased_dense_legacy.jl
#
# Pre-session LEGACY reduction-stationary AliasedBS × Dense serial kernel,
# extracted verbatim from git HEAD 5b6720d (append!-copy finalize, before the
# refactor / finalize buffer-swap / output-stationary work). Kept ONLY for
# back-to-back A/B of the finalize cost. Renamed _contract_dense_serial_legacy!;
# a trailing _lmap arg is accepted (and ignored) so the current dispatcher,
# which passes lmap, can call it. Route here with SB_ALIASED_LEGACY=1.
# Relies on module-level helpers that still exist (_advance_run, _ckey is inline,
# _aliased_template_view, _aliased_contribute!, _commit_aliased_dicts_lazy!, etc.).

function _contract_dense_serial_legacy!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    A             :: AliasedBlockSparse,
    Bp,
    blksize       :: Int,
    M :: Int, N :: Int, K :: Int, Nm :: Int, Mmov_total :: Int,
    Nmov_total    :: Int,
    n_sp :: Int, NB :: Int,
    join_posA, sp_strides,
    _bconv        :: Bool, _noconv :: Bool,
    mode          :: Symbol,
    can_blas      :: Bool,
    c_src_kind, c_src_idx,
    movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
    _fuse :: Bool, _direct :: Bool, _sp :: Bool,
    _prealloc_buf :: Bool, _cas_stats :: Bool, _fusion_diag :: Bool,
    pending       :: Vector{TC},
    key_to_alias, key_to_accum, combined_tid_map,
    Cscratch      :: Matrix{TC},
    _fgroups,
    _lmap,                      # accepted + ignored (legacy predates interleaved-Cdense support)
) where {TC,NC,N2C,PC}

    n_pending = 0
    _cas_niter = 0

    # ckey closure and ck_to_cid are defined inside this function (not passed in).
    @inline _ckey(akey, m_a_lin, m_b_lin) = ntuple(j -> begin
        kind = c_src_kind[j]; idx = c_src_idx[j]
        if kind === :A
            akey[idx]
        elseif kind === :movA
            Int((m_a_lin - 1) ÷ movA_strides[idx] % movA_dims_vec[idx]) + 1
        else
            Int((m_b_lin - 1) ÷ movB_strides[idx] % movB_dims_vec[idx]) + 1
        end
    end, Val(PC))

    # ── 7′) Fast paths: m_b-fused GEMM and/or BS-style β=1 direct output ──────
    # Composable via (_fuse, _direct); "neither" falls through to the loop below.
    if _fuse || _direct
        # P2 pre-pass: dedup output keys → cid; zero a contiguous output slab in `pending`.
        # Skipped when _sp (single-pass): cid assignment + per-block zeroing move inline.
        ck_to_cid = _direct ? Dict{NTuple{PC,Int}, Int}() : nothing
        if _direct && !_sp
            _pt0 = _RF_ON[] ? time_ns() : UInt64(0)
            # Two-pass sorted prepass: collect unique keys, sort by prefix linear index,
            # assign cids in sorted order. Eliminates sortperm + 3 indexed copies from
            # finalize (those 4 per-call allocations were the dominant finalize overhead).
            _ckeys_pre = NTuple{PC,Int}[]
            iAp = firstindex(A.keys); nAp = lastindex(A.keys)
            @inbounds while iAp <= nAp
                iAp2 = _advance_run(A.keys, iAp, nAp, join_posA)
                for ii in iAp:(iAp2-1)
                    akey = A.keys[ii]
                    for m_a_lin in 1:Mmov_total, m_b_lin in 1:Nm
                        ck = _ckey(akey, m_a_lin, m_b_lin)
                        haskey(ck_to_cid, ck) || (ck_to_cid[ck] = 0; push!(_ckeys_pre, ck))
                    end
                end
                iAp = iAp2
            end
            _pdims_pre = ntuple(i -> C.dims[i], Val(PC))
            sort!(_ckeys_pre; by = k -> _prefix_lin(k, _pdims_pre))
            for (i, ck) in enumerate(_ckeys_pre); ck_to_cid[ck] = i; end
            n_pending = length(_ckeys_pre)
            if get(ENV, "SB_CAS_STATS", "0") == "1"
                # work items = Σ over A-keys of Mmov_total × Nm (one (akey,m_a,m_b) each)
                _CAS_NWORK[] += length(A.keys) * Mmov_total * Nm
                _CAS_NPEND[] += n_pending
            end
            resize!(pending, n_pending * blksize); fill!(pending, zero(TC))
            _RF_ON[] && (_RF_PRE_NS[] += Float64(time_ns() - _pt0))
        end
        # Fusion scratch and pending-mode group dedup.
        _at0 = _RF_ON[] ? time_ns() : UInt64(0)
        Ffull = if _fuse
            _frows, _fcols = mode == :AthenB ? (M, N*Nm) : (N*Nm, M)
            if get(ENV, "SB_ALIASED_KERNEL_POOL", "0") == "1"
                _fb = _ws_ffull(TC); resize!(_fb, _frows * _fcols)
                reshape(_fb, _frows, _fcols)   # shares _fb memory; Array ⇒ stays Matrix{TC}
            else
                Matrix{TC}(undef, _frows, _fcols)
            end
        else
            Matrix{TC}(undef, 0, 0)
        end
        _RF_ON[] && (_RF_ALLOC_NS[] += Float64(time_ns() - _at0))
        fgmap = (_fuse && !_direct) ? Dict{NTuple{3,Int}, Int}() : nothing

        iA = firstindex(A.keys); nA = lastindex(A.keys)
        _loop_t0 = _RF_ON[] ? time_ns() : UInt64(0)
        @inbounds while iA <= nA
            iA2   = _advance_run(A.keys, iA, nA, join_posA)
            akey0 = A.keys[iA]
            sp_lin = 1
            for t in 1:n_sp
                sp_lin += (akey0[join_posA[t]] - 1) * sp_strides[t]
            end
            if n_sp == 0
                Bsub = Bp
            else
                sp_vals = ntuple(t -> akey0[join_posA[t]], n_sp)
                idx     = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
                @views Bsub = Bp[idx...]
            end
            Bsub_3d = reshape(Bsub, K, N, Nm)
            # L3(a): hoist the per-(ii,m_a) redundant convert of Bfull to ONCE per
            # shared-prefix run — Bfull depends only on Bsub, not ii/m_a. Bit-identical;
            # removes the convert from the GEMM inner block. Only used in the _fuse path.
            _ct0 = _RF_ON[] ? time_ns() : UInt64(0)
            Bfull_run = _fuse ? (_bconv ? convert(Matrix{TC}, reshape(Bsub, K, N * Nm)) :
                                          reshape(Bsub, K, N * Nm)) : nothing
            _RF_ON[] && (_RF_CONV_NS[] += Float64(time_ns() - _ct0))

            for ii in iA:(iA2-1)
                akey = A.keys[ii]; tidA = A.alias_ids[ii]; αA = convert(TC, A.scalars[ii])
                tmpl_A_3d = reshape(_aliased_template_view(A, tidA), M, Mmov_total, K)
                for m_a_lin in 1:Mmov_total
                    Amat = @view tmpl_A_3d[:, m_a_lin, :]            # (M, K)
                    if _fuse
                        _rf_gemm!(M, N * Nm, K)                      # roofline: Amat(M,K)·Bfull(K,N·Nm)
                        _gt0 = _RF_ON[] ? time_ns() : UInt64(0)      # now times pure mul! (convert hoisted)
                        if mode == :AthenB
                            mul!(Ffull, Amat, Bfull_run)
                        else
                            mul!(Ffull, transpose(Bfull_run), transpose(Amat))
                        end
                        _RF_ON[] && (_RF_GEMM_NS[] += Float64(time_ns() - _gt0))
                        if !_direct
                            # pending mode: dedup the (tidA,sp_lin,m_a) group → Nm consecutive tids.
                            gkey = (tidA, sp_lin, m_a_lin)
                            base = get(fgmap, gkey, 0)
                            if base == 0
                                base = n_pending + 1; n_pending += Nm; fgmap[gkey] = base
                                need = (base - 1 + Nm) * blksize
                                length(pending) < need && resize!(pending, max(need, 2*length(pending), blksize))
                                if mode == :AthenB
                                    copyto!(pending, (base-1)*blksize + 1, vec(Ffull), 1, Nm*blksize)
                                else
                                    for mb in 1:Nm
                                        boff = (base-1+mb-1)*blksize; rb = (mb-1)*N
                                        for col in 1:M, row in 1:N
                                            pending[boff + (col-1)*N + row] = Ffull[rb + row, col]
                                        end
                                    end
                                end
                            end
                            for m_b_lin in 1:Nm
                                ck = _ckey(akey, m_a_lin, m_b_lin)
                                @timeit TIMER "cas.contribute" _aliased_contribute!(key_to_alias, key_to_accum,
                                    pending, ck, base + (m_b_lin - 1), αA, blksize)
                            end
                        end
                    end
                    for m_b_lin in 1:Nm
                        if _direct
                            ck  = _ckey(akey, m_a_lin, m_b_lin)
                            if _sp
                                # Single-pass: assign cid on first sight, zero the new block.
                                cid = get(ck_to_cid, ck, 0)
                                if cid == 0
                                    n_pending += 1; cid = n_pending
                                    ck_to_cid[ck] = cid
                                    need = n_pending * blksize
                                    length(pending) < need &&
                                        resize!(pending, max(need, 2 * length(pending), blksize))
                                    @views fill!(pending[(cid-1)*blksize+1 : cid*blksize], zero(TC))
                                end
                                outoff = (cid - 1) * blksize
                            else
                                outoff = (ck_to_cid[ck] - 1) * blksize
                            end
                            if _fuse
                                # Accumulate α·(m_b slab of Ffull) into output block — vectorized
                                # broadcast (the scalar element-loop was the cause of `both`'s
                                # regression). AthenB slab = contiguous columns; BthenA = row band.
                                _acct0 = _RF_ON[] ? time_ns() : UInt64(0)
                                if mode == :AthenB
                                    cb   = (m_b_lin - 1) * N
                                    oblk = reshape(view(pending, outoff+1:outoff+blksize), M, N)
                                    @views oblk .+= αA .* Ffull[:, cb+1:cb+N]
                                else
                                    rb   = (m_b_lin - 1) * N
                                    oblk = reshape(view(pending, outoff+1:outoff+blksize), N, M)
                                    @views oblk .+= αA .* Ffull[rb+1:rb+N, :]
                                end
                                _RF_ON[] && (_RF_ACC_NS[] += Float64(time_ns() - _acct0))
                            else
                                Bmat = @view Bsub_3d[:, :, m_b_lin]
                                _rf_gemm!(M, N, K)                   # roofline FLOP count (non-fuse path)
                                if mode == :AthenB
                                    Cblk = reshape(view(pending, outoff+1:outoff+blksize), M, N)
                                    _bconv ? mul!(Cblk, Amat, convert(Matrix{TC}, Bmat), αA, one(TC)) :
                                             mul!(Cblk, Amat, Bmat, αA, one(TC))
                                else
                                    Cblk = reshape(view(pending, outoff+1:outoff+blksize), N, M)
                                    _bconv ? mul!(Cblk, transpose(convert(Matrix{TC}, Bmat)), transpose(Amat), αA, one(TC)) :
                                             mul!(Cblk, transpose(Bmat), transpose(Amat), αA, one(TC))
                                end
                            end
                        end
                    end
                end
            end
            iA = iA2
        end
        _RF_ON[] && (_RF_LOOP_NS[] += Float64(time_ns() - _loop_t0))

        if _sp && get(ENV, "SB_CAS_STATS", "0") == "1"
            _CAS_NWORK[] += length(A.keys) * Mmov_total * Nm
            _CAS_NPEND[] += n_pending
        end
        if _direct
            _ft0 = _RF_ON[] ? time_ns() : UInt64(0)
            # C.templates/keys/alias_ids/scalars were cleared at contract_shared! entry.
            append!(C.templates, view(pending, 1:n_pending*blksize))
            C.n_templates = n_pending
            resize!(C.keys, n_pending); resize!(C.alias_ids, n_pending); resize!(C.scalars, n_pending)
            for (k, cid) in ck_to_cid
                C.keys[cid] = k; C.alias_ids[cid] = cid; C.scalars[cid] = one(TC)
            end
            if _sp
                # Single-pass assigns cids in first-seen order; sort needed.
                pdims = ntuple(i -> C.dims[i], Val(PC))
                p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
                C.keys = C.keys[p]; C.alias_ids = C.alias_ids[p]; C.scalars = C.scalars[p]
            end
            # !_sp: sorted prepass already assigned cids in sorted order; no sort needed.
            _RF_ON[] && (_RF_FIN_NS[] += Float64(time_ns() - _ft0))
        else
            @timeit TIMER "cas.commit" _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
        end
        return C
    end

    # ── 7) Main loop: runs of A sharing the same shared-prefix values ─────────
    iA = firstindex(A.keys); nA = lastindex(A.keys)

    @inbounds while iA <= nA
        iA2   = _advance_run(A.keys, iA, nA, join_posA)
        akey0 = A.keys[iA]

        # Column-major linear index of this run's shared-prefix values in B
        sp_lin = 1
        for t in 1:n_sp
            sp_lin += (akey0[join_posA[t]] - 1) * sp_strides[t]
        end

        # Slice Bp along its leading n_sp dims; result has shape (red..., keepB..., movB...).
        if n_sp == 0
            Bsub = Bp
        else
            sp_vals = ntuple(t -> akey0[join_posA[t]], n_sp)
            idx     = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
            @views Bsub = Bp[idx...]
        end
        # Reshape Bsub to (K, N, Nmov_total) so we can iterate over moved_keepB
        # values by indexing the last dim.
        Bsub_3d = reshape(Bsub, K, N, Nmov_total)

        for ii in iA:(iA2-1)
            akey = A.keys[ii]
            tidA = A.alias_ids[ii]
            αA   = convert(TC, A.scalars[ii])

            # A template (full): laid out as (M, Mmov, K). Reshape per (m_a, *)
            # slicing later.
            tmpl_A_3d = reshape(_aliased_template_view(A, tidA), M, Mmov_total, K)

            for m_a_lin in 1:Mmov_total, m_b_lin in 1:Nmov_total
                ct_key = (tidA, sp_lin, m_a_lin, m_b_lin)
                combined_tid = get(combined_tid_map, ct_key, 0)
                if combined_tid == 0
                    n_pending += 1
                    combined_tid = n_pending
                    combined_tid_map[ct_key] = combined_tid
                    if _fusion_diag
                        gk = (tidA, m_a_lin)
                        g  = get(_fgroups, gk, nothing)
                        if g === nothing
                            _fgroups[gk] = Int[1, sp_lin, sp_lin]
                        else
                            g[1] += 1; g[2] = min(g[2], sp_lin); g[3] = max(g[3], sp_lin)
                        end
                    end
                    Amat = @view tmpl_A_3d[:, m_a_lin, :]            # (M, K)
                    Bmat = @view Bsub_3d[:, :, m_b_lin]              # (K, N)
                    if _prealloc_buf
                        # Write GEMM result directly into the preallocated pending buffer.
                        off  = (combined_tid - 1) * blksize
                        need = combined_tid * blksize
                        length(pending) < need && resize!(pending, max(need, 2 * length(pending), blksize))
                        dest = view(pending, off+1:off+blksize)   # contiguous, length blksize
                        if can_blas
                            if _noconv
                                # P1: pass strided template/B views straight into BLAS (no copy).
                                if mode == :AthenB
                                    if _bconv
                                        mul!(Cscratch, Amat, convert(Matrix{TC}, Bmat))
                                    else
                                        mul!(Cscratch, Amat, Bmat)
                                    end
                                else
                                    if _bconv
                                        mul!(Cscratch, transpose(convert(Matrix{TC}, Bmat)), transpose(Amat))
                                    else
                                        mul!(Cscratch, transpose(Bmat), transpose(Amat))
                                    end
                                end
                            else
                                # pre-P1 baseline (A/B): convert both operands.
                                if mode == :AthenB
                                    mul!(Cscratch, convert(Matrix{TC}, Amat), convert(Matrix{TC}, Bmat))
                                else
                                    mul!(Cscratch, convert(Matrix{TC}, transpose(Bmat)), convert(Matrix{TC}, transpose(Amat)))
                                end
                            end
                            copyto!(dest, 1, vec(Cscratch), 1, blksize)   # column-major flat == reshape layout
                        else
                            fill!(dest, zero(TC))
                            if mode == :AthenB
                                for k in 1:K
                                    _rank1_add_generic!(dest, one(TC), @view(Amat[:, k]), @view(Bmat[k, :]))
                                end
                            else
                                for k in 1:K
                                    _rank1_add_generic!(dest, one(TC), @view(Bmat[k, :]), @view(Amat[:, k]))
                                end
                            end
                        end
                    else
                    new_tmpl = Vector{TC}(undef, C.blksize)
                    if mode == :AthenB
                        Cmat = reshape(new_tmpl, M, N)
                        if can_blas
                            if _bconv
                                mul!(Cmat, Amat, convert(Matrix{TC}, Bmat))
                            else
                                mul!(Cmat, Amat, Bmat)
                            end
                        else
                            fill!(new_tmpl, zero(TC))
                            for k in 1:K
                                _rank1_add_generic!(new_tmpl, one(TC),
                                                    @view(Amat[:, k]), @view(Bmat[k, :]))
                            end
                        end
                    else
                        Cmat = reshape(new_tmpl, N, M)
                        if can_blas
                            if _bconv
                                mul!(Cmat, transpose(convert(Matrix{TC}, Bmat)), transpose(Amat))
                            else
                                mul!(Cmat, transpose(Bmat), transpose(Amat))
                            end
                        else
                            fill!(new_tmpl, zero(TC))
                            for k in 1:K
                                _rank1_add_generic!(new_tmpl, one(TC),
                                                    @view(Bmat[k, :]), @view(Amat[:, k]))
                            end
                        end
                    end
                    append!(pending, new_tmpl)
                    end
                end

                ckey = ntuple(j -> begin
                    kind = c_src_kind[j]
                    idx  = c_src_idx[j]
                    if kind === :A
                        akey[idx]
                    elseif kind === :movA
                        Int((m_a_lin - 1) ÷ movA_strides[idx] % movA_dims_vec[idx]) + 1
                    else   # :movB
                        Int((m_b_lin - 1) ÷ movB_strides[idx] % movB_dims_vec[idx]) + 1
                    end
                end, Val(PC))

                _cas_stats && (_cas_niter += 1)
                @timeit TIMER "cas.contribute" _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                     ckey, combined_tid, αA, C.blksize)
            end   # m_a, m_b loop
        end   # ii loop

        iA = iA2
    end   # main while

    if _cas_stats
        _CAS_NCALLS[] += 1; _CAS_NTIDA[] += A.n_templates
        _CAS_NGEMM[]  += n_pending; _CAS_NITER[] += _cas_niter
    end
    if _fusion_diag
        for (_, g) in _fgroups
            _CAS_NGRP[] += 1
            g[1] == 1 && (_CAS_NGRP1[] += 1)
            _CAS_GMAX[] = max(_CAS_GMAX[], g[1])
            _CAS_SPRANGE[] += (g[3] - g[2])
        end
        # ensure GEMM count is available even if SB_CAS_STATS is off
        _cas_stats || (_CAS_NGEMM[] += n_pending)
    end
    @timeit TIMER "cas.commit" _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end
