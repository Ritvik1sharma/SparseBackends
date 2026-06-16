# tensoralgebra/contract_aliased_dense_shared.jl
#
# AliasedBlockSparse × Dense multi-label contraction kernel.
# Extracted from contract_aliased_shared.jl; contains:
#   - _ws_pending workspace helper
#   - Debug/roofline consts and stats functions
#   - _aliased_threaded_chunk top-level worker
#   - _contract_dense_threaded! (threaded two-pass kernel)
#   - _contract_dense_serial!   (serial fast-path + legacy loop kernel)
#   - contract_shared!(C::AliasedBS, A::AliasedBS, B::Dense, ...) dispatcher

# Cross-call scratch reuse (SB_ALIASED_PREALLOC_BUF=1): the matvec calls these
# kernels ~hundreds of times per eigsolve, each rebuilding a `pending` template
# buffer from empty → the per-call `append!`/alloc churn dominates allocations
# (≈53% of the contraction at N=12/md=60). A task-local persistent `pending`
# (one per element type per task) makes `resize!` a no-op after warmup, so the
# buffer is reused across calls with ~0 allocation. Task-local ⇒ thread-safe,
# no signature threading, numerics unchanged (same GEMM, same iteration order).
@inline function _ws_pending(::Type{TC}) where {TC}
    get!(() -> TC[], task_local_storage(), (:aliased_ws_pending, TC))::Vector{TC}
end

# #2 transient-scratch pool: the fusion GEMM scratch `Ffull` is allocated fresh
# per contract_shared! call (~10% of sweep alloc churn) yet never escapes — it is
# fully overwritten by `mul!` and consumed within the call. Back it by a
# task-local buffer (reshape to the needed Matrix shape) so resize! is a no-op
# after warmup. Task-local ⇒ thread-safe (each task gets its own); numerics and
# iteration order unchanged. Gated by SB_ALIASED_KERNEL_POOL.
@inline function _ws_ffull(::Type{TC}) where {TC}
    get!(() -> TC[], task_local_storage(), (:aliased_ws_ffull, TC))::Vector{TC}
end

# Non-allocating identity-permutation check (replaces `perm != collect(1:N)`,
# which allocated a Vector per call just for the comparison).
@inline function _is_identity_perm(p::AbstractVector{<:Integer})
    @inbounds for i in eachindex(p)
        p[i] == i || return false
    end
    return true
end

# Redundancy counters (SB_CAS_STATS=1): quantify how much the per-block loop
# repeats work. ntidA = Σ A.n_templates over calls; ngemm = total GEMMs (unique
# combined_tids); niter = total block-iterations. Then:
#   gemms_per_template = ngemm / ntidA   → how many times each A-template is
#                                          re-converted/re-GEMM'd (convert + batching redundancy)
#   iters_per_gemm     = niter / ngemm   → combined_tid dedup factor (≈1 ⇒ none)
const _CAS_NCALLS = Ref(0); const _CAS_NTIDA = Ref(0)
const _CAS_NGEMM  = Ref(0); const _CAS_NITER = Ref(0)
# Fusion / locality diagnostic (SB_FUSION_DIAG=1):
#   NGRP   = Σ over calls of #fusion groups (distinct (tidA,m_a) left operands)
#   NGRP1  = #groups with exactly one work item (not fusable)
#   GMAX   = largest fusion group seen (max B-slices sharing a left operand)
#   SPRANGE= Σ over groups of (max_sp_lin - min_sp_lin) → how scattered the B-slices
#            a group must gather are (locality cost proxy; 0 ⇒ already contiguous)
# fusion_factor = NGEMM/NGRP (max BLAS-call reduction by operand fusion); >1 ⇒ fusable.
const _CAS_NGRP = Ref(0); const _CAS_NGRP1 = Ref(0); const _CAS_GMAX = Ref(0)
const _CAS_SPRANGE = Ref(0)
# Output-block dedup (SB_CAS_STATS=1, _direct path): NWORK = Σ (akey × m_a × m_b)
# work items; NPEND = Σ n_pending (distinct output blocks). out_dedup = NWORK/NPEND.
# ≈1 ⇒ each output block gets exactly ONE contribution ⇒ the β=1 read-modify-write
# accumulate + the zero-fill prepass are both unnecessary (a scaled write suffices).
const _CAS_NWORK = Ref(0); const _CAS_NPEND = Ref(0)
function reset_cas_stats!()
    _CAS_NCALLS[]=0; _CAS_NTIDA[]=0; _CAS_NGEMM[]=0; _CAS_NITER[]=0
    _CAS_NGRP[]=0; _CAS_NGRP1[]=0; _CAS_GMAX[]=0; _CAS_SPRANGE[]=0
    _CAS_NWORK[]=0; _CAS_NPEND[]=0
end

# Roofline counters (SB_ROOFLINE=1): exact GEMM FLOPs + compulsory byte traffic
# for the aliased contraction kernel. FLOPs = Σ 8·M·N·K (ComplexF64 GEMM, 8 real
# flops per complex MAC). Bytes = Σ 16·(M·K + K·N + M·N) (read A,B + write C, 16
# B/ComplexF64) — a compulsory-traffic (no-reuse) model → upper bound on AI.
# achieved GFLOP/s = RF_FLOPS / kernel_time; AI = RF_FLOPS / RF_BYTES.
const _RF_ON    = Ref(false)
const _RF_FLOPS = Ref(0.0); const _RF_BYTES = Ref(0.0); const _RF_NGEMM = Ref(0)
const _RF_GEMM_NS = Ref(0.0)   # ns spent in the mul! (GEMM) calls only
# Non-GEMM kernel-time split (SB_ROOFLINE=1; zero cost when off):
#   CONV = Bfull_run convert-copy of the env slice (per shared-prefix run)
#   ACC  = β=1 scalar-weighted accumulate of Ffull slabs into output blocks
#   PRE  = _direct output-key dedup pre-pass (ck_to_cid build + slab zero-fill)
#   ALLOC= Ffull/Cscratch scratch-matrix allocation
const _RF_CONV_NS = Ref(0.0); const _RF_ACC_NS = Ref(0.0)
const _RF_PRE_NS  = Ref(0.0); const _RF_ALLOC_NS = Ref(0.0)
#   FIN  = _direct finalize (append! C.templates + key-write + sortperm)
#   LOOP = fast-path outer while-loop wall-clock; subtract conv+gemm+accum for
#          the raw bookkeeping residual (_ckey build, dict lookup, view/reshape)
const _RF_FIN_NS = Ref(0.0); const _RF_LOOP_NS = Ref(0.0)
# Threaded kernel overhead (SB_ROOFLINE=1, SB_ALIASED_NTHREADS>1):
#   BLAS_SWITCH = time for the two BLAS.set_num_threads calls (save/restore)
#   SPAWN       = wall-clock from first @spawn to last fetch (parallel exec + dispatch),
#                 minus REDUCE — i.e. the portion spent in task execution overhead
#   REDUCE      = sequential per-task buffer accumulation into `pending`
const _RF_BLAS_SWITCH_NS = Ref(0.0)
const _RF_SPAWN_NS = Ref(0.0); const _RF_REDUCE_NS = Ref(0.0)
# SETUP = contract_shared! dispatcher work BEFORE the kernel dispatch (classify,
#         A/B permute, build maps + key-sourcing + strides + scratch). Measured
#         explicitly (entry→dispatch) rather than inferred as a timer gap.
const _RF_SETUP_NS = Ref(0.0)
# A-permute (input ψ reorg to kernel canonical layout, contract_shared! step 3):
# time + how often permA is non-identity (i.e. an actual permutedims fires).
const _RF_PERMA_NS = Ref(0.0); const _RF_PERMA_HITS = Ref(0); const _RF_PERMA_CALLS = Ref(0)
const _SETUP_DBG_N = Ref(0)   # SB_SETUP_DBG: one-shot dump of permA/permB index orderings
reset_roofline!() = (_RF_ON[] = get(ENV,"SB_ROOFLINE","0")=="1"; _RF_FLOPS[]=0.0; _RF_BYTES[]=0.0; _RF_NGEMM[]=0; _RF_GEMM_NS[]=0.0;
                     _RF_CONV_NS[]=0.0; _RF_ACC_NS[]=0.0; _RF_PRE_NS[]=0.0; _RF_ALLOC_NS[]=0.0; _RF_FIN_NS[]=0.0; _RF_LOOP_NS[]=0.0;
                     _RF_BLAS_SWITCH_NS[]=0.0; _RF_SPAWN_NS[]=0.0; _RF_REDUCE_NS[]=0.0; _RF_SETUP_NS[]=0.0;
                     _RF_PERMA_NS[]=0.0; _RF_PERMA_HITS[]=0; _RF_PERMA_CALLS[]=0)
@inline function _rf_gemm!(M::Int, N::Int, K::Int)
    if _RF_ON[]
        _RF_FLOPS[] += 8.0*M*N*K
        _RF_BYTES[] += 16.0*(M*K + K*N + M*N)
        _RF_NGEMM[] += 1
    end
    return nothing
end
function show_roofline()
    f=_RF_FLOPS[]; b=_RF_BYTES[]; ng=_RF_NGEMM[]; g=_RF_GEMM_NS[]/1e9
    println("[ROOFLINE] GEMMs=$ng  total_FLOPs=", round(f/1e9,digits=3), " GFLOP  ",
            "compulsory_bytes=", round(b/1e9,digits=3), " GB  AI=", b==0 ? 0 : round(f/b,digits=3), " FLOP/byte")
    println("[ROOFLINE] avg_FLOPs/GEMM=", ng==0 ? 0 : round(f/ng,digits=0),
            "   GEMM-only(mul!) time=", round(g,digits=3), " s  (achieved-in-GEMM=",
            g==0 ? 0 : round(f/1e9/g,digits=1), " GFLOP/s)")
    loop_raw = _RF_LOOP_NS[] - _RF_CONV_NS[] - _RF_GEMM_NS[] - _RF_ACC_NS[]
    println("[ROOFLINE] non-GEMM split:  Bconv=", round(_RF_CONV_NS[]/1e9,digits=3),
            " s  accum=", round(_RF_ACC_NS[]/1e9,digits=3),
            " s  prepass=", round(_RF_PRE_NS[]/1e9,digits=3),
            " s  scratch_alloc=", round(_RF_ALLOC_NS[]/1e9,digits=3),
            " s  finalize=", round(_RF_FIN_NS[]/1e9,digits=3),
            " s  loop_bookkeeping=", round(max(0.0,loop_raw)/1e9,digits=3), " s",
            "  [loop_total=", round(_RF_LOOP_NS[]/1e9,digits=3), " s]")
    println("[ROOFLINE] SETUP (dispatcher prep, entry→kernel)=", round(_RF_SETUP_NS[]/1e9,digits=3),
            " s  (classify + A/B permute + build maps/strides/scratch; redundant per-call within a bond)")
    println("[ROOFLINE]   A-permute (input ψ reorg, line 859)=", round(_RF_PERMA_NS[]/1e9,digits=3),
            " s   fired ", _RF_PERMA_HITS[], "/", _RF_PERMA_CALLS[], " calls (non-identity permA)")
    println("[ROOFLINE]   align: ALIGN_OK=", _ALIGN_OK[], " fallback=", _ALIGN_FALLBACK[],
            "   recast_permutes=", _RECAST_PERMUTE_HITS[],
            "   (ALIGN_OK>0 ⇒ output reorder fired; A-permute still high ⇒ order ≠ kernel canonical)")
    if _RF_BLAS_SWITCH_NS[] + _RF_SPAWN_NS[] + _RF_REDUCE_NS[] > 0.0
        println("[ROOFLINE] threaded overhead:  blas_switch=", round(_RF_BLAS_SWITCH_NS[]/1e9,digits=3),
                " s  parallel_exec=", round(_RF_SPAWN_NS[]/1e9,digits=3),
                " s  reduce=", round(_RF_REDUCE_NS[]/1e9,digits=3), " s")
    end
end
function show_cas_stats()
    nc=_CAS_NCALLS[]; nt=_CAS_NTIDA[]; ng=_CAS_NGEMM[]; ni=_CAS_NITER[]
    println("[CAS_STATS] calls=$nc  Σn_templates=$nt  GEMMs=$ng  block_iters=$ni")
    println("[CAS_STATS] gemms_per_template = ", nt==0 ? 0 : round(ng/nt, digits=2),
            "   (= convert/GEMM redundancy per template; >1 ⇒ batching/caching win)")
    println("[CAS_STATS] iters_per_gemm     = ", ng==0 ? 0 : round(ni/ng, digits=2),
            "   (= combined_tid dedup; ≈1 ⇒ no dedup)")
    nw=_CAS_NWORK[]; np=_CAS_NPEND[]
    println("[CAS_STATS] out_dedup          = ", np==0 ? 0 : round(nw/np, digits=3),
            "   (work_items=$nw / out_blocks=$np; ≈1 ⇒ accum can be a write, skip zero-fill)")
    ngrp=_CAS_NGRP[]; ng1=_CAS_NGRP1[]; gmax=_CAS_GMAX[]; spr=_CAS_SPRANGE[]
    if ngrp > 0
        println("[FUSION_DIAG] fusion groups=$ngrp  GEMMs=$ng  fusion_factor = ",
                round(ng/ngrp, digits=2), "  (= max BLAS-call reduction by operand fusion)")
        println("[FUSION_DIAG]   size-1 (non-fusable) groups = $ng1 (", round(100*ng1/ngrp,digits=1),
                "%)   max group = $gmax   mean B-slice sp_lin-range/group = ",
                round(spr/ngrp, digits=1), "  (0 ⇒ contiguous; large ⇒ scattered gather)")
    end
end

# Standalone (top-level) chunk worker for the threaded aliased×dense matvec (Direction 1c).
# Defined at MODULE scope on purpose: all its loop/temporary variables (iA, ii, Amat, Bsub,
# Bfull_run, ck, outoff, oblk, …) are then genuine locals of THIS function. The previous
# implementation used a nested closure inside contract_shared!, and Julia closures *capture*
# (do not shadow) names that are also assigned in the enclosing function — and the enclosing
# kernel's serial main loop assigns those exact names. So concurrent @spawn'd closure calls
# clobbered ~21 shared bindings (including the GEMM operands and the scatter target) → a data
# race. Using a top-level function (everything passed in as arguments) eliminates that.
#
# Each task ALLOCATES its OWN `Pt` (output) and `Ff` (GEMM scratch) here and RETURNS `Pt`;
# the caller reduces over the returned per-task buffers. Allocating inside this top-level
# function (rather than at the call site) keeps `Pt`/`Ff` out of the enclosing kernel's
# variable scope, so there is no capture-sharing. `ck_to_cid` is built serially before any
# task runs and only READ here (safe). `ckey` is the read-only key-mapping closure. BLAS is
# pinned to 1 thread by the caller, so the nt Julia tasks run nt independent single-threaded
# GEMMs concurrently on disjoint memory. (Perf TODO: Pt is a fresh zeros(n_pending·blksize)
# per call — pool per-task buffers passed as args once correctness is confirmed.)
function _aliased_threaded_chunk(
        A, Bp, run_starts, lo::Int, hi::Int, ck_to_cid, n_pending::Int, blksize::Int,
        M::Int, N::Int, K::Int, Nm::Int, Mmov_total::Int, n_sp::Int, NB::Int,
        join_posA, _bconv::Bool, mode::Symbol, ckey, ::Type{TC}) where {TC}
    Pt = zeros(TC, n_pending * blksize)
    Ff = Matrix{TC}(undef, (mode == :AthenB ? (M, N * Nm) : (N * Nm, M))...)
    @inbounds for r in lo:hi
        iA = run_starts[r]; iA2 = run_starts[r + 1]
        akey0 = A.keys[iA]
        if n_sp == 0
            Bsub = Bp
        else
            sp_vals = ntuple(t -> akey0[join_posA[t]], n_sp)
            idx     = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
            @views Bsub = Bp[idx...]
        end
        Bfull_run = _bconv ? convert(Matrix{TC}, reshape(Bsub, K, N * Nm)) :
                             reshape(Bsub, K, N * Nm)
        for ii in iA:(iA2 - 1)
            akey = A.keys[ii]; tidA = A.alias_ids[ii]
            αA = convert(TC, A.scalars[ii])
            tmpl_A_3d = reshape(_aliased_template_view(A, tidA), M, Mmov_total, K)
            for m_a_lin in 1:Mmov_total
                Amat = @view tmpl_A_3d[:, m_a_lin, :]
                _rf_gemm!(M, N * Nm, K)
                _gt0 = _RF_ON[] ? time_ns() : UInt64(0)
                if mode == :AthenB
                    mul!(Ff, Amat, Bfull_run)
                else
                    mul!(Ff, transpose(Bfull_run), transpose(Amat))
                end
                _RF_ON[] && (_RF_GEMM_NS[] += Float64(time_ns() - _gt0))
                for m_b_lin in 1:Nm
                    ck = ckey(akey, m_a_lin, m_b_lin)
                    outoff = (ck_to_cid[ck] - 1) * blksize
                    if mode == :AthenB
                        cb = (m_b_lin - 1) * N
                        oblk = reshape(view(Pt, outoff+1:outoff+blksize), M, N)
                        @views oblk .+= αA .* Ff[:, cb+1:cb+N]
                    else
                        rb = (m_b_lin - 1) * N
                        oblk = reshape(view(Pt, outoff+1:outoff+blksize), N, M)
                        @views oblk .+= αA .* Ff[rb+1:rb+N, :]
                    end
                end
            end
        end
    end
    return Pt
end

# ─────────────────────────────────────────────────────────────────────────────
# _contract_dense_threaded!
# Outer-loop parallel kernel over A-key runs (Direction 1c).
# Called from contract_shared!(C::AliasedBS, A::AliasedBS, B::Dense, ...) when
# _threaded is true. Race-free by construction: each task runs the top-level
# worker _aliased_threaded_chunk which allocates its own output/scratch buffers
# and returns them; a serial reduction then sums them.
# ─────────────────────────────────────────────────────────────────────────────
function _contract_dense_threaded!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    A             :: AliasedBlockSparse,
    Bp,
    blksize       :: Int,
    M :: Int, N :: Int, K :: Int, Nm :: Int, Mmov_total :: Int,
    n_sp :: Int, NB :: Int,
    join_posA, sp_strides,
    _bconv        :: Bool,
    mode          :: Symbol,
    c_src_kind, c_src_idx,
    movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
    _kernel_nt    :: Int,
    pending       :: Vector{TC},
) where {TC,NC,N2C,PC}

    # ckey closure — re-defined here (captures parameters, not enclosing-function locals).
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

    ck_to_cid = Dict{NTuple{PC,Int}, Int}()
    # ── Phase 1 (serial): assign cids in sorted order; collect run starts ──
    # Sorted prepass: collect unique keys, sort by prefix linear index, assign sorted
    # cids. Eliminates sortperm+3 indexed copies from Phase 3 finalize.
    _ckeys_pre_thr = NTuple{PC,Int}[]
    run_starts = Int[]
    let iAp = firstindex(A.keys), nAp = lastindex(A.keys)
        @inbounds while iAp <= nAp
            push!(run_starts, iAp)
            iAp2 = _advance_run(A.keys, iAp, nAp, join_posA)
            for ii in iAp:(iAp2-1)
                akey = A.keys[ii]
                for m_a_lin in 1:Mmov_total, m_b_lin in 1:Nm
                    ck = _ckey(akey, m_a_lin, m_b_lin)
                    haskey(ck_to_cid, ck) || (ck_to_cid[ck] = 0; push!(_ckeys_pre_thr, ck))
                end
            end
            iAp = iAp2
        end
    end
    let _pdims_thr = ntuple(i -> C.dims[i], Val(PC))
        sort!(_ckeys_pre_thr; by = k -> _prefix_lin(k, _pdims_thr))
        for (i, ck) in enumerate(_ckeys_pre_thr); ck_to_cid[ck] = i; end
    end
    n_pending = length(_ckeys_pre_thr)
    nruns = length(run_starts)
    push!(run_starts, lastindex(A.keys) + 1)   # sentinel: end of last run
    nt = max(1, min(_kernel_nt, Threads.nthreads(), nruns))
    chunk = cld(nruns, nt)
    # ── Phase 2 (parallel): each chunk of runs handled by `_aliased_threaded_chunk`
    # (a top-level function — see its definition for why it is NOT a nested closure).
    # Each task returns its own output buffer; Phase 3 reduces them into `pending`.
    resize!(pending, n_pending * blksize); fill!(pending, zero(TC))
    _blas_t0 = _RF_ON[] ? time_ns() : UInt64(0)
    _blas_save = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)   # nt Julia tasks × 1 BLAS thread each
    _RF_ON[] && (_RF_BLAS_SWITCH_NS[] += Float64(time_ns() - _blas_t0))
    try
        if get(ENV, "SB_ALIASED_THREADED_SERIAL", "0") == "1"
            # Diagnostic: identical worker + partition, run sequentially (no @spawn).
            for c in 1:nt
                lo = (c - 1) * chunk + 1
                lo > nruns && continue
                hi = min(c * chunk, nruns)
                Pt = _aliased_threaded_chunk(A, Bp, run_starts, lo, hi, ck_to_cid,
                    n_pending, blksize, M, N, K, Nm, Mmov_total, n_sp, NB, join_posA,
                    _bconv, mode, _ckey, TC)
                @inbounds for i in eachindex(Pt); pending[i] += Pt[i]; end
            end
        else
            _par_t0 = _RF_ON[] ? time_ns() : UInt64(0)
            tasks = Task[]
            for c in 1:nt
                lo = (c - 1) * chunk + 1
                lo > nruns && continue
                hi = min(c * chunk, nruns)
                let lo = lo, hi = hi
                    push!(tasks, Threads.@spawn _aliased_threaded_chunk(
                        A, Bp, run_starts, lo, hi, ck_to_cid, n_pending, blksize,
                        M, N, K, Nm, Mmov_total, n_sp, NB, join_posA, _bconv, mode,
                        _ckey, TC))
                end
            end
            local _red_ns_local = 0.0
            for tk in tasks
                Pt = fetch(tk)::Vector{TC}
                _rt0 = _RF_ON[] ? time_ns() : UInt64(0)
                @inbounds for i in eachindex(Pt); pending[i] += Pt[i]; end
                _RF_ON[] && (_red_ns_local += Float64(time_ns() - _rt0))
            end
            if _RF_ON[]
                _par_total = Float64(time_ns() - _par_t0)
                _RF_SPAWN_NS[]  += _par_total - _red_ns_local
                _RF_REDUCE_NS[] += _red_ns_local
            end
        end
    finally
        _blas_t0r = _RF_ON[] ? time_ns() : UInt64(0)
        LinearAlgebra.BLAS.set_num_threads(_blas_save)
        _RF_ON[] && (_RF_BLAS_SWITCH_NS[] += Float64(time_ns() - _blas_t0r))
    end
    if get(ENV, "SB_CAS_STATS", "0") == "1"
        _CAS_NWORK[] += length(A.keys) * Mmov_total * Nm
        _CAS_NPEND[] += n_pending
    end
    # ── Phase 3 finalize ──
    # Phase 1 (sorted prepass) already assigned cids in sorted order.
    _ft0 = _RF_ON[] ? time_ns() : UInt64(0)
    # C.templates/keys/alias_ids/scalars were cleared at contract_shared! entry.
    append!(C.templates, view(pending, 1:n_pending * blksize))
    C.n_templates = n_pending
    resize!(C.keys, n_pending); resize!(C.alias_ids, n_pending); resize!(C.scalars, n_pending)
    for (k, cid) in ck_to_cid
        C.keys[cid] = k; C.alias_ids[cid] = cid; C.scalars[cid] = one(TC)
    end
    # sortperm eliminated: Phase 1 assigned sorted cids.
    _RF_ON[] && (_RF_FIN_NS[] += Float64(time_ns() - _ft0))
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# _contract_dense_serial!
# Serial fast-path (m_b-fused GEMM + BS-style β=1 direct output) and legacy
# generic loop. Called from contract_shared!(C::AliasedBS, A::AliasedBS,
# B::Dense, ...) when _threaded is false.
# ─────────────────────────────────────────────────────────────────────────────
function _contract_dense_serial!(
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

# ─────────────────────────────────────────────────────────────────────────────
# contract_shared! : AliasedBlockSparse × Dense  →  AliasedBlockSparse
# ─────────────────────────────────────────────────────────────────────────────

"""
    contract_shared!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared_labels)

Multi-label contraction `C = A ⊗_{shared_labels} B` where `A` is
`AliasedBlockSparse`, `B` is a dense array, and `C` is `AliasedBlockSparse`.

All labels in `shared_labels` must be reduced (absent from `labelsC`).
They are partitioned by where they appear in `A`:
- `shared_prefix` — in A's sparse prefix → these index into B slices
- `shared_dense`  — in A's dense tail    → GEMM reduction axes

Combined template key: `(tidA, sp_lin)` where `sp_lin` is the 1-based
column-major linear index into B's shared-prefix dimensions.  When there are
no shared-prefix labels `sp_lin == 1` always, giving at most `n_A_templates`
combined templates (perfect aliasing preservation).
"""
# Fission fallback: when output_inds_hint pushes B labels into C's prefix
# (which the aliased kernel doesn't yet support natively), delegate the
# contract to the BS path (which DOES implement hint+fission), then
# trivially re-aliasify the BS result (one template per block).
#
# Loses alias compression for this single contract step but preserves C's
# requested axis classification, so subsequent contracts in the matvec
# chain don't see cross-region label mismatches.
function contract_shared!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC       :: AbstractVector,
    A             :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA       :: AbstractVector,
    B             :: AbstractArray{TB,NB},
    labelsB       :: AbstractVector,
    mapA          :: Dict,
    mapB          :: Dict,
    shared_labels :: Vector;
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    allowed_keys_C=nothing,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

    # ── 0) Preconditions ──────────────────────────────────────────────────────
    @inbounds for lab in shared_labels
        @assert !(lab in labelsC) "shared label $lab must be reduced (not in labelsC)"
    end

    # Clear output
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    _setup0 = _RF_ON[] ? time_ns() : UInt64(0)   # explicit setup timer (entry→dispatch)

    # ── 1) Classify shared labels ─────────────────────────────────────────────
    shared_prefix = eltype(shared_labels)[]
    shared_dense  = eltype(shared_labels)[]
    @inbounds for lab in shared_labels
        @assert haskey(mapA, lab) "shared label $lab must exist in A"
        @assert haskey(mapB, lab) "shared label $lab must exist in B"
        if mapA[lab] <= PA
            push!(shared_prefix, lab)
        else
            push!(shared_dense, lab)
        end
    end

    # ── 2) Keep / reduction sets ──────────────────────────────────────────────
    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB
    redset  = Set(shared_dense)

    red_dense = [lab for lab in Adense0 if lab in redset]
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]
    keepB0    = [lab for lab in Bdense0 if !(lab in Set(shared_labels))]

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved &&
        error("Cdense interleaves A/B kept dims; unsupported — reorder C labels so A-kept and B-kept form contiguous groups")

    # Fission detection: some keepA / keepB labels may have been moved to C's
    # prefix (caller used output_inds_hint at the wrapper level). Compute the
    # moved label lists. The native fission path is taken when either moved
    # set is non-empty.
    desired_keepA_set = Set(desired_keepA)
    desired_keepB_set = Set(desired_keepB)
    moved_keepA = [lab for lab in keepA0 if !(lab in desired_keepA_set)]
    moved_keepB = [lab for lab in keepB0 if !(lab in desired_keepB_set)]
    has_fission = !isempty(moved_keepA) || !isempty(moved_keepB) || allowed_keys_C !== nothing
    # DIAGNOSTIC: force BS-delegation fallback to bypass native fission kernel.
    # Toggle with SB_ALIASED_NATIVE_FISSION=1 to use native; default off (BS fallback).
    if has_fission &&
       (allowed_keys_C !== nothing || get(ENV, "SB_ALIASED_NATIVE_FISSION", "1") != "1")  # native fission default ON (hardened 2026-06)
        return _aliased_shared_via_bs_fission!(C, labelsC, A, labelsA, B, labelsB,
            mapA, mapB, shared_labels;
            output_inds_hint=output_inds_hint, allowed_keys_C=allowed_keys_C)
    end

    @assert Set(vcat(desired_keepA, moved_keepA)) == Set(keepA0)
    @assert Set(vcat(desired_keepB, moved_keepB)) == Set(keepB0)

    # ── 3) Permute A: prefix=[keep_pref..., shared_pref...], dense=[desired_keepA..., moved_keepA..., red...] ──
    permA = _find_perm_for_A_join_and_dense_order(
        A, labelsA, mapA, shared_prefix,
        vcat(desired_keepA, moved_keepA), red_dense)
    _RF_ON[] && (_RF_PERMA_CALLS[] += 1)
    _pa0 = _RF_ON[] ? time_ns() : UInt64(0)
    if !_is_identity_perm(permA)   # non-allocating identity check (was: permA != collect(1:NA))
        _RF_ON[] && (_RF_PERMA_HITS[] += 1)
        A       = permutedims(A, permA)
        labelsA = labelsA[permA]
        mapA    = Dict(l => i for (i, l) in enumerate(labelsA))
    end
    _RF_ON[] && (_RF_PERMA_NS[] += Float64(time_ns() - _pa0))

    # ── 4) Permute B to layout (shared_prefix..., red_dense..., desired_keepB..., moved_keepB...) ──
    sp_axes_B = Int[mapB[lab] for lab in shared_prefix]
    rd_axes_B = Int[mapB[lab] for lab in red_dense]
    kb_axes_B = Int[mapB[lab] for lab in desired_keepB]
    mvb_axes_B = Int[mapB[lab] for lab in moved_keepB]
    permB     = vcat(sp_axes_B, rd_axes_B, kb_axes_B, mvb_axes_B)
    @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"
    Bp = _is_identity_perm(permB) ? B :
         (@timeit TIMER "cas.permuteB" permutedims(B, permB))

    if get(ENV, "SB_SETUP_DBG", "0") == "1" && _SETUP_DBG_N[] < 6
        _SETUP_DBG_N[] += 1
        println("[SETUP #", _SETUP_DBG_N[], "] PA=", PA, " NA=", NA, " NB=", NB,
                "\n   permA=", permA, "  (identity? ", _is_identity_perm(permA), ")",
                "\n   permB=", permB, "  (identity? ", _is_identity_perm(permB), ")",
                "\n   shared_prefix=", shared_prefix, " red_dense=", red_dense,
                "\n   desired_keepA=", desired_keepA, " moved_keepA=", moved_keepA,
                " desired_keepB=", desired_keepB, " moved_keepB=", moved_keepB)
    end

    n_sp    = length(shared_prefix)
    n_red   = length(red_dense)
    n_keepA = length(desired_keepA)
    n_movA  = length(moved_keepA)
    n_keepB = length(desired_keepB)
    n_movB  = length(moved_keepB)

    # ── 5) Dense contraction shapes ───────────────────────────────────────────
    # A.dense (after permA) = [desired_keepA..., moved_keepA..., red_dense...]
    dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))
    dimsB_p     = size(Bp)                               # (sp..., red..., keepB..., movB...)

    M    = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
    Mmov = (n_movA  == 0) ? 1 : prod(dimsA_dense[n_keepA+1:n_keepA+n_movA])
    K    = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+n_movA+1:end])
    N    = (n_keepB == 0) ? 1 : prod(dimsB_p[(n_sp + n_red + 1):(n_sp + n_red + n_keepB)])
    Nmov = (n_movB  == 0) ? 1 : prod(dimsB_p[(n_sp + n_red + n_keepB + 1):end])

    if n_red > 0
        @assert K == prod(dimsB_p[(n_sp + 1):(n_sp + n_red)]) "Reduction extent mismatch between A dense tail and B"
    end
    @assert C.blksize == M * N "C.blksize must equal M*N (M=$M, N=$N, got C.blksize=$(C.blksize))"

    # ── 6) C prefix sourcing.
    # Classify each of the PC C-prefix slots by where its value comes from:
    #   :A → from akey at position c_src_idx[j] (A's prefix position).
    #   :movA → decoded from m_a_lin (moved_keepA index = c_src_idx[j]).
    #   :movB → decoded from m_b_lin (moved_keepB index = c_src_idx[j]).
    # labelsC[1:PC] can interleave these freely (mode='AthenB'/'BthenA' is set
    # by _cdense_grouping_and_orders).
    c_src_kind = Vector{Symbol}(undef, PC)
    c_src_idx  = Vector{Int}(undef, PC)
    shared_p_set = Set(shared_prefix)
    movA_set     = Set(moved_keepA)
    movB_set     = Set(moved_keepB)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if lab in movA_set
            c_src_kind[j] = :movA
            c_src_idx[j]  = findfirst(==(lab), moved_keepA)
        elseif lab in movB_set
            c_src_kind[j] = :movB
            c_src_idx[j]  = findfirst(==(lab), moved_keepB)
        else
            @assert haskey(mapA, lab) "C prefix label $lab must exist in A"
            apos = mapA[lab]
            @assert apos <= PA "C prefix label $lab must come from A sparse prefix"
            @assert !(lab in shared_p_set) "C prefix label $lab cannot be a reduced shared-prefix label"
            c_src_kind[j] = :A
            c_src_idx[j]  = apos
        end
    end
    # Strides for decoding linear index into per-axis values.
    movA_dims_vec = Int[dimsA_dense[n_keepA + i] for i in 1:n_movA]
    movB_dims_vec = Int[dimsB_p[n_sp + n_red + n_keepB + i] for i in 1:n_movB]
    join_posA = Int[mapA[lab] for lab in shared_prefix]   # positions of shared_prefix in A.key

    # Precompute strides into B's shared-prefix dimensions for sp_lin computation.
    # sp_lin is the 1-based col-major linear index in (dimsB_p[1], ..., dimsB_p[n_sp]).
    sp_strides = Vector{Int}(undef, max(n_sp, 1))
    if n_sp > 0
        sp_strides[1] = 1
        for t in 2:n_sp
            sp_strides[t] = sp_strides[t-1] * dimsB_p[t-1]
        end
    end

    can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)

    # Perf gates (both BLAS-only; default off → original path below). Read early so the
    # per-call dicts the fast paths don't use can be skipped (allocation reduction).
    #  FUSE_MB: fuse the moved-keepB (m_b) loop into ONE GEMM (M,K)·(K,N·Nmov).
    #  DIRECT_OUTPUT (P2): BS-style β=1 mul! into a pre-deduped output slab; skips
    #    pending/contribute/commit (so combined_tid_map/key_to_alias/key_to_accum unused).
    # HARDENED (2026-06): m_b-fusion + BS-style β=1 direct output are the validated winning
    # path (3.44 s/sweep, 2× over the legacy pending/contribute/commit loop, energy ~1e-12,
    # aliasing preserved). Baked in (no env toggle) whenever BLAS applies; the legacy loop
    # below now serves ONLY as the non-BLAS (generic rank-1) fallback.
    _fuse   = can_blas
    _direct = can_blas
    # SB_ALIASED_SINGLE_PASS=1 (default off): merge the _direct cid-assignment prepass
    # INTO the main loop. The two-pass _direct walks every (akey,m_a,m_b) work item twice
    # (prepass builds ck_to_cid + zeroes the slab; main loop recomputes _ckey to look the
    # cid up). Single-pass assigns cids on first sight and zeroes each new block as it is
    # first touched — one walk, one _ckey per item, no extra memory. Bit-identical: cids
    # are still first-seen order (identical iteration order), output is sorted at finalize.
    _sp = can_blas && get(ENV, "SB_ALIASED_SINGLE_PASS", "0") == "1"
    # SB_ALIASED_NTHREADS=N (default 1 ⇒ serial): a KERNEL-ONLY scheduling knob for the aliased matvec's
    # outer-loop parallelism over A-key runs. It is deliberately SEPARATE from
    # JULIA_NUM_THREADS / OPENBLAS_NUM_THREADS — it controls only how many tasks THIS
    # kernel spawns, nothing else in the flow. Capped at the threads Julia was launched
    # with (`--threads=N` at launch is still required for the tasks to run in parallel;
    # this flag does not and must not change that env). FP-close to serial (the Phase-3
    # reduction reorders sums), NOT bit-identical — per the plan's Direction-1c gate.
    _kernel_nt = let n = tryparse(Int, get(ENV, "SB_ALIASED_NTHREADS", "1"))
        (n === nothing || n < 1) ? 1 : n
    end
    _threaded = can_blas && _kernel_nt > 1 && Threads.nthreads() > 1

    # Combined template deduplication: (tidA, sp_lin, m_a_lin, m_b_lin) → pending_tid.
    # In the no-fission case (n_movA = n_movB = 0), this reduces to (tidA, sp_lin).
    # Only the original loop uses combined_tid_map; only original+fusion-pending use the
    # key_to_alias/accum dicts → skip allocating them for the fast paths that don't.
    combined_tid_map = Dict{NTuple{4,Int}, Int}()
    # Persistent (task-local) pending buffer when prealloc is on, so its capacity
    # is reused across the many matvec calls (resize! ≈ no-op after warmup).
    pending  = get(ENV, "SB_ALIASED_PREALLOC_BUF", "1") == "1" ?   # default ON (hardened 2026-06: bit-identical, −56% alloc / −35%/sweep)
               (let p = _ws_pending(TC); empty!(p); p end) : TC[]

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # Precompute strides for decoding linear keys back to per-axis values.
    movA_strides = ones(Int, max(n_movA, 1))
    for d in 2:n_movA; movA_strides[d] = movA_strides[d-1] * movA_dims_vec[d-1]; end
    movB_strides = ones(Int, max(n_movB, 1))
    for d in 2:n_movB; movB_strides[d] = movB_strides[d-1] * movB_dims_vec[d-1]; end
    Mmov_total = (n_movA == 0) ? 1 : Mmov
    Nmov_total = (n_movB == 0) ? 1 : Nmov

    # Preallocated template buffer (SB_ALIASED_PREALLOC_BUF=1): GEMM into a
    # reused scratch matrix and write the result directly into a preallocated
    # `pending` buffer (resize! amortized geometric growth), eliminating the
    # per-template `Vector(undef,blksize)` alloc and the `append!` reallocation
    # (~26% of kernel allocations). Numerically identical — same GEMM, only the
    # buffer management changes.
    _prealloc_buf = get(ENV, "SB_ALIASED_PREALLOC_BUF", "1") == "1"   # default ON (hardened 2026-06)
    _cas_stats = get(ENV, "SB_CAS_STATS", "0") == "1"   # redundancy counting (off ⇒ zero overhead)
    blksize = C.blksize
    # P1 (BS-mirrored, hardened 2026-06): pass the strided template/B slices straight
    # into mul! instead of convert(Matrix{TC},·)-copying them. The convert was the
    # dominant matvec cost (~27 GiB / the bulk of the 42%-of-runtime kernel at
    # N=12/md=40), and `gemms_per_template≈8.72` means each A-template was re-copied
    # 8.72×. The A-template slice `@view tmpl_A_3d[:,m,:]` is always a unit-first-stride
    # StridedMatrix{TC} (contiguous template storage) → BLAS-ready with no copy
    # (verified: stride(·,1)==1, mul! routes to gemm!, bit-identical). The B slice is
    # unit-first-stride only when there are no shared-prefix axes (n_sp==0 ⇒ Bsub==Bp
    # contiguous); when n_sp>0 the slice is a non-strided ReshapedArray view and must
    # be copied for BLAS. The branch is hoisted (constant per call) for type stability.
    _bconv = n_sp > 0
    # HARDENED (2026-06): P1 (pass strided template/B views straight into BLAS, no
    # convert-copy) is always on — it's bit-identical and −30% allocation. (Used only in
    # the legacy non-BLAS-fallback loop's BLAS branch, which is itself now unreachable.)
    _noconv = true
    # Fusion/locality diagnostic (read-only; off ⇒ zero overhead). Per call, group
    # work items by shared left operand (tidA,m_a) and track count + sp_lin span.
    _fusion_diag = get(ENV, "SB_FUSION_DIAG", "0") == "1"
    _fgroups = _fusion_diag ? Dict{Tuple{Int,Int}, Vector{Int}}() : nothing  # (tidA,m_a)->[count,min_sp,max_sp]
    # Cscratch is consumed ONLY by the legacy non-fuse GEMM branch (runs when
    # !_fuse). The fused fast path (the matvec; _fuse=can_blas=true) uses Ffull +
    # pending and never touches Cscratch — so the old `can_blas && _prealloc_buf`
    # gate allocated a full M×N scratch on every matvec call that was immediately
    # garbage (dead per-call churn). Gate on !_fuse so it's empty on the hot path.
    Cscratch = (!_fuse && _prealloc_buf) ?
        Matrix{TC}(undef, (mode == :AthenB ? (M, N) : (N, M))...) :
        Matrix{TC}(undef, 0, 0)

    Nm = Nmov_total

    _RF_ON[] && (_RF_SETUP_NS[] += Float64(time_ns() - _setup0))
    if _threaded
        return _contract_dense_threaded!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total,
            n_sp, NB, join_posA, sp_strides, _bconv, mode,
            c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
            _kernel_nt, pending)
    else
        # kbd.serialcall: total time in the serial kernel BODY. Comparing this to
        # the global _RF phase sum (prepass+loop+finalize) localises the ~76s gap
        # between kbd.general_contract and the timed phases: if serialcall ≈ phase
        # sum, the gap is dispatcher setup (above this call); if serialcall ≫ phase
        # sum, it's GC / unbucketed work inside the function.
        return @timeit TIMER "kbd.serialcall" _contract_dense_serial!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total, Nmov_total,
            n_sp, NB, join_posA, sp_strides, _bconv, _noconv, mode, can_blas,
            c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
            _fuse, _direct, _sp, _prealloc_buf, _cas_stats, _fusion_diag,
            pending, key_to_alias, key_to_accum, combined_tid_map, Cscratch, _fgroups)
    end
end
