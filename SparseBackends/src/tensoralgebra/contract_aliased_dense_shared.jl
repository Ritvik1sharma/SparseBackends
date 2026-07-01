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

# Redundancy counters (reported under SB_ROOFLINE): quantify how much the per-block loop
# repeats work. ntidA = Σ A.n_templates over calls; ngemm = total GEMMs (unique
# combined_tids); niter = total block-iterations. Then:
#   gemms_per_template = ngemm / ntidA   → how many times each A-template is
#                                          re-converted/re-GEMM'd (convert + batching redundancy)
#   iters_per_gemm     = niter / ngemm   → combined_tid dedup factor (≈1 ⇒ none)
const _CAS_NCALLS = Ref(0); const _CAS_NTIDA = Ref(0)
const _CAS_NGEMM  = Ref(0); const _CAS_NITER = Ref(0)
# Fusion / locality diagnostic: REMOVED for now (counters retained but never
# incremented — _fusion_diag is hard-off). Re-enable by restoring the accounting
# block and a gate if the operand-fusion analysis is needed again.
const _CAS_NGRP = Ref(0); const _CAS_NGRP1 = Ref(0); const _CAS_GMAX = Ref(0)
const _CAS_SPRANGE = Ref(0)
# Output-block dedup (reported under SB_ROOFLINE, _direct path): NWORK = Σ (akey × m_a × m_b)
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

# THE single timing/instrumentation switch. SB_ROOFLINE=1 is the ONLY env flag:
# it turns on the per-phase _RF timers (GEMM-only vs A-permute / Bconv / accum /
# prepass / finalize+sortperm / loop bookkeeping) AND the CAS redundancy counters,
# and both are reported when it is set. There is no separate SB_CAS_STATS env var
# (CAS counting is gated on this) and SB_FUSION_DIAG is removed for now (it built a
# per-call dict that perturbed the path being measured).
@inline _roofline_on() = get(ENV, "SB_ROOFLINE", "0") == "1"
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
            "   (ALIGN_OK>0 ⇒ output reorder fired)")
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
    if _roofline_on()
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


@inline function _gemm_mode!(C, A, B, α, β, ::Val{:AthenB}, ::Val{bconv}, TC) where {bconv}
    B2 = bconv ? convert(Matrix{TC}, B) : B
    mul!(C, A, B2, α, β)
end
@inline function _gemm_mode!(C, A, B, α, β, ::Val{:BthenA}, ::Val{bconv}, TC) where {bconv}
    B2 = bconv ? convert(Matrix{TC}, B) : B
    mul!(C, transpose(B2), transpose(A), α, β)
end
# 3-arg (overwrite, α=1/β=0) variants. NOT equivalent to passing one(TC)/zero(TC)
# to the 5-arg form above: the 3-arg `mul!` dispatches through MulAddMul{true,false}
# (a strong-zero, scaling-free BLAS path) which is bit-for-bit what the pre-refactor
# fused GEMM used. Keep these for the fused path so output stays bit-identical.
@inline function _gemm_mode!(C, A, B, ::Val{:AthenB}, ::Val{bconv}, TC) where {bconv}
    B2 = bconv ? convert(Matrix{TC}, B) : B
    mul!(C, A, B2)
end
@inline function _gemm_mode!(C, A, B, ::Val{:BthenA}, ::Val{bconv}, TC) where {bconv}
    B2 = bconv ? convert(Matrix{TC}, B) : B
    mul!(C, transpose(B2), transpose(A))
end

function _iter_runs(f, A, join_posA, n_sp, sp_strides, Bp, NB, K, N, Nm)
    iA = firstindex(A.keys); nA = lastindex(A.keys)
    @inbounds while iA <= nA
        iA2   = _advance_run(A.keys, iA, nA, join_posA)
        akey0 = A.keys[iA]
        sp_lin = 1
        for t in 1:n_sp; sp_lin += (akey0[join_posA[t]] - 1) * sp_strides[t]; end
        Bsub   = n_sp == 0 ? Bp : (@views Bp[ntuple(t -> akey0[join_posA[t]], n_sp)...,
                                                ntuple(_ -> Colon(), NB - n_sp)...])
        f(iA, iA2, akey0, sp_lin, reshape(Bsub, K, N, Nm))
        iA = iA2
    end
end

# ── BLAS fast-path helpers (the can_blas branch of _contract_dense_serial!) ───
# Each specializes on MODE (:AthenB / :BthenA) so the GEMM + scatter compile
# branch-free per mode (chosen once via a function barrier at the call site).
@inline _gemm_timer_label(::Val{:AthenB}) = "mul!"
@inline _gemm_timer_label(::Val{:BthenA}) = "mul_T!"

# Fused-GEMM output buffer Ffull holds ALL Nm moved-keepB slabs at once:
# AthenB ⇒ (M, N·Nm), BthenA ⇒ (N·Nm, M). Backed by a task-local pool (default;
# resize! ≈ no-op after warmup, and it's fully overwritten by each GEMM so reuse
# is safe). SB_ALIASED_KERNEL_POOL=0 forces a fresh allocation per call.
function _alloc_ffull(::Type{TC}, M::Int, N::Int, Nm::Int, mode::Symbol) where {TC}
    frows, fcols = mode === :AthenB ? (M, N * Nm) : (N * Nm, M)
    if get(ENV, "SB_ALIASED_KERNEL_POOL", "1") == "1"
        fb = _ws_ffull(TC); resize!(fb, frows * fcols)
        return reshape(fb, frows, fcols)          # dense-Vector reshape ⇒ Matrix{TC}
    end
    return Matrix{TC}(undef, frows, fcols)
end

# Single-pass: cid for output key `ck`, assigned on first sight (in iteration
# order) and its output block zeroed. cids run 1,2,… == length(ck_to_cid) order.
@inline function _cid_single_pass!(ck_to_cid, pending::Vector{TC}, ck, blksize::Int) where {TC}
    cid = get(ck_to_cid, ck, 0)
    if cid == 0
        cid = length(ck_to_cid) + 1
        ck_to_cid[ck] = cid
        need = cid * blksize
        length(pending) < need && resize!(pending, max(need, 2 * length(pending), blksize))
        @views fill!(pending[(cid - 1) * blksize + 1 : cid * blksize], zero(TC))
    end
    return cid
end

# Accumulate αA·(the m_b-th slab of Ffull) into the output block at `outoff`.
# AthenB slab = N contiguous columns; BthenA slab = an N-row band.
@inline function _scatter_slab!(pending, outoff::Int, αA, Ffull, m_b_lin::Int,
                                M::Int, N::Int, ::Val{:AthenB}, lmap)
    cb = (m_b_lin - 1) * N
    if lmap === nothing
        oblk = reshape(view(pending, outoff + 1 : outoff + M * N), M, N)
        @views oblk .+= αA .* Ffull[:, cb + 1 : cb + N]
    else
        # Interleaved Cdense: the GEMM result is in [keepA, keepB] order but the
        # output block's dense tail is in the (interleaved) Cdense order, so each
        # GEMM element lands at the precomputed block position lmap[gemm_linear].
        # Same element count as the contiguous write — just strided.
        @inbounds for b in 1:N, a in 1:M
            pending[outoff + lmap[a + (b - 1) * M]] += αA * Ffull[a, cb + b]
        end
    end
end
@inline function _scatter_slab!(pending, outoff::Int, αA, Ffull, m_b_lin::Int,
                                M::Int, N::Int, ::Val{:BthenA}, lmap)
    # lmap is always nothing here: interleaved Cdense is realized via the :AthenB
    # GEMM + lmap path (chosen in contract_shared!), never :BthenA.
    rb   = (m_b_lin - 1) * N
    oblk = reshape(view(pending, outoff + 1 : outoff + M * N), N, M)
    @views oblk .+= αA .* Ffull[rb + 1 : rb + N, :]
end

# Precompute the linear-index map for an interleaved-Cdense strided scatter:
# lmap[g] = the column-major position within the output block (Cdense order) of
# the element at column-major position g in the GEMM result ([keepA…, keepB…]
# order). Built once per contract_shared! call; consumed by _scatter_slab!.
function _build_interleave_lmap(Cdense::AbstractVector, desired_keepA::AbstractVector,
                                desired_keepB::AbstractVector, cdims::Vector{Int})
    n = length(Cdense)
    dimof  = Dict(Cdense[i] => cdims[i] for i in 1:n)
    gorder = vcat(desired_keepA, desired_keepB)               # GEMM axis order
    gdims  = Int[dimof[l] for l in gorder]
    gstride = ones(Int, n); for i in 2:n; gstride[i] = gstride[i-1] * gdims[i-1]; end
    cstride = ones(Int, n); for i in 2:n; cstride[i] = cstride[i-1] * cdims[i-1]; end
    gposof = Dict(gorder[g] => g for g in 1:n)                # Cdense leg -> GEMM axis
    gpos   = Int[gposof[Cdense[p]] for p in 1:n]
    total  = isempty(gdims) ? 1 : prod(gdims)
    lmap   = Vector{Int}(undef, total)
    @inbounds for l in 1:total
        lc = 1
        for p in 1:n
            g     = gpos[p]
            idx_g = ((l - 1) ÷ gstride[g]) % gdims[g]         # 0-based index along axis g
            lc   += idx_g * cstride[p]
        end
        lmap[l] = lc
    end
    return lmap
end

# The fused-GEMM run loop, specialized on MODE. One GEMM per (A-template, m_a)
# produces all Nm slabs; each slab is scattered (β=1, weighted by αA) into its
# deduplicated output block. ck_to_cid is pre-filled (two-pass) or grown here
# (single-pass). Mutates `pending`/`ck_to_cid` in place only ⇒ no captured-Int box.
function _blas_run!(A, Bp, pending::Vector{TC}, ck_to_cid, Ffull, _ckey,
                    M::Int, N::Int, K::Int, Nm::Int, Mmov_total::Int,
                    n_sp::Int, NB::Int, join_posA, sp_strides, blksize::Int,
                    _bconv::Bool, _sp::Bool, ::Val{MODE}, lmap=nothing) where {TC,MODE}
    _iter_runs(A, join_posA, n_sp, sp_strides, Bp, NB, K, N, Nm) do iA, iA2, _akey0, _sp_lin, Bsub_3d
        _ct0  = _RF_ON[] ? time_ns() : UInt64(0)
        Bfull = reshape(Bsub_3d, K, N * Nm)
        _bconv && (Bfull = convert(Matrix{TC}, Bfull))     # contiguous copy for BLAS
        _RF_ON[] && (_RF_CONV_NS[] += Float64(time_ns() - _ct0))
        for ii in iA:(iA2 - 1)
            akey = A.keys[ii]; αA = convert(TC, A.scalars[ii])
            tmpl_A_3d = reshape(_aliased_template_view(A, A.alias_ids[ii]), M, Mmov_total, K)
            for m_a_lin in 1:Mmov_total
                Amat = @view tmpl_A_3d[:, m_a_lin, :]      # (M, K)
                _rf_gemm!(M, N * Nm, K)
                _gt0 = _RF_ON[] ? time_ns() : UInt64(0)
                @timeit TIMER _gemm_timer_label(Val(MODE)) _gemm_mode!(
                    Ffull, Amat, Bfull, Val(MODE), Val(false), TC)
                _RF_ON[] && (_RF_GEMM_NS[] += Float64(time_ns() - _gt0))
                for m_b_lin in 1:Nm
                    ck  = _ckey(akey, m_a_lin, m_b_lin)
                    cid = _sp ? _cid_single_pass!(ck_to_cid, pending, ck, blksize) : ck_to_cid[ck]
                    _acct0 = _RF_ON[] ? time_ns() : UInt64(0)
                    _scatter_slab!(pending, (cid - 1) * blksize, αA, Ffull, m_b_lin, M, N, Val(MODE), lmap)
                    _RF_ON[] && (_RF_ACC_NS[] += Float64(time_ns() - _acct0))
                end
            end
        end
    end
    return nothing
end

# Finalize the direct-output path: publish `pending` as C's templates and write
# the deduplicated keys. Two-pass already assigned cids in prefix-sorted order
# (no sort); single-pass assigned first-seen order, so sort the keys here.
# Zero-copy publish: `pending` already holds the result contiguously (it IS the
# task-local pool buffer under SB_ALIASED_PREALLOC_BUF), so hand it straight to
# C instead of append!-copying the whole buffer every call (that copy was the
# dominant finalize cost). C's old (emptied-at-entry) template buffer is recycled
# back into the pool so the next call reuses its capacity — and so the pool no
# longer aliases the buffer now owned by C.
function _finalize_direct!(C::AliasedBlockSparse{TC,NC,N2C,PC}, pending::Vector{TC},
                           ck_to_cid, _sp::Bool) where {TC,NC,N2C,PC}
    _ft0 = _RF_ON[] ? time_ns() : UInt64(0)
    np = length(ck_to_cid); bs = C.blksize
    resize!(pending, np * bs)
    recycled = C.templates
    C.templates = pending
    empty!(recycled)
    task_local_storage((:aliased_ws_pending, TC), recycled)
    C.n_templates = np
    resize!(C.keys, np); resize!(C.alias_ids, np); resize!(C.scalars, np)
    for (k, cid) in ck_to_cid
        C.keys[cid] = k; C.alias_ids[cid] = cid; C.scalars[cid] = one(TC)
    end
    if _sp
        pdims = ntuple(i -> C.dims[i], Val(PC))
        p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
        C.keys = C.keys[p]; C.alias_ids = C.alias_ids[p]; C.scalars = C.scalars[p]
    end
    _RF_ON[] && (_RF_FIN_NS[] += Float64(time_ns() - _ft0))
    return C
end

# Generic rank-1 accumulate of one output block: dest = Σ_k A[:,k]⊗B[k,:]
# (AthenB) or B[k,:]⊗A[:,k] (BthenA), matching the BLAS GEMM's index order.
function _rank1_block!(dest, Amat, Bmat, K::Int, mode::Symbol)
    fill!(dest, zero(eltype(dest)))
    o = one(eltype(dest))
    if mode === :AthenB
        @inbounds for k in 1:K; _rank1_add_generic!(dest, o, @view(Amat[:, k]), @view(Bmat[k, :])); end
    else
        @inbounds for k in 1:K; _rank1_add_generic!(dest, o, @view(Bmat[k, :]), @view(Amat[:, k])); end
    end
    return dest
end

# ── Generic (non-BLAS) fallback ──────────────────────────────────────────────
# Reached only when can_blas is false (non-BlasFloat / mismatched eltypes). Builds
# each combined template by a generic rank-1 accumulate, dedupes via combined_tid,
# and contributes through the lazy alias/accum dicts. Runs a handful of times per
# sweep, so the runtime `mode` branch in _rank1_block! is a non-issue.
function _contract_dense_generic!(
        C::AliasedBlockSparse{TC,NC,N2C,PC}, A, Bp, blksize::Int,
        M::Int, N::Int, K::Int, Nm::Int, Mmov_total::Int, n_sp::Int, NB::Int,
        join_posA, sp_strides, mode::Symbol, _prealloc_buf::Bool, _cas_stats::Bool,
        pending::Vector{TC}, key_to_alias, key_to_accum, combined_tid_map, _ckey,
    ) where {TC,NC,N2C,PC}
    n_pending  = 0
    _cas_niter = 0
    @timeit TIMER "main_loop" begin
        _iter_runs(A, join_posA, n_sp, sp_strides, Bp, NB, K, N, Nm) do iA, iA2, _akey0, sp_lin, Bsub_3d
            for ii in iA:(iA2 - 1)
                tidA = A.alias_ids[ii]; αA = convert(TC, A.scalars[ii]); akey = A.keys[ii]
                tmpl_A_3d = reshape(_aliased_template_view(A, tidA), M, Mmov_total, K)
                for m_a_lin in 1:Mmov_total, m_b_lin in 1:Nm
                    ct_key       = (tidA, sp_lin, m_a_lin, m_b_lin)
                    combined_tid = get(combined_tid_map, ct_key, 0)
                    if combined_tid == 0
                        n_pending += 1; combined_tid = n_pending
                        combined_tid_map[ct_key] = combined_tid
                        Amat = @view tmpl_A_3d[:, m_a_lin, :]
                        Bmat = @view Bsub_3d[:, :, m_b_lin]
                        if _prealloc_buf
                            need = combined_tid * blksize
                            length(pending) < need && resize!(pending, max(need, 2 * length(pending), blksize))
                            _rank1_block!(view(pending, (combined_tid - 1) * blksize + 1 : combined_tid * blksize),
                                          Amat, Bmat, K, mode)
                        else
                            tmpl = Vector{TC}(undef, blksize)
                            _rank1_block!(tmpl, Amat, Bmat, K, mode)
                            append!(pending, tmpl)
                        end
                    end
                    ckey = _ckey(akey, m_a_lin, m_b_lin)
                    _cas_stats && (_cas_niter += 1)
                    @timeit TIMER "cas.contribute" _aliased_contribute!(
                        key_to_alias, key_to_accum, pending, ckey, combined_tid, αA, blksize)
                end
            end
        end
    end
    if _cas_stats
        _CAS_NCALLS[] += 1; _CAS_NTIDA[] += A.n_templates
        _CAS_NGEMM[]  += n_pending; _CAS_NITER[] += _cas_niter
    end
    @timeit TIMER "cas.commit" _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# _contract_dense_serial_outstat!  —  OUTPUT-STATIONARY AliasedBS × Dense kernel
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the reduction-stationary (run-grouped) kernel + its per-work-item
# `_ckey` + `ck_to_cid` hash (measured 30.9 s / N=12 md40 6-sweep — the single
# largest matvec cost) AND its `_scatter_slab!` scatter, with NO hash and NO
# scatter:
#
#   • Group A's blocks by their KEPT prefix (output prefix) instead of by the
#     reduced shared-prefix. With A.keys stored col-major (kept = fast axes,
#     shared = slow axes), a STABLE sort by the kept-prefix linear index gathers
#     each output group's members in shared-prefix-ASCENDING order — i.e. the
#     exact accumulation order the old kernel used ⇒ bit-identical reduction.
#   • The output slot is then pure ARITHMETIC: cid = group_base +
#     (m_a-1)·Nmov + m_b. No `_ckey` tuple, no Dict lookup.
#   • The GEMM writes DIRECTLY into a contiguous slice of the output buffer and
#     the shared-prefix reduction is `mul!`'s β=1 accumulation over the group's
#     members — no separate accumulate/scatter pass.
#       :AthenB — one (M,K)·(K,N·Nmov) GEMM per (group,m_a) lands as the Nmov
#                 consecutive (M,N) blocks contiguously (Nmov batching kept ONLY
#                 because it needs no non-local scatter).
#       :BthenA — the batched slabs would be row-strided (non-contiguous), so we
#                 emit one (N,K)·(K,M)→(N,M) GEMM per (m_a,m_b) block, each
#                 written contiguously into its own block. Still no scatter.
#   • β=0 on the first member overwrites (so the output buffer needs no
#     pre-zeroing); β=1 on the rest accumulates.
#
# Same call signature as _contract_dense_serial! (drop-in dispatch), plus a trailing
# `out_dense_perm` (default nothing). Two regimes, selected by that arg:
#   • out_dense_perm === nothing  (default / non-SB_OUTSTAT_SCHED path): unchanged behaviour
#     — GEMM in the dispatcher-chosen `mode` (:AthenB/:BthenA), plain finalize. Byte-identical
#     to before, so runs that don't opt in are unaffected.
#   • out_dense_perm !== nothing  (SB_OUTSTAT_SCHED): the kernel OWNS every requested order.
#     It always GEMMs in natural :AthenB [keepA, keepB] order, then applies ONE `permutedims`
#     phase (`out_dense_perm`, computed from labels in contract_shared!) to reach the requested
#     order — AthenB→identity, BthenA→swap, interleaved→interleave. `permutedims` on the aliased
#     storage permutes each template's dense tail AND remaps+sorts the prefix keys, so it
#     subsumes the finalize sort. Self-contained: no `lmap`, no scatter, no fallback.

function _contract_dense_serial_outstat!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    A             :: AliasedBlockSparse{TA,NA,N2A,PA},
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
    lmap          = nothing,
    out_dense_perm = nothing,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA}

    # SB_OUTSTAT_SCHED path ⇔ out_dense_perm provided: own every order via a permute phase.
    _use_perm = out_dense_perm !== nothing

    nkeys = length(A.keys)
    n_keep_pref = PA - n_sp                      # kept (output) prefix axes = 1:n_keep_pref

    # Output key for one (group-representative akey, m_a, m_b). Built once per
    # output block (n_pending times), NOT per work-item.
    @inline _ckey(akey, m_a_lin, m_b_lin) = ntuple(j -> begin
        kind = c_src_kind[j]; idx = c_src_idx[j]
        kind === :A    ? akey[idx] :
        kind === :movA ? Int((m_a_lin - 1) ÷ movA_strides[idx] % movA_dims_vec[idx]) + 1 :
                         Int((m_b_lin - 1) ÷ movB_strides[idx] % movB_dims_vec[idx]) + 1
    end, Val(PC))

    if nkeys == 0
        empty!(C.templates); C.n_templates = 0
        empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)
        return C
    end

    # ── 1) Group A's entries by kept-prefix (output prefix) ───────────────────
    # keep-linear index (col-major over kept prefix axes 1:n_keep_pref).
    keep_strides = Vector{Int}(undef, max(n_keep_pref, 1))
    if n_keep_pref > 0
        keep_strides[1] = 1
        @inbounds for j in 2:n_keep_pref
            keep_strides[j] = keep_strides[j-1] * A.dims[j-1]
        end
    end
    @inline _keeplin(ii) = begin
        kl = 1
        @inbounds for j in 1:n_keep_pref
            kl += (A.keys[ii][j] - 1) * keep_strides[j]
        end
        kl
    end
    keeplins = Vector{Int}(undef, nkeys)
    @timeit TIMER "outstat.keylin" (@inbounds for ii in 1:nkeys; keeplins[ii] = _keeplin(ii); end)
    # STABLE sort ⇒ within an equal-keep group the members stay in their original
    # (shared-prefix-ascending) order ⇒ bit-identical reduction order.
    order = @timeit TIMER "outstat.sort" sortperm(keeplins; alg = Base.Sort.MergeSort)

    blk_per_group = Mmov_total * Nmov_total
    # number of distinct kept-prefix groups
    n_groups = 1
    @inbounds for r in 2:nkeys
        (keeplins[order[r]] != keeplins[order[r-1]]) && (n_groups += 1)
    end
    n_pending = n_groups * blk_per_group

    resize!(pending, n_pending * blksize)
    @timeit TIMER "outstat.fill" fill!(pending, zero(TC))   # β=1 accumulate starts from 0

    # ── 2) Accumulate: output-stationary, GEMM-direct, β-reduction over members ──
    # Under the permute path the kernel always produces natural :AthenB [keepA, keepB];
    # the requested order (incl. BthenA / interleaved) is realized by the permute phase
    # below. Non-permute path keeps the dispatcher-chosen mode (unchanged behaviour).
    _kmode = _use_perm ? :AthenB : mode
    if _kmode === :AthenB
        _outstat_run!(C, A, Bp, pending, order, keeplins, n_keep_pref,
                      blksize, M, N, K, Nm, Mmov_total, Nmov_total, n_sp, NB,
                      join_posA, blk_per_group, _bconv, Val(:AthenB))
    else
        _outstat_run!(C, A, Bp, pending, order, keeplins, n_keep_pref,
                      blksize, M, N, K, Nm, Mmov_total, Nmov_total, n_sp, NB,
                      join_posA, blk_per_group, _bconv, Val(:BthenA))
    end

    # ── 3) Finalize: write keys (arithmetic cid in natural order), publish templates ──
    @timeit TIMER "outstat.finalize" begin
        recycled = C.templates
        C.templates = pending
        empty!(recycled)
        task_local_storage((:aliased_ws_pending, TC), recycled)
        C.n_templates = n_pending
        resize!(C.keys, n_pending); resize!(C.alias_ids, n_pending); resize!(C.scalars, n_pending)
        let gidx = 0, r = 1
            @inbounds while r <= nkeys
                r2 = r
                while r2 < nkeys && keeplins[order[r2+1]] == keeplins[order[r]]; r2 += 1; end
                rep = A.keys[order[r]]                   # any member: :A key parts share kept prefix
                gbase = gidx * blk_per_group
                for m_a in 1:Mmov_total, m_b in 1:Nmov_total
                    cid = gbase + (m_a - 1) * Nmov_total + m_b
                    C.keys[cid] = _ckey(rep, m_a, m_b)
                    C.alias_ids[cid] = cid
                    C.scalars[cid] = one(TC)
                end
                gidx += 1; r = r2 + 1
            end
        end
    end

    # ── 4) Output order ──
    if _use_perm && !all(i -> out_dense_perm[i] == i, eachindex(out_dense_perm))
        # PERMUTE PHASE (separate from the multiply, mirrors the input permA phase):
        # C currently holds natural [keepA, keepB] dense data; relabel C's dense dims to
        # that natural order, then permutedims to the requested order. permutedims permutes
        # each template's dense tail AND remaps+sorts the prefix keys — so it also does the
        # canonical key sort. Returns a fresh storage with the requested (target) dims.
        orig_dims = C.dims
        invp = invperm(out_dense_perm)
        C.dims = ntuple(k -> k <= PC ? orig_dims[k] : orig_dims[PC + invp[k - PC]], Val(NC))
        full_perm = vcat(collect(1:PC), Int[PC + out_dense_perm[i] for i in 1:N2C])
        return @timeit TIMER "outstat.permute" permutedims(C, full_perm)
    else
        # No relayout needed (natural order == requested): canonical key sort only.
        pdims = ntuple(i -> C.dims[i], Val(PC))
        p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
        C.keys = C.keys[p]; C.alias_ids = C.alias_ids[p]; C.scalars = C.scalars[p]
        return C
    end
end

# Output-stationary accumulation, specialized on MODE. For each kept-prefix group,
# each member (shared-prefix-ascending) contributes a GEMM accumulated (β) directly
# into the group's contiguous output region — no scatter, no key lookup.
function _outstat_run!(C, A, Bp, pending::Vector{TC}, order, keeplins,
                       n_keep_pref::Int, blksize::Int, M::Int, N::Int, K::Int, Nm::Int,
                       Mmov_total::Int, Nmov_total::Int, n_sp::Int, NB::Int, join_posA,
                       blk_per_group::Int, _bconv::Bool, ::Val{MODE}) where {TC,MODE}
    nkeys = length(order)
    gidx = 0
    r = 1
    # Bslab cache: the contiguous B-slab depends ONLY on the shared-prefix key
    # values (sp_vals); the same sp_vals recurs across many A-blocks within this
    # contraction, so convert each distinct slab once and reuse (kills the bulk
    # of outstat.conv). Keyed by the linearized sp index over Bp's first n_sp dims.
    _Bdims = size(Bp)
    _Bcache = (MODE === :AthenB && _bconv && n_sp > 0) ? Dict{Int,Matrix{TC}}() : nothing
    @inbounds while r <= nkeys
        # extent of this kept-prefix group within `order`
        r2 = r
        while r2 < nkeys && keeplins[order[r2+1]] == keeplins[order[r]]; r2 += 1; end
        gbase = gidx * blk_per_group               # first block index (0-based) of group

        for oi in r:r2
            ii  = order[oi]
            αA  = convert(TC, A.scalars[ii])
            β   = one(TC)                           # pending pre-zeroed ⇒ accumulate from 0
            akey = A.keys[ii]
            tmpl_A_3d = reshape(_aliased_template_view(A, A.alias_ids[ii]), M, Mmov_total, K)
            # B sub-block for this member's shared-prefix key (dense ⇒ just an offset/strided view)
            if n_sp == 0
                Bsub = Bp
            else
                sp_vals = ntuple(t -> akey[join_posA[t]], n_sp)
                idx     = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
                Bsub    = @views Bp[idx...]
            end
            if MODE === :AthenB
                if _Bcache === nothing
                    Bslab = @timeit TIMER "outstat.conv" (_bconv ?
                                convert(Matrix{TC}, reshape(Bsub, K, N * Nmov_total)) :
                                Matrix{TC}(reshape(Bsub, K, N * Nmov_total)))
                else
                    sp_lin = 1; _st = 1
                    for t in 1:n_sp
                        sp_lin += (Int(sp_vals[t]) - 1) * _st
                        _st *= _Bdims[t]
                    end
                    Bslab = get!(_Bcache, sp_lin) do
                        @timeit TIMER "outstat.conv" convert(Matrix{TC}, reshape(Bsub, K, N * Nmov_total))
                    end
                end
                for m_a in 1:Mmov_total
                    off    = (gbase + (m_a - 1) * Nmov_total) * blksize
                    # GEMM directly into the contiguous output region (zero-copy, BLAS-native).
                    region = unsafe_wrap(Matrix{TC}, pointer(pending, off + 1), (M, N * Nmov_total))
                    Amat   = @view tmpl_A_3d[:, m_a, :]      # (M,K)
                    @timeit TIMER "outstat.gemm" mul!(region, Amat, Bslab, αA, β)
                end
            else  # :BthenA — per-block GEMM (batched slabs would be row-strided ⇒ would need a scatter)
                Bslab3 = @timeit TIMER "outstat.conv" (_bconv ?
                             convert(Array{TC,3}, reshape(Bsub, K, N, Nmov_total)) :
                             Array{TC,3}(reshape(Bsub, K, N, Nmov_total)))
                for m_a in 1:Mmov_total
                    Amat = @view tmpl_A_3d[:, m_a, :]        # (M,K)
                    for m_b in 1:Nmov_total
                        off   = (gbase + (m_a - 1) * Nmov_total + (m_b - 1)) * blksize
                        block = unsafe_wrap(Matrix{TC}, pointer(pending, off + 1), (N, M))
                        Bmb   = @view Bslab3[:, :, m_b]      # (K,N)
                        @timeit TIMER "outstat.gemm_T" mul!(block, transpose(Bmb), transpose(Amat), αA, β)
                    end
                end
            end
        end
        gidx += 1; r = r2 + 1
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# _contract_dense_serial!
# Serial AliasedBS × Dense kernel. Two paths: a BLAS fast path (fused per-m_a
# GEMM + β=1 direct deduplicated output) and a generic rank-1 fallback. Called
# from contract_shared! when _threaded is false. In the dispatcher
# _fuse == _direct == can_blas, so BLAS path ⟺ can_blas; the legacy (_fuse,
# !_direct)/(!_fuse,_direct) hybrids and the in-fallback BLAS branches were dead
# and have been removed (the args _fuse/_direct/_noconv/_fusion_diag/Cscratch/
# _fgroups are retained only to keep the dispatcher call site unchanged).
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
    lmap          = nothing,
) where {TC,NC,N2C,PC}

    # Output key for one work item (akey, m_a_lin, m_b_lin); shared by both paths.
    @timeit TIMER "ckey_closure!" begin
        @inline _ckey(akey, m_a_lin, m_b_lin) = ntuple(j -> begin
            kind = c_src_kind[j]; idx = c_src_idx[j]
            kind === :A    ? akey[idx] :
            kind === :movA ? Int((m_a_lin - 1) ÷ movA_strides[idx] % movA_dims_vec[idx]) + 1 :
                             Int((m_b_lin - 1) ÷ movB_strides[idx] % movB_dims_vec[idx]) + 1
        end, Val(PC))
    end

    # Generic fallback (non-BlasFloat / mismatched eltypes).
    if !can_blas
        return _contract_dense_generic!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total,
            n_sp, NB, join_posA, sp_strides, mode, _prealloc_buf, _cas_stats,
            pending, key_to_alias, key_to_accum, combined_tid_map, _ckey)
    end

    # ── BLAS fast path: fused per-(template, m_a) GEMM + β=1 direct output ────
    ck_to_cid = Dict{NTuple{PC,Int}, Int}()

    # Two-pass: dedup output keys up front, assign cids in prefix-sorted order (so
    # finalize needs no sort) and zero the output slab. Single-pass (default,
    # SB_ALIASED_SINGLE_PASS) skips this — cids are assigned inline in _blas_run!.
    if !_sp
        @timeit TIMER "prepass" begin
            _pt0  = _RF_ON[] ? time_ns() : UInt64(0)
            ckeys = NTuple{PC,Int}[]
            iAp = firstindex(A.keys); nAp = lastindex(A.keys)
            @inbounds while iAp <= nAp
                iAp2 = _advance_run(A.keys, iAp, nAp, join_posA)
                for ii in iAp:(iAp2 - 1)
                    akey = A.keys[ii]
                    for m_a_lin in 1:Mmov_total, m_b_lin in 1:Nm
                        ck = _ckey(akey, m_a_lin, m_b_lin)
                        haskey(ck_to_cid, ck) || (ck_to_cid[ck] = 0; push!(ckeys, ck))
                    end
                end
                iAp = iAp2
            end
            pdims = ntuple(i -> C.dims[i], Val(PC))
            sort!(ckeys; by = k -> _prefix_lin(k, pdims))
            for (i, ck) in enumerate(ckeys); ck_to_cid[ck] = i; end
            _cas_stats && (_CAS_NWORK[] += length(A.keys) * Mmov_total * Nm;
                           _CAS_NPEND[] += length(ckeys))
            resize!(pending, length(ckeys) * blksize); fill!(pending, zero(TC))
            _RF_ON[] && (_RF_PRE_NS[] += Float64(time_ns() - _pt0))
        end
    end

    _at0  = _RF_ON[] ? time_ns() : UInt64(0)
    Ffull = _alloc_ffull(TC, M, N, Nm, mode)
    _RF_ON[] && (_RF_ALLOC_NS[] += Float64(time_ns() - _at0))

    @timeit TIMER "outer_loop" begin
        _loop_t0 = _RF_ON[] ? time_ns() : UInt64(0)
        # Function barrier on mode ⇒ the GEMM + scatter compile branch-free per mode.
        if mode === :AthenB
            _blas_run!(A, Bp, pending, ck_to_cid, Ffull, _ckey, M, N, K, Nm, Mmov_total,
                       n_sp, NB, join_posA, sp_strides, blksize, _bconv, _sp, Val(:AthenB), lmap)
        else
            _blas_run!(A, Bp, pending, ck_to_cid, Ffull, _ckey, M, N, K, Nm, Mmov_total,
                       n_sp, NB, join_posA, sp_strides, blksize, _bconv, _sp, Val(:BthenA), lmap)
        end
        _RF_ON[] && (_RF_LOOP_NS[] += Float64(time_ns() - _loop_t0))
        _sp && _cas_stats && (_CAS_NWORK[] += length(A.keys) * Mmov_total * Nm;
                              _CAS_NPEND[] += length(ck_to_cid))
        _finalize_direct!(C, pending, ck_to_cid, _sp)
    end
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
    # OUTPUT-STATIONARY relayout perm (self-contained; replaces the lmap path for outstat).
    # The output-stationary kernel always GEMMs in natural [desired_keepA…, desired_keepB…]
    # dense order; `out_dense_perm` maps that natural order to the REQUESTED `Cdense` order
    # (identity when Cdense is already AthenB-grouped, a swap for BthenA, an interleave for
    # interleaved). The kernel applies it as a single `permutedims` phase, so it owns every
    # requested order with no scatter and no fallback. Computed from labels — no `lmap`.
    _os_nat_dense  = vcat(desired_keepA, desired_keepB)
    out_dense_perm = Int[findfirst(==(Cdense[j]), _os_nat_dense) for j in 1:length(Cdense)]
    if get(ENV, "SB_PERM_DBG", "0") == "1"
        _isid = all(j -> out_dense_perm[j] == j, eachindex(out_dense_perm))
        println("[PERM_DBG bond=", get(ENV, "SB_BONDTYPE", "?"), " step=", get(ENV, "SB_STEP", "?"),
                "] mode=", mode, " identity=", _isid, " out_dense_perm=", out_dense_perm,
                "\n    requested Cdense=", Cdense, "  natural[keepA;keepB]=", _os_nat_dense,
                "  desired_keepA=", desired_keepA, " desired_keepB=", desired_keepB)
        flush(stdout)
    end
    # Interleaved Cdense (A-kept and B-kept dense legs interleaved in the output):
    # the GEMM still computes keepA × keepB (mode :AthenB); the interleaved output
    # layout is realized by a strided write (lmap) in the scatter. This lets the
    # caller emit an output order that makes the NEXT step's permA the identity
    # even when the reduction legs split across operand origins (the both-origin
    # case `_canon_inds_for_next_A` would otherwise bail on). Realized on the
    # serial BLAS path only; see the lmap guard at the dispatch below.
    _interleaved = (mode === :interleaved)
    _interleaved && (mode = :AthenB)

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
    @timeit TIMER "cas.permA" begin
        permA = _find_perm_for_A_join_and_dense_order(
            A, labelsA, mapA, shared_prefix,
            vcat(desired_keepA, moved_keepA), red_dense)
        _RF_ON[] && (_RF_PERMA_CALLS[] += 1)
        _pa0 = _RF_ON[] ? time_ns() : UInt64(0)
        if !_is_identity_perm(permA)   # non-allocating identity check (was: permA != collect(1:NA))
            _RF_ON[] && (_RF_PERMA_HITS[] += 1)
            get(ENV, "SB_PERMA_DBG", "0") == "1" &&
                println("[PERMA_FIRED bond=", get(ENV,"SB_BONDTYPE","?"),
                        " step=", get(ENV,"SB_STEP","?"), "] permA=", permA)
            A       = permutedims(A, permA)
            labelsA = labelsA[permA]
            mapA    = Dict(l => i for (i, l) in enumerate(labelsA))
        end
        _RF_ON[] && (_RF_PERMA_NS[] += Float64(time_ns() - _pa0))
    end

    # ── 4) Permute B to layout (shared_prefix..., red_dense..., desired_keepB..., moved_keepB...) ──
    @timeit TIMER "cas.permB" begin
        sp_axes_B = Int[mapB[lab] for lab in shared_prefix]
        rd_axes_B = Int[mapB[lab] for lab in red_dense]
        kb_axes_B = Int[mapB[lab] for lab in desired_keepB]
        mvb_axes_B = Int[mapB[lab] for lab in moved_keepB]
        permB     = vcat(sp_axes_B, rd_axes_B, kb_axes_B, mvb_axes_B)
        @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"
        Bp = _is_identity_perm(permB) ? B :
            (@timeit TIMER "cas.permuteB" permutedims(B, permB))

        if get(ENV, "SB_SETUP_DBG", "0") == "1" &&
           _SETUP_DBG_N[] < parse(Int, get(ENV, "SB_SETUP_DBG_MAX", "9"))
            _SETUP_DBG_N[] += 1
            spS=Set(shared_prefix); rdS=Set(red_dense); kaS=Set(desired_keepA); maS=Set(moved_keepA)
            # role of each A leg; P=sparse-prefix slot, d=dense-tail slot, [dim] from A.dims
            rolA = lab -> lab in spS ? "shared·contract" : lab in rdS ? "red·contract" :
                        lab in kaS ? "keep" : lab in maS ? "moved→Cprefix" : "keepPrefix"
            arep = join([string(i<=PA ? "P" : "d", "[", A.dims[i], "]", rolA(labelsA[i])) for i in 1:NA], "  ")
            crep = join([string(i<=PC ? "P" : "d", "[", C.dims[i], "]") for i in 1:NC], "  ")
            println("[SETUP #", _SETUP_DBG_N[], "] bond=", get(ENV, "SB_BOND", "?"),
                    " step=", get(ENV, "SB_STEP", "?"), "  PA=", PA, " NA=", NA, " → PC=", PC, " NC=", NC,
                    "   permA=", permA, " (id? ", _is_identity_perm(permA), ")")
        end
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
    @timeit TIMER "cas.csrc" begin
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
    # SB_ALIASED_SINGLE_PASS (default ON): merge the _direct cid-assignment prepass
    # INTO the main loop. The two-pass _direct walks every (akey,m_a,m_b) work item twice
    # (prepass builds ck_to_cid + zeroes the slab; main loop recomputes _ckey to look the
    # cid up). Single-pass assigns cids on first sight and zeroes each new block as it is
    # first touched — one walk, one _ckey per item, no extra memory. Bit-identical: cids
    # are still first-seen order (identical iteration order), output is sorted at finalize.
    # Default flipped 2026-06-22 (N=12 md=40 4-sweep, SB_ROOFLINE): single-pass removes the
    # prepass (measured 1.29s) at the cost of one finalize sortperm (+0.63s) → net ~0.66s/run,
    # BIT-IDENTICAL energy (validated all 4 sweeps). A modest win, NOT the ~14s originally
    # hypothesized — the prepass was never the bottleneck (finalize ~7.8s and accum ~4.2s are).
    # Set SB_ALIASED_SINGLE_PASS=0 to restore the old two-pass.
    _sp = can_blas && get(ENV, "SB_ALIASED_SINGLE_PASS", "1") == "1"
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
    # `pending` buffer (resize! amortized geometric growth)
    _prealloc_buf = get(ENV, "SB_ALIASED_PREALLOC_BUF", "1") == "1"   # default ON (hardened 2026-06)
    _cas_stats = _roofline_on()   # CAS redundancy counting is reported under SB_ROOFLINE (no separate env var)
    blksize = C.blksize
    # P1 (BS-mirrored, hardened 2026-06): pass the strided template/B slices straight
    # into mul! instead of convert(Matrix{TC},·)-copying them. 
    _bconv = n_sp > 0
    _noconv = true
    _fusion_diag = false
    _fgroups = nothing
    Cscratch = (!_fuse && _prealloc_buf) ?
        Matrix{TC}(undef, (mode == :AthenB ? (M, N) : (N, M))...) :
        Matrix{TC}(undef, 0, 0)

    Nm = Nmov_total

    # Interleaved Cdense → precompute the GEMM→block strided-scatter map (serial
    # BLAS path only). _canon_inds_for_next_A gates the interleaved emission to
    # serial, so it must not reach the threaded/generic paths.
    lmap = nothing
    if _interleaved
        (_threaded || !can_blas) &&
            error("interleaved Cdense reached non-serial/non-BLAS path (should be gated in _canon_inds_for_next_A)")
        cdims = Int[C.dims[PC + i] for i in 1:length(Cdense)]
        lmap  = _build_interleave_lmap(Cdense, desired_keepA, desired_keepB, cdims)
    end

    _RF_ON[] && (_RF_SETUP_NS[] += Float64(time_ns() - _setup0))
    if _threaded
        return @timeit TIMER "kbd.threadedcontractdensecall" _contract_dense_threaded!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total,
            n_sp, NB, join_posA, sp_strides, _bconv, mode,
            c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
            _kernel_nt, pending)
    else
        # ── SB_OUTSTAT_SCHED=1: output-stationary OWNS every can_blas aliased×dense call ──
        # (any requested order, realized by the kernel's permute phase via out_dense_perm).
        # Additive + default OFF ⇒ runs that don't set it are byte-identical to before.
        # Mutually exclusive with the legacy A/B hook so we never accidentally run legacy.
        _sched_on = get(ENV, "SB_OUTSTAT_SCHED", "0") == "1"
        (_sched_on && get(ENV, "SB_ALIASED_LEGACY", "0") == "1") &&
            error("SB_ALIASED_LEGACY and SB_OUTSTAT_SCHED are mutually exclusive — set only one")
        if _sched_on && can_blas
            return @timeit TIMER "kbd.serial_outstat" _contract_dense_serial_outstat!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total, Nmov_total,
                n_sp, NB, join_posA, sp_strides, _bconv, _noconv, mode, can_blas,
                c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
                _fuse, _direct, _sp, _prealloc_buf, _cas_stats, _fusion_diag,
                pending, key_to_alias, key_to_accum, combined_tid_map, Cscratch, _fgroups, nothing, out_dense_perm)
        end
        # A/B HOOK (SB_ALIASED_LEGACY=1): route to the pre-session legacy reduction-
        # stationary kernel (append!-copy finalize, from git HEAD) for back-to-back
        # finalize-cost comparison. Takes precedence over outstat. Pair with
        # SB_ALIASED_OUTSTAT=0 so the "current" side is the swap-finalize
        # _contract_dense_serial!, NOT the output-stationary kernel.
        # GATED on lmap === nothing: the legacy kernel predates interleaved-Cdense
        # support and ignores lmap, so interleaved contractions MUST fall through to
        # the current _contract_dense_serial! (else wrong energy). The A/B then
        # isolates the finalize change on the non-interleaved contractions.
        if lmap === nothing && get(ENV, "SB_ALIASED_LEGACY", "0") == "1"
            return @timeit TIMER "kbd.serial_legacy" _contract_dense_serial_legacy!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total, Nmov_total,
                n_sp, NB, join_posA, sp_strides, _bconv, _noconv, mode, can_blas,
                c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
                _fuse, _direct, _sp, _prealloc_buf, _cas_stats, _fusion_diag,
                pending, key_to_alias, key_to_accum, combined_tid_map, Cscratch, _fgroups, lmap)
        end
        # OUTPUT-STATIONARY kernel (default): keep-grouped, GEMM-direct, no scatter/hash.
        # Handles the BLAS non-interleaved case; interleaved (lmap) and non-BLAS fall
        # through to the legacy reduction-stationary _contract_dense_serial!.
        # Toggle off with SB_ALIASED_OUTSTAT=0 to A/B against the legacy kernel.
        # OUTPUT-STATIONARY kernel (default ON). Bit-identical (to ~1e-12); back-to-back
        # under matched throttle it runs on par with the legacy kernel (~46 vs ~45 s/sweep,
        # within noise) — the earlier "regression" was a throttle artifact. Profiled below
        # to find where its time goes (redundant per-member B-convert, fill, β-RMW, per-call
        # sort are the suspects). Toggle SB_ALIASED_OUTSTAT=0 for the legacy kernel.
        _outstat = can_blas && lmap === nothing && get(ENV, "SB_ALIASED_OUTSTAT", "1") == "1"
        if _outstat
            return @timeit TIMER "kbd.serial_outstat" _contract_dense_serial_outstat!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total, Nmov_total,
                n_sp, NB, join_posA, sp_strides, _bconv, _noconv, mode, can_blas,
                c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
                _fuse, _direct, _sp, _prealloc_buf, _cas_stats, _fusion_diag,
                pending, key_to_alias, key_to_accum, combined_tid_map, Cscratch, _fgroups, lmap)
        end
        # kbd.serial_current: total time in the current (reduction-stationary) serial kernel.
        return @timeit TIMER "kbd.serial_current" _contract_dense_serial!(C, A, Bp, blksize, M, N, K, Nm, Mmov_total, Nmov_total,
            n_sp, NB, join_posA, sp_strides, _bconv, _noconv, mode, can_blas,
            c_src_kind, c_src_idx, movA_dims_vec, movB_dims_vec, movA_strides, movB_strides,
            _fuse, _direct, _sp, _prealloc_buf, _cas_stats, _fusion_diag,
            pending, key_to_alias, key_to_accum, combined_tid_map, Cscratch, _fgroups, lmap)
    end
end
