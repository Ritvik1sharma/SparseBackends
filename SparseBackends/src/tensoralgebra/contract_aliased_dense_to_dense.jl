# tensoralgebra/contract_aliased_dense_to_dense.jl
#
# Direct kernel: AliasedBlockSparse × Dense → Dense
# Mirrors contract_bs_dense_to_dense! in structure. Two-phase strategy:
#
#   1. Group A blocks by (tidA, sp_vals). For each group, perform ONE GEMM
#      `template_A[tidA] × Bsub(sp_vals) → partial[M×N]` (alias amplification).
#   2. For each block in the group, accumulate `α_i * partial` into the
#      corresponding C slice via BLAS `axpy!`.
#
# Compared to the BS path which does one BLAS GEMM per block:
#   - GEMM count is reduced by compression_ratio (n_blocks / n_templates).
#   - Each block pays only an axpy (M*N work) instead of a GEMM (M*K*N work).
# Net win when K (size of reduced dense dim) is large relative to 1.
#
# To avoid runtime type instability (a previous version used
# `Dict{Tuple{Int, NTuple{n_sp,Int}}, Matrix{TC}}` with `n_sp` a runtime Int —
# this is non-concrete and triggers generic dispatch on every lookup), we
# linearise sp_vals to a single Int via column-major encoding into B's
# shared-prefix dims, giving a flat `Dict{Tuple{Int,Int}, Matrix{TC}}`.

using LinearAlgebra: mul!, BLAS

# Debug: SB_ALIASED_DEBUG=1 enables verbose prints for the first
# SB_ALIASED_DEBUG_MAX calls (default 12 — enough to cover one matvec at
# position 3 in a fresh sweep), then goes silent.  Inspect what index
# orders the kernel sees vs the canonical layout it permutes to.
const _ADD_DBG_COUNT = Ref(0)

# Cumulative alias-structure counters for the aliased×dense kernel. Updated
# once per call (cheap, no per-block work). Reset via _reset_alias_stats!().
mutable struct _AliasStats
    calls         :: Int
    total_blocks  :: Int
    total_tmpls   :: Int  # sum of A.n_templates seen
    unique_tidA   :: Int  # sum of len(unique A.alias_ids per call)
    total_MKN     :: Int
end
const _ALIAS_STATS = _AliasStats(0, 0, 0, 0, 0)
function _reset_alias_stats!()
    s = _ALIAS_STATS
    s.calls = 0; s.total_blocks = 0; s.total_tmpls = 0
    s.unique_tidA = 0; s.total_MKN = 0
end
function _report_alias_stats()
    s = _ALIAS_STATS
    s.calls == 0 && return
    avg_blocks  = s.total_blocks / s.calls
    avg_tmpls   = s.total_tmpls / s.calls
    avg_unique  = s.unique_tidA / s.calls
    println("\n========== aliased×dense per-call alias structure ==========")
    println("  calls            = ", s.calls)
    println("  Σ blocks         = ", s.total_blocks,
            "    avg/call = ", round(avg_blocks; digits=2))
    println("  Σ A.n_templates  = ", s.total_tmpls,
            "    avg/call = ", round(avg_tmpls; digits=2))
    println("  Σ unique tidA    = ", s.unique_tidA,
            "    avg/call = ", round(avg_unique; digits=2))
    println("  block/template ratio (avg) = ",
            round(avg_blocks / max(avg_tmpls, 1e-9); digits=3),
            "  ← how many blocks per unique template per call")
    println("  block/unique-tidA   (avg) = ",
            round(avg_blocks / max(avg_unique, 1e-9); digits=3),
            "  ← reuse achievable by batching same-tidA blocks")
    println("  Σ MKN work       = ", s.total_MKN)
end
@inline function _record_alias_call!(A::AliasedBlockSparse, M, K, N)
    s = _ALIAS_STATS
    s.calls += 1
    nb = length(A.keys)
    s.total_blocks += nb
    s.total_tmpls  += A.n_templates
    # Count unique tidA actually referenced in this call (could be < n_templates
    # if some templates are dead, or = if all referenced).
    seen = Set{Int}()
    @inbounds for i in 1:nb
        push!(seen, A.alias_ids[i])
    end
    s.unique_tidA += length(seen)
    s.total_MKN   += nb * M * K * N
end
@inline _add_dbg_enabled() = get(ENV, "SB_ALIASED_DEBUG", "0") == "1"
@inline _add_dbg_max()     = parse(Int, get(ENV, "SB_ALIASED_DEBUG_MAX", "12"))

# Linear-search position lookup. For N ≤ 8 labels this beats Dict-build +
# hash lookup. Returns 0 if not found (caller checks via `!= 0`).
@inline function _posin(l, labels)
    @inbounds for i in eachindex(labels)
        labels[i] == l && return i
    end
    return 0
end
@inline _has(l, labels) = _posin(l, labels) != 0

# Diagnostic: when SB_PERM_PROFILE=1, print the first SB_PERMB_DBG_MAX kernel
# calls in which permute_B fires (i.e. env layout != [red_dense, keepB,
# shared_prefix]), then the aggregate pattern + per-step tables.
const _PERMB_DBG_COUNT = Ref(0)

# Pattern profiler: when SB_PERM_PROFILE=1, accumulate (permB, perm_C, dims)
# tuples and their occurrence counts so we can see which orderings dominate.
# Print a sorted-by-count report on demand via `_report_perm_profile()`.
mutable struct _PermSig
    permB    :: Vector{Int}
    perm_C   :: Vector{Int}
    n_sp     :: Int
    n_rd     :: Int
    n_keepA  :: Int
    n_keepB  :: Int
    n_cpfx   :: Int
end
Base.hash(s::_PermSig, h::UInt) =
    hash(s.permB, hash(s.perm_C, hash(s.n_sp, hash(s.n_rd, hash(s.n_keepA,
        hash(s.n_keepB, hash(s.n_cpfx, h)))))))
Base.:(==)(a::_PermSig, b::_PermSig) =
    a.permB == b.permB && a.perm_C == b.perm_C &&
    a.n_sp == b.n_sp && a.n_rd == b.n_rd && a.n_keepA == b.n_keepA &&
    a.n_keepB == b.n_keepB && a.n_cpfx == b.n_cpfx
const _PERM_PROFILE = Dict{_PermSig, Int}()
# Per-matvec-step permB tally (keyed by ENV["SB_STEP"]): value = [calls, permB_fired].
# Only populated when SB_PERM_PROFILE=1. Answers "which step permutes".
const _PERMB_STEP = Dict{String, Vector{Int}}()
function _reset_perm_profile!()
    empty!(_PERM_PROFILE)
    empty!(_PERMB_STEP)
end
function _report_perm_profile()
    isempty(_PERM_PROFILE) && (println("[perm_profile] (empty)"); return)
    total = sum(values(_PERM_PROFILE))
    items = sort(collect(_PERM_PROFILE); by = x -> -x[2])
    println("\n========== perm_profile (", length(items), " unique patterns, ",
            total, " calls) ==========")
    println(rpad("count", 8), rpad("frac", 8),
            rpad("(nkA,nkB,ncp,nsp,nrd)", 22),
            rpad("permB", 30), "perm_C")
    @inbounds for (s, c) in items[1:min(end, 20)]
        sig = string("(", s.n_keepA, ",", s.n_keepB, ",", s.n_cpfx, ",", s.n_sp, ",", s.n_rd, ")")
        permB_is_id = s.permB == collect(1:length(s.permB))
        perm_C_is_id = s.perm_C == collect(1:length(s.perm_C))
        permB_str = string(s.permB, permB_is_id ? " ✓id" : "")
        perm_C_str = string(s.perm_C, perm_C_is_id ? " ✓id" : "")
        println(rpad(c, 8), rpad(string(round(c/total*100; digits=2), "%"), 8),
                rpad(sig, 22), rpad(permB_str, 30), perm_C_str)
    end
    println("==========")
    if !isempty(_PERMB_STEP)
        println("---------- permB by matvec step (SB_STEP) ----------")
        println(rpad("step", 8), rpad("calls", 8), rpad("permB_fired", 14), "fire_frac")
        for st in sort(collect(keys(_PERMB_STEP)))
            c, f = _PERMB_STEP[st]
            println(rpad(st, 8), rpad(c, 8), rpad(f, 14), round(f/max(c,1)*100; digits=1), "%")
        end
        println("----------")
    end
end
@inline _perm_profile_enabled() = get(ENV, "SB_PERM_PROFILE", "0") == "1"

function contract_aliased_dense_to_dense!(
    C        :: AbstractArray{TC},
    labelsC  :: AbstractVector{Label},
    A        :: AliasedBlockSparse{TA,NA,NA2,PA},
    labelsA  :: AbstractVector{Label},
    B        :: AbstractArray{TB,NB},
    labelsB  :: AbstractVector{Label},
) where {TC,TA,NA,NA2,PA,TB,NB}
    NC = ndims(C)
    if isempty(A.keys)
        fill!(C, zero(TC))
        return C
    end

    @timeit TIMER "add.classify" begin
        shared_prefix = [l for l in labelsA[1:PA]     if  _has(l, labelsB) && !_has(l, labelsC)]
        c_prefix      = [l for l in labelsA[1:PA]     if  _has(l, labelsC)]
        keepA         = [l for l in labelsA[PA+1:end] if  _has(l, labelsC)]
        red_dense     = [l for l in labelsA[PA+1:end] if !_has(l, labelsC)]
        keepB         = [l for l in labelsB           if  _has(l, labelsC)]

        n_sp    = length(shared_prefix)
        n_rd    = length(red_dense)
        n_keepA = length(keepA)
        n_keepB = length(keepB)
        n_cpfx  = length(c_prefix)
    end

    @timeit TIMER "add.permute_A" begin
        dense_perm = vcat([_posin(l, labelsA) - PA for l in keepA],
                          [_posin(l, labelsA) - PA for l in red_dense])
        if dense_perm != collect(1:NA-PA)
            full_perm = vcat(collect(1:PA), dense_perm .+ PA)
            A       = permutedims(A, full_perm)
            labelsA = labelsA[full_perm]
        end
    end

    @timeit TIMER "add.permute_B" begin
        permB = Vector{Int}(undef, NB)
        let i = 1
            @inbounds for l in red_dense;     permB[i] = _posin(l, labelsB); i += 1; end
            @inbounds for l in keepB;         permB[i] = _posin(l, labelsB); i += 1; end
            @inbounds for l in shared_prefix; permB[i] = _posin(l, labelsB); i += 1; end
        end
        if permB == collect(1:NB)
            Bp = B
        else
            # Per-call permute_B detail: folded under the single SB_PERM_PROFILE
            # flag (was its own SB_PERMB_DBG knob). Still capped via
            # SB_PERMB_DBG_MAX so it prints the first N firings, then the
            # aggregate + per-step tables come from _report_perm_profile.
            if _perm_profile_enabled() &&
               _PERMB_DBG_COUNT[] < parse(Int, get(ENV, "SB_PERMB_DBG_MAX", "12"))
                _PERMB_DBG_COUNT[] += 1
                println("\n[permB #", _PERMB_DBG_COUNT[], "] permute_B firing  (SB_STEP=", get(ENV, "SB_STEP", "?"), ")")
                println("  labelsA = ", labelsA, "  (PA = ", PA, ")")
                println("  labelsB = ", labelsB)
                println("  labelsC = ", labelsC)
                println("  red_dense     = ", red_dense)
                println("  keepB         = ", keepB)
                println("  shared_prefix = ", shared_prefix)
                println("  desired B order = ", vcat(red_dense, keepB, shared_prefix),
                        "  (red_dense first, shared_prefix last)")
                println("  actual permB    = ", permB)
            end
            # Bp = PermutedDimsArray(B, permB)
            dimsBp     = ntuple(i -> size(B, permB[i]), Val(NB))
            nB_total   = length(B)
            permB_buf  = _bdd_permB_buffer(TB, nB_total)
            Bp         = reshape(view(permB_buf, 1:nB_total), dimsBp)
            Base.permutedims!(Bp, B, permB)
        end
    end


    @timeit TIMER "add.setup" begin
        dimsA_d = A.dims[PA+1:end]
        M = n_keepA == 0 ? 1 : prod(dimsA_d[1:n_keepA])
        K = n_rd    == 0 ? 1 : prod(size(B, _posin(l, labelsB)) for l in red_dense)
        N = n_keepB == 0 ? 1 : prod(size(B, _posin(l, labelsB)) for l in keepB)

        join_posA         = [_posin(l, labelsA) for l in shared_prefix]
        c_prefix_pos_in_A = [_posin(l, labelsA) for l in c_prefix]

        # Column-major strides into sp dims for linearising sp_vals → Int.
        sp_stride = Vector{Int}(undef, n_sp)
        let s = 1
            @inbounds for t in 1:n_sp
                sp_stride[t] = s
                s *= size(B, _posin(shared_prefix[t], labelsB))
            end
        end

        # Column-major strides into c-prefix dims for linearising cpfx_idx → Int.
        cpfx_stride = Vector{Int}(undef, n_cpfx)
        let s = 1
            @inbounds for j in 1:n_cpfx
                cpfx_stride[j] = s
                s *= A.dims[c_prefix_pos_in_A[j]]
            end
        end

        # Canonical C layout: keepA, keepB first then c_prefix last (contiguous M*N slice).
        canon_labels = vcat(keepA, keepB, c_prefix)
        perm_C       = [_posin(l, labelsC) for l in canon_labels]
        if _flop_count_enabled()
            add_reshuffle!(permB != collect(1:NB), perm_C != collect(1:NC))
        end
        if _perm_profile_enabled()
            sig = _PermSig(copy(permB), copy(perm_C), n_sp, n_rd, n_keepA, n_keepB, n_cpfx)
            _PERM_PROFILE[sig] = get(_PERM_PROFILE, sig, 0) + 1
            _st = get(ENV, "SB_STEP", "?")
            _v  = get!(_PERMB_STEP, _st, Int[0, 0])
            _v[1] += 1
            (permB != collect(1:NB)) && (_v[2] += 1)
        end

        # ── Direct-write detection (skips permute_back) ──────────────────────
        # Goal: write each block's GEMM result straight into the (pre-zeroed) C
        # with NO permute_back. This is possible as a single BLAS GEMM whenever
        # one kept group (keepA or keepB) is the unit-stride LEADING run [1..r]
        # of labelsC (kernel order) and the OTHER kept group is a contiguous run
        # [p..p+c-1] (kernel order) with p > r — then every remaining axis is a
        # c_prefix selector and labelsC factorizes as
        #     C ≅ reshape(C, reshR, reshGAP, reshCcols, reshTAIL)
        # (reshR = leading row block, stride 1; reshGAP/reshTAIL = c_prefix axes
        # before/after the column run; reshCcols = the column run). For a block we
        # fix (GAP=g, TAIL=t) and mul! into the reshR×reshCcols slice
        # view(C4,:,g,:,t), whose column stride is reshR·reshGAP — a *uniform*
        # stride, so BLAS handles it directly:
        #   • GAP == 1  → stride == reshR → CONTIGUOUS block (the fast common
        #                 case; written via a flat reshape with no SubArray);
        #   • GAP  > 1  → strided (a c_prefix axis sits between the kept groups);
        #                 still one GEMM, just ldc = reshR·reshGAP > reshR.
        # keepB-leading is served by the transposed GEMM Cᵀ = Bᵀ·Aᵀ (BLAS flag,
        # no data motion). Per-block ii accumulation order is unchanged ⇒
        # bit-identical to the permute_back path.
        # Only layouts where a kept group is INTERNALLY PERMUTED (column run not
        # contiguous in kernel order ⇒ non-uniform column stride ⇒ no single GEMM
        # can scatter it) fall back to canonical Ctgt + permute_back below.
        direct_rows   = :none           # :keepA | :keepB | :none
        direct_strided = false          # true ⇒ GAP>1 (strided slice); false ⇒ contiguous
        cpfx_strideC  = Int[]           # column-major stride of each c_prefix axis in C
        reshR = 1; reshGAP = 1; reshCcols = 1; reshTAIL = 1
        if perm_C != collect(1:NC)
            kApos = [_posin(l, labelsC) for l in keepA]
            kBpos = [_posin(l, labelsC) for l in keepB]
            row_kind = :none; r = 0; colpos = Int[]
            if n_keepA > 0 && kApos == collect(1:n_keepA)
                row_kind = :keepA; r = n_keepA; colpos = kBpos
            elseif n_keepB > 0 && kBpos == collect(1:n_keepB)
                row_kind = :keepB; r = n_keepB; colpos = kApos
            end
            # The other kept group must be empty, or a contiguous run (kernel
            # order) starting after the row group. Contiguity in kernel order is
            # what guarantees a single uniform column stride.
            if row_kind != :none
                p = isempty(colpos) ? r + 1 : colpos[1]
                c = length(colpos)
                if isempty(colpos) || (p > r && colpos == collect(p : p+c-1))
                    _dpos(lo, hi) = (lo > hi) ? 1 : prod(size(C, i) for i in lo:hi)
                    reshR      = _dpos(1, r)
                    reshGAP    = _dpos(r+1, p-1)
                    reshCcols  = _dpos(p, p+c-1)
                    reshTAIL   = _dpos(p+c, NC)
                    direct_rows    = row_kind
                    direct_strided = reshGAP != 1
                    sC = Vector{Int}(undef, NC); sC[1] = 1
                    @inbounds for i in 2:NC; sC[i] = sC[i-1] * size(C, i-1); end
                    cpfx_strideC = [sC[_posin(l, labelsC)] for l in c_prefix]
                end
            end
        end

        if perm_C == collect(1:NC) || direct_rows != :none
            # Identity layout OR strided-writable layout → write straight into
            # the caller's C (already zeroed by wrapped_contract_aliased's
            # `zeros(TC, dimsC...)`); no scratch, no permute_back.
            Ctgt = C
            canon_owns_buffer = false
        else
            canon_dims = ntuple(i -> size(C, perm_C[i]), Val(NC))
            n_canon    = prod(canon_dims)
            flat_buf   = _bdd_ctgt_buffer(TC, n_canon)
            Ctgt       = reshape(view(flat_buf, 1:n_canon), canon_dims)
            @timeit TIMER "add.zero_C" fill!(Ctgt, zero(TC))
            canon_owns_buffer = true
        end
    end

    # ── Debug print: what the kernel sees vs what it wants ──────────────────
    if _add_dbg_enabled() && _ADD_DBG_COUNT[] < _add_dbg_max()
        _ADD_DBG_COUNT[] += 1
        idx = _ADD_DBG_COUNT[]
        println("\n[SB_ALIASED_DEBUG call #", idx, "]  ProjMPO matvec contraction")
        println("  A (aliased H):  PA=", PA, "  N=", NA,
                "  n_templates=", A.n_templates, "  n_blocks=", length(A.keys))
        println("    labelsA = ", labelsA, "   A.dims = ", A.dims)
        println("    (first PA labels = sparse prefix, last NA-PA = dense tail)")
        println("  B (env):  N=", NB, "  size=", size(B))
        println("    labelsB = ", labelsB)
        println("  Classification:")
        println("    shared_prefix (red, in A.prefix) = ", shared_prefix)
        println("    c_prefix      (kept, in A.prefix) = ", c_prefix)
        println("    keepA         (kept, in A.tail)   = ", keepA)
        println("    red_dense     (red, in A.tail)    = ", red_dense)
        println("    keepB         (kept, in B)        = ", keepB)
        println("  permB: actual = ", permB)
        println("         desired B order = ", vcat(red_dense, keepB, shared_prefix),
                "  (red_dense FASTEST, shared_prefix LAST)")
        println("         ",
                permB == collect(1:NB) ?
                  "✓ env already canonical — permute_B is identity (no cost)" :
                  "✗ permute_B is non-identity — env needs physical reorder")
        println("  C: labelsC = ", labelsC, "  NC=", NC,
                "  MNK = (", M, ", ", N, ", ", K, ")  per-block work = ", M*K*N)
        println("    canon_labels (kernel's preferred C order) = ", canon_labels)
        println("         ",
                perm_C == collect(1:NC) ?
                  "✓ output C is already in canonical layout (no permute_back)" :
                  "✗ output C requires permute_back at end")
    end

    nA = length(A.keys)
    if get(ENV, "SB_ALIAS_STATS", "0") == "1"
        _record_alias_call!(A, M, K, N)
    end
    if _flop_count_enabled()
        # Actual per-block GEMM work the aliased kernel performs.
        actual_macs = nA * M * K * N
        # MACs a fully-dense contraction of this same step would cost: the
        # sparse channel (shared_prefix contracted, c_prefix kept) un-factored.
        SP   = n_sp   == 0 ? 1 : prod(size(B, _posin(l, labelsB)) for l in shared_prefix)
        Cpfx = n_cpfx == 0 ? 1 : prod(A.dims[c_prefix_pos_in_A[j]] for j in 1:n_cpfx)
        denseequiv_macs = M * K * N * SP * Cpfx
        add_aliased_macs!(actual_macs, denseequiv_macs)
    end

    # Lean per-block path: one fused mul!(C_slice, A_mat, B_mat, α, 1) per
    # block, with the slice metadata derived inline from integer offsets into
    # the flat Bp / C / Ctgt buffers. No ntuple, no Colon() splat, no multi-axis
    # SubArray construction in the hot loop.
    #
    # Three paths share the same per-block sp_lin / base arithmetic:
    #   contiguous direct — flat reshape of C at offset `base` (GAP==1); leanest;
    #   strided direct    — view(C4,:,g,:,t) into the 4-region reshape (GAP>1);
    #   :none             — accumulate into scratch Ctgt, permute_back after.
    # Both direct paths skip permute_back/scratch/zero_C and use the transposed
    # GEMM Cᵀ=Bᵀ·Aᵀ for keepB-leading.
    Bp_vec = vec(Bp)
    KN = K * N
    MN = M * N
    @timeit TIMER "add.main_loop" begin
      if direct_rows != :none && !direct_strided
        # Contiguous fast path: the reshR×reshCcols slice is a contiguous MN
        # block at flat offset `base` (c_prefix trails the kept groups).
        row_is_B = direct_rows == :keepB
        C_vec = vec(C)
        @inbounds for ii in 1:nA
          akey = A.keys[ii]
          α    = convert(TC, A.scalars[ii])
          sp_lin = 0
          for t in 1:n_sp; sp_lin += (akey[join_posA[t]] - 1) * sp_stride[t]; end
          base = 0
          for j in 1:n_cpfx; base += (akey[c_prefix_pos_in_A[j]] - 1) * cpfx_strideC[j]; end
          Boff = sp_lin * KN
          Bmat = reshape(view(Bp_vec, Boff + 1 : Boff + KN), K, N)
          Amat = reshape(_aliased_template_view(A, A.alias_ids[ii]), M, K)
          Cmat = reshape(view(C_vec, base + 1 : base + MN), reshR, reshCcols)
          if row_is_B
            mul!(Cmat, transpose(Bmat), transpose(Amat), α, one(TC))   # N×M = Bᵀ·Aᵀ
          else
            mul!(Cmat, Amat, Bmat, α, one(TC))                         # M×N
          end
        end
      elseif direct_rows != :none
        # Strided direct path: a c_prefix axis sits between the kept groups, so
        # the slice has a uniform column stride reshR·reshGAP > reshR. Still one
        # GEMM per block — into the strided view(C4,:,g,:,t) of the pre-zeroed C.
        row_is_B = direct_rows == :keepB
        C4  = reshape(C, reshR, reshGAP, reshCcols, reshTAIL)
        RGC = reshR * reshGAP * reshCcols
        @inbounds for ii in 1:nA
          akey = A.keys[ii]
          α    = convert(TC, A.scalars[ii])
          sp_lin = 0
          for t in 1:n_sp; sp_lin += (akey[join_posA[t]] - 1) * sp_stride[t]; end
          base = 0
          for j in 1:n_cpfx; base += (akey[c_prefix_pos_in_A[j]] - 1) * cpfx_strideC[j]; end
          g = (base ÷ reshR) % reshGAP
          t = base ÷ RGC
          Boff = sp_lin * KN
          Bmat = reshape(view(Bp_vec, Boff + 1 : Boff + KN), K, N)
          Amat = reshape(_aliased_template_view(A, A.alias_ids[ii]), M, K)
          Cmat = view(C4, :, g + 1, :, t + 1)   # reshR×reshCcols, col stride reshR·reshGAP
          if row_is_B
            mul!(Cmat, transpose(Bmat), transpose(Amat), α, one(TC))
          else
            mul!(Cmat, Amat, Bmat, α, one(TC))
          end
        end
      else
        Ctgt_vec = vec(Ctgt)
        @inbounds for ii in 1:nA
          akey = A.keys[ii]
          α    = convert(TC, A.scalars[ii])
          sp_lin = 0
          for t in 1:n_sp; sp_lin += (akey[join_posA[t]] - 1) * sp_stride[t]; end
          cp_lin = 0
          for j in 1:n_cpfx; cp_lin += (akey[c_prefix_pos_in_A[j]] - 1) * cpfx_stride[j]; end
          Boff = sp_lin * KN
          Coff = cp_lin * MN
          Bmat = reshape(view(Bp_vec,   Boff + 1 : Boff + KN), K, N)
          Cmat = reshape(view(Ctgt_vec, Coff + 1 : Coff + MN), M, N)
          Amat = reshape(_aliased_template_view(A, A.alias_ids[ii]), M, K)
          mul!(Cmat, Amat, Bmat, α, one(TC))
        end
      end
    end

    if canon_owns_buffer
        @timeit TIMER "add.permute_back" begin
            inv_perm_C = invperm(perm_C)
            Base.permutedims!(C, Ctgt, inv_perm_C)
        end
    end

    return C
end
