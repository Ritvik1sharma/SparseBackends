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
@inline _add_dbg_enabled() = get(ENV, "SB_ALIASED_DEBUG", "0") == "1"
@inline _add_dbg_max()     = parse(Int, get(ENV, "SB_ALIASED_DEBUG_MAX", "12"))

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
        mapA = Dict(l => i for (i, l) in enumerate(labelsA))
        mapB = Dict(l => i for (i, l) in enumerate(labelsB))
        mapC = Dict(l => i for (i, l) in enumerate(labelsC))

        shared_prefix = [l for l in labelsA[1:PA]     if  haskey(mapB, l) && !haskey(mapC, l)]
        c_prefix      = [l for l in labelsA[1:PA]     if  haskey(mapC, l)]
        keepA         = [l for l in labelsA[PA+1:end] if  haskey(mapC, l)]
        red_dense     = [l for l in labelsA[PA+1:end] if !haskey(mapC, l)]
        keepB         = [l for l in labelsB           if  haskey(mapC, l)]

        n_sp    = length(shared_prefix)
        n_rd    = length(red_dense)
        n_keepA = length(keepA)
        n_keepB = length(keepB)
        n_cpfx  = length(c_prefix)
    end

    @timeit TIMER "add.permute_A" begin
        dense_perm = vcat([mapA[l] - PA for l in keepA],
                          [mapA[l] - PA for l in red_dense])
        if dense_perm != collect(1:NA-PA)
            full_perm = vcat(collect(1:PA), dense_perm .+ PA)
            A       = permutedims(A, full_perm)
            labelsA = labelsA[full_perm]
            mapA    = Dict(l => i for (i, l) in enumerate(labelsA))
        end
    end

    @timeit TIMER "add.permute_B" begin
        permB = Vector{Int}(undef, NB)
        let i = 1
            @inbounds for l in red_dense;     permB[i] = mapB[l]; i += 1; end
            @inbounds for l in keepB;         permB[i] = mapB[l]; i += 1; end
            @inbounds for l in shared_prefix; permB[i] = mapB[l]; i += 1; end
        end
        if permB == collect(1:NB)
            Bp = B
        else
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
        K = n_rd    == 0 ? 1 : prod(size(B, mapB[l]) for l in red_dense)
        N = n_keepB == 0 ? 1 : prod(size(B, mapB[l]) for l in keepB)

        join_posA         = [mapA[l] for l in shared_prefix]
        c_prefix_pos_in_A = [mapA[l] for l in c_prefix]

        # Column-major strides into sp dims for linearising sp_vals → Int.
        sp_dims = ntuple(t -> size(B, mapB[shared_prefix[t]]), n_sp)
        sp_stride = Vector{Int}(undef, n_sp)
        let s = 1
            @inbounds for t in 1:n_sp
                sp_stride[t] = s
                s *= sp_dims[t]
            end
        end

        # Canonical C layout: keepA, keepB first then c_prefix last (contiguous M*N slice).
        canon_labels = vcat(keepA, keepB, c_prefix)
        perm_C       = [mapC[l] for l in canon_labels]
        if perm_C == collect(1:NC)
            Ctgt = C
            canon_owns_buffer = false
            # C is freshly zeroed by the caller (wrapped_contract_aliased's
            # `zeros(TC, dimsC...)`), so no re-fill needed here.
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

    # BS-mirror path: one fused mul!(C_slice, A_mat, B_mat, α, 1) per block.
    # The alias-amplified cache path was removed — it provided 0 reuse on
    # PHP-style workloads (each block has unique (tidA, sp_lin), compression
    # ≈ 1.0×) and was pure overhead. See git history if needed.
    #
    # SB_GEMM_FINE_TIMERS=1 splits per-block work into fetch / Bsub view /
    # tmpl view / C_slice view / reshape / mul!. Off by default to keep the
    # tight loop overhead-free.
    @timeit TIMER "add.main_loop" begin
      if get(ENV, "SB_GEMM_FINE_TIMERS", "0") == "1"
        @inbounds for ii in 1:nA
            @timeit TIMER "add.fetch" begin
                akey     = A.keys[ii]
                tidA     = A.alias_ids[ii]
                α        = convert(TC, A.scalars[ii])
                sp_vals  = ntuple(t -> akey[join_posA[t]], n_sp)
                cpfx_idx = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)
            end
            @timeit TIMER "add.bsub_view" begin
                Bsub = (n_sp == 0) ? Bp :
                       @view Bp[ntuple(_ -> Colon(), NB - n_sp)..., sp_vals...]
            end
            @timeit TIMER "add.tmpl_view" begin
                tmpl = _aliased_template_view(A, tidA)
            end
            @timeit TIMER "add.c_slice_view" begin
                C_slice = @view Ctgt[ntuple(_ -> Colon(), n_keepA + n_keepB)..., cpfx_idx...]
            end
            @timeit TIMER "add.reshape" begin
                Bmat = reshape(Bsub, K, N)
                Amat = reshape(tmpl, M, K)
                Cmat = reshape(C_slice, M, N)
            end
            @timeit TIMER "add.mul!" begin
                mul!(Cmat, Amat, Bmat, α, one(TC))
            end
        end
      else
        @inbounds for ii in 1:nA
            akey     = A.keys[ii]
            tidA     = A.alias_ids[ii]
            α        = convert(TC, A.scalars[ii])
            sp_vals  = ntuple(t -> akey[join_posA[t]], n_sp)
            cpfx_idx = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)
            Bsub  = (n_sp == 0) ? Bp :
                    @view Bp[ntuple(_ -> Colon(), NB - n_sp)..., sp_vals...]
            tmpl  = _aliased_template_view(A, tidA)
            C_slice = @view Ctgt[ntuple(_ -> Colon(), n_keepA + n_keepB)..., cpfx_idx...]
            @timeit TIMER "add.gemm" begin
                mul!(reshape(C_slice, M, N), reshape(tmpl, M, K), reshape(Bsub, K, N), α, one(TC))
            end
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
