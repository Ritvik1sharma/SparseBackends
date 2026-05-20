# Offline replay benchmark for the BS×Dense GEMM kernel.
#
# Compares the *current* per-block GEMM strategy (kernel-A) against a
# *fused-K* strategy that concatenates same-c_prefix A blocks along K and
# issues one GEMM per c_prefix slot (kernel-B). Runs on real DMRG inputs
# dumped via SB_GEMM_DUMP=<path> from contract_bs_dense_to_dense!.
#
# Usage:
#   julia --project=temp/edited_packages temp/edited_packages/test_fused_k_replay.jl \
#         temp/edited_packages/results/gemm_dump_md40.jls
#
# Reports per-record and aggregate: time(A), time(B), speedup, max abs diff.

using LinearAlgebra: mul!, BLAS
using Serialization: deserialize
using Printf: @printf

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/gemm_dump_md40.jls"
const NREPEAT = parse(Int, get(ENV, "REPLAY_NREPEAT", "5"))
const VERBOSE = get(ENV, "REPLAY_VERBOSE", "0") == "1"

BLAS.set_num_threads(1)

# -------------------- replay kernels --------------------

# kernel-A: identical to production inner loop (one mul! per A block,
# β=1 accumulating into same C_slice across shared cpfx blocks).
function replay_per_block!(C, rec)
    M, K, N           = rec.M, rec.K, rec.N
    n_sp, n_cpfx      = rec.n_sp, rec.n_cpfx
    n_keepA, n_keepB  = rec.n_keepA, rec.n_keepB
    NB                = rec.NB
    Bp                = rec.Bp
    fill!(C, zero(eltype(C)))
    colons_C = ntuple(_ -> Colon(), n_keepA + n_keepB)
    colons_B = ntuple(_ -> Colon(), NB - n_sp)
    @inbounds for blk in rec.blocks
        Bsub = n_sp == 0 ? Bp : @view Bp[colons_B..., blk.sp_vals...]
        B_mat = reshape(Bsub, K, N)
        C_slice = n_cpfx == 0 ? C :
                  @view C[colons_C..., blk.cpfx_idx...]
        mul!(reshape(C_slice, M, N), blk.A_mat, B_mat, one(eltype(C)), one(eltype(C)))
    end
    return C
end

# kernel-A_no_zero: identical to A but skips fill!(C, 0) — measures
# the zero_C cost in isolation. (Output is garbage; do not check correctness.)
function replay_per_block_no_zero!(C, rec)
    M, K, N           = rec.M, rec.K, rec.N
    n_sp, n_cpfx      = rec.n_sp, rec.n_cpfx
    n_keepA, n_keepB  = rec.n_keepA, rec.n_keepB
    NB                = rec.NB
    Bp                = rec.Bp
    colons_C = ntuple(_ -> Colon(), n_keepA + n_keepB)
    colons_B = ntuple(_ -> Colon(), NB - n_sp)
    @inbounds for blk in rec.blocks
        Bsub = n_sp == 0 ? Bp : @view Bp[colons_B..., blk.sp_vals...]
        B_mat = reshape(Bsub, K, N)
        C_slice = n_cpfx == 0 ? C :
                  @view C[colons_C..., blk.cpfx_idx...]
        mul!(reshape(C_slice, M, N), blk.A_mat, B_mat, one(eltype(C)), one(eltype(C)))
    end
    return C
end

# kernel-A_view_B: slice through PermutedDimsArray(B_orig, permB) instead of
# the materialized Bp. Tests whether eliminating the bdd.permute_B materialization
# (1.59 s + 113 MiB in production) is a win once mul! has to handle a strided B.
function replay_view_B!(C, rec)
    M, K, N           = rec.M, rec.K, rec.N
    n_sp, n_cpfx      = rec.n_sp, rec.n_cpfx
    n_keepA, n_keepB  = rec.n_keepA, rec.n_keepB
    NB                = rec.NB
    # PermutedDimsArray gives a strided view with no copy
    Bp_view = PermutedDimsArray(rec.B_orig, Tuple(rec.permB))
    fill!(C, zero(eltype(C)))
    colons_C = ntuple(_ -> Colon(), n_keepA + n_keepB)
    colons_B = ntuple(_ -> Colon(), NB - n_sp)
    @inbounds for blk in rec.blocks
        Bsub = n_sp == 0 ? Bp_view : @view Bp_view[colons_B..., blk.sp_vals...]
        # Cannot reshape a strided non-contiguous view directly. mul! accepts
        # AbstractMatrix and will fall back to a generic path if strided. Try
        # reshape via copy if it errors.
        B_mat = try
            reshape(Bsub, K, N)
        catch
            # Materialize this single slice (the rest of the dispatch costs
            # nothing). Should still beat materializing all of Bp upfront.
            reshape(collect(Bsub), K, N)
        end
        C_slice = n_cpfx == 0 ? C :
                  @view C[colons_C..., blk.cpfx_idx...]
        mul!(reshape(C_slice, M, N), blk.A_mat, B_mat, one(eltype(C)), one(eltype(C)))
    end
    return C
end

# kernel-B: partition blocks by cpfx_idx; for each group of size g > 1,
# stack A blocks horizontally to (M, K*g) and B slices vertically to (K*g, N)
# and call mul! once. Groups of size 1 fall back to a single mul!.
function replay_fused_k!(C, rec; scratch_A=Ref(Matrix{eltype(rec.blocks[1].A_mat)}(undef,0,0)),
                                  scratch_B=Ref(Matrix{eltype(rec.Bp)}(undef,0,0)))
    M, K, N           = rec.M, rec.K, rec.N
    n_sp, n_cpfx      = rec.n_sp, rec.n_cpfx
    n_keepA, n_keepB  = rec.n_keepA, rec.n_keepB
    NB                = rec.NB
    Bp                = rec.Bp
    blocks            = rec.blocks

    fill!(C, zero(eltype(C)))
    colons_C = ntuple(_ -> Colon(), n_keepA + n_keepB)
    colons_B = ntuple(_ -> Colon(), NB - n_sp)

    # group block indices by cpfx_idx — preserves insertion order
    groups = Dict{NTuple{n_cpfx,Int}, Vector{Int}}()
    @inbounds for (ii, blk) in enumerate(blocks)
        push!(get!(() -> Int[], groups, blk.cpfx_idx), ii)
    end

    @inbounds for (cpfx_idx, idxs) in groups
        g = length(idxs)
        C_slice = n_cpfx == 0 ? C :
                  @view C[colons_C..., cpfx_idx...]
        C_mat = reshape(C_slice, M, N)
        if g == 1
            blk = blocks[idxs[1]]
            Bsub = n_sp == 0 ? Bp : @view Bp[colons_B..., blk.sp_vals...]
            mul!(C_mat, blk.A_mat, reshape(Bsub, K, N),
                 one(eltype(C)), one(eltype(C)))
        else
            Kg = K * g
            # grow scratch if needed
            sA = scratch_A[]
            if size(sA, 1) < M || size(sA, 2) < Kg
                scratch_A[] = Matrix{eltype(sA)}(undef, max(M, size(sA,1)), max(Kg, size(sA,2)))
                sA = scratch_A[]
            end
            sB = scratch_B[]
            if size(sB, 1) < Kg || size(sB, 2) < N
                scratch_B[] = Matrix{eltype(sB)}(undef, max(Kg, size(sB,1)), max(N, size(sB,2)))
                sB = scratch_B[]
            end
            A_stack = @view sA[1:M, 1:Kg]
            B_stack = @view sB[1:Kg, 1:N]
            for (k, ii) in enumerate(idxs)
                blk  = blocks[ii]
                cols = (k-1)*K+1 : k*K
                copyto!(@view(A_stack[:, cols]), blk.A_mat)
                Bsub = n_sp == 0 ? Bp : @view Bp[colons_B..., blk.sp_vals...]
                copyto!(@view(B_stack[cols, :]), reshape(Bsub, K, N))
            end
            mul!(C_mat, A_stack, B_stack,
                 one(eltype(C)), one(eltype(C)))
        end
    end
    return C
end

# -------------------- driver --------------------

records = Any[]
open(PATH, "r") do io
    while !eof(io)
        push!(records, deserialize(io))
    end
end
println("Loaded $(length(records)) records from $PATH")

# Stats on the dump
total_blocks = sum(length(r.blocks) for r in records)
group_sizes  = Int[]
for r in records
    if r.n_cpfx == 0
        push!(group_sizes, length(r.blocks))
    else
        groups = Dict{NTuple{r.n_cpfx,Int}, Int}()
        for blk in r.blocks
            groups[blk.cpfx_idx] = get(groups, blk.cpfx_idx, 0) + 1
        end
        append!(group_sizes, values(groups))
    end
end
@printf("Total A blocks:        %d\n", total_blocks)
@printf("Total cpfx groups:     %d\n", length(group_sizes))
@printf("Avg group size:        %.2f\n", total_blocks / length(group_sizes))
@printf("Max group size:        %d\n", maximum(group_sizes))
@printf("Groups of size 1:      %d (%.1f%%)\n",
        count(==(1), group_sizes), 100*count(==(1), group_sizes)/length(group_sizes))
@printf("Groups of size > 4:    %d (%.1f%%)\n",
        count(>(4), group_sizes), 100*count(>(4), group_sizes)/length(group_sizes))
@printf("Blocks in groups > 1:  %d (%.1f%%)\n",
        sum(g for g in group_sizes if g > 1; init=0),
        100*sum(g for g in group_sizes if g > 1; init=0)/total_blocks)

# Pre-allocate C per record (canon dims), and scratch for fused
function alloc_C(rec)
    Array{rec.TC}(undef, rec.canon_dims...)
end

# Warm up BLAS / JIT — one pass through everything
let CA = alloc_C(records[1]), CB = alloc_C(records[1])
    for rec in records[1:min(20, length(records))]
        sizeA = rec.canon_dims
        Cw = Array{rec.TC}(undef, sizeA...)
        replay_per_block!(Cw, rec)
        replay_fused_k!(Cw, rec)
    end
end

# Correctness: one pass each, compare
max_diff = let
    md = 0.0
    for rec in records
        CA = alloc_C(rec)
        CB = alloc_C(rec)
        replay_per_block!(CA, rec)
        replay_fused_k!(CB, rec)
        d = maximum(abs.(CA .- CB); init=0.0)
        md = max(md, d)
    end
    md
end
@printf("\nCorrectness: max |A - B| = %.3e\n", max_diff)

# Time per kernel (NREPEAT passes through the whole record set)
function bench(kernel!, records, nrepeat)
    Cs = [alloc_C(rec) for rec in records]
    # Trigger compilation on this code path
    for (rec, C) in zip(records, Cs)
        kernel!(C, rec)
    end
    t = @elapsed begin
        for _ in 1:nrepeat
            for (rec, C) in zip(records, Cs)
                kernel!(C, rec)
            end
        end
    end
    return t
end

scratch_A = Ref(Matrix{ComplexF64}(undef,0,0))
scratch_B = Ref(Matrix{ComplexF64}(undef,0,0))
# Use type-flexible wrappers
A_call!(C, rec) = replay_per_block!(C, rec)
B_call!(C, rec) = replay_fused_k!(C, rec; scratch_A=scratch_A, scratch_B=scratch_B)

# kernel-A_fresh_C: alloc fresh C every call → measures the wc.alloc_dense cost
# that pooling would eliminate.
function bench_fresh_C(records, nrepeat)
    # Don't reuse C across calls — measures alloc cost
    t = @elapsed begin
        for _ in 1:nrepeat
            for rec in records
                C = alloc_C(rec)
                replay_per_block!(C, rec)
            end
        end
    end
    return t
end

A_no_zero_call!(C, rec) = replay_per_block_no_zero!(C, rec)
A_view_B_call!(C, rec)  = replay_view_B!(C, rec)

# Correctness check for view_B
let
    rec = records[10]
    CA = alloc_C(rec); CV = alloc_C(rec)
    replay_per_block!(CA, rec)
    replay_view_B!(CV, rec)
    d = maximum(abs.(CA .- CV); init=0.0)
    @printf("view_B correctness on rec[10]: max |A - viewB| = %.3e\n", d)
end

println("\nBenchmarking with NREPEAT=$NREPEAT, single-thread BLAS")
tA   = bench(A_call!,         records, NREPEAT)
tAnz = bench(A_no_zero_call!, records, NREPEAT)
tAvB = bench(A_view_B_call!,  records, NREPEAT)
tAfC = bench_fresh_C(records, NREPEAT)
tB   = bench(B_call!,         records, NREPEAT)

base_us(t) = 1e6*t/(NREPEAT*length(records))
@printf("kernel-A (baseline, pooled C, mat'd Bp):  %.3f s  (%.1f μs/call)\n", tA,   base_us(tA))
@printf("kernel-A_no_zero (skip fill!(C,0)):       %.3f s  (%.1f μs/call)  Δ vs A = %.3f s\n",
        tAnz, base_us(tAnz), tA - tAnz)
@printf("kernel-A_view_B (PermutedDimsArray Bp):   %.3f s  (%.1f μs/call)  Δ vs A = %.3f s\n",
        tAvB, base_us(tAvB), tA - tAvB)
@printf("kernel-A_fresh_C (alloc C per call):      %.3f s  (%.1f μs/call)  Δ vs A = %.3f s  (pool savings)\n",
        tAfC, base_us(tAfC), tAfC - tA)
@printf("kernel-B (fused-K):                       %.3f s  (%.1f μs/call)  speedup %.2fx (%s)\n",
        tB, base_us(tB), tA/tB, tB < tA ? "faster" : "slower")
