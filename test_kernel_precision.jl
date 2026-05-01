"""
test_kernel_precision.jl

Measures the numerical error introduced by `contract_prefix_outer_bd!`
compared to a reference Dense×Dense computation.

Structure:
  A: NewBlockSparseSorted, PA=2 (axes: :left, :rlab), N2A=1 (dense axis: :da)
     dims = (left_dim=4, rlab_dim=3, da_dim=4)
     All 12 (left, rlab) blocks present → 4 ckey groups, each nb=3
  B: Dense matrix, dims = (rlab_dim=3, b_dim=3)
  C: NewBlockSparseSorted, PC=1 (:left), N2C=2 (:da, :b)
     dims = (left_dim=4, da_dim=4, b_dim=3), blksize=12

Reference: dense A × dense B contracted over rlab axis, pure Julia triple loop.

Tests:
  [current]  current kernel (BLAS fused GEMM path)
  [kahan]    Kahan-compensated kernel (Fix A)  ← patched inline below
  [dense_env] materialize A to dense, use standard Julia matmul (proxy for Fix B)
"""

using SparseBackends
using Random, LinearAlgebra, Printf

# ---- Access internal symbols needed for direct kernel test ----------------
using SparseBackends: NewBlockSparseSorted, _ensure_block!, _block_view

# ---- Build the NewBlockSparseSorted A tensor ------------------------------
function make_A(T::Type, left_dim, rlab_dim, da_dim; seed=42)
    rng = MersenneTwister(seed)
    # PA=2, N2A=1 → N=3 axes: (left, rlab, da), blksize=da_dim
    A = NewBlockSparseSorted{T,3,1}((left_dim, rlab_dim, da_dim))
    for li in 1:left_dim
        for rv in 1:rlab_dim
            id = _ensure_block!(A, (li, rv))
            v  = _block_view(A, id)
            for k in 1:da_dim
                v[k] = randn(rng, T)
            end
        end
    end
    return A
end

# ---- Reference: materialize A to dense array and contract with B ----------
function reference_contraction(A::NewBlockSparseSorted{T,3,1}, B::Matrix{T},
                                left_dim, rlab_dim, da_dim, b_dim) where T
    # A_dense[left, rlab, da], B[rlab, b] → C_ref[left, da, b]
    A_dense = zeros(T, left_dim, rlab_dim, da_dim)
    for (key, id) in zip(A.keys, A.ids)
        li, rv = key
        v = _block_view(A, id)
        for k in 1:da_dim
            A_dense[li, rv, k] = v[k]
        end
    end

    C_ref = zeros(T, left_dim, da_dim, b_dim)
    for li in 1:left_dim
        for rv in 1:rlab_dim
            for da in 1:da_dim
                for b in 1:b_dim
                    C_ref[li, da, b] += A_dense[li, rv, da] * B[rv, b]
                end
            end
        end
    end
    return C_ref
end

# ---- Extract C result as dense array for comparison ----------------------
function c_to_dense(C::NewBlockSparseSorted, left_dim, da_dim, b_dim)
    out = zeros(eltype(C), left_dim, da_dim, b_dim)
    for (key, id) in zip(C.keys, C.ids)
        (li,) = key
        blk = _block_view(C, id)
        # C.blksize = da_dim * b_dim, layout [da, b] = dense_order :AB
        for da in 1:da_dim
            for b in 1:b_dim
                out[li, da, b] = blk[(b-1)*da_dim + da]
            end
        end
    end
    return out
end

# ---- Kahan kernel (inlined copy of contract_prefix_outer_bd! with Kahan) --
function contract_prefix_outer_bd_kahan!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC, A::NewBlockSparseSorted{TA,NA,N2A,PA}, labelsA,
    B::StridedArray{TB,NB}, labelsB, mapA, mapB, rlab
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
    # (Re-implementation of the BLAS path using Kahan compensated summation)
    axisAr = mapA[rlab]
    @assert axisAr == PA "rlab must be last prefix axis (call after permuting)"
    axisBr = mapB[rlab]
    R = size(B, axisBr)

    # permute B so rlab is last
    permB = (axisBr == NB) ? nothing :
            [d for d in 1:NB if d != axisBr]
    B_r_last = (permB === nothing) ? B : permutedims(B, vcat(permB, [axisBr]))
    chunkB = length(B_r_last) ÷ R
    Bmat = reshape(B_r_last, chunkB, R)

    Adense = labelsA[PA+1:NA]
    Bwo = [lab for lab in labelsB if lab != rlab]
    Cdense = collect(labelsC[PC+1:NC])
    dense_order = (Cdense == vcat(Adense, Bwo)) ? :AB :
                  (Cdense == vcat(Bwo, Adense)) ? :BA :
                  error("C dense tail mismatch")

    chunkA = A.blksize
    @assert C.blksize == chunkA * chunkB

    empty!(C.keys); empty!(C.ids); empty!(C.data)

    src = Vector{Int}(undef, PC)
    for j in 1:PC
        src[j] = mapA[labelsC[j]]
    end

    α1 = one(TC)

    group_order = Vector{NTuple{PC,Int}}()
    groups = Dict{NTuple{PC,Int}, Vector{Int}}()
    for iA in eachindex(A.keys)
        akey = A.keys[iA]
        ckey = ntuple(j -> akey[src[j]], Val(PC))
        idxs = get(groups, ckey, nothing)
        if idxs === nothing
            groups[ckey] = Int[iA]
            push!(group_order, ckey)
        else
            push!(idxs, iA)
        end
    end

    for ckey in group_order
        idxs = groups[ckey]
        nb   = length(idxs)
        cid  = _ensure_block!(C, ckey)
        Cvec = _block_view(C, cid)

        Amat       = Matrix{TC}(undef, chunkA, nb)
        Bmat_local = Matrix{TC}(undef, chunkB, nb)
        for k in 1:nb
            iA_k = idxs[k]
            rv   = A.keys[iA_k][PA]
            Avec = _block_view(A, A.ids[iA_k])
            for i in 1:chunkA; Amat[i, k] = convert(TC, Avec[i]); end
            for j in 1:chunkB; Bmat_local[j, k] = Bmat[j, rv]; end
        end

        # Kahan-compensated outer product accumulation
        if dense_order === :AB
            Cmat = reshape(Cvec, chunkA, chunkB)
            comp = zeros(TC, chunkA, chunkB)
            for k in 1:nb
                @inbounds for j in 1:chunkB
                    @inbounds for i in 1:chunkA
                        y = α1 * Amat[i,k] * Bmat_local[j,k] - comp[i,j]
                        t = Cmat[i,j] + y
                        comp[i,j] = (t - Cmat[i,j]) - y
                        Cmat[i,j] = t
                    end
                end
            end
        else
            Cmat = reshape(Cvec, chunkB, chunkA)
            comp = zeros(TC, chunkB, chunkA)
            for k in 1:nb
                @inbounds for j in 1:chunkA
                    @inbounds for i in 1:chunkB
                        y = α1 * Bmat_local[i,k] * Amat[j,k] - comp[i,j]
                        t = Cmat[i,j] + y
                        comp[i,j] = (t - Cmat[i,j]) - y
                        Cmat[i,j] = t
                    end
                end
            end
        end
    end
    return C
end

# ---- Dense proxy for Fix B: materialize A before contracting -------------
function contract_dense_proxy(A::NewBlockSparseSorted{T,3,1}, B::Matrix{T},
                               left_dim, rlab_dim, da_dim, b_dim) where T
    # Same reference_contraction but structured like dense BLAS would do it:
    # one large GEMM on the materialized A.
    A_mat = zeros(T, left_dim * da_dim, rlab_dim)
    for (key, id) in zip(A.keys, A.ids)
        li, rv = key
        v = _block_view(A, id)
        for da in 1:da_dim
            A_mat[(li-1)*da_dim + da, rv] = v[da]
        end
    end
    C_mat = A_mat * B   # (left_dim*da_dim, b_dim)
    C_ref = reshape(C_mat, left_dim, da_dim, b_dim)
    return C_ref
end

# ---- Run tests ------------------------------------------------------------
function run_precision_test(; T=Float64, left_dim=4, rlab_dim=3, da_dim=4, b_dim=3, seed=42)
    println("\n=== Kernel Precision Test ===")
    println("T=$T  left_dim=$left_dim  rlab_dim=$rlab_dim  da_dim=$da_dim  b_dim=$b_dim")

    rng = MersenneTwister(seed + 100)
    B = randn(rng, T, rlab_dim, b_dim)

    A = make_A(T, left_dim, rlab_dim, da_dim; seed=seed)

    # --- labels and maps ---
    labelsA = [:left, :rlab, :da]
    labelsB = [:rlab, :b]
    labelsC = [:left, :da, :b]
    rlab    = :rlab
    mapA = Dict(:left => 1, :rlab => 2, :da => 3)
    mapB = Dict(:rlab => 1, :b => 2)

    # --- Reference answer (triple loop, no BLAS) ---
    C_ref = reference_contraction(A, B, left_dim, rlab_dim, da_dim, b_dim)

    # --- BLAS proxy (Fix B analogue: one big matmul) ---
    C_dense = contract_dense_proxy(A, B, left_dim, rlab_dim, da_dim, b_dim)
    err_dense = norm(C_ref - C_dense)
    println("  [dense_blas proxy]   ‖C_ref - C_dense_blas‖ = $err_dense")

    # --- Current kernel (BLAS fused GEMM via contract!) ---
    # Construct a fresh C
    C_current = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    # Need to call via SparseBackends contract! dispatch which routes to contract_prefix_outer_bd!
    # We call directly using the internal function (accessible since it's in the same package scope)
    # Use _permute_r_to_last_prefix first (as the function expects PA == axisAr already done)
    # The contract_prefix_outer_bd! function calls _permute_r_to_last_prefix internally,
    # so we can call it directly without pre-permuting.
    SparseBackends.contract_prefix_outer_bd!(
        C_current, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_cur_dense = c_to_dense(C_current, left_dim, da_dim, b_dim)
    err_current = norm(C_ref - C_cur_dense)
    err_vs_dense = norm(C_dense - C_cur_dense)
    println("  [current kernel]     ‖C_ref - C_current‖    = $err_current")
    println("  [current vs BLAS]    ‖C_dense_blas - C_current‖ = $err_vs_dense")

    # --- Kahan kernel ---
    C_kahan = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    contract_prefix_outer_bd_kahan!(
        C_kahan, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_kahan_dense = c_to_dense(C_kahan, left_dim, da_dim, b_dim)
    err_kahan = norm(C_ref - C_kahan_dense)
    err_kahan_vs_dense = norm(C_dense - C_kahan_dense)
    println("  [kahan kernel]       ‖C_ref - C_kahan‖       = $err_kahan")
    println("  [kahan vs BLAS]      ‖C_dense_blas - C_kahan‖ = $err_kahan_vs_dense")

    println("  --- Summary ---")
    @printf("  current vs ref:  %.3e\n", err_current)
    @printf("  kahan   vs ref:  %.3e\n", err_kahan)
    @printf("  dense   vs ref:  %.3e\n", err_dense)
    @printf("  kahan improvement ratio vs current: %.1fx\n",
            err_current > 0 ? err_current / max(err_kahan, 1e-300) : NaN)

    return (; err_current, err_kahan, err_dense)
end

# ---- Also test with larger nb (more blocks per group) to stress Kahan -----
function run_precision_test_large_nb(; T=Float64, left_dim=8, rlab_dim=8, da_dim=6, b_dim=6, seed=77)
    println("\n=== Kernel Precision Test (large nb=$rlab_dim) ===")
    println("T=$T  left_dim=$left_dim  rlab_dim=$rlab_dim  da_dim=$da_dim  b_dim=$b_dim")

    rng = MersenneTwister(seed + 100)
    B = randn(rng, T, rlab_dim, b_dim)
    A = make_A(T, left_dim, rlab_dim, da_dim; seed=seed)

    labelsA = [:left, :rlab, :da]
    labelsB = [:rlab, :b]
    labelsC = [:left, :da, :b]
    rlab    = :rlab
    mapA = Dict(:left => 1, :rlab => 2, :da => 3)
    mapB = Dict(:rlab => 1, :b => 2)

    C_ref = reference_contraction(A, B, left_dim, rlab_dim, da_dim, b_dim)

    C_current = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    SparseBackends.contract_prefix_outer_bd!(
        C_current, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_cur_dense = c_to_dense(C_current, left_dim, da_dim, b_dim)
    err_current = norm(C_ref - C_cur_dense)

    C_kahan = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    contract_prefix_outer_bd_kahan!(
        C_kahan, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_kahan_dense = c_to_dense(C_kahan, left_dim, da_dim, b_dim)
    err_kahan = norm(C_ref - C_kahan_dense)

    println("  [current kernel]  ‖error‖ = $err_current")
    println("  [kahan kernel]    ‖error‖ = $err_kahan")
    @printf("  improvement: %.1fx\n", err_current > 0 ? err_current / max(err_kahan, 1e-300) : NaN)

    return (; err_current, err_kahan)
end

run_precision_test()
run_precision_test_large_nb()

# ---- Test with larger values (scaling the A tensor, matching DMRG magnitudes) ----
println("\n=== Kernel Precision Test (scale=50, matching DMRG MPO magnitudes) ===")
let T=Float64, left_dim=4, rlab_dim=3, da_dim=4, b_dim=3, seed=42, scale=50.0
    rng = MersenneTwister(seed + 100)
    B = randn(rng, T, rlab_dim, b_dim)
    A = make_A(T, left_dim, rlab_dim, da_dim; seed=seed)
    A.data .*= scale

    labelsA = [:left, :rlab, :da]
    labelsB = [:rlab, :b]
    labelsC = [:left, :da, :b]
    rlab    = :rlab
    mapA = Dict(:left => 1, :rlab => 2, :da => 3)
    mapB = Dict(:rlab => 1, :b => 2)

    C_ref = reference_contraction(A, B, left_dim, rlab_dim, da_dim, b_dim)

    C_current = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    SparseBackends.contract_prefix_outer_bd!(
        C_current, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_cur_dense = c_to_dense(C_current, left_dim, da_dim, b_dim)
    err_current = norm(C_ref - C_cur_dense)

    C_kahan = NewBlockSparseSorted{T,3,2}((left_dim, da_dim, b_dim))
    contract_prefix_outer_bd_kahan!(
        C_kahan, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_kahan_dense = c_to_dense(C_kahan, left_dim, da_dim, b_dim)
    err_kahan = norm(C_ref - C_kahan_dense)

    @printf("  current ‖error‖: %.3e\n", err_current)
    @printf("  kahan   ‖error‖: %.3e\n", err_kahan)
    @printf("  improvement: %.1fx\n", err_current > 0 ? err_current / max(err_kahan, 1e-300) : NaN)
end

# ---- Test with PA=3, PC=2 matching actual SB_TRACE params ----
println("\n=== Kernel Precision Test (PA=3, PC=2, 12 ckey-groups, nb=3, matching DMRG) ===")
let T=Float64, seed=99
    d1_dim, d2_dim, rlab_dim, da_dim, b_dim = 4, 3, 3, 4, 3

    rng = MersenneTwister(seed + 100)
    B = randn(rng, T, rlab_dim, b_dim) .* 5.0

    A = NewBlockSparseSorted{T,4,1}((d1_dim, d2_dim, rlab_dim, da_dim))
    rng2 = MersenneTwister(seed)
    for d1 in 1:d1_dim, d2 in 1:d2_dim, rv in 1:rlab_dim
        id = _ensure_block!(A, (d1, d2, rv))
        v = _block_view(A, id)
        for k in 1:da_dim; v[k] = randn(rng2, T) * 20.0; end
    end

    labelsA = [:d1, :d2, :rlab, :da]
    labelsB = [:rlab, :b]
    labelsC = [:d1, :d2, :da, :b]
    rlab    = :rlab
    mapA = Dict(:d1 => 1, :d2 => 2, :rlab => 3, :da => 4)
    mapB = Dict(:rlab => 1, :b => 2)

    C_ref = zeros(T, d1_dim, d2_dim, da_dim, b_dim)
    A_dense_t = zeros(T, d1_dim, d2_dim, rlab_dim, da_dim)
    for (key, id) in zip(A.keys, A.ids)
        d1, d2, rv = key
        v = _block_view(A, id)
        for k in 1:da_dim; A_dense_t[d1, d2, rv, k] = v[k]; end
    end
    for d1 in 1:d1_dim, d2 in 1:d2_dim, rv in 1:rlab_dim, da in 1:da_dim, b in 1:b_dim
        C_ref[d1, d2, da, b] += A_dense_t[d1, d2, rv, da] * B[rv, b]
    end

    C_current = NewBlockSparseSorted{T,4,2}((d1_dim, d2_dim, da_dim, b_dim))
    SparseBackends.contract_prefix_outer_bd!(
        C_current, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_cur = zeros(T, d1_dim, d2_dim, da_dim, b_dim)
    for (key, id) in zip(C_current.keys, C_current.ids)
        d1, d2 = key
        blk = _block_view(C_current, id)
        for b in 1:b_dim, da in 1:da_dim
            C_cur[d1, d2, da, b] = blk[(b-1)*da_dim + da]
        end
    end

    C_kahan_t = NewBlockSparseSorted{T,4,2}((d1_dim, d2_dim, da_dim, b_dim))
    contract_prefix_outer_bd_kahan!(
        C_kahan_t, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab
    )
    C_kah = zeros(T, d1_dim, d2_dim, da_dim, b_dim)
    for (key, id) in zip(C_kahan_t.keys, C_kahan_t.ids)
        d1, d2 = key
        blk = _block_view(C_kahan_t, id)
        for b in 1:b_dim, da in 1:da_dim
            C_kah[d1, d2, da, b] = blk[(b-1)*da_dim + da]
        end
    end

    @printf("  current ‖error‖: %.3e\n", norm(C_ref - C_cur))
    @printf("  kahan   ‖error‖: %.3e\n", norm(C_ref - C_kah))
    ratio = norm(C_ref - C_cur) / max(norm(C_ref - C_kah), 1e-300)
    @printf("  improvement: %.1fx\n", ratio)
end

println("\nDone.")
