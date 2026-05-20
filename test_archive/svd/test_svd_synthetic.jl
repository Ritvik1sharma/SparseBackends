# Synthetic minimal test for blocksparse_svd to pinpoint the truncation bug.
# Builds a phi with the same shape as the failing DMRG case but tiny, then
# manually checks per-block U·S·V_t against the SVD output.

using SparseBackends, ITensors
using LinearAlgebra
using Random

# Build a sparse phi with hand-picked block values. Shape mirrors the failing
# DMRG case (nls=2, nrs=2, nld=1, nrd=1) but with small dims.
function build_phi(; d_l1=3, d_l2=2, d_r1=2, d_r2=2, d_ld=1, d_rd=2,
                     density::Float64 = 1.0,  # fraction of blocks populated
                     seed::Int = 0)
    rng = MersenneTwister(seed)
    # Build Index objects with consistent tags
    Il1 = Index(d_l1; tags="Link,l=0")
    Il2 = Index(d_l2; tags="S=1,Site,n=1")
    Ir1 = Index(d_r1; tags="Link,l=2")
    Ir2 = Index(d_r2; tags="S=1,Site,n=2")
    Ild = Index(d_ld; tags="Link,l=0")
    Ird = Index(d_rd; tags="Link,l=2")
    inds = (Il1, Il2, Ir1, Ir2, Ild, Ird)

    # NewBlockSparseSorted with P=4, N2=2
    bs = SparseBackends.NewBlockSparseSorted{ComplexF64, 6, 2}((d_l1, d_l2, d_r1, d_r2, d_ld, d_rd))

    # Populate blocks with random values
    blocks_set = Tuple{Int,Int,Int,Int}[]
    for k1 in 1:d_l1, k2 in 1:d_l2, k3 in 1:d_r1, k4 in 1:d_r2
        if rand(rng) < density
            push!(blocks_set, (k1, k2, k3, k4))
        end
    end

    for (k1, k2, k3, k4) in blocks_set
        for s in 1:d_ld, t in 1:d_rd
            bs[k1, k2, k3, k4, s, t] = randn(rng, ComplexF64)
        end
    end

    w = SparseBackends.WrappedBlockSparse(bs, inds)
    phi = ITensors._itensor_from_external_storage(w)
    Linds = [Il1, Il2, Ild]  # U side
    return phi, Linds, inds, blocks_set
end

# Compute ⟨phi|recon⟩ and ‖phi‖², ‖recon‖² safely using dense conversions
function tensor_compare_safe(a::ITensor, b::ITensor)
    arr_a = SparseBackends.to_dense(a)
    arr_b = SparseBackends.to_dense(b)
    # If shapes differ, can't compare directly
    if size(arr_a) != size(arr_b)
        # Reshape to flat
        va = vec(arr_a); vb = vec(arr_b)
        if length(va) != length(vb)
            return (err = NaN, na2 = sum(abs2, va), nb2 = sum(abs2, vb), cross = NaN, ok = false)
        end
        cross = real(dot(va, vb))
        diff2 = real(dot(va - vb, va - vb))
        return (err = sqrt(diff2)/max(norm(va), 1e-300), na2 = sum(abs2, va), nb2 = sum(abs2, vb),
                cross = cross, ok = true)
    end
    cross = real(dot(vec(arr_a), vec(arr_b)))
    diff2 = real(dot(vec(arr_a - arr_b), vec(arr_a - arr_b)))
    err = sqrt(diff2) / max(norm(vec(arr_a)), 1e-300)
    return (err = err, na2 = sum(abs2, vec(arr_a)), nb2 = sum(abs2, vec(arr_b)), cross = cross, ok = true)
end

# Manually do per-left-key SVD of phi (matching what blocksparse_svd does)
# and compute the EXPECTED reconstruction. This bypasses BlockSparse storage
# entirely so we know the math is right.
function manual_perblock_recon(bs::SparseBackends.NewBlockSparseSorted{T,N,N2,P,K},
                                nls::Int, nrs::Int, nld::Int, nrd::Int,
                                maxdim::Int) where {T,N,N2,P,K}
    dims = bs.dims
    d_left  = prod(ntuple(i -> dims[P+i], nld); init=1)
    d_right = prod(ntuple(i -> dims[P+nld+i], nrd); init=1)

    # Bin blocks by left key
    bins = Dict{NTuple{nls,K}, Vector{Tuple{NTuple{nrs,K}, Int}}}()
    for (key, id) in SparseBackends.blocks_sorted(bs)
        lk = ntuple(i -> key[i], nls)
        rk = ntuple(i -> key[nls+i], nrs)
        push!(get!(bins, lk, Tuple{NTuple{nrs,K}, Int}[]), (rk, id))
    end
    l_vals = sort!(collect(keys(bins)))

    # Per-block SVD
    block_svds = Vector{Tuple{NTuple{nls,K}, Matrix{T}, Vector{Float64}, Matrix{T}, Vector{NTuple{nrs,K}}}}()
    for lk in l_vals
        entries = bins[lk]
        r_list = sort!(unique!([rk for (rk, _) in entries]))
        n_Ri = length(r_list)
        r_pos = Dict(r => j for (j, r) in enumerate(r_list))
        M_i = zeros(T, d_left, n_Ri * d_right)
        for (rk, id) in entries
            rj = r_pos[rk]
            blk = reshape(SparseBackends._block_view(bs, id), d_left, d_right)
            M_i[:, (rj-1)*d_right+1 : rj*d_right] .= blk
        end
        F = svd(M_i)
        push!(block_svds, (lk, F.U, F.S, F.Vt, r_list))
    end

    # Global truncation
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev=true, by=x -> x[1])
    keep_count = min(maxdim, length(all_svs))
    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    # Reconstruct DENSELY: build full (d_l1*d_l2*d_left × d_r1*d_r2*d_right)
    # matrix piece by piece, including only kept SVs.
    sparse_dims_left = ntuple(i -> dims[i], nls)
    sparse_dims_right = ntuple(i -> dims[nls+i], nrs)
    full_left  = prod(sparse_dims_left) * d_left
    full_right = prod(sparse_dims_right) * d_right
    Mfull_kept = zeros(T, full_left, full_right)

    for (bi, (lk, U_i, S_i, Vt_i, r_list)) in enumerate(block_svds)
        ki = k_kept[bi]
        ki == 0 && continue
        # Build the rank-ki approximation of M_i
        Mi_recon = U_i[:, 1:ki] * Diagonal(S_i[1:ki]) * Vt_i[1:ki, :]
        # Embed into Mfull_kept at the appropriate sparse-key position
        # left index: lin from (lk[1], lk[2], dl_index)
        # right index: lin from (rk[1], rk[2], dr_index) for each rk in r_list
        for dl in 1:d_left
            left_lin = (lk[2]-1)*sparse_dims_left[1]*d_left + (lk[1]-1)*d_left + dl
            for (rj, rk) in enumerate(r_list)
                for dr in 1:d_right
                    right_lin = (rk[2]-1)*sparse_dims_right[1]*d_right + (rk[1]-1)*d_right + dr
                    col = (rj-1)*d_right + dr
                    Mfull_kept[left_lin, right_lin] = Mi_recon[dl, col]
                end
            end
        end
    end
    return Mfull_kept, full_left, full_right, sparse_dims_left, sparse_dims_right, d_left, d_right, k_kept, block_svds
end

# Build the dense full matrix from the original BlockSparse phi.
function densify_phi_matrix(bs::SparseBackends.NewBlockSparseSorted{T,N,N2,P,K},
                              nls::Int, nrs::Int, nld::Int, nrd::Int) where {T,N,N2,P,K}
    dims = bs.dims
    d_left  = prod(ntuple(i -> dims[P+i], nld); init=1)
    d_right = prod(ntuple(i -> dims[P+nld+i], nrd); init=1)
    sparse_dims_left = ntuple(i -> dims[i], nls)
    sparse_dims_right = ntuple(i -> dims[nls+i], nrs)
    full_left  = prod(sparse_dims_left) * d_left
    full_right = prod(sparse_dims_right) * d_right
    M = zeros(T, full_left, full_right)
    for (key, id) in SparseBackends.blocks_sorted(bs)
        lk = ntuple(i -> key[i], nls)
        rk = ntuple(i -> key[nls+i], nrs)
        for dl in 1:d_left
            left_lin = (lk[2]-1)*sparse_dims_left[1]*d_left + (lk[1]-1)*d_left + dl
            for dr in 1:d_right
                right_lin = (rk[2]-1)*sparse_dims_right[1]*d_right + (rk[1]-1)*d_right + dr
                blk_lin = (dr - 1)*d_left + dl  # dense suffix in column-major
                M[left_lin, right_lin] = SparseBackends._block_view(bs, id)[blk_lin]
            end
        end
    end
    return M
end

let
    println("=" ^ 70)
    println("Synthetic phi: dims=(3,2,2,2,1,2), nls=2, nrs=2, nld=1, nrd=1")
    println("=" ^ 70)
    phi, Linds, inds, blocks = build_phi(; d_l1=3, d_l2=2, d_r1=2, d_r2=2,
                                          d_ld=1, d_rd=2, density=1.0, seed=42)
    bs = ITensors.get_external_storage(phi).blocksparse
    println("phi has $(length(blocks)) non-zero blocks")
    println("phi storage P=$(SparseBackends._P(bs)) N2=$(SparseBackends._N2(bs)) dims=$(bs.dims)")

    nls, nrs, nld, nrd = 2, 2, 1, 1

    # Reference: dense matrix M from phi, full dense SVD
    M = densify_phi_matrix(bs, nls, nrs, nld, nrd)
    F = svd(M)
    println("\nDense reference SVD: ‖M‖²=", sum(abs2, M), "  rank=", count(F.S .> 1e-10),
            "  SVs=", round.(F.S; sigdigits=3))

    # First: deep dump for maxdim=1
    println("\n" * "=" ^ 70)
    println("DEEP DUMP at maxdim=1")
    println("=" ^ 70)
    L1, R1, _ = SparseBackends.itensor_blocksparse_svd(
        phi, Linds; ortho="left", maxdim=1, mindim=1, cutoff=0.0,
        tags = ITensors.TagSet("Link,l=1"),
    )
    rec1 = L1 * R1
    println("phi   inds: ", ITensors.inds(phi))
    println("L     inds: ", ITensors.inds(L1))
    println("R     inds: ", ITensors.inds(R1))
    println("recon inds: ", ITensors.inds(rec1))
    wL = ITensors.get_external_storage(L1).blocksparse
    wR = ITensors.get_external_storage(R1).blocksparse
    if ITensors.has_external_storage(rec1)
        wRec = ITensors.get_external_storage(rec1).blocksparse
        println("recon storage: P=$(SparseBackends._P(wRec)) N2=$(SparseBackends._N2(wRec))  dims=$(wRec.dims)  n_blocks=$(length(wRec.keys))")
        println("  keys: ", wRec.keys)
        println("  data: ", wRec.data)
    else
        println("recon is dense: dims=", size(SparseBackends.to_dense(rec1)))
    end
    println("L storage: dims=$(wL.dims) keys=$(wL.keys) data=$(wL.data)")
    println("R storage: dims=$(wR.dims) keys=$(wR.keys) data=$(wR.data)")

    # Manual: which lk has the biggest first SV?
    Mfull_kept_1, _, _, _, _, _, _, k_kept, block_svds = manual_perblock_recon(bs, nls, nrs, nld, nrd, 1)
    biggest_bi = findfirst(x -> x > 0, k_kept)
    println("manual: biggest bi=$biggest_bi  lk_of_bi=$(block_svds[biggest_bi][1])  S_of_bi=$(block_svds[biggest_bi][3])")

    # Test: contraction self-consistency under permutation.
    # Take phi, permute its storage to a different axis order, and check
    # ⟨phi, phi_permuted⟩ == ‖phi‖². If wrong, the contraction is buggy.
    println("\n  -- Storage-permute self-consistency test --")
    phi_norm2 = real(ITensors.scalar(ITensors.dag(phi) * phi))
    println("    ‖phi‖² (direct) = $phi_norm2")
    # Permute phi to a different sparse-axis order: swap pos 2 and 3
    new_inds = (inds[1], inds[3], inds[2], inds[4], inds[5], inds[6])
    phi_perm = SparseBackends.permute(phi, new_inds...)
    cross_self = real(ITensors.scalar(ITensors.dag(phi) * phi_perm))
    nperm2 = real(ITensors.scalar(ITensors.dag(phi_perm) * phi_perm))
    println("    ‖phi_perm‖² = $nperm2")
    println("    ⟨phi, phi_perm⟩ = $cross_self  (should equal ‖phi‖² = $phi_norm2)")
    if !isapprox(cross_self, phi_norm2; atol=1e-6 * phi_norm2)
        println("    ✗ BUG: BlockSparse contraction does NOT respect Index-identity matching")
        println("        when storage orders differ!")
    end

    # Direct comparison: find phi's block at (Il1=2, Il2=1, Ir1=1, Ir2=1)
    # and recon's block at the same physical position. Compare data.
    println("\n  -- Direct value comparison --")
    phi_bs = ITensors.get_external_storage(phi).blocksparse
    rec_bs = ITensors.get_external_storage(rec1).blocksparse
    # phi storage order: (Il1, Il2, Ir1, Ir2, Ild, Ird) — keys are (Il1, Il2, Ir1, Ir2)
    # rec storage order: (Il1, Ir1, Il2, Ir2, Ild, Ird) — keys are (Il1, Ir1, Il2, Ir2)
    # For physical position (Il1=2, Il2=1, Ir1=1, Ir2=1):
    #   phi key = (2, 1, 1, 1), rec key = (2, 1, 1, 1)
    # For (Il1=2, Il2=1, Ir1=2, Ir2=1):
    #   phi key = (2, 1, 2, 1), rec key = (2, 2, 1, 1)
    # For (Il1=2, Il2=1, Ir1=1, Ir2=2):
    #   phi key = (2, 1, 1, 2), rec key = (2, 1, 1, 2)
    # For (Il1=2, Il2=1, Ir1=2, Ir2=2):
    #   phi key = (2, 1, 2, 2), rec key = (2, 2, 1, 2)
    function find_block(bs, key)
        for (i, k) in enumerate(bs.keys)
            k == key && return SparseBackends._block_view(bs, bs.ids[i])
        end
        return nothing
    end
    for (phys, phi_key, rec_key) in [
        ((2,1,1,1), (2,1,1,1), (2,1,1,1)),
        ((2,1,2,1), (2,1,2,1), (2,2,1,1)),
        ((2,1,1,2), (2,1,1,2), (2,1,1,2)),
        ((2,1,2,2), (2,1,2,2), (2,2,1,2)),
    ]
        phi_data = find_block(phi_bs, phi_key)
        rec_data = find_block(rec_bs, rec_key)
        println("    phys (Il1,Il2,Ir1,Ir2)=$phys")
        println("      phi[$phi_key] = $phi_data")
        println("      rec[$rec_key] = $rec_data")
    end

    println()
    # Test at several maxdim values
    for maxdim in [1, 2, 3, 5, 10, 100]
        # Manual perblock recon (math-only, no BlockSparse storage path)
        Mfull_kept, _, _, _, _, _, _, k_kept, block_svds = manual_perblock_recon(bs, nls, nrs, nld, nrd, maxdim)
        cross_manual = real(dot(vec(M), vec(Mfull_kept)))
        norm2_manual = sum(abs2, vec(Mfull_kept))
        norm2_M      = sum(abs2, vec(M))
        manual_proj_check = isapprox(cross_manual, norm2_manual; atol=1e-10*norm2_M)

        # Full-rank dense SVD truncated to maxdim
        kk = min(maxdim, length(F.S))
        M_dense_trunc = F.U[:, 1:kk] * Diagonal(F.S[1:kk]) * F.Vt[1:kk, :]
        cross_dense = real(dot(vec(M), vec(M_dense_trunc)))
        norm2_dense = sum(abs2, vec(M_dense_trunc))

        # Now: actual itensor_blocksparse_svd
        L, R, _ = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = maxdim, mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=1"),
        )
        recon_sp = L * R
        cmp_sp = tensor_compare_safe(phi, recon_sp)
        # Also: use ITensor index-aware contraction
        cross_it  = real(ITensors.scalar(ITensors.dag(phi) * recon_sp))
        nb2_it    = real(ITensors.scalar(ITensors.dag(recon_sp) * recon_sp))
        # And: permute recon to phi's storage order, then compute inner product
        rec_permuted = SparseBackends.permute(recon_sp, ITensors.inds(phi)...)
        cross_aligned = real(ITensors.scalar(ITensors.dag(phi) * rec_permuted))
        nb2_aligned = real(ITensors.scalar(ITensors.dag(rec_permuted) * rec_permuted))

        println("\n--- maxdim=$maxdim ---")
        println("  manual perblock: ‖rec‖²=$(round(norm2_manual; sigdigits=4))  ⟨M,rec⟩=$(round(cross_manual; sigdigits=4))  proj_check=$manual_proj_check")
        println("  dense top-k SVD: ‖rec‖²=$(round(norm2_dense; sigdigits=4))  ⟨M,rec⟩=$(round(cross_dense; sigdigits=4))")
        println("  sparse SVD impl: ‖rec‖²=$(round(cmp_sp.nb2; sigdigits=4))  ⟨phi,rec⟩(unsafe)=$(round(cmp_sp.cross; sigdigits=4))  err=$(round(cmp_sp.err; sigdigits=3))")
        println("    ITensor contraction:  ⟨phi,rec⟩=$(round(cross_it; sigdigits=4))  ‖rec‖²=$(round(nb2_it; sigdigits=4))")
        println("    Dense (aligned inds): ⟨phi,rec⟩=$(round(cross_aligned; sigdigits=4))  ‖rec‖²=$(round(nb2_aligned; sigdigits=4))")
        # If manual perblock differs from sparse impl, that's the bug location
        if !isapprox(cmp_sp.nb2, norm2_manual; atol=1e-6 * norm2_M)
            println("  ✗ DISCREPANCY: sparse impl ‖rec‖²=$(cmp_sp.nb2) ≠ manual perblock ‖rec‖²=$norm2_manual")
        end
    end
end
nothing
