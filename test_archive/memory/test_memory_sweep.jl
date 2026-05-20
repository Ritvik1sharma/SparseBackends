# Track memory of psi (sparse vs dense) PER SWEEP during DMRG.
# Sparse path uses the all-sparse blocksparse_svd inside DMRG (currently the
# energy is wrong due to remaining contraction bug, but memory tracking is
# still meaningful because the sparse-key pattern grows naturally with BD).
#
# Two runs:
#   1. Sparse-psi DMRG (all-sparse path engaged via env override)
#   2. Dense-psi DMRG (reference)
# Log psi memory after each sweep.

using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    psi_dn = copy(psi0)
    for j in 1:N
        psi_dn = replaceprime(ConsOps2[j] * psi_dn, 1 => 0)
    end
    return sites, H, H_sparse, psi_sp, psi_dn
end

mps_bytes(M::MPS) = sum(Base.summarysize(M[i]) for i in 1:length(M))

function snapshot(psi::MPS, label::String)
    types = [ITensors.has_external_storage(psi[i]) ? "BS" : "D" for i in 1:length(psi)]
    link_dims = [prod(dim(I) for I in commoninds(psi[b], psi[b+1])) for b in 1:length(psi)-1]
    maxbd = maximum(link_dims; init=1)
    n_block_info = []
    for i in 1:length(psi)
        if ITensors.has_external_storage(psi[i])
            w = ITensors.get_external_storage(psi[i])
            if w isa SparseBackends.WrappedBlockSparse
                push!(n_block_info, (n_blocks=length(w.blocksparse.keys),
                                      blksize=w.blocksparse.blksize,
                                      stored=length(w.blocksparse.data)))
            end
        end
    end
    total_blocks = sum(b.n_blocks for b in n_block_info; init=0)
    total_stored = sum(b.stored for b in n_block_info; init=0)
    avg_blksize = total_blocks > 0 ? total_stored/total_blocks : 0
    bytes = mps_bytes(psi)
    println("  [$label]  maxBD=$maxbd  bytes=$(round(bytes/1024; digits=2)) KB",
            "  types=$types",
            total_blocks > 0 ? "  total_blocks=$total_blocks  avg_blksize=$(round(avg_blksize;digits=1))  total_stored_values=$total_stored" : "")
    return bytes
end

let
    N = 2
    sites, H, H_sparse, psi_sp, psi_dn = build_setup(N)
    println("="^78)
    println("DMRG memory tracking, maxdim=80, N=$N")
    println("="^78)
    println("\nInitial:")
    sp0 = snapshot(psi_sp, "sparse")
    dn0 = snapshot(psi_dn, "dense ")

    println("\n--- DENSE-psi DMRG (reference) ---")
    psi_cur = copy(psi_dn)
    println("  initial:")
    snapshot(psi_cur, "dense ")
    for sw in 1:8
        E, psi_cur, _, _ = dmrg(H, psi_cur;
            nsweeps=1, maxdim=[80], mindim=[1], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        print("  sweep $sw: E=$(round(real(E); sigdigits=8))  ")
        snapshot(psi_cur, "dense ")
    end

    println("\n--- SPARSE-psi DMRG (dense fallback path; psi stays dense after sweep 1) ---")
    psi_cur = copy(psi_sp)
    println("  initial:")
    snapshot(psi_cur, "sparse")
    for sw in 1:8
        E, psi_cur, _, _ = dmrg(H_sparse, psi_cur;
            nsweeps=1, maxdim=[80], mindim=[1], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        print("  sweep $sw: E=$(round(real(E); sigdigits=8))  ")
        snapshot(psi_cur, "sparse")
    end
end
nothing
