# Memory tracking across DMRG sweeps with Option A
# (sparse SVD for bulk, dense fallback for boundary).
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
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, H, psi_sp, copy(psi0)
end

mps_bytes(M::MPS) = sum(Base.summarysize(M[i]) for i in 1:length(M))

function describe_psi(psi::MPS, label::String)
    types = [ITensors.has_external_storage(psi[i]) ? "BS" : "D" for i in 1:length(psi)]
    bds = [prod(dim(I) for I in commoninds(psi[b], psi[b+1])) for b in 1:length(psi)-1]
    bytes = mps_bytes(psi) / 1024
    println("  [$label] maxBD=$(maximum(bds; init=1))  bytes=$(round(bytes;digits=2))KB  types=$types")
    bytes
end

let
    N = 2
    H_sparse, H, psi_sp, psi0 = build_setup(N)
    println("="^72)
    println("Memory tracking per sweep (Option A: bulk-sparse, boundary-dense)")
    println("="^72)

    println("\n[A] Sparse-psi DMRG (with new Option A)")
    psi_cur = copy(psi_sp)
    describe_psi(psi_cur, "init")
    for sw in 1:8
        E, psi_cur, _, _ = dmrg(H_sparse, psi_cur;
            nsweeps=1, maxdim=[20], mindim=[20], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        print("  sweep $sw E=$(round(real(E); sigdigits=8)) ")
        describe_psi(psi_cur, "psi_sp")
    end

    println("\n[B] Dense-psi DMRG (reference)")
    psi_cur = copy(psi0)
    describe_psi(psi_cur, "init")
    for sw in 1:8
        E, psi_cur, _, _ = dmrg(H, psi_cur;
            nsweeps=1, maxdim=[20], mindim=[20], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        print("  sweep $sw E=$(round(real(E); sigdigits=8)) ")
        describe_psi(psi_cur, "psi_dn")
    end
end
nothing
