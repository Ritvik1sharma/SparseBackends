# Compare DMRG energies on the same projected Hamiltonian, run with
# BlockSparse psi (current sparse path) vs Dense psi (reference).
# Both use the SAME H. Only psi storage differs.
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
    H1_dense       = contract(P_sparse'', H', :coo, :dense)
    H_sparse       = replaceprime(contract(P_sparse, H1_dense, :coo, :blocksparse), 3 => 1)
    H_dense        = replaceprime(contract(P_sparse, H1_dense, :coo, :dense),       3 => 1)

    psi0    = random_mps(sites)
    psi_sp  = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense),       1 => 0)
    psi_d   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense),       1 => 0)
    # Force the dense reference psi to a dense ITensor MPS (no external storage)
    psi_d_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_d])

    return H_sparse, H_dense, psi_sp, psi_d_dense
end

length(ARGS) < 1 && error("Usage: julia test_dmrg_dense_vs_sparse.jl <N_plaq>")
const _N_DDS = parse(Int, ARGS[1])

let
    H_sparse, H_dense, psi_sp, psi_d = build_setup(_N_DDS)

    println("psi_sp external-storage?  ", [ITensors.has_external_storage(psi_sp[i]) for i in 1:length(psi_sp)])
    println("psi_d  external-storage?  ", [ITensors.has_external_storage(psi_d[i])  for i in 1:length(psi_d)])
    println("H_sparse external-storage? ", [ITensors.has_external_storage(H_sparse[i]) for i in 1:length(H_sparse)])
    println("H_dense  external-storage? ", [ITensors.has_external_storage(H_dense[i])  for i in 1:length(H_dense)])

    println("\n=== Dense DMRG (reference) ===")
    E_d, _, _, _ = dmrg(H_dense, psi_d;
        nsweeps=10, maxdim=[20,20,40,40,80,80,160,160,160,160],
        mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=1)
    println("Dense final E = $E_d")

    println("\n=== Sparse DMRG (with mirror kernel) ===")
    E_s, _, _, _ = dmrg(H_sparse, psi_sp;
        nsweeps=10, maxdim=[20,20,40,40,80,80,160,160,160,160],
        mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=1)
    println("Sparse final E = $E_s")

    println("\nDifference: |E_dense - E_sparse| = ", abs(E_d - E_s))
end
nothing
