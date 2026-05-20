# Sparse-only variant of test_b_phase.jl — skips the dense DMRG so we can
# get SVD_DIAG feedback faster. Reuses build_setup from test_b_phase.jl.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_setup_sparse(N::Int)
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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

let
    N_plaq = 3
    println("Building setup for N_plaq=$N_plaq ...")
    H_sp, psi_sp = build_setup_sparse(N_plaq)
    println("psi_sp is_sparse_mps? ", SparseBackends.is_sparse_mps(psi_sp))

    sweeps = Sweeps(2)
    setmaxdim!(sweeps, 10, 20)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)

    println("\n=== Sparse DMRG (Path B) ===")
    E_s, _ = dmrg(H_sp, psi_sp, sweeps; outputlevel=1)
    println("Sparse final E = $E_s")
end
nothing
