# Step-by-step DMRG comparison: sparse vs dense.
# Runs the SAME sequence of DMRG operations independently on each, comparing
# the eigsolve energy at every bond to find the first deviation point.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
using KrylovKit: eigsolve

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
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_sparse, H_dense, psi_sp, psi_dense
end

bs_info(T) = ITensors.has_external_storage(T) ?
    "BS($(length(ITensors.get_external_storage(T).blocksparse.keys)) blk)" : "D"

describe(label, psi) = println("  [$label] ", [bs_info(T) for T in psi])

# Full state norm <psi|psi> (works for both BS and D)
function state_norm2(psi::MPS)
    N = length(psi)
    n2 = scalar(dag(psi[1]) * psi[1])
    for i in 2:N
        n2 = scalar(dag(psi[i]) * psi[i] * n2 / scalar(n2) * scalar(n2))
    end
    return abs(n2)
end

# Inner-product based comparison: just track eigsolve energies (canonical metric)
function step_eigsolve(PH, psi, b)
    phi = psi[b] * psi[b+1]
    vals, vecs = eigsolve(PH, phi, 1, :SR;
        ishermitian=true, tol=1e-14, krylovdim=3, maxiter=1, verbosity=0)
    return vals[1], vecs[1]
end

let
    H_sp, H_d, psi_sp, psi_d = build_setup(2)
    N = length(psi_sp)

    println("=== Initial ===")
    describe("psi_sp", psi_sp); describe("psi_d ", psi_d)

    println("\n=== orthogonalize(psi, 1) ===")
    psi_sp = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_d  = ITensorMPS.orthogonalize(psi_d,  1)
    describe("psi_sp", psi_sp); describe("psi_d ", psi_d)

    PH_sp = position!(ProjMPO(H_sp), psi_sp, 1)
    PH_d  = position!(ProjMPO(H_d),  psi_d,  1)

    for b in 1:(N-1)
        println("\n=== Bond $b (forward) ===")
        # Position to current bond if not already there
        if b > 1
            PH_sp = position!(PH_sp, psi_sp, b)
            PH_d  = position!(PH_d,  psi_d,  b)
        end

        E_sp, phi_sp = step_eigsolve(PH_sp, psi_sp, b)
        E_d,  phi_d  = step_eigsolve(PH_d,  psi_d,  b)
        println("  eigsolve   E_sp = $E_sp")
        println("  eigsolve   E_d  = $E_d")
        println("  |ΔE|         = $(abs(E_sp - E_d))")

        replacebond!(PH_sp, psi_sp, b, phi_sp;
            ortho="left", maxdim=20, mindim=1, cutoff=1e-12, normalize=true)
        replacebond!(PH_d,  psi_d,  b, phi_d;
            ortho="left", maxdim=20, mindim=1, cutoff=1e-12, normalize=true)
        describe("psi_sp after", psi_sp)
        describe("psi_d  after", psi_d)
    end
end
nothing
