# Minimal reproducer for the itensor_blocksparse_svd partition bug.
# Builds a small sparse phi shaped like the bad cases observed in DMRG, then
# calls itensor_blocksparse_svd directly and compares L*R to phi.
using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random
include("utils.jl")

# Simulate a bond-2 phi by contracting two sparse psi sites built from a real projection.
function sandwich_mpo(P, H)
    H1 = contract(P'', H', :coo, :dense)
    replaceprime(contract(P, H1, :coo, :blocksparse), 3 => 1)
end
function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
function multiplyVecMPOtoMPO(v); r = v[1]; for j in 2:length(v); r = mulMPO(r, v[j]); end; r; end

let
    Random.seed!(42)
    N = 2; sites = siteinds("S=1", 2*N+2)
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    P_sparse = multiplyVecMPOtoMPO(ConsOps1)
    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)

    # Try various bonds
    for b in 1:length(psi_sp)-1
        phi = psi_sp[b] * psi_sp[b+1]
        Linds = uniqueinds(psi_sp[b], psi_sp[b+1])
        L, R, spec = SparseBackends.itensor_blocksparse_svd(phi, Linds;
            ortho="left", maxdim=400, mindim=1, cutoff=1e-14,
            tags=ITensors.TagSet("Link,l=$b"))
        recon = L * R
        # Index-aware error
        phi_n2 = real(ITensors.scalar(ITensors.dag(phi) * phi))
        rec_n2 = real(ITensors.scalar(ITensors.dag(recon) * recon))
        cross  = real(ITensors.scalar(ITensors.dag(phi)  * recon))
        err = sqrt(max(phi_n2 - 2*cross + rec_n2, 0.0)) / max(sqrt(phi_n2), 1e-300)
        println("bond $b: err=$err  ‖phi‖²=$phi_n2  ‖rec‖²=$rec_n2  ⟨phi,rec⟩=$cross")
        println("    inds(phi) dims: ", map(ITensors.dim, inds(phi)))
        println("    Linds   dims: ", map(ITensors.dim, Linds))
    end
end
nothing
