# N=4 sparse-only DMRG with new channel-aware sweep path.
# Dense baseline: E ≈ -6.41785306.  Old right_binned gap: ~1.14e-4 at 2 sweeps.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_psi(N::Int)
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

real_linkdims(psi) = [prod(ITensors.dim(I) for I in commoninds(psi[i], psi[i+1]); init=1)
                       for i in 1:length(psi)-1]

let
    N = 4
    println("=== N=$N sparse-only (channel-aware sweep path, relax_iso_cap=true) ===")
    H_sp, psi_sp = build_psi(N)
    println("Initial REAL linkdims = ", real_linkdims(psi_sp))

    nsweeps_ = 6
    sweeps = Sweeps(nsweeps_)
    setmaxdim!(sweeps, [min(10*k, 40) for k in 1:nsweeps_]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)

    E_s, psi_sp_out = dmrg(H_sp, psi_sp, sweeps; outputlevel=1)

    println("\nSparse final E = $E_s")
    println("Dense baseline E ≈ -6.417853060444436 (from earlier multiN run)")
    println("|gap| = ", abs(E_s - (-6.417853060444436)))
    println("Sparse final REAL linkdims = ", real_linkdims(psi_sp_out))
end
nothing
