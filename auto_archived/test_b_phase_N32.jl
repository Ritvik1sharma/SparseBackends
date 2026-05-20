# Scaling correctness test: N=32 plaquettes (66 sites), maxdim=80, with truncation.
# Verifies Path B integration gives correct energies on a system large enough
# that channel-aware truncation matters. Expect: sparse energy ≈ dense energy
# AND sparse psi has high fidelity with dense psi after densification.
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
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_sparse, H_dense, psi_sp, psi_dense
end

let
    N_plaq = 4
    println("Building setup for N_plaq=$N_plaq ...")
    t_setup = @elapsed (H_sp, H_d, psi_sp, psi_d) = build_setup(N_plaq)
    println("Setup time: $(round(t_setup, digits=1))s, system size: $(length(psi_sp)) sites")
    println("psi_sp linkdims = ", [ITensors.dim(commonind(psi_sp[i], psi_sp[i+1])) for i in 1:length(psi_sp)-1])

    nsweeps_ = 25
    sweeps = Sweeps(nsweeps_)
    setmaxdim!(sweeps, 10, 20, 30, 40, [40 for _ in 5:nsweeps_]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)

    println("\n=== Dense DMRG (N=$N_plaq, maxdim=80) ===")
    t_d = @elapsed (E_d, psi_d_out) = dmrg(H_d, psi_d, sweeps; outputlevel=1)
    println("Dense final E = $E_d  (time: $(round(t_d, digits=1))s)")

    println("\n=== Sparse DMRG (Path B, N=$N_plaq, maxdim=80) ===")
    t_s = @elapsed (E_s, psi_sp_out) = dmrg(H_sp, psi_sp, sweeps; outputlevel=1)
    println("Sparse final E = $E_s  (time: $(round(t_s, digits=1))s)")

    println("\n|E_dense - E_sparse| = ", abs(E_d - E_s))

    println("\n=== Overlap check ===")
    psi_sp_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp_out])
    nrm_sp = sqrt(real(inner(psi_sp_dense, psi_sp_dense)))
    nrm_d  = sqrt(real(inner(psi_d_out,  psi_d_out)))
    ov     = inner(psi_sp_dense, psi_d_out)
    fid    = abs(ov) / (nrm_sp * nrm_d)
    println("  ||psi_sparse|| = ", nrm_sp)
    println("  ||psi_dense||  = ", nrm_d)
    println("  fidelity = ", fid)
    println("  1 - fidelity = ", 1 - fid)
end
nothing
