# Test whether orthogonalize! preserves the state ψ (up to scaling).
# Compute <ψ_before, ψ_after> / ||ψ_before|| / ||ψ_after||.
# If = 1: ψ is preserved → any "leakage" measurement is bug elsewhere.
# If = 0: ψ genuinely changed during orthogonalize.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
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
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp, P_sparse
end

for N_plaq in [2, 3]
    println("\n========== N=$N_plaq plaquettes ($(2*N_plaq+2) sites) ==========")
    psi_sp, _ = build_setup(N_plaq)
    # Densify BEFORE doing anything destructive — keep the dense copy as the
    # "before" reference. Then orthogonalize modifies a fresh sparse copy.
    psi_b_d = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    psi_after  = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_a_d = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_after])

    n_b = sqrt(real(inner(psi_b_d, psi_b_d)))
    n_a = sqrt(real(inner(psi_a_d, psi_a_d)))
    ov  = inner(psi_b_d, psi_a_d)
    fid = abs(ov) / (n_b * n_a)
    println("  ||ψ_before||   = $n_b")
    println("  ||ψ_after||    = $n_a")
    println("  <ψ_before, ψ_after> = $ov")
    println("  |<ψ_b, ψ_a>| / (||·||·||·||) = $fid")
    println("  $(fid > 0.99 ? "→ STATE PRESERVED (fidelity ≈ 1)" : "→ STATE CHANGED (fidelity << 1)")")
end
nothing
