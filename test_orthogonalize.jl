# Tests whether orthogonalize! preserves ψ ∈ image(P).
# Usage: julia test_orthogonalize.jl <N>
# Three checks on the same ψ before/after orthogonalize:
#   (1) Fidelity |⟨ψ_before|ψ_after⟩| / (‖ψ_before‖·‖ψ_after‖)  — should be ≈ 1
#   (2) ‖ψ‖² and ‖Pψ‖² before/after                               — should be invariant
#   (3) Image leakage ‖ψ − Pψ‖/‖ψ‖                               — should be ≈ 0
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_orthogonalize.jl <N_plaq>")
const N_PLAQ = parse(Int, ARGS[1])

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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    psi0   = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp, P_sparse
end

densify_mps(M) = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in M])
densify_mpo(M) = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in M])

function measure_Ppsi_norm2(P_dense::MPO, psi_dense::MPS)
    Ppsi = replaceprime(contract(P_dense, copy(psi_dense), :dense, :dense), 1 => 0)
    return real(inner(Ppsi, Ppsi))
end

function image_leakage(psi_dense::MPS, P_dense::MPO; label::String="")
    n_psi2 = real(inner(psi_dense, psi_dense))
    Ppsi   = replaceprime(contract(P_dense, copy(psi_dense), :dense, :dense), 1 => 0)
    n_Ppsi2 = real(inner(Ppsi, Ppsi))
    ov     = inner(psi_dense, Ppsi)
    diff2  = n_psi2 + n_Ppsi2 - 2*real(ov)
    rel    = sqrt(max(diff2, 0.0)) / sqrt(n_psi2)
    println("  [$label]  ‖ψ‖²=$(round(n_psi2,sigdigits=6))  ‖Pψ‖²=$(round(n_Ppsi2,sigdigits=6))  ‖ψ−Pψ‖/‖ψ‖=$(round(rel,sigdigits=4))")
    return rel
end

let
    println("=== N=$N_PLAQ plaquettes ($(2*N_PLAQ+2) sites) ===\n")
    psi_sp, P_sparse = build_setup(N_PLAQ)
    P_dense = densify_mpo(P_sparse)

    psi_d_before = densify_mps(psi_sp)

    # --- Check 1: Fidelity ---
    psi_after_sp = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_d_after  = densify_mps(psi_after_sp)

    n_b  = sqrt(real(inner(psi_d_before, psi_d_before)))
    n_a  = sqrt(real(inner(psi_d_after,  psi_d_after)))
    ov   = inner(psi_d_before, psi_d_after)
    fid  = abs(ov) / (n_b * n_a)
    println("--- Check 1: Fidelity ---")
    println("  ‖ψ_before‖ = $n_b   ‖ψ_after‖ = $n_a")
    println("  ⟨ψ_before|ψ_after⟩ = $ov")
    println("  fidelity = $fid")
    println("  $(fid > 0.99 ? "→ STATE PRESERVED" : "→ STATE CHANGED (fidelity << 1)")\n")

    # --- Check 2: Norm and ‖Pψ‖² ---
    println("--- Check 2: Norm and ‖Pψ‖² ---")
    norm2_before  = real(inner(psi_d_before, psi_d_before))
    norm2_after   = real(inner(psi_d_after,  psi_d_after))
    Pnorm2_before = measure_Ppsi_norm2(P_dense, psi_d_before)
    Pnorm2_after  = measure_Ppsi_norm2(P_dense, psi_d_after)
    println("  Before:  ‖ψ‖² = $norm2_before   ‖Pψ‖² = $Pnorm2_before")
    println("  After:   ‖ψ‖² = $norm2_after   ‖Pψ‖² = $Pnorm2_after")
    println("  ‖Pψ‖²/‖ψ‖² ratio after/before = $(Pnorm2_after/Pnorm2_before)\n")

    # --- Check 3: Image leakage ---
    println("--- Check 3: Image leakage ‖ψ − Pψ‖/‖ψ‖ ---")
    image_leakage(psi_d_before, P_dense; label="before orthogonalize")
    image_leakage(psi_d_after,  P_dense; label="after  orthogonalize")
end
nothing
