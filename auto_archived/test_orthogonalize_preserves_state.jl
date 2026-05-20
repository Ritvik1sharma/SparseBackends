# Direct test of the claim "orthogonalize loses 100% of image(P) content".
#
# Build ψ ∈ image(P). Measure these BEFORE and AFTER orthogonalize:
#   (a) ||ψ||²                            — should be invariant
#   (b) inner(ψ_before, ψ_after)          — should equal ||ψ||² up to phase
#   (c) ||Pψ||² densely                   — should be invariant up to P²-scaling
#
# (b) is the decisive check. If ψ_before and ψ_after have inner product 0,
# the state genuinely changed. If inner = ||ψ||², state preserved → the
# other session's ||Pψ||² → 0 is a measurement issue not a real leakage.

using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

function build_P_and_psi(N::Int)
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
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return P_sparse, psi_sp
end

densify_mps(M) = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in M])
densify_mpo(M) = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in M])

function measure_Ppsi(P_dense, psi_dense)
    # Both arguments must be dense (sparse storage lacks dag method here).
    Ppsi = contract(P_dense, copy(psi_dense); cutoff=1e-14)
    return real(inner(Ppsi, Ppsi))
end

function probe(N::Int)
    println("\n========= N=$N (2N+2=$(2*N+2) sites) =========")
    P, psi = build_P_and_psi(N)
    P_dense = densify_mpo(P)

    # Capture a dense copy of the ORIGINAL ψ for inner-product comparison post-orthogonalize.
    psi_dense_before = densify_mps(psi)
    norm_before    = real(inner(psi_dense_before, psi_dense_before))
    Pnorm_before_d = measure_Ppsi(P_dense, psi_dense_before)
    println("Before orthogonalize:")
    println("  ||ψ||² (dense)  = $norm_before")
    println("  ||Pψ||² (dense) = $Pnorm_before_d")

    # Apply orthogonalize in-place on the sparse psi.
    orthogonalize!(psi, 1)

    # Densify the orthogonalized sparse ψ and compare.
    psi_dense_after = densify_mps(psi)
    norm_after    = real(inner(psi_dense_after, psi_dense_after))
    Pnorm_after_d = measure_Ppsi(P_dense, psi_dense_after)
    ov            = inner(psi_dense_before, psi_dense_after)
    println("After orthogonalize:")
    println("  ||ψ||² (dense)             = $norm_after")
    println("  ||Pψ||² (dense)            = $Pnorm_after_d")
    println("  ⟨ψ_before|ψ_after⟩         = $ov")
    println("  |⟨ψ_before|ψ_after⟩|/||ψ||² = $(abs(ov)/norm_before)")
    println("  ||Pψ||² ratio after/before  = $(Pnorm_after_d / Pnorm_before_d)")
end

let
    for N in (2, 3)
        probe(N)
    end
end
nothing
