# Direct measure of ||ψ - P·ψ|| / ||ψ|| to check whether ψ stays in image(P).
# Tested both right after construction (should be 0) and after orthogonalize
# (where mult-starvation / channel-aware SVD might cause leakage).
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

function image_p_leakage(psi::MPS, P::MPO; label::String="")
    # Densify psi and P, then compute Pψ directly, and measure ||ψ - Pψ||/||ψ||
    # as a parameter-free leakage metric. If ψ ∈ image(P) and P is the projector,
    # then Pψ = ψ → leakage = 0.
    psi_d = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi])
    P_d   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in P])
    n_psi2  = real(inner(psi_d, psi_d))
    Pψ = replaceprime(contract(P_d, copy(psi_d), :dense, :dense), 1 => 0)
    n_Pψ2 = real(inner(Pψ, Pψ))
    # ||ψ - Pψ||² = ||ψ||² + ||Pψ||² - 2·Re<ψ|Pψ>
    ov  = inner(psi_d, Pψ)
    diff2 = n_psi2 + n_Pψ2 - 2*real(ov)
    rel  = sqrt(max(diff2, 0.0)) / sqrt(n_psi2)
    println("  [$label]")
    println("           ||ψ||²    = $n_psi2")
    println("           ||Pψ||²   = $n_Pψ2")
    println("           <ψ|Pψ>    = $ov")
    println("           ||ψ - Pψ||² = $diff2")
    println("           ||ψ - Pψ||/||ψ|| = $rel")
end

for N_plaq in [2, 3]
    println("\n========== N=$N_plaq plaquettes ($(2*N_plaq+2) sites) ==========")
    psi_sp, P_sparse = build_setup(N_plaq)
    image_p_leakage(psi_sp, P_sparse; label="after P·random_mps")

    psi_sp_o = ITensorMPS.orthogonalize(psi_sp, 1)
    image_p_leakage(psi_sp_o, P_sparse; label="after orthogonalize(psi, 1)")
end
nothing
