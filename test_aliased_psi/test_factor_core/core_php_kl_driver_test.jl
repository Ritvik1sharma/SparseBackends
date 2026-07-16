# core_php_kl_driver_test.jl — core-mode DMRG on the KL model (OFF-diagonal / flip P).
# Companion to core_php_driver_test.jl (PXP). With the (rv_b,rv_{b+1})→template map
# EMITTED by the ψ[b]·ψ[b+1] aliased contract (emit_window_map=true) and read back by
# window_write_map, KL should work — the flip-P routing can't be reconstructed post-hoc
# (φ sums over the internal FSM bond), so it is produced at the multiply.
# Reference: fully-converged dense-KL ground E = −7.764220969914 (test_dense_kl.jl
# --N-plaq 5 --maxdim 40 --n-sweeps 25, psign=+1). NOTE the 6-sweep dense value
# (−7.763745...) is NOT converged; core-mode converges faster (~sweep 3) to the same E.

using SparseBackends, ITensors, ITensorMPS, Random, Printf, LinearAlgebra
include("../../test_sparse_psi/utils.jl")
include("../../test_sparse_ham/aliased_helpers.jl")

# build_models from diag_heff_php.jl, but ALSO returning P_sparse.
function build_models(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    cs = 0.5 * psign
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    psi0 = random_mps(sites)
    psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, P_sparse, psi_ali
end

let
    N_plaq = 5                       # 2N+2 = 12 sites, to match "N=12"
    H, P, psi = build_models(N_plaq, +1)
    @printf("KL model: %d sites (N_plaq=%d)\n", length(psi), N_plaq)
    try
        E, _ = ITensorMPS.dmrg_core_php(H, P, psi; nsweeps=6, maxdim=40)
        @printf("[KL core-mode] final E = %.10f  %s\n", E,
                abs(E - (-7.764220969914)) < 1e-6 ? "PASS ✓" : "CHECK (ref -7.764220969914)")
    catch e
        println("KL core-mode ERROR — full backtrace:")
        Base.showerror(stdout, e, catch_backtrace())
        println()
    end
end
nothing
