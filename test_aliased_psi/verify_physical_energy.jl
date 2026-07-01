# PHYSICALITY VERIFIER for aliased Path-B energy.
#
# The reported aliased E comes from the Path-B M-corrected generalized eigensolve
# (A = M^{-1}·H_eff). This script independently recomputes the TRUE Rayleigh
# quotient  E_chk = <psi|H|psi> / <psi|psi>  on the *decompressed dense* psi using
# stock ITensorMPS `inner` (bare H, no M, no Path-B machinery).
#
# Decisive: the Rayleigh quotient of any real state is >= the true ground energy.
#   - E_chk == E_reported  => reported energy is a genuine expectation value
#                              => physical (>= E_exact), just a better variational state.
#   - E_chk  > E_reported  => reported energy is a Path-B/M^{-1} artifact (unphysical).

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra: norm
using Random, Printf

include("../test_sparse_psi/utils.jl")

# Canonical build_setup (verbatim from test_aliased_kl.jl line 83), also returning P.
function build_setup(N::Int, psign::Int, spin::Int)
    Random.seed!(42)
    sites = spin == 2 ? siteinds("S=1/2", 2*N+2) :
            spin == 3 ? siteinds("S=1",   2*N+2) : error("Unsupported spin: $spin")
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
    mulMPO(A, B) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H    = MPO(os, sites)
    psi0 = random_mps(sites)
    psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, P_sparse, psi_ali
end

honest_bd(psi) = maximum((cis = commoninds(psi[i], psi[i+1]);
                          isempty(cis) ? 0 : prod(ITensors.dim, cis)) for i in 1:length(psi)-1)

# Decompress an aliased MPS to a plain dense MPS (per-tensor unfuse).
function densify_mps(psi)
    MPS([ITensors.has_external_storage(psi[i]) ?
         SparseBackends.to_dense_itensors_unfused(psi[i]) : psi[i] for i in 1:length(psi)])
end

let
    N      = parse(Int, get(ENV, "VN", "12"))
    md     = parse(Int, get(ENV, "VMD", "50"))
    nsw    = parse(Int, get(ENV, "VSW", "8"))
    psign  = get(ENV, "VPS", "-1") == "-1" ? -1 : +1
    spin   = parse(Int, get(ENV, "VSPIN", "3"))

    println("=== PHYSICALITY VERIFY  N=$N md=$md sweeps=$nsw psign=$psign ===")
    H, P, psi = build_setup(N, psign, spin)
    println("sites=$(length(psi))")

    E = NaN
    for i in 1:nsw
        sw = Sweeps(1); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
        (E, psi, _, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        @printf("  [sweep %2d] E_reported=%.12f  maxtruncerr=%.3e\n", i, E, terr)
        flush(stdout)
    end

    println("\n--- independent recompute on decompressed dense psi (bare H, stock inner) ---")
    psi_d = densify_mps(psi)
    nrm2  = real(inner(psi_d, psi_d))
    Hexp  = real(inner(psi_d, H, psi_d))
    E_chk = Hexp / nrm2
    @printf("honest_bd(psi)      = %d\n", honest_bd(psi))
    @printf("<psi|psi>           = %.12f   (norm=%.6f)\n", nrm2, sqrt(nrm2))
    @printf("<psi|H|psi>         = %.12f\n", Hexp)
    @printf("E_reported (PathB)  = %.12f\n", E)
    @printf("E_recompute (bareH) = %.12f\n", E_chk)
    @printf("diff (recompute - reported) = %+.3e\n", E_chk - E)

    # --- THREE-WAY triangulation -------------------------------------------------
    # Inject the ALIASED matvec into the numerator: <psi_d | H | psi_ali> routes
    # H|psi_ali> through the same aliased contraction kernel DMRG uses, with the
    # densified bra. Norm stays dense (aliased x aliased is not a standard kernel).
    # This bypasses the M^-1 generalized eigensolve, so:
    #   E_RQ_dense  vs E_RQ_aliased  ->  aliased contraction KERNEL correctness
    #   E_reported  vs E_RQ_aliased  ->  M^-1 generalized-eigensolve correctness
    println("\n--- three-way triangulation (kernel vs M^-1) ---")
    E_chk_ali = NaN
    try
        num_ali   = real(inner(psi_d, H, psi))   # psi is the aliased MPS; psi_d = densify(psi)
        E_chk_ali = num_ali / nrm2
        @printf("<psi_d|H|psi_ali> (aliased kernel)  = %.12f\n", num_ali)
    catch err
        @printf("[aliased-kernel inner failed: %s]\n", err)
    end
    @printf("E_RQ_dense   (full densify, no kernel/M) = %.12f\n", E_chk)
    @printf("E_RQ_aliased (aliased kernel, no M^-1)   = %.12f\n", E_chk_ali)
    @printf("E_reported   (PathB: kernel + M^-1)      = %.12f\n", E)
    @printf("  kernel diff  (E_RQ_dense - E_RQ_aliased)  = %+.3e   [aliased contraction kernel]\n", E_chk - E_chk_ali)
    @printf("  M^-1   diff  (E_reported - E_RQ_aliased)  = %+.3e   [M^-1 eigensolve]\n", E - E_chk_ali)

    tol = 1e-6 * abs(E)
    if E_chk - E > tol
        println("\n*** WARNING: recompute is ABOVE reported by more than tol ($(tol)) ***")
        println("*** reported energy is NOT a true expectation value -> Path-B artifact ***")
    elseif abs(E_chk - E) <= tol
        println("\nOK: recompute MATCHES reported within tol -> reported E is a genuine")
        println("    Rayleigh quotient => physical (>= E_exact), a better variational state.")
    else
        println("\nNote: recompute is BELOW reported (reported was conservative); also fine.")
    end
end
