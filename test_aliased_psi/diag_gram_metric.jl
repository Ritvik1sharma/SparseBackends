# diag_gram_metric.jl
#
# Decisive correctness diagnostic for the aliased Path-B DMRG energy bug.
#
# Tests, at one or more bonds, on the FRESHLY-CONSTRUCTED (non-canonical) psi:
#   A. Gram-cache correctness: densify(Lgram_ali) == densify(Lgram_dense), same Rgram.
#      (README working hypothesis #7 — "gram values wrong for aliased psi".)
#   B. Metric validity + eigensolver machinery:
#        <phi|M|phi> via the ACTUAL aliased M_dot machinery
#                    (build_minv_half_pair_factored + apply_minv_preserve_bs ×4)
#        vs <phi|M|phi> via a dense reference contraction
#        vs <Psi|Psi> (full-state norm^2).
#      All three must agree if M is the right metric AND apply_minv is correct.
#   C. H_eff consistency: <phi|H_eff|phi> (product(PH,phi)) vs <Psi|H|Psi>.
#   D. Ground-truth generalized eigenproblem: build H_eff and M as dense matrices
#      (from the dense psi path), solve eigen(H_eff, M), report smallest real
#      eigenvalue. This is what Path-B SHOULD return at this bond. Then run the
#      ACTUAL aliased Path-B eigsolve (mirroring dmrg.jl) and compare.

using SparseBackends, ITensors, ITensorMPS
using Random, Printf
using LinearAlgebra
using KrylovKit: eigsolve, InnerProductVec

include("../test_sparse_psi/utils.jl")

const N_PLAQ = parse(Int, get(ENV, "DIAG_N_PLAQ", "2"))

function build_setup(N::Int, psign::Int)
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
        t += 0.5,         "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
        t += 0.5 * psign, "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    return H, P_sparse, psi0, sites
end

# Densify an aliased / BS ITensor to a plain dense ITensor with same inds.
function densify(T::ITensor)
    ITensors.has_external_storage(T) || return T
    s = ITensors.get_external_storage(T)
    if s isa SparseBackends.WrappedAliasedBlockSparse
        return ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...)
    elseif s isa SparseBackends.WrappedBlockSparse
        return ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...)
    end
    return T
end

# Norm of (densified) difference, aligning index order.
function diffnorm(a::ITensor, b::ITensor)
    da = densify(a); db = densify(b)
    common = collect(inds(da))
    db2 = permute(db, common...; allow_alias=true)
    return norm(array(da) .- array(db2))
end

H, P_sparse, psi0, sites = build_setup(N_PLAQ, +1)
N = length(psi0)
println("=== diag_gram_metric  N_plaq=$N_PLAQ  n_sites=$N ===\n")

psi_d   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense),   1 => 0)
psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)

# ---------------------------------------------------------------------------
# Full dense state from the (densified) MPS, and full H, for ground-truth
# <Psi|Psi> and <Psi|H|Psi>.
# ---------------------------------------------------------------------------
function full_state(psi)
    Ψ = densify(psi[1])
    for k in 2:length(psi); Ψ = Ψ * densify(psi[k]); end
    return Ψ
end
Ψ_d   = full_state(psi_d)
Ψ_ali = full_state(psi_ali)
@printf("full-state diff (ali vs dense) = %.3e\n", norm(array(Ψ_d) .- array(permute(Ψ_ali, inds(Ψ_d)...; allow_alias=true))))

function build_full_op(H, N)
    Hf = H[1]
    for k in 2:N; Hf = Hf * H[k]; end
    return Hf
end
# DIAG_GRAM_ONLY=1: skip the full-dense operator build (it OOMs for N≥4 — the
# full H is 2^(2N+2)-dim). The gram/metric construction checks (Part A) don't
# need it, so this lets us verify M's correctness at scale.
const _GRAM_ONLY = get(ENV, "DIAG_GRAM_ONLY", "0") == "1"
H_full = _GRAM_ONLY ? nothing : build_full_op(H, N)
function state_norm2(Ψ); return real(scalar(dag(Ψ) * Ψ)); end
function state_energy(Ψ)
    Ψp = prime(Ψ, sites...)
    return real(scalar(dag(Ψp) * (H_full * Ψ)))
end
if !_GRAM_ONLY
    nn_d   = state_norm2(Ψ_d);   ee_d   = state_energy(Ψ_d)
    nn_ali = state_norm2(Ψ_ali); ee_ali = state_energy(Ψ_ali)
    @printf("<Psi|Psi>:   dense=%.10f  ali=%.10f\n", nn_d, nn_ali)
    @printf("<Psi|H|Psi>: dense=%.10f  ali=%.10f\n", ee_d, ee_ali)
    @printf("Rayleigh E = <H>/<1>: dense=%.10f  ali=%.10f\n\n", ee_d/nn_d, ee_ali/nn_ali)
end

# ---------------------------------------------------------------------------
# A. Gram-cache correctness.
# ---------------------------------------------------------------------------
println("--- A. Gram-cache correctness (aliased vs dense, densified) ---")
gc_d   = SparseBackends.init_gram_cache(psi_d)
gc_ali = SparseBackends.init_gram_cache(psi_ali)
for i in 1:N+1
    dl = diffnorm(gc_ali.L[i], gc_d.L[i])
    dr = diffnorm(gc_ali.R[i], gc_d.R[i])
    nl = norm(densify(gc_d.L[i])); nr = norm(densify(gc_d.R[i]))
    @printf("  i=%d  |dLgram|=%.3e (|L|=%.3e)   |dRgram|=%.3e (|R|=%.3e)\n", i, dl, nl, dr, nr)
end
if _GRAM_ONLY
    println("\n[DIAG_GRAM_ONLY] gram/M-construction check complete; skipping H-dependent parts (full-dense op OOMs at N≥4).")
    println("If all |dLgram|/|dRgram| above are ~0, the aliased gram/metric M is constructed correctly at this N.")
    exit(0)
end

# ---------------------------------------------------------------------------
# Helper: dense application of a gram tensor G (paired bond/bond') to y.
# ---------------------------------------------------------------------------
function dense_M_apply(G::ITensor, y::ITensor)
    order(G) == 0 && return y * scalar(G)
    Gy = G * y
    return replaceprime(Gy, 1 => 0; tags="Link")
end

# ---------------------------------------------------------------------------
# Per-bond tests B, C, D.
# ---------------------------------------------------------------------------
function build_phi_ali(psi, b)
    Aw = ITensors.get_external_storage(psi[b])
    Bw = ITensors.get_external_storage(psi[b+1])
    Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)
    return Cw isa ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
end

function run_bond(b, psi_d, psi_ali, gc_d, gc_ali, H, sites, nn_ali, ee_ali)
    println("\n================ BOND b=$b ================")

    phi_d   = psi_d[b] * psi_d[b+1]
    phi_ali = build_phi_ali(psi_ali, b)

    Lgram_d   = SparseBackends.get_left_gram(gc_d, b)
    Rgram_d   = SparseBackends.get_right_gram(gc_d, b)
    Lgram_ali = SparseBackends.get_left_gram(gc_ali, b)
    Rgram_ali = SparseBackends.get_right_gram(gc_ali, b)

    # ----- B. metric validity + apply_minv machinery -----
    # B1: dense reference quadratic form <phi|M|phi> with M = Lgram (x) Rgram.
    Mphi_dref = dense_M_apply(densify(Lgram_d), dense_M_apply(densify(Rgram_d), densify(phi_d)))
    q_dref = real(scalar(dag(densify(phi_d)) * Mphi_dref))

    # B2: aliased M_dot machinery (EXACTLY mirrors dmrg.jl Path-B).
    Mhalf_L, Linv_L, Mhalf_R, Linv_R =
        SparseBackends.build_minv_half_pair_factored(Lgram_ali, Rgram_ali; phi_template=phi_ali)
    function M_dot(x, y)
        My = SparseBackends.apply_minv_preserve_bs(Mhalf_L, y,  y)
        My = SparseBackends.apply_minv_preserve_bs(Mhalf_L, My, y)
        My = SparseBackends.apply_minv_preserve_bs(Mhalf_R, My, y)
        My = SparseBackends.apply_minv_preserve_bs(Mhalf_R, My, y)
        return inner(x, My)
    end
    q_ali = real(M_dot(phi_ali, phi_ali))

    # B3: dense-path M_dot (same machinery but dense gram + dense phi) — isolates
    # whether the bug is in apply_minv-on-aliased vs build_minv itself.
    MhL_d, _, MhR_d, _ =
        SparseBackends.build_minv_half_pair_factored(Lgram_d, Rgram_d; phi_template=phi_d)
    function M_dot_d(x, y)
        My = SparseBackends.apply_minv_preserve_bs(MhL_d, y,  y)
        My = SparseBackends.apply_minv_preserve_bs(MhL_d, My, y)
        My = SparseBackends.apply_minv_preserve_bs(MhR_d, My, y)
        My = SparseBackends.apply_minv_preserve_bs(MhR_d, My, y)
        return inner(x, My)
    end
    q_dmach = real(M_dot_d(phi_d, phi_d))

    @printf("B. <phi|M|phi>:  dense-ref=%.10f   ali-machinery=%.10f   dense-machinery=%.10f   (<Psi|Psi>=%.10f)\n",
            q_dref, q_ali, q_dmach, nn_ali)

    # ----- C. H_eff consistency -----
    PH_d = ProjMPO(H); position!(PH_d, psi_d, b)
    PH_ali = ProjMPO(H); position!(PH_ali, psi_ali, b)
    Hphi_d   = ITensorMPS.product(PH_d, phi_d)
    Hphi_ali = ITensorMPS.product(PH_ali, phi_ali)
    he_d   = real(scalar(dag(densify(phi_d))   * densify(Hphi_d)))
    he_ali = real(scalar(dag(densify(phi_ali)) * densify(Hphi_ali)))
    @printf("C. <phi|H_eff|phi>:  dense=%.10f   ali=%.10f   (<Psi|H|Psi>=%.10f)\n",
            he_d, he_ali, ee_ali)

    # ----- D. ground-truth generalized eigenproblem from DENSE matrices -----
    phi_inds = collect(inds(phi_d))
    D = prod(dim, phi_inds; init=1)
    @printf("D. phi-space dim D=%d\n", D)
    lam_dense = NaN
    if D <= 4000
        Heff_mat = zeros(ComplexF64, D, D)
        M_mat    = zeros(ComplexF64, D, D)
        cidx = CartesianIndices(Tuple(dim(I) for I in phi_inds))
        for j in 1:D
            ej = ITensor(ComplexF64, phi_inds...)
            ej[cidx[j]] = 1.0
            hj = ITensorMPS.product(PH_d, ej)
            mj = dense_M_apply(densify(Lgram_d), dense_M_apply(densify(Rgram_d), ej))
            Heff_mat[:, j] = vec(array(hj, phi_inds...))
            M_mat[:, j]    = vec(array(mj, phi_inds...))
        end
        herm(A) = (A + A')/2
        Heff_mat = herm(Heff_mat); M_mat = herm(M_mat)
        # symmetric-definite generalized eig via M^{-1/2}.
        Fm = eigen(Hermitian(M_mat))
        tol = 1e-10 * maximum(real, Fm.values)
        invsqrt = [real(l) > tol ? 1/sqrt(real(l)) : 0.0 for l in Fm.values]
        Minvhalf = Fm.vectors * Diagonal(invsqrt) * Fm.vectors'
        B = herm(Minvhalf * Heff_mat * Minvhalf)
        evals = eigen(Hermitian(B)).values
        lam_dense = minimum(real, evals)
        @printf("   dense generalized eig: smallest=%.10f   (full range [%.4f, %.4f], M rank=%d/%d)\n",
                lam_dense, lam_dense, maximum(real, evals),
                count(>(tol), real.(Fm.values)), D)
    else
        println("   (skipped dense matrix build: D too large)")
    end

    # ----- D2. actual aliased Path-B eigsolve (mirror dmrg.jl) -----
    phi_wrapped = InnerProductVec(phi_ali, M_dot)
    H_op = function(v)
        Hv = ITensorMPS.product(PH_ali, v[])
        if ITensors.has_external_storage(Hv) && ITensors.has_external_storage(phi_ali)
            Tw = ITensors.get_external_storage(phi_ali)
            Cw = ITensors.get_external_storage(Hv)
            if Cw isa SparseBackends.WrappedAliasedBlockSparse && Tw isa SparseBackends.WrappedAliasedBlockSparse
                Hv = ITensors._itensor_from_external_storage(
                    SparseBackends.recast_aliased_to_template(Cw, Tw))
            end
        end
        return InnerProductVec(Hv, M_dot)
    end
    try
        vals, vecs = eigsolve(H_op, phi_wrapped, 1, :SR;
            ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0)
        @printf("D2. aliased Path-B eigsolve smallest (CURRENT, no Minv) = %.10f\n", real(vals[1]))
    catch err
        @printf("D2. aliased Path-B eigsolve FAILED: %s\n", err)
    end

    # ----- D3. FIX hypothesis: apply A = M^{-1} H_eff with M-inner product -----
    # M^{-1} = Linv_L^2 (x) Linv_R^2 (Linv = M^{-1/2}). A = M^{-1}H_eff is
    # M-self-adjoint, so Lanczos in the M-inner product finds the generalized
    # eigenvalues H x = λ M x. This is the correction the current code omits.
    function Minv_apply(z)
        Mz = SparseBackends.apply_minv_preserve_bs(Linv_L, z,  phi_ali)
        Mz = SparseBackends.apply_minv_preserve_bs(Linv_L, Mz, phi_ali)
        Mz = SparseBackends.apply_minv_preserve_bs(Linv_R, Mz, phi_ali)
        Mz = SparseBackends.apply_minv_preserve_bs(Linv_R, Mz, phi_ali)
        return Mz
    end
    A_op = function(v)
        Hv = ITensorMPS.product(PH_ali, v[])
        if ITensors.has_external_storage(Hv) && ITensors.has_external_storage(phi_ali)
            Tw = ITensors.get_external_storage(phi_ali)
            Cw = ITensors.get_external_storage(Hv)
            if Cw isa SparseBackends.WrappedAliasedBlockSparse && Tw isa SparseBackends.WrappedAliasedBlockSparse
                Hv = ITensors._itensor_from_external_storage(
                    SparseBackends.recast_aliased_to_template(Cw, Tw))
            end
        end
        Av = Minv_apply(Hv)
        return InnerProductVec(Av, M_dot)
    end
    try
        vals, vecs = eigsolve(A_op, InnerProductVec(phi_ali, M_dot), 1, :SR;
            ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0)
        # Report generalized Rayleigh quotient of the returned vector (the
        # eigenvalue of A IS the generalized eigenvalue, but recompute to be safe).
        x = vals isa AbstractVector ? vals[1] : vals
        @printf("D3. aliased Path-B eigsolve smallest (FIX: A=Minv*Heff) = %.10f\n", real(x))
    catch err
        @printf("D3. aliased Path-B eigsolve (FIX) FAILED: %s\n", err)
    end

    # THIS HAS BEEN ARCHIVED
    # # ----- D4. generalized Rayleigh-Ritz oracle (PARKED) -----
    # # ARCHIVED 2026-06: the Rayleigh-Ritz direction is parked. It matches the
    # # dense oracle at b=1/b=3 (~1e-11) and the B_op energy end-to-end, but is
    # # ~2.5× slower than B_op, and b=2 exposes a PRE-EXISTING aliased H_eff matvec
    # # bug that is upstream of RR (raw==recast, both ≠ dense ‖Hφ‖²). Gated behind
    # # BMF_RAYLEIGH_RITZ=1 so the default diagnostic (D1–D3) is unchanged. See the
    # # README "ARCHIVED: Rayleigh-Ritz" section to resume.
    # if get(ENV, "BMF_RAYLEIGH_RITZ", "0") == "1"
    #     recast_ali = function(Hv)
    #         if ITensors.has_external_storage(Hv) && ITensors.has_external_storage(phi_ali)
    #             Tw = ITensors.get_external_storage(phi_ali)
    #             Cw = ITensors.get_external_storage(Hv)
    #             if Cw isa SparseBackends.WrappedAliasedBlockSparse && Tw isa SparseBackends.WrappedAliasedBlockSparse
    #                 return ITensors._itensor_from_external_storage(
    #                     SparseBackends.recast_aliased_to_template(Cw, Tw))
    #             end
    #         end
    #         return Hv
    #     end
    #     Hop_rr = v -> recast_ali(ITensorMPS.product(PH_ali, v))
    #     try
    #         vals, vecs = SparseBackends.rayleigh_ritz_local_eigsolve(
    #             Hop_rr, phi_ali, Lgram_ali, Rgram_ali;
    #             which=:SR, tol=1e-12,
    #             krylovdim=8,
    #             maxiter=100, b=b)
    #         lam_rr = real(vals[1])
    #         okstr = isnan(lam_dense) ? "(no dense ref)" :
    #                 (abs(lam_rr - lam_dense) < 1e-8 ? "MATCH ✓" : "MISMATCH ✗")
    #         @printf("D4. Rayleigh-Ritz smallest = %.10f   (dense ref=%.10f, |Δ|=%.2e) %s\n",
    #                 lam_rr, lam_dense, isnan(lam_dense) ? NaN : abs(lam_rr - lam_dense), okstr)
    #         # Confirm the raw-gram Mop matches the dense_M_apply reference on phi.
    #         Mop_rr = v -> SparseBackends.apply_minv_preserve_bs(Lgram_ali,
    #                         SparseBackends.apply_minv_preserve_bs(Rgram_ali, v, phi_ali), phi_ali)
    #         q_rr = real(inner(phi_ali, Mop_rr(phi_ali)))
    #         @printf("    <phi|M|phi> via raw-gram Mop = %.10f   (dense-ref=%.10f, |Δ|=%.2e)\n",
    #                 q_rr, q_dref, abs(q_rr - q_dref))
    #     catch err
    #         @printf("D4. Rayleigh-Ritz FAILED: %s\n", err)
    #     end
    # end
end

for b in [1, 2, 3]
    b > N-1 && continue
    run_bond(b, psi_d, psi_ali, gc_d, gc_ali, H, sites, nn_ali, ee_ali)
end

nothing
