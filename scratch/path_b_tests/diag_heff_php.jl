# diag_heff_php.jl
#
# Decisive localization diagnostic for the WRONG-ENERGY of the aliased ψ × aliased
# PHP combination (test_sparse_ham_psi). Mirrors Part B/C of
# ../test_aliased_psi/diag_gram_metric.jl, but with the PROJECTED Hamiltonian PHP
# (aliased) instead of the bare H, so it exercises the NEW aliased-env matvec.
#
# At bonds b=1..3 on the freshly-constructed (non-canonical) ψ it compares, for
# the SAME initial φ, the aliased-PHP path vs the dense-PHP reference:
#   A. φ sanity:        <φ|φ> aliased vs dense (must match — same state).
#   B. metric M:        <φ|M|φ> via the aliased build_minv/apply_minv machinery
#                       vs a dense reference, vs <Ψ|Ψ>.
#   C. H_eff (THE KEY): <φ|H_eff|φ> = <φ|product(PH,φ)>, aliased-PHP vs dense-PHP,
#                       and the densified ‖Hφ_ali − Hφ_den‖. If these DIFFER, the
#                       aliased-PHP/aliased-env matvec computes the wrong action
#                       (operator/matvec bug). If they MATCH but DMRG energy is
#                       still wrong, the bug is in the Path-B M⁻¹/eigsolve layer.
#
# Run (machine must be free — see SKILLS contention rule):
#   julia --project=.. diag_heff_php.jl                    # N_plaq=2
#   julia --project=.. diag_heff_php.jl --N-plaq 2

ENV["BMF_BOP_PROJECT"]           = get(ENV, "BMF_BOP_PROJECT", "1")
ENV["BMF_MINV_RTOL"]             = get(ENV, "BMF_MINV_RTOL", "1e-2")
ENV["SB_ALIASED_PERCM_CAP"]      = get(ENV, "SB_ALIASED_PERCM_CAP", "1")
ENV["SB_USE_QR"]                 = get(ENV, "SB_USE_QR", "1")
ENV["SB_BALANCED_OWNERSHIP"]     = get(ENV, "SB_BALANCED_OWNERSHIP", "1")
ENV["SB_ADAPTIVE_RANK"]          = get(ENV, "SB_ADAPTIVE_RANK", "1")
ENV["SB_FUSE_LINKS"]             = get(ENV, "SB_FUSE_LINKS", "1")
# SB_ALIASED_AA_ENV / SB_ALIASED_AA_HINT hardened 2026-06 — always on now.

using SparseBackends, ITensors, ITensorMPS
using Random, Printf, LinearAlgebra
using KrylovKit: eigsolve
using ArgParse

include("../test_sparse_psi/utils.jl")
include("../test_sparse_ham/aliased_helpers.jl")

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N-plaq"
            help = "Number of plaquettes (N)"
            arg_type = Int
            default = 2
    end
    return parse_args(s)
end

# Was DIAG_N_PLAQ env var.
const N_PLAQ = parse_command_line()["N-plaq"]

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
    H    = MPO(os, sites)
    psi0 = random_mps(sites)

    psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    H_ali = MPO(length(H))
    for i in 1:length(H)
        H1      = SparseBackends.contract_aliased_itensor(P_sparse[i]'', H[i]', :coo, :dense)
        H_eff_i = SparseBackends.contract_aliased_itensor(P_sparse[i], H1, :coo, :aliased)
        H_ali[i] = replaceprime(H_eff_i, 3 => 1)
    end
    get(ENV, "SB_FUSE_LINKS", "0") == "1" && fuse_sparse_links!(H_ali)

    H1d       = contract(P_sparse'', H', :coo, :dense)
    H_sparse  = replaceprime(contract(P_sparse, H1d, :coo, :blocksparse), 3 => 1)
    psi_sp    = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return (; sites, H, H_ali, psi_ali, H_dense, psi_dense)
end

densify(T::ITensor) = begin
    ITensors.has_external_storage(T) || return T
    s = ITensors.get_external_storage(T)
    s isa SparseBackends.WrappedAliasedBlockSparse ? ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...) :
    s isa SparseBackends.WrappedBlockSparse        ? ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...) : T
end

# Densify and contract <a|b> robustly by matching index TAGS+plev (the aliased
# PHP uses FusedSparse link tags; the dense PHP uses plain Link tags, so ids and
# even counts can differ — fall back to a tag-aligned dense scalar).
function expval(bra::ITensor, op_applied::ITensor)
    da = densify(bra); db = densify(op_applied)
    s = dag(da) * db
    return order(s) == 0 ? real(scalar(s)) : NaN  # NaN ⇒ leftover uncontracted inds
end

function build_phi_ali(psi, b)
    Aw = ITensors.get_external_storage(psi[b]); Bw = ITensors.get_external_storage(psi[b+1])
    Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)
    return Cw isa ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
end

let
    m = build_models(N_PLAQ, +1)
    N = length(m.psi_ali)
    println("=== diag_heff_php  N_plaq=$N_PLAQ  n_sites=$N  (aliased PHP vs dense PHP) ===\n")
    println("H_ali sparse-prefix widths (P) per site:")
    for i in 1:N
        T = m.H_ali[i]
        if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
            w = T.tensor.data
            @printf("  H[%d] P=%d  N=%d  inds=%s\n", i, SparseBackends._abs_head_len(w),
                    ndims(w.aliased), string([(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(T)]))
        end
    end

    for b in 1:min(3, N-1)
        println("\n================ BOND b=$b ================")
        phi_d   = m.psi_dense[b] * m.psi_dense[b+1]
        phi_ali = build_phi_ali(m.psi_ali, b)

        # A. φ sanity
        nn_d   = expval(phi_d, phi_d)
        nn_ali = expval(phi_ali, phi_ali)
        @printf("A. <phi|phi>:        dense=%.10f   ali=%.10f\n", nn_d, nn_ali)

        # C. H_eff via product(PH, φ) — aliased-PHP env vs dense-PHP env.
        PH_d   = ProjMPO(m.H_dense); position!(PH_d, m.psi_dense, b)
        PH_ali = ProjMPO(m.H_ali);   position!(PH_ali, m.psi_ali,  b)
        Hphi_d   = ITensorMPS.product(PH_d, phi_d)
        Hphi_ali = ITensorMPS.product(PH_ali, phi_ali)
        he_d   = expval(phi_d,   Hphi_d)
        he_ali = expval(phi_ali, Hphi_ali)
        @printf("C. <phi|H_eff|phi>:  dense=%.10f   ali=%.10f   |diff|=%.3e\n",
                he_d, he_ali, abs(he_d - he_ali))
        @printf("   Rayleigh <H_eff>/<phi|phi>:  dense=%.10f   ali=%.10f\n", he_d/nn_d, he_ali/nn_ali)
        # value-level: densified Hφ difference, tag+plev aligned
        da = densify(Hphi_d); db = densify(Hphi_ali)
        try
            db2 = permute(db, inds(da)...; allow_alias=true)
            @printf("   ‖Hphi_ali − Hphi_dense‖ (aligned) = %.3e  (‖Hphi_dense‖=%.3e)\n",
                    norm(array(da) .- array(db2)), norm(array(da)))
        catch e
            println("   (could not align Hphi inds for value diff: ", sprint(showerror, e), ")")
        end

        # ----- D. Path-B layer: dense generalized eig vs actual dmrg.jl B_op -----
        gc_d   = SparseBackends.init_gram_cache(m.psi_dense)
        gc_ali = SparseBackends.init_gram_cache(m.psi_ali)
        Lg_d = SparseBackends.get_left_gram(gc_d, b);   Rg_d = SparseBackends.get_right_gram(gc_d, b)
        Lg_a = SparseBackends.get_left_gram(gc_ali, b); Rg_a = SparseBackends.get_right_gram(gc_ali, b)

        dense_M_apply(G, y) = order(G) == 0 ? y * scalar(G) :
            replaceprime(G * y, 1 => 0; tags="Link")
        phi_inds = collect(inds(phi_d)); D = prod(dim, phi_inds; init=1)
        @printf("D. phi-space dim D=%d\n", D)
        if D <= 4000
            Heff_mat = zeros(ComplexF64, D, D); M_mat = zeros(ComplexF64, D, D)
            cidx = CartesianIndices(Tuple(dim(I) for I in phi_inds))
            for j in 1:D
                ej = ITensor(ComplexF64, phi_inds...); ej[cidx[j]] = 1.0
                hj = ITensorMPS.product(PH_d, ej)
                mj = dense_M_apply(densify(Lg_d), dense_M_apply(densify(Rg_d), ej))
                Heff_mat[:, j] = vec(array(hj, phi_inds...))
                M_mat[:, j]    = vec(array(mj, phi_inds...))
            end
            herm(A) = (A + A')/2; Heff_mat = herm(Heff_mat); M_mat = herm(M_mat)
            Fm = eigen(Hermitian(M_mat)); tol = 1e-10 * maximum(real, Fm.values)
            invsqrt = [real(l) > tol ? 1/sqrt(real(l)) : 0.0 for l in Fm.values]
            Minvhalf = Fm.vectors * Diagonal(invsqrt) * Fm.vectors'
            Bmat = herm(Minvhalf * Heff_mat * Minvhalf)
            evals = eigen(Hermitian(Bmat)).values
            @printf("   dense generalized eig: smallest=%.10f  (range [%.4f,%.4f], M rank=%d/%d)\n",
                    minimum(real, evals), minimum(real, evals), maximum(real, evals),
                    count(>(tol), real.(Fm.values)), D)
        end

        Mhalf_L, Linv_L, Mhalf_R, Linv_R =
            SparseBackends.build_minv_half_pair_factored(Lg_a, Rg_a; phi_template=phi_ali)
        recast_to_phi(Hv) = begin
            if ITensors.has_external_storage(Hv) && ITensors.has_external_storage(phi_ali)
                Tw = ITensors.get_external_storage(phi_ali); Cw = ITensors.get_external_storage(Hv)
                if Cw isa SparseBackends.WrappedAliasedBlockSparse && Tw isa SparseBackends.WrappedAliasedBlockSparse
                    return ITensors._itensor_from_external_storage(SparseBackends.align_aliased_axes(Cw, Tw))
                end
            end
            return Hv
        end
        apply_half(opL, opR, z) = begin
            z = SparseBackends.apply_minv_preserve_bs(opL, z, phi_ali)
            z = SparseBackends.apply_minv_preserve_bs(opR, z, phi_ali); z
        end
        # D1. recast value-preservation
        Hphi_raw  = ITensorMPS.product(PH_ali, phi_ali)
        Hphi_rcst = recast_to_phi(Hphi_raw)
        @printf("D1. recast value drift ‖recast(Hφ)−Hφ‖ = %.3e\n",
                norm(array(densify(Hphi_rcst), inds(densify(Hphi_raw))...) .- array(densify(Hphi_raw))))
        B_op(y) = apply_half(Linv_L, Linv_R, recast_to_phi(ITensorMPS.product(PH_ali, apply_half(Linv_L, Linv_R, y))))
        y0 = apply_half(Mhalf_L, Mhalf_R, phi_ali)
        try
            vals, _ = eigsolve(B_op, y0, 1, :SR; ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0)
            @printf("D2. actual aliased B_op smallest eig = %.10f\n", real(vals[1]))
        catch e
            println("D2. B_op eigsolve FAILED: ", sprint(showerror, e))
        end
    end
end
nothing
