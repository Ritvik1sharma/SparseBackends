# diag_gram_offblock.jl — is the metric M's tail WITHIN-channel or CROSS-channel?
#
# Each bond link is a DOUBLED index: a small CHANNEL index (dim 4 or 2, the sparse
# key) × a large MULTIPLICITY index (dense template extent).  M has axes
# (ch, mu ; ch', mu').  Test block-diagonality in the channel index:
#   off-channel weight = sqrt( Σ_{i≠j} ‖M[i,:,j,:]‖² / ‖M‖² )
#     ~0  ⇒ M block-diagonal in channel; tail is WITHIN-channel (a single
#            template's mult-columns not mutually orthonormal). Per-template fix.
#     >0  ⇒ CROSS-channel overlap: channel i's states overlap channel j's
#            (mismatched templates overlap). Needs a COUPLED orthogonalization.
# Also per within-channel block: deviation from c·I.  L (left) vs R (right), all bonds.
#
# Run:  julia --project=.. diag_gram_offblock.jl
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf
include("test_aliased_kl.jl")

N = 12; psign = -1; spin = 3
_dense(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x

function analyze(lbl, G)
    Gd  = _dense(G)
    unp = filter(I -> plev(I) == 0, collect(inds(Gd)))
    length(unp) == 2 || (println("  [$lbl] unexpected #unprimed=$(length(unp))"); return)
    # channel = smaller dim, mult = larger dim
    ch = dim(unp[1]) <= dim(unp[2]) ? unp[1] : unp[2]
    mu = ch === unp[1] ? unp[2] : unp[1]
    nc = dim(ch); nm = dim(mu)
    A  = Array(Gd, ch, mu, prime(ch), prime(mu))    # (nc, nm, nc, nm)
    total = norm(A)^2
    off = 0.0
    for i in 1:nc, j in 1:nc
        i == j && continue
        off += norm(@view A[i, :, j, :])^2
    end
    # DEFINITIVE clean/spread test: eigenvalues of the full M (= c·Π ⟺ 1 nonzero cluster)
    d  = nc * nm
    Mf = reshape(A, d, d); Mf = (Mf + Mf') / 2
    ev = sort(real.(eigvals(Hermitian(Mf))); rev = true)
    cg = maximum(abs, ev)
    nz = ev[abs.(ev) .> 1e-8 * cg]
    nclust = length(unique(round.(nz ./ cg, digits = 3)))
    # within-channel diagonal block shape (NOTE: not a clean/spread test — a channel-
    # coupling projector has non-∝I diagonal blocks even when M is a clean projector)
    c = 0.0
    for i in 1:nc; c = max(c, maximum(abs, diag(@view A[i, :, i, :]))); end
    wdev = 0.0
    for i in 1:nc
        B = A[i, :, i, :]; B = (B + B') / 2
        wdev = max(wdev, norm(B - c*Matrix(I, nm, nm)) / (c*sqrt(nm)))
    end
    @printf("  [%s] ch=%d mu=%d  rank=%d  cg=%.5g  #nz-clusters=%-3d %s | OFF-CHAN/‖M‖=%.3e  within-blk-dev=%.2e\n",
            lbl, nc, nm, length(nz), cg, nclust,
            nclust == 1 ? "CLEAN c·Π" : "SPREAD  ", sqrt(max(off,0)/total), wdev)
end

H, psi0 = build_setup(N, psign, spin)
res = run_sweeps(H, psi0, 12, 40; label = "conv")
psi = res.psi
covec = SparseBackends.build_covector_cache(H)
println("\n########## converged  E=$(res.E) ##########")
for b in (5, 6, 7, 8, 9, 10)
    par = isodd(b) ? "ODD " : "EVEN"
    orthogonalize!(psi, b)
    PH = ProjMPO(H); ITensorMPS.position!(PH, psi, b)
    Lg = ITensorMPS.lproj(PH) * covec.eL[b]
    Rg = ITensorMPS.rproj(PH) * covec.eR[b + 2]
    println("\n=== bond $b ($par) ===")
    analyze("Lg", Lg)
    analyze("Rg", Rg)
end
