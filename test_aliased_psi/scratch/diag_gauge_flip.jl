# diag_gauge_flip.jl — is the Lgram/Rgram disparity purely the CANONICALIZATION
# DIRECTION (a gauge op), or intrinsic to the physical cut?
#
# For ONE fixed physical link λ = link(k,k+1), build the SAME metric two ways:
#   (A) LEFT-canonical view  : orthocenter at k+1 ⇒ sites 1..k left-canonical;
#       λ is the right link of a left-canonical block.  Lg = lproj(PH@k+1)·eL[k+1]
#   (B) RIGHT-canonical view : orthocenter at k   ⇒ sites k+1..N right-canonical;
#       λ is the left link of a right-canonical block. Rg = rproj(PH@k-1)·eR[k+1]
# Same λ, opposite gauge.  If (A) SPREAD and (B) CLEAN ⇒ disparity is the
# canonicalization direction (the SVD ortho op), and clean c·Π is a GAUGE choice.
#
# Run:  julia --project=.. diag_gauge_flip.jl
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf
include("test_aliased_kl.jl")
N = 12; psign = -1; spin = 3
_dense(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x

function verdict(lbl, G)
    Gd  = _dense(G)
    unp = filter(I -> plev(I) == 0, collect(inds(Gd)))
    length(unp) == 2 || (println("  [$lbl] #unprimed=$(length(unp))"); return)
    ch = dim(unp[1]) <= dim(unp[2]) ? unp[1] : unp[2]
    mu = ch === unp[1] ? unp[2] : unp[1]
    d  = dim(ch) * dim(mu)
    A  = reshape(Array(Gd, ch, mu, prime(ch), prime(mu)), d, d); A = (A + A') / 2
    ev = sort(real.(eigvals(Hermitian(A))); rev = true); cg = maximum(abs, ev)
    nz = ev[abs.(ev) .> 1e-8*cg]
    nclust = length(unique(round.(nz ./ cg, digits = 3)))
    @printf("  [%s]  ch=%d mu=%d  rank=%d  cg=%.5g  #nz-clusters=%-3d %s\n",
            lbl, dim(ch), dim(mu), length(nz), cg, nclust, nclust==1 ? "CLEAN c·Π" : "SPREAD")
end

H, psi0 = build_setup(N, psign, spin)
res = run_sweeps(H, psi0, 12, 40; label="conv"); psi = res.psi
covec = SparseBackends.build_covector_cache(H)
println("\nconverged E=$(res.E)\n")
PH = ProjMPO(H)
for k in (6, 7, 8)      # link λ between site k and k+1
    println("=== physical link λ = ($k,$(k+1)) — SAME link, two gauges ===")
    # (A) left-canonical view: orthocenter at k+1
    ITensorMPS.position!(PH, psi, k+1)
    LgA = ITensorMPS.lproj(PH) * covec.eL[k+1]
    verdict("LEFT-canonical  (Lg-style)", LgA)
    # (B) right-canonical view: orthocenter at k
    ITensorMPS.position!(PH, psi, k-1)
    RgB = ITensorMPS.rproj(PH) * covec.eR[k+1]
    verdict("RIGHT-canonical (Rg-style)", RgB)
end
