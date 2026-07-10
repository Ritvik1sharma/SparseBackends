# Is the gram computation (env-slice rproj·eR) correct?  Compute Rgram TWO ways and
# compare: (A) DMRG env-slice = rproj(PH)·eR[b+2];  (B) DIRECT ψ†ψ over the right sites
# (no H, no co-vector). If A≠B → the env-slice gram computation is buggy. Also print
# the core bond dims (is it b.d.-1?) and each gram's proj-defect.
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf
include("test_aliased_kl.jl")
N = 2; psign = -1; spin = 3
H, psi = build_setup(N, psign, spin)
_dense(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x

# core bond dims: for each site print the aliased blksize (dense mult) + link dims
println("=== initial ψ per-site structure (core b.d. = blksize / mult) ===")
for i in 1:length(psi)
    a = psi[i].tensor.data.aliased
    linkdims = [dim(I) for I in inds(psi[i]) if hastags(I,"Link")]
    @printf("  site %2d: n_keys=%d n_tmpl=%d blksize(mult)=%d  linkdims=%s\n",
            i, length(a.keys), a.n_templates, a.blksize, string(linkdims))
end

function report(lbl, G)
    Gd=_dense(G); gi=collect(inds(Gd)); unp=filter(I->plev(I)==0,gi); prm=filter(I->plev(I)==1,gi)
    d=prod(I->dim(I),unp;init=1); A=reshape(Array(Gd,unp...,prm...),d,d); A=(A+A')/2
    ev=sort(real.(eigvals(Hermitian(A)));rev=true); c=maximum(abs,ev)
    pd=norm(A*A-c*A)/max(norm(c*A),eps())
    @printf("  [%s] d=%d c=%.4g proj-defect=%.3e  inds=%s\n", lbl, d, c, pd, string([(dim(I),string(tags(I))) for I in unp]))
    return A
end

b = 2
orthogonalize!(psi, b)
PH = ProjMPO(H); ITensorMPS.position!(PH, psi, b)
covec = SparseBackends.build_covector_cache(H)

println("\n=== Rgram at b=$b, TWO ways ===")
# (A) env-slice
RgA = ITensorMPS.rproj(PH) * covec.eR[b+2]
AA = report("A: env-slice rproj·eR", RgA)

# (B) direct ψ†ψ over sites b+2..N (right environment), no H, no co-vector
function direct_right_gram(psi, b)
    R = ITensors.ITensor(1.0)
    for i in length(psi):-1:(b+2)
        A = psi[i]
        R = R * A * dag(prime(A, "Link"))   # ψ links unprimed, ψ† links primed; physical contracts
    end
    return R
end
Rd = direct_right_gram(psi, b)
AB = report("B: direct ψ†ψ (no H)", Rd)

# compare A vs B (align indices)
try
    diff = norm(_dense(RgA) - _dense(Rd)) / max(norm(_dense(Rd)), eps())
    @printf("\n  ‖A − B‖/‖B‖ = %.3e   (≈0 ⇒ env-slice matches direct ψ†ψ; >0 ⇒ env-slice BUG)\n", diff)
catch e
    println("\n  (A,B index mismatch — comparing spectra instead)")
end
