# core_eigsolve_test.jl — STEP 4a: the core-mode LOCAL eigensolve on one bond.
# Krylov vectors = dense cores; matvec f(core) = product(cpm, form_φ(core)).
# form_φ(core) = replaceprime(P[b]·P[b+1]·core, 1=>0)  (dense φ = P·core; matvec
# densifies at step 1 anyway — the aliased×dense→dense first implementation).
# Asserts:
#   (A) f(core0) == product(cpm, ψ[b]ψ[b+1])   (form_φ reproduces the aliased φ)
#   (B) f is Hermitian in plain Frobenius (metric I)
#   (C) KrylovKit ground eig λ == dense eig of the same operator matrix

using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
using KrylovKit: eigsolve

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]
_nz(dims, coords) = (A=zeros(Float64,dims...); for c in coords; A[(c .+ 1)...]=1.0; end; A)
const Rf = _nz((3,3,2),[(0,0,0),(1,1,1),(2,2,0)])
const Rb = _nz((3,3,2,2),[(0,0,0,0),(0,0,1,0),(1,1,0,1),(1,1,1,0),(2,2,0,0)])
const Rl = _nz((3,3,2),[(0,0,0),(0,0,1),(1,1,0),(1,1,1),(2,2,0)])
function NotEqlsLoop_R1(sites)
    N=length(sites); bonds=[Index(2,"Link,l=$i") for i in 1:N-1]; W=Vector{ITensor}(undef,N)
    W[1]=ITensor(Rf, sites[1], sites[1]', bonds[1])
    for j in 2:N-1; W[j]=ITensor(Rb, sites[j], sites[j]', bonds[j-1], bonds[j]); end
    W[N]=ITensor(Rl, sites[N], sites[N]', bonds[N-1]); MPO(W)
end
function build_setup_pxp(N)
    Random.seed!(42); sites=siteinds("S=1",N)
    HT=OpSum(); HT+=1,"Xp",1
    for j in 0:N-2; HT+=1,"Px",j+1,"LP",j+2; HT+=1,"RP",j+1,"Xp",j+2; end
    HT+=1,"Px",N; H=MPO(HT,sites)
    P=NotEqlsLoop_R1(sites); psi0=random_mps(sites)
    psi=replaceprime(contract(P,copy(psi0),:coo,:aliased;denseLinksB=0),1=>0)
    return H, psi, P
end
_dns(T) = ITensors.has_external_storage(T) ?
    (s=ITensors.get_external_storage(T); s isa SparseBackends.WrappedAliasedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...) :
     s isa SparseBackends.WrappedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...) : T) : T

let
    N = 12; b = div(N, 2)
    H, psi, P = build_setup_pxp(N)
    ITensorMPS.orthogonalize!(psi, b)
    cpm = ITensorMPS.CoreProjMPO(H, P; nsite=2)
    ITensorMPS.position!(cpm, psi, b)

    # core0 (dense 2-site core) and the matvec
    core0 = SparseBackends.read_core(ITensors.get_external_storage(psi[b])) *
            SparseBackends.read_core(ITensors.get_external_storage(psi[b+1]))
    cinds = inds(core0)
    form_phi(core) = replaceprime(P[b] * P[b+1] * core, 1 => 0)   # dense φ = P·core
    f(core) = ITensorMPS.product(cpm, form_phi(core))

    # (A) f(core0) == product(cpm, aliased φ)
    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    v_ref = ITensorMPS.product(cpm, phi)
    v_f   = f(core0)
    dA = norm(_dns(v_f) - _dns(v_ref)) / max(norm(_dns(v_ref)), eps())
    @printf("(A) ‖f(core0) − product(cpm,φ)‖/‖·‖ = %.2e   %s\n", dA, dA < 1e-10 ? "PASS ✓" : "FAIL ✗")

    # Build the dense operator matrix W (loop over core basis) for the reference eig.
    D = prod(dim, cinds)
    ci = CartesianIndices(Tuple(dim(i) for i in cinds))
    W = zeros(Float64, D, D)
    for j in 1:D
        ej = ITensor(Float64, cinds...); ej[ci[j]] = 1.0
        W[:, j] = vec(array(f(ej), cinds...))
    end
    asym = norm(W - W') / max(norm(W), eps())
    @printf("(B) operator Hermiticity ‖W−Wᵀ‖/‖W‖ = %.2e   %s\n", asym, asym < 1e-8 ? "PASS ✓" : "FAIL ✗")
    λ_dense = minimum(real, eigen(Symmetric((W+W')/2)).values)

    # (C) KrylovKit ground eig on f
    vals, vecs, info = eigsolve(f, core0, 1, :SR; ishermitian=true, tol=1e-12, krylovdim=30)
    λ_kk = real(vals[1])
    @printf("(C) λ  KrylovKit=% .10f   dense eig=% .10f   |Δ|=%.2e   %s\n",
            λ_kk, λ_dense, abs(λ_kk-λ_dense), abs(λ_kk-λ_dense) < 1e-8 ? "PASS ✓" : "FAIL ✗")
    @printf("    (converged=%s)\n", info.converged >= 1)

    # sanity: physical energy of the optimized state ψ_g = P·cg  (numerator ⟨cg|W|cg⟩)
    cg = vecs[1]; cg = cg / norm(cg)
    num = real(scalar(dag(cg) * f(cg)))          # = ⟨ψ_g|H|ψ_g⟩ (physical numerator)
    @printf("    ⟨cg|W|cg⟩ (phys numerator) = % .8f   λ=% .8f\n", num, λ_kk)
end
nothing
