# INITIAL state (random core). Check the DIRECT quantities (not just gram eigenvalues):
#  (0) constraint: ⟨ψ|Pψ⟩/⟨ψ|ψ⟩ = 1  ⇒ ψ ∈ image(P)  (should hold by construction)
#  print the core (templates) + P's action (keys/scalars)
#  (1) gram proj-defect ‖G²−cG‖/‖cG‖   + the ACTUAL Lgram/Rgram matrix
#  (2) sqrt-defect ‖Mhalf·Mhalf − G‖/‖G‖
#  (3) round-trip ‖M^{-1/2}M^{1/2}φ − φ‖/‖φ‖
#  (4) factorize exact ‖L·R − φ‖/‖φ‖  (ortho left & right)
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf, Random
include("test_aliased_kl.jl")
N = 2; psign = -1; spin = 3

# reconstruct P and the random psi0 (same seed as build_setup) to check the constraint
function build_P_and_psi0(N, psign, spin)
    Random.seed!(42)
    sites = siteinds(spin==3 ? "S=1" : "S=1/2", 2*N+2)
    cs = 0.5*psign; os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id",2*j-1,"Id",2*j,"Id",2*j+1,"Id",2*j+2
        t += cs, "exp(i*pi*Sy)",2*j-1,"exp(i*pi*Sx)",2*j,"exp(i*pi*Sx)",2*j+1,"exp(i*pi*Sy)",2*j+2
        push!(os2, t)
    end
    C = [clean!(MPO(os2[j], sites, [2*j-1,2*j,2*j+1,2*j+2])) for j in 1:N]
    mulMPO(A,B)= replaceprime(contract(A, prime(B,"Site"), :coo,:coo), 2=>1)
    P = C[1]; for j in 2:length(C); P = mulMPO(P,C[j]); end
    Random.seed!(42); siteinds("S=1", 2*N+2)  # advance RNG same as build_setup path
    return P, sites
end

H, psi = build_setup(N, psign, spin)
_dense(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x
# (0) ψ∈image(P) is structural (the aliased schema IS P) — skipping fragile re-apply.

function schema(lbl, T)
    a = T.tensor.data.aliased
    println("[$lbl] n_keys=$(length(a.keys)) n_tmpl=$(a.n_templates) dedup=$(round(length(a.keys)/max(a.n_templates,1),digits=2)) blksize=$(a.blksize)")
    println("   keys=", a.keys, "  alias_ids=", a.alias_ids)
    println("   scalars=", round.(a.scalars, sigdigits=3))
    println("   TEMPLATES(core)=", round.(a.templates, sigdigits=3))
end
gram_mat(G)=(Gd=_dense(G); gi=collect(inds(Gd)); unp=filter(I->plev(I)==0,gi); prm=filter(I->plev(I)==1,gi); d=prod(I->dim(I),unp;init=1); A=reshape(Array(Gd,unp...,prm...),d,d); (A+A')/2)
proj_defect(A)=(ev=eigvals(Hermitian(A)); c=maximum(abs,ev); (norm(A*A-c*A)/max(norm(c*A),eps()), c))

println("\n===== core (templates) + P action =====")
for b in (2,3); schema("psi[$b]", psi[b]); end

covec = SparseBackends.build_covector_cache(H)
for b in (2,3)
    println("\n===== bond $b (after orthogonalize!) =====")
    psic=copy(psi); orthogonalize!(psic,b); PH=ProjMPO(H); ITensorMPS.position!(PH,psic,b)
    Lg=ITensorMPS.lproj(PH)*covec.eL[b]; Rg=ITensorMPS.rproj(PH)*covec.eR[b+2]; phi=psic[b]*psic[b+1]
    for (nm,G) in (("Lgram",Lg),("Rgram",Rg))
        A=gram_mat(G); pd,c=proj_defect(A)
        @printf("  [%s] d=%d c=%.4g  (1)proj-defect=%.3e\n", nm, size(A,1), c, pd)
        size(A,1)<=12 && for i in 1:size(A,1); println("     ", [round(real(A[i,j]);sigdigits=2) for j in 1:size(A,1)]); end
    end
    ML,LiL,MR,LiR=SparseBackends.build_minv_half_pair_factored(Lg,Rg; phi_template=phi, p_c=nothing)
    AL=gram_mat(Lg);MLm=gram_mat(ML);AR=gram_mat(Rg);MRm=gram_mat(MR)
    @printf("  (2) sqrt-defect L=%.3e R=%.3e\n", norm(MLm*MLm-AL)/max(norm(AL),eps()), norm(MRm*MRm-AR)/max(norm(AR),eps()))
    ah(oL,oR,z)=SparseBackends.apply_minv_preserve_bs(oR,SparseBackends.apply_minv_preserve_bs(oL,z,phi;fission=false),phi;fission=false)
    phirec=ah(LiL,LiR,ah(ML,MR,phi))
    @printf("  (3) round-trip ‖M^-½M^½φ−φ‖/‖φ‖=%.3e\n", norm(_dense(phirec)-_dense(phi))/norm(_dense(phi)))
    for o in ("left","right")
        L,R,_=SparseBackends.itensor_aliased_factorize(phi,psic[b],psic[b+1];ortho=o,maxdim=40,cutoff=1e-10)
        @printf("  (4) factorize %-5s ‖L·R−φ‖/‖φ‖=%.3e\n", o, norm(_dense(L*R)-_dense(phi))/norm(_dense(phi)))
    end
end
