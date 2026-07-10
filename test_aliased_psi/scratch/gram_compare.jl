# gram_compare.jl — is the covector env-slice gram (lproj·eL) == the direct ψ†ψ
# transfer?  KL is the control (must match ~1e-17). If PXP does NOT match, the
# identity-covector selector is producing a WRONG M for PXP's off-diagonal-MPO H
# — the root cause of the -8.8/-8 non-convergence (both B_op and RR inherit it).
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf, Random
include("test_aliased_kl.jl")   # KL build_setup(N,psign,spin) — guarded

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]
_nz(dims, coords) = (A=zeros(Float64,dims...); for c in coords; A[(c .+ 1)...]=1.0; end; A)
function NotEqlsLoop_R1(sites)
    N=length(sites)
    Rf=_nz((3,3,2),[(0,0,0),(1,1,1),(2,2,0)])
    Rb=_nz((3,3,2,2),[(0,0,0,0),(0,0,1,0),(1,1,0,1),(1,1,1,0),(2,2,0,0)])
    Rl=_nz((3,3,2),[(0,0,0),(0,0,1),(1,1,0),(1,1,1),(2,2,0)])
    bonds=[Index(2,"Link,l=$i") for i in 1:N-1]; W=Vector{ITensor}(undef,N)
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
    return H, psi
end
_dense(x)= ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x
_pbmul(A,B)= *(A, B; preserve_bs_output=true)
function ali_transfer(psi, from, to)  # direct ψ†ψ folded from boundary
    step = from <= to ? 1 : -1
    T = _pbmul(psi[from], SparseBackends._dag_link_primed(psi[from])); i=from
    while i != to; i += step; T = _pbmul(T, psi[i]); T = _pbmul(T, SparseBackends._dag_link_primed(psi[i])); end
    return T
end
function compare(name, H, psi, bonds)
    covec = SparseBackends.build_covector_cache(H)
    println("\n########## $name ##########")
    for b in bonds
        psic=copy(psi); orthogonalize!(psic,b)
        PH=ProjMPO(H); ITensorMPS.position!(PH,psic,b)
        lp = ITensorMPS.lproj(PH)
        Gcov = lp isa ITensorMPS.OneITensor ? nothing : _dense(lp * covec.eL[b])
        Gdir = _dense(ali_transfer(psic, 1, b-1))
        if Gcov === nothing; println("  [b=$b] edge (OneITensor) — skip"); continue; end
        # align: ITensor subtraction matches by index id
        rel = try
            norm(Gcov - Gdir) / max(norm(Gdir), eps())
        catch e
            # index mismatch → report the index sets instead
            println("  [b=$b] index mismatch: cov=$(inds(Gcov))  dir=$(inds(Gdir))"); NaN
        end
        @printf("  [b=%d] ‖cov-dir‖/‖dir‖ = %.3e   ‖cov‖=%.5g  ‖dir‖=%.5g   %s\n",
                b, rel, norm(Gcov), norm(Gdir), rel < 1e-8 ? "MATCH ✓" : "MISMATCH ✗ (covector M is WRONG)")
    end
end

compare("KL (control — must match)", build_setup(6, -1, 3)..., (5,6,7))
compare("PXP (suspect)",            build_setup_pxp(12)...,   (4,6,8))
