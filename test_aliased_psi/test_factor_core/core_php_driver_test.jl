# core_php_driver_test.jl — validate the PROMOTED ITensorMPS.dmrg_core_php on PXP N=12.
# Confirms the production driver (factor_core_dmrg.jl) reproduces E ≈ −14.77 and that
# ψ stays aliased + deduped.

using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra

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

let
    N = 12
    # (a) natural maxdim=40 (PXP constrained ground is low-rank → converges below 40)
    H, psi, P = build_setup_pxp(N)
    E1, psif = ITensorMPS.dmrg_core_php(H, P, psi; nsweeps=6, maxdim=40)
    @printf("\n[natural maxdim40] final E = %.8f  %s\n", E1,
            abs(E1 - (-14.77196927)) < 1e-5 ? "PASS ✓ (bit-identical to pre-kernel-change)" : "CHECK")
    nal = count(j -> ITensors.has_external_storage(psif[j]) &&
                     ITensors.get_external_storage(psif[j]) isa SparseBackends.WrappedAliasedBlockSparse, 1:N)
    b = div(N,2); w = ITensors.get_external_storage(psif[b])
    @printf("aliased sites: %d/%d   ψ[%d] dedup = %.2fx\n", nal, N, b,
            length(w.aliased.keys)/max(w.aliased.n_templates,1))

    # (b) FORCED bd=40 (mindim=maxdim=40) → true bd=40 per-sweep time
    println("\n--- forced bd=40 (mindim=maxdim=40) ---")
    H2, psi2, P2 = build_setup_pxp(N)
    E2, _ = ITensorMPS.dmrg_core_php(H2, P2, psi2; nsweeps=6, mindim=40, maxdim=40)
    @printf("[forced bd40] final E = %.8f  %s\n", E2,
            abs(E2 - (-14.77196927)) < 1e-5 ? "PASS ✓" : "CHECK")
end
nothing
