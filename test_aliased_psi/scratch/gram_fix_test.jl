# gram_fix_test.jl — does the DIRECT ψ†ψ gram (gram_from_h=false) fix PXP?
# Same solver for both (run_mode=:bop_aliased, minv_from_p=nothing = eigen), so the
# ONLY variable is the gram source. KL is the regression control (direct == covector
# for KL, so KL must still reach -17.18). PXP is the test: covector gave -8.8; does
# the correct direct M reach the dense ground -14.772?
using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
import KrylovKit; KrylovKit.set_num_threads(1)
include("test_aliased_kl.jl")   # KL build_setup — guarded

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

function run_it(name, H, psi, nsw, ref, gfh)
    println("\n########## $name  gram_from_h=$gfh  (dense ref E=$ref) ##########")
    Efin=NaN
    for i in 1:nsw
        s=Sweeps(1); setmaxdim!(s,40); setmindim!(s,1); setcutoff!(s,1e-10)
        st = @timed (E, psi, _esw, terr) = dmrg(H, psi, s; outputlevel=0, use_early_exit=false,
                                                run_mode=:bop_aliased, minv_from_p=nothing, gram_from_h=gfh)
        Efin = E
        @printf("  [%s sw %d] t=%7.3fs  E=%.10f  Δref=%.2e\n", name, i, st.time, E, abs(E-ref))
    end
    @printf("  → [%s] final E=%.8f   dense ref=%.8f   %s\n", name, Efin, ref,
            abs(Efin-ref) < 1e-3 ? "MATCH ✓" : "MISMATCH ✗")
end

run_it("PXP", build_setup_pxp(12)...,   8, -14.771969, false)
run_it("KL",  build_setup(12, +1, 3)..., 6, -17.182782, false)
