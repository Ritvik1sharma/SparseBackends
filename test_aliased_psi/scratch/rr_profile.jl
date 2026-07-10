# rr_profile.jl — where does RR's per-sweep time go? (gram build vs eigensolve vs
# factorize).  Suspect: the FRESH O(N²) direct ψ†ψ gram rebuild (gram_from_h=false,
# not yet incrementalized) dominates, especially for tiny PXP.
using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
using TimerOutputs: reset_timer!, print_timer
import KrylovKit; KrylovKit.set_num_threads(1)
include("test_aliased_kl.jl")

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

function profile_rr(name, H, psi, nsw)
    println("\n########## $name RR profile (gram_from_h=false, fresh transfer) ##########")
    for i in 1:nsw
        s=Sweeps(1); setmaxdim!(s,40); setmindim!(s,1); setcutoff!(s,1e-10)
        st=@timed (E,psi,_e,_t)=dmrg(H,psi,s; outputlevel=0, use_early_exit=false,
                                     run_mode=:rr, minv_from_p=nothing, gram_from_h=false)
        @printf("  [%s sw %d] t=%.3fs E=%.8f\n", name, i, st.time, E)
        if i==1; reset_timer!(ITensorMPS.PROJMPO_TIMER); reset_timer!(SparseBackends.TIMER); end
    end
    println("\n===== $name PROJMPO_TIMER (steady, sweeps 2..$nsw) ====="); print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n===== $name SparseBackends.TIMER (steady) ====="); print_timer(SparseBackends.TIMER)
end

profile_rr("PXP", build_setup_pxp(12)..., 4)
profile_rr("KL",  build_setup(12,+1,3)..., 4)
