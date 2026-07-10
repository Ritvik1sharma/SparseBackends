# pxp_iso_check.jl — does PXP aliased ψ actually need M (Path-B), or is it iso?
#   (a) measure how far Lgram is from I at several bonds (iso ⟺ Lgram=I ⟺ M=I)
#   (b) run DMRG with run_mode=:iso (NO M correction) and see if it reaches the
#       dense constrained ground E≈-14.772 (N=12). If :iso converges → PXP does
#       NOT need M; the broken bop path was solving a needless generalized problem.
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf, Random

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

# (a) iso check: Lgram = ψ†ψ over sites 1..b-1 (folded from the boundary).
_pbmul(A,B)= *(A, B; preserve_bs_output=true)
function ali_transfer(psi, from, to)
    step = from <= to ? 1 : -1
    T = _pbmul(psi[from], SparseBackends._dag_link_primed(psi[from])); i=from
    while i != to; i += step; T = _pbmul(T, psi[i]); T = _pbmul(T, SparseBackends._dag_link_primed(psi[i])); end
    return T
end
function iso_dist(lbl, G)
    Gd=_dense(G); unp=filter(I->plev(I)==0,collect(inds(Gd)))
    length(unp)!=2 && (println("  [$lbl] #unp=$(length(unp))"); return)
    d=prod(dim.(unp)); A=reshape(Array(Gd,unp...,prime.(unp)...),d,d); A=(A+A')/2
    ev=sort(real.(eigvals(Hermitian(A)));rev=true); mx=maximum(ev)
    # iso ⟺ all nonzero eigenvalues == 1 (Lgram = I on its support)
    nz=ev[ev.>1e-8*mx]
    @printf("  [%s] dim=%d  eig(max)=%.4g eig(min_nz)=%.4g  #nz=%d/%d  ‖Lgram-I‖/‖I‖=%.3e  (0 ⇒ ISO, M=I)\n",
            lbl, d, mx, minimum(nz), length(nz), d, norm(A - I(d))/sqrt(d))
end

N=12
H, psi = build_setup_pxp(N)
println("=== (a) is PXP aliased ψ iso?  (Lgram vs I) ===")
for b in (4,6,8)
    psic=copy(psi); orthogonalize!(psic,b)
    Lg = ali_transfer(psic, 1, b-1)
    iso_dist("Lgram b=$b (after orthogonalize!)", Lg)
end

function run_iso(H, psi0)
    println("\n=== (b) DMRG with run_mode=:iso (NO M correction) ===")
    psi_iso = copy(psi0)
    for sw in 1:6
        s=Sweeps(1); setmaxdim!(s,40); setmindim!(s,1); setcutoff!(s,1e-10)
        try
            E,psi_iso,_,_=dmrg(H,psi_iso,s; outputlevel=0, use_early_exit=false, run_mode=:iso)
            @printf("  [:iso sweep %d] E=%.10f\n", sw, E)
        catch err
            print("  [:iso sweep $sw] THREW: "); Base.showerror(stdout, err); println(); break
        end
    end
end
run_iso(H, psi)
println("\n  dense constrained ground (reference): E = -14.771969")
