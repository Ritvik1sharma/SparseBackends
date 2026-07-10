# pxp_both_states.jl — PXP ground AND excited, aliased, with FIXED bond dim
# (mindim=maxdim=bd) so the per-sweep work is real bond-dim-bd (PXP's natural rank
# is ~5; without fixing bd the timing is overhead-dominated). Direct gram
# (gram_from_h=false, the correct M). Compares RR vs B_op-eigen.
#   dense refs (N=12):  ground -14.771969   excited -13.781210
using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
import KrylovKit; KrylovKit.set_num_threads(1)

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

const BD = 40
function sweep1(H, psi, mode; ortho=nothing, rr_dense_iter::Bool=false)
    s=Sweeps(1); setmaxdim!(s,BD); setmindim!(s,BD); setcutoff!(s,1e-10)  # FIXED bond dim
    if ortho === nothing
        st=@timed (E,psi2,_e,_t)=dmrg(H,psi,s; outputlevel=0, use_early_exit=false, run_mode=mode, minv_from_p=nothing, gram_from_h=false, rr_dense_iter=rr_dense_iter)
    else
        st=@timed (E,psi2,_e,_t)=dmrg(H,ortho,psi,s; outputlevel=0, use_early_exit=false, weight=20.0, run_mode=mode, minv_from_p=nothing, gram_from_h=false, rr_dense_iter=rr_dense_iter)
    end
    return E, psi2, st.time
end
function run_state(label, H, psi0, mode, nsw, ref; ortho=nothing, rr_dense_iter::Bool=false)
    psi=psi0; cum1=0.0; Efin=NaN
    for i in 1:nsw
        E, psi, t = sweep1(H, psi, mode; ortho=ortho, rr_dense_iter=rr_dense_iter); Efin=E
        i>1 && (cum1 += t)
        @printf("  [%s %s sw %d] t=%7.3fs  E=%.10f  Δref=%.2e\n", label, mode, i, t, E, abs(E-ref))
    end
    @printf("  → [%s %s] avg/sw(excl1)=%.3fs  final E=%.8f  ref=%.8f  %s\n\n",
            label, mode, nsw>1 ? cum1/(nsw-1) : NaN, Efin, ref, abs(Efin-ref)<1e-3 ? "MATCH ✓" : "off")
    return psi
end

# (mode, rr_dense_iter) configs. rr_dense_iter only meaningful for :rr — runs the
# local eigensolve fully DENSE (dense φ/grams/H·v) and forces aliasing ONLY at φ
# recovery (snap_dense_to_aliased), to test whether per-iteration aliasing is what
# causes the RR energy plateau. Pass a subset via ARGS, e.g. `julia pxp_both_states.jl rr-dense`.
_all_cfgs = [(:rr, false), (:rr, true), (:bop_aliased, false)]
_sel = isempty(ARGS) ? _all_cfgs :
       filter(c -> (c==(:rr,false)        && "rr"        in ARGS) ||
                   (c==(:rr,true)         && "rr-dense"  in ARGS) ||
                   (c==(:bop_aliased,false) && "bop"      in ARGS), _all_cfgs)
for (mode, rrdi) in _sel
    _tag = mode===:rr ? "rr$(rrdi ? "(dense-iter)" : "(aliased-iter)")" : String(mode)
    println("\n############## PXP $_tag  (mindim=maxdim=$BD → capped at natural rank) ##############")
    H, psi0 = build_setup_pxp(12)   # build ONCE — reuse H+sites for ground & excited
    psi_gs = run_state("GROUND",  H, copy(psi0), mode, 18, -14.771969; rr_dense_iter=rrdi)
    run_state("EXCITED", H, copy(psi0), mode, 18, -13.781210; ortho=[psi_gs], rr_dense_iter=rrdi)
end
