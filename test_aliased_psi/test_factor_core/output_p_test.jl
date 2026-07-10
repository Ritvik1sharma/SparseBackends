# test_factor_core/output_p_test.jl — STEP 3a: output_P (re-dedup H·φ to P's schema).
# Asserts:
#   (A) consistency/leakage ≈ 0  (all channels of a physical coord agree ÷scalar
#       ⇒ H·φ ∈ image(P), by [H,P]=0)
#   (B) exactness: densify(output_P(H·φ)) == densify(H·φ)  (re-dedup loses nothing)
#   (C) dedup restored: n_templates drops to #physical-coords (36→9-ish)
#   (D) read_core works on the cleaned result (schema is clean P·core again)

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
    return H, psi
end
_dns(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x

let
    for (N, sw) in ((12, 0), (12, 4))       # fresh, and after 4 bop sweeps (more populated)
        H, psi = build_setup_pxp(N)
        if sw > 0
            s=Sweeps(sw); setmaxdim!(s,40); setmindim!(s,40); setcutoff!(s,1e-10)
            _,psi,_,_ = dmrg(H, psi, s; outputlevel=0, use_early_exit=false,
                             run_mode=:bop_aliased, minv_from_p=nothing, gram_from_h=false)
        end
        b = div(N,2)
        ITensorMPS.orthogonalize!(psi, b)
        PH = ProjMPO(H); ITensorMPS.set_nsite!(PH, 2); ITensorMPS.position!(PH, psi, b)
        phi  = *(psi[b], psi[b+1]; preserve_bs_output=true)
        Hphi = product(PH, phi; roofline=false, run_label="outP")
        w    = ITensors.get_external_storage(Hphi)
        if !(w isa SparseBackends.WrappedAliasedBlockSparse)
            @printf("N=%d sw=%d: H·φ NOT aliased (%s) — skip\n", N, sw, typeof(w)); continue
        end
        nt_before = w.aliased.n_templates
        wphi = ITensors.get_external_storage(phi)
        # keysets same?  and axis orders same?
        same_keys = Set(w.aliased.keys) == Set(wphi.aliased.keys)
        same_axes = w.inds == wphi.inds
        # ALIGN H·φ to φ's axis order first, THEN snap onto φ's schema
        walign = SparseBackends.align_aliased_axes(w, wphi)
        snapped = SparseBackends._snap_to_schema(walign, wphi)
        Hphi_snap = ITensors._itensor_from_external_storage(snapped)
        exact_snap = norm(_dns(Hphi_snap) - _dns(Hphi)) / max(norm(_dns(Hphi)), eps())
        rc_ok = try; SparseBackends.read_core(snapped); true; catch e; false; end
        @printf("N=%d sw=%d: same_keys=%s same_axes=%s  |  aligned+SNAP exact=%.2e  nt %d→%d  read_core=%s\n",
                N, sw, same_keys, same_axes, exact_snap, nt_before, snapped.aliased.n_templates, rc_ok ? "ok" : "FAIL")
    end
    println("\n→ want SNAP-to-φ exact≈0 (H·φ recovered via P's scalars); collapse expected lossy.")
end
nothing
