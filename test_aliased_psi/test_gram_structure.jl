# test_gram_structure.jl
# Verify the gram-structure prediction for TWO constraint types, via the observable
# offblock decomposition of the measured Lgram/Rgram (env-slice = ψ†ψ):
#
#   General:  Lgram[a m; a' m'] = ⟨core^m | T^L_{a,a'} | core^{m'}⟩,  T^L_{a,a'}=(P_L^a)†P_L^{a'}.
#   KL  (P = ∏ ½(I+psign·G_j), G_j off-diagonal commuting unitary involutions):
#        T = G_{D(a,a')}  → CROSS-CHANNEL coupling → OFF-CHAN=√(1-1/nc), M→c·Π at convergence.
#   PXP (P = single diagonal deterministic-FSM constraint MPO):
#        T = δ_{a,a'}·Π_a → BLOCK-DIAGONAL in channel → OFF-CHAN≈0.
#
# PASS: KL shows OFF-CHAN≈√(1-1/nc)>0 (→ #cl=1 cΠ at convergence); PXP shows OFF-CHAN≈0.
#
# STORAGE COROLLARY (analyze_aliased, merged from the former test_compress.jl):
# the same structure has two ORTHOGONAL aliased-storage signatures —
#   sparsity = channel-block occupancy (key density),  dedup = keys/templates.
#   KL  → occupancy 1 (dense keys, cross-channel) + high dedup (compress nt→nk/nc).
#   PXP → occupancy 1/nc (sparse keys, block-diagonal) + dedup measured (was open).
using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf, Random
include("test_aliased_kl.jl")   # KL build_setup(N,psign,spin), run_sweeps

# ── PXP setup (self-contained; diagonal FSM constraint) ──────────────────────
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
_ddims(t)= (a=t.tensor.data.aliased; (length(a.keys), a.n_templates))

# ── ALIASED-STORAGE corollary: sparsity (channel-block occupancy) AND dedup ──
# The structure prediction has TWO orthogonal storage consequences (density of
# keys ≠ template sharing):
#   KL  cross-channel T=G_{D(a,a')}: ALL nc² channel-pairs populated (occupancy=1,
#        DENSE keys) but combined(a,a') depends only on D=a⊕a' → nc distinct
#        templates → compress collapses nt→nk/nc (HIGH dedup).
#   PXP block-diagonal T=δ_{a,a'}Π_a: only the nc DIAGONAL channel-pairs populated
#        (occupancy=1/nc, SPARSE keys); dedup across the surviving diagonal blocks
#        is measured here (open question — was never run before this merge).
# Built as the aliased single-site self-gram A·dag(A') with preserve_bs_output
# (plain `*` densifies) — the same contraction whose AA-kernel dedup we assert.
function analyze_aliased(lbl, A)
    G = *(A, dag(prime(A,"Link")); preserve_bs_output=true)
    if !(ITensors.has_external_storage(G) && G.tensor.data isa SparseBackends.WrappedAliasedBlockSparse)
        println("  [$lbl] self-gram NOT aliased (densified) — nk/nt N/A"); return
    end
    # DEDUP axis only. (Sparsity/occupancy lives in analyze() on the env-slice
    # metric, whose clean 2-index (ch,mu) bond structure this single-site self-gram
    # A·dag(A') does NOT share — it keeps both links open.) nk is the raw key count.
    nk0,nt0 = _ddims(G); Gd0 = copy(_dense(G))
    SparseBackends.compress_aliased_templates!(G)
    _,nt1 = _ddims(G); err = norm(_dense(G)-Gd0)/max(norm(Gd0),eps())
    @printf("  [%s] nk=%d nt=%d dedup=%.2f → compress nt=%d dedup=%.2f  val-err=%.1e\n",
            lbl, nk0,nt0, nk0/max(nt0,1), nt1, nk0/max(nt1,1), err)
end

function analyze(lbl, G)
    Gd=_dense(G); unp=filter(I->plev(I)==0,collect(inds(Gd)))
    if length(unp)!=2; println("  [$lbl] #unp=$(length(unp)) dims=$(dim.(unp))"); return; end
    ch=dim(unp[1])<=dim(unp[2]) ? unp[1] : unp[2]; mu=ch===unp[1] ? unp[2] : unp[1]
    nc,nm=dim(ch),dim(mu); A=Array(Gd,ch,mu,prime(ch),prime(mu)); tot=norm(A)^2
    off=0.0; for i in 1:nc,j in 1:nc; i==j&&continue; off+=norm(@view A[i,:,j,:])^2; end
    # SPARSITY axis: channel-block occupancy = fraction of (i,j) channel-pairs with
    # nonzero weight.  KL cross-channel → ~1 (dense keys); PXP block-diagonal Lg →
    # nc/nc²=1/nc (sparse keys); PXP Rg cross (FSM directionality) → higher.
    occ=0; for i in 1:nc,j in 1:nc; norm(@view A[i,:,j,:])>1e-8*sqrt(max(tot,eps())) && (occ+=1); end
    occ=occ/(nc*nc)
    d=nc*nm; Mf=reshape(A,d,d); Mf=(Mf+Mf')/2; ev=sort(real.(eigvals(Hermitian(Mf)));rev=true); cg=maximum(abs,ev)
    nz=ev[abs.(ev).>1e-8*cg]; ncl=length(unique(round.(nz./cg,digits=3)))
    c=0.0; for i in 1:nc; c=max(c,maximum(abs,diag(@view A[i,:,i,:]))); end
    wd=0.0; for i in 1:nc; B=A[i,:,i,:];B=(B+B')/2; wd=max(wd,norm(B-c*Matrix(I,nm,nm))/(c*sqrt(nm))); end
    @printf("  [%s] nc=%d mu=%d rank=%d cg=%.4g #cl=%d %s OFF-CHAN=%.3e √(1-1/nc)=%.3e occupancy=%.2f within-dev=%.2e\n",
            lbl,nc,nm,length(nz),cg,ncl, ncl==1 ? "cΠ" : "  ", sqrt(max(off,0)/tot), sqrt(1-1/max(nc,1)), occ, wd)
end

function run_model(name, H, psi0, bonds, nsw)
    covec=SparseBackends.build_covector_cache(H); cur=psi0
    println("\n########## $name ##########")
    for sw in 0:nsw
        if sw>0
            s=Sweeps(1); setmaxdim!(s,40); setmindim!(s,1); setcutoff!(s,1e-10)
            E,cur,_,_=dmrg(H,cur,s; outputlevel=0, use_early_exit=false, run_mode=:bop_aliased, minv_from_p=nothing)
            @printf(">>> %s sweep %d E=%.8f\n", name, sw, E)
        else; println(">>> $name INIT"); end
        for b in bonds
            psic=copy(cur); orthogonalize!(psic,b); PH=ProjMPO(H); ITensorMPS.position!(PH,psic,b)
            analyze("$name Lg b=$b", ITensorMPS.lproj(PH)*covec.eL[b])
            analyze("$name Rg b=$b", ITensorMPS.rproj(PH)*covec.eR[b+2])
            analyze_aliased("$name Ali b=$b", psic[b])   # sparsity + dedup corollary
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# APPLY-PRESERVATION: does applying the (aliased) metric M to an aliased φ from a
# real DMRG step KEEP φ aliased (sparse keys + dedup>1), or densify it?
#
# In production the per-bond metric M=Lgram⊗Rgram is the DENSE env-slice
# (lproj·eL), so the eigsolve's M^{±1/2}·φ apply densifies φ (all keys populated,
# dedup→1). The from-P factors are scalar·gram (build_half_pair_single_fromP), so
# M^{±1/2} inherits the gram's storage — IF the gram is built aliased. Here we
# build Lgram/Rgram as the aliased ψ†ψ transfer (`*` with preserve_bs_output;
# = the DMRG gram to ~1e-17) and push φ through the SAME production apply
# (apply_minv_preserve_bs) to see if aliasing survives. The scalar c^{∓...} is
# cosmetic for the aliasing question, so we apply the RAW aliased gram.
_is_ali(t) = ITensors.has_external_storage(t) && t.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
function _alistat(t)
    _is_ali(t) || return (-1, -1, NaN)
    a = t.tensor.data.aliased; (length(a.keys), a.n_templates, length(a.keys)/max(a.n_templates,1))
end
_pbmul(A, B) = *(A, B; preserve_bs_output=true)
# ψ†ψ transfer folded from the BOUNDARY site `from` inward to `to`. Folding from
# the boundary is essential: each step then collapses back to a 2-link gram, so
# the aliased template count stays bounded. (Folding outward instead keeps the
# outer link as a spectator whose mult multiplies the template count every step →
# it blows past the UInt8 alias-id capacity by ~3 sites. That is a construction
# artifact, not a kernel limit.)  Left gram: ali_transfer(psi,1,b-1); right gram:
# ali_transfer(psi,N,b+1) — both leave the open link on φ's matching bond.
function ali_transfer(psi, from, to)
    step = from <= to ? 1 : -1
    T = _pbmul(psi[from], SparseBackends._dag_link_primed(psi[from]))
    i = from
    while i != to
        i += step
        T = _pbmul(T, psi[i])
        T = _pbmul(T, SparseBackends._dag_link_primed(psi[i]))
    end
    return T
end
function check_apply(name, psi, b)
    N = length(psi); orthogonalize!(psi, b); phi = psi[b]
    if !_is_ali(phi); println("  [$name b=$b] φ NOT aliased — skip"); return; end
    Lg = b > 1 ? ali_transfer(psi, 1, b-1) : ITensors.ITensor(1.0)
    Rg = b < N ? ali_transfer(psi, N, b+1) : ITensors.ITensor(1.0)
    @printf("  gram aliased?  Lg=%s Rg=%s\n", _is_ali(Lg), _is_ali(Rg))
    nk,nt,dd = _alistat(phi)
    # production apply: M_L·φ then M_R·(…), through the SAME path DMRG uses.
    z = SparseBackends.apply_minv_preserve_bs(Lg, phi, phi)
    nkL,ntL,ddL = _alistat(z)
    z = SparseBackends.apply_minv_preserve_bs(Rg, z, phi)
    nkz,ntz,ddz = _alistat(z)
    # correctness: aliased result must equal the dense-gram·dense-φ apply.
    zref = ITensors.replaceprime(_dense(Lg) * _dense(phi), 1=>0; tags="Link")
    zref = ITensors.replaceprime(_dense(Rg) * zref,        1=>0; tags="Link")
    err = norm(_dense(z) - zref) / max(norm(zref), eps())
    # Is the collapsed dedup a KERNEL ARTIFACT (value-equal templates the input-pair
    # kernel didn't share → compress recovers it) or GENUINE (channels truly mixed →
    # compress can't help)?
    SparseBackends.compress_aliased_templates!(z)
    _,ntc,ddc = _alistat(z)
    # Is each gram a clean scaled projector c·Π at this bond? (#clusters==1 ⇒ clean.)
    _ncl(G) = (Gd=_dense(G); u=filter(I->plev(I)==0,collect(inds(Gd)));
               length(u)!=2 ? -1 : (d=prod(dim.(u)); M=reshape(Array(Gd,u...,prime.(u)...),d,d);
               ev=abs.(eigvals(Hermitian((M+M')/2))); mx=maximum(ev);
               length(unique(round.(ev[ev.>1e-8*mx]./mx,digits=3)))))
    @printf("  [%s b=%d] gram #clusters Lg=%d Rg=%d (1 ⇒ clean c·Π)\n", name,b, _ncl(Lg),_ncl(Rg))
    @printf("  [%s b=%d] φ: nk=%d nt=%d dedup=%.2f → M_L·φ: nk=%d dedup=%.2f → M_R·: nk=%d nt=%d dedup=%.2f → +compress nt=%d dedup=%.2f | aliased=%s val-err=%.1e\n",
            name, b, nk,nt,dd, nkL,ddL, nkz,ntz,ddz, ntc,ddc, _is_ali(z), err)
    verdict = _is_ali(z) && ddc > 1 ? "PASS (aliasing preserved: sparse + dedup>1 after compress)" :
              _is_ali(z)             ? "PARTIAL (aliased + sparse but dedup≤1 even after compress → channels genuinely mixed)" :
                                       "FAIL (densified)"
    println("     → $verdict")
end
function run_apply_check(name, H, psi0, b, nsw)
    s=Sweeps(nsw); setmaxdim!(s,40); setmindim!(s,1); setcutoff!(s,1e-10)
    E,psi,_,_=dmrg(H,psi0,s; outputlevel=0, use_early_exit=false, run_mode=:bop_aliased, minv_from_p=nothing)
    println("\n########## $name — apply-preservation after $nsw sweep(s) (E=$(round(E,digits=6))) ##########")
    check_apply(name, psi, b)
end

Hkl, psikl = build_setup(6, -1, 3)          # KL: 14 sites
run_model("KL", Hkl, psikl, (6,7), 3)
Hpxp, psipxp = build_setup_pxp(12)          # PXP: 12 sites
run_model("PXP", Hpxp, psipxp, (5,6), 3)

# APPLY-PRESERVATION check. Contrast a NON-converged φ (spread gram → apply mixes
# channels → dedup collapses) against a CONVERGED φ (gram → clean c·Π → for
# schema-following φ∈range(Π), Lgram·φ=c·φ is a pure scalar → dedup preserved).
run_apply_check("KL",  build_setup(6, -1, 3)...,  7, 1)
run_apply_check("KL",  build_setup(6, -1, 3)...,  7, 8)
run_apply_check("PXP", build_setup_pxp(12)...,    6, 1)
run_apply_check("PXP", build_setup_pxp(12)...,    6, 8)
