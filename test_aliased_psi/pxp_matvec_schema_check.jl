# pxp_matvec_schema_check.jl — STEP 1 linchpin (READ-ONLY diagnostic).
#
# Tests the assumption behind the "aliased ψ + template map" method:
#   for PXP, does the raw local matvec H·φ (aliased, NO M frame) stay in P's key
#   schema, with a STABLE alias_id→template map, so that reading H·φ's templates
#   yields H·core directly (dedup=1 ⇒ bijective map)?
#
# READ-ONLY: builds φ = ψ[b]·ψ[b+1] and Hφ = product(PH, φ) as NEW tensors and
# inspects their aliased storage. It never mutates ψ's stored templates, never
# reorders/ snaps anything — so it cannot violate the (not-yet-built) template map.
#
# What it measures, at a mid bond, over several successive matvecs φ→Hφ→H²φ…:
#   • key-set of Hφ vs φ: keys in Hφ but NOT in φ (schema INFLATION) + their norm
#     (zero-norm ⇒ inflation is a representational artifact, snap is exact;
#      full-norm ⇒ genuine out-of-φ-schema weight → map not directly reusable)
#   • keys in φ missing from Hφ
#   • dedup (nb / n_templates): preserved (PXP expects ~1) or changed?
#   • whether the key-set STABILIZES across iterations or keeps growing.

using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
import KrylovKit; KrylovKit.set_num_threads(1)

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

# READ-ONLY inspection of an aliased ITensor's storage.
function ali(T)
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        if w isa SparseBackends.WrappedAliasedBlockSparse
            a = w.aliased; nb = length(a.keys); nt = a.n_templates; bs = a.blksize
            bnorm(i) = (off=(Int(a.alias_ids[i])-1)*bs;
                        abs(a.scalars[i])*sqrt(sum(abs2, @view a.templates[off+1:off+bs])))
            return (; aliased=true, nb, nt, dedup=nb/max(nt,1), bs,
                      keys=a.keys, keyset=Set(a.keys), bnorm,
                      total=sqrt(sum(bnorm(i)^2 for i in 1:nb; init=0.0)))
        end
    end
    return (; aliased=false, nb=0, nt=0, dedup=NaN, keyset=Set(), total=norm(T))
end

function compare(prev, cur; tag="")
    if !cur.aliased
        println("  [$tag] Hφ is DENSE (not aliased) — matvec densified; template map broken."); return
    end
    infl = [k for k in cur.keys if !(k in prev.keyset)]          # in Hφ, not in φ
    miss = [k for k in prev.keys if !(k in cur.keyset)]          # in φ, not in Hφ
    # norm of inflation blocks (are the new keys zero or full?)
    idx_of = Dict(k=>i for (i,k) in enumerate(cur.keys))
    infl_norm = isempty(infl) ? 0.0 : sqrt(sum(cur.bnorm(idx_of[k])^2 for k in infl))
    @printf("  [%s] φ: nb=%d nt=%d dedup=%.2f | Hφ: nb=%d nt=%d dedup=%.2f\n",
            tag, prev.nb, prev.nt, prev.dedup, cur.nb, cur.nt, cur.dedup)
    @printf("       keys in Hφ∉φ (INFLATION): %d  (‖them‖/‖Hφ‖ = %.3e)   keys in φ∉Hφ: %d\n",
            length(infl), infl_norm/max(cur.total,eps()), length(miss))
    @printf("       key-set identical to φ? %s\n", isempty(infl) && isempty(miss) ? "YES ✓ (map directly reusable)" :
            (infl_norm/max(cur.total,eps()) < 1e-10 ? "no, but inflation is ZERO-norm ✓ (snap exact)" :
             "NO — full-norm inflation ✗ (map not directly reusable)"))
end

_stg(T) = ITensors.has_external_storage(T) ?
          (T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse ? "ALIASED" :
           T.tensor.data isa SparseBackends.WrappedBlockSparse ? "blocksparse" : string(typeof(T.tensor.data))) : "dense"

let
    N  = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 12
    bd = length(ARGS)>=2 ? parse(Int,ARGS[2]) : 40
    b  = div(N,2)

    # ── Part A: storage trace through the pipeline (documents the collapse) ──
    println("\n===== PART A: does ψ stay aliased through DMRG? (N=$N bd=$bd, site b=$b) =====")
    H, psi = build_setup_pxp(N)
    println("after build (P·ψ₀):        psi[b] storage = ", _stg(psi[b]))
    let s=Sweeps(1); setmaxdim!(s,bd); setmindim!(s,bd); setcutoff!(s,1e-10)
        _,psi1,_,_ = dmrg(H, copy(psi), s; outputlevel=0, use_early_exit=false,
                          run_mode=:bop_aliased, minv_from_p=nothing, gram_from_h=false)
        println("after 1 bop_aliased sweep: psi[b] storage = ", _stg(psi1[b]),
                "   (collapse ⇒ current Path-B loses aliasing)")
    end

    # ── Part B: raw matvec schema on the FRESH aliased ψ (pre-collapse) ──
    println("\n===== PART B: raw H·φ key-schema on the fresh ALIASED ψ (no sweeps) =====")
    H2, psi2 = build_setup_pxp(N)
    println("psi2[b], psi2[b+1] storage = ", _stg(psi2[b]), " , ", _stg(psi2[b+1]))
    PH = ProjMPO(H2); ITensorMPS.set_nsite!(PH, 2); ITensorMPS.position!(PH, psi2, b)
    phi = *(psi2[b], psi2[b+1]; preserve_bs_output=true)     # two-site block, KEEP aliased (READ)
    p0 = ali(phi)
    @printf("φ = ψ[b]·ψ[b+1]:  storage=%s  nb=%d  n_templates=%d  dedup=%.2f\n",
            _stg(phi), p0.nb, p0.nt, p0.dedup)
    if !p0.aliased
        println("  φ came out non-aliased even with preserve_bs_output — the two-site combine densifies.")
        println("  (That itself is a finding: the P·core two-site block can't be held aliased by the current combine.)")
    end
    cur = phi; prev_ins = p0
    for it in 1:4
        Hphi = product(PH, cur; roofline=false, run_label="schemachk")   # matvec (READ)
        @printf("  matvec %d: H·(prev) storage = %s\n", it, _stg(Hphi))
        ci = ali(Hphi)
        (prev_ins.aliased && ci.aliased) ? compare(prev_ins, ci; tag="matvec $it") : nothing
        cur = Hphi; prev_ins = ci
    end
    println("\nINTERPRETATION:")
    println("  • Part A: 'after 1 sweep = dense' confirms the known collapse — current Path-B does NOT maintain aliasing.")
    println("    Your template-map scheme AVOIDS this: mutating templates (never the densifying aliased Lanczos add).")
    println("  • Part B: if H·φ stays ALIASED with φ's key-set (zero-norm inflation) ⇒ the raw matvec preserves P's schema")
    println("    ⇒ template map is directly reusable. If it densifies/leaks ⇒ the matvec itself needs the schema fix.")
end
nothing
