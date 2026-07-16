# ph_matvec_oneshot_test.jl — STEP 3a+3c (single-site, READ-ONLY validation).
#
# Validates the factor-core matvec operator O = P·H (P on H's OUTPUT indices only)
# on ONE bond, single-site, before any dmrg.jl wiring:
#   (A) CHANNEL-FREE: v_o' = Lenv·φ·O[b]·Renv carries NO P-FSM (channel) index —
#       the bra-side P closes every channel into the envs.
#   (B) ACTION CORRECT: ⟨c|v_o'⟩ (core-space, c=read_core(ψ[b])) equals the dense
#       physical ⟨φ|H_eff|φ⟩ = ⟨c|P†HP|c⟩  (numerator, no c-scale).
#   (C) SCALE: ⟨φ|φ⟩/⟨c|c⟩ = c = 2^env  (checks M = c·Π on image(P)).
#
# build_ph_output is prototyped inline here for fast prime-iteration; promote to
# SparseBackends/src/aliased/factor_core.jl once green.

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
    return H, psi, P, sites
end
_dns(T) = ITensors.has_external_storage(T) ?
    (s=ITensors.get_external_storage(T); s isa SparseBackends.WrappedAliasedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...) :
     s isa SparseBackends.WrappedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...) : T) : T

# O = P·H (P on H's OUTPUT indices only) — now promoted to
# ITensorMPS/src/abstractprojmpo/core_projmpo.jl; use it from there.
const build_ph_output = ITensorMPS.build_ph_output

let
    N = 12; b = div(N, 2)
    H, psi, P, sites = build_setup_pxp(N)
    ITensorMPS.orthogonalize!(psi, b)

    # ---- aliased single-site envs from BARE H ----
    PHa = ProjMPO(H); ITensorMPS.set_nsite!(PHa, 1); ITensorMPS.position!(PHa, psi, b)
    Lenv = ITensorMPS.lproj(PHa); Renv = ITensorMPS.rproj(PHa)
    phi  = psi[b]

    O  = build_ph_output(P, H)
    Ob = O[b]
    println("── O[b] indices (want: site ket@0/bra@1, H-MPO-bonds@0, P-FSM@1) ──")
    for i in inds(Ob)
        @printf("   dim=%-3d plev=%d tags=%-24s id=%s\n",
                dim(i), plev(i), string(tags(i)), string(id(i)%100000))
    end

    # ---- matvec  v_o' = Lenv · φ · O[b] · Renv ----
    v = phi
    Lenv !== nothing && (v = Lenv * v)
    v = v * Ob
    Renv !== nothing && (v = Renv * v)

    # (A) channel-free?
    pfsm_ids = Set(ITensors.id(l) for l in ITensorMPS.linkinds(P))
    leftover = [i for i in inds(v) if ITensors.id(i) in pfsm_ids]
    @printf("\n(A) v_o' indices (%d): %s\n", length(inds(v)),
            join([@sprintf("[d%d p%d %s]", dim(i), plev(i), string(tags(i))) for i in inds(v)], " "))
    @printf("    channel-free = %s   (leftover P-FSM idx: %d)\n",
            isempty(leftover) ? "YES ✓" : "NO ✗", length(leftover))

    # (B) action: ⟨c|v_o'⟩ vs dense physical ⟨φ|H_eff|φ⟩
    c   = SparseBackends.read_core(ITensors.get_external_storage(phi))   # core on (s_b, core-links)@0
    v0  = replaceprime(_dns(v), 1 => 0)                                   # bra sites/core-links -> 0
    num_ali = real(scalar(dag(c) * v0))

    psi_d = MPS([_dns(psi[j]) for j in 1:N])
    PHd = ProjMPO(H); ITensorMPS.set_nsite!(PHd, 1); ITensorMPS.position!(PHd, psi_d, b)
    phid = psi_d[b]
    Hphid = ITensorMPS.product(PHd, phid)
    num_ref  = real(scalar(dag(phid) * Hphid))
    norm_ref = real(scalar(dag(phid) * phid))
    cc       = real(scalar(dag(c) * c))

    @printf("\n(B) ⟨c|v_o'⟩ (aliased O-path) = % .8f\n", num_ali)
    @printf("    ⟨φ|H_eff|φ⟩ (dense ref)    = % .8f\n", num_ref)
    rel = abs(num_ali - num_ref) / max(abs(num_ref), eps())
    @printf("    rel err = %.2e   %s\n", rel, rel < 1e-8 ? "PASS ✓" : "FAIL ✗")

    @printf("\n(C) ⟨φ|φ⟩/⟨c|c⟩ = %.6f  (expect power of 2 = c=2^env);  log2 = %.4f\n",
            norm_ref / cc, log2(norm_ref / cc))
end

# ============================ TWO-SITE ============================
let
    println("\n================= TWO-SITE =================")
    N = 12; b = div(N, 2)
    H, psi, P, sites = build_setup_pxp(N)
    ITensorMPS.orthogonalize!(psi, b)

    PHa = ProjMPO(H); ITensorMPS.set_nsite!(PHa, 2); ITensorMPS.position!(PHa, psi, b)
    Lenv = ITensorMPS.lproj(PHa); Renv = ITensorMPS.rproj(PHa)
    phi  = *(psi[b], psi[b+1]; preserve_bs_output=true)

    O = build_ph_output(P, H)
    v = phi
    Lenv !== nothing && (v = Lenv * v)
    v = v * O[b]
    v = v * O[b+1]
    Renv !== nothing && (v = Renv * v)

    pfsm_ids = Set(ITensors.id(l) for l in ITensorMPS.linkinds(P))
    leftover = [i for i in inds(v) if ITensors.id(i) in pfsm_ids]
    @printf("(A) v_o' indices (%d): %s\n", length(inds(v)),
            join([@sprintf("[d%d p%d %s]", dim(i), plev(i), string(tags(i))) for i in inds(v)], " "))
    @printf("    channel-free = %s   (leftover P-FSM idx: %d)\n",
            isempty(leftover) ? "YES ✓" : "NO ✗", length(leftover))

    # 2-site core c = read_core(ψ[b]) · read_core(ψ[b+1])  (contract the core-center link)
    cL = SparseBackends.read_core(ITensors.get_external_storage(psi[b]))
    cR = SparseBackends.read_core(ITensors.get_external_storage(psi[b+1]))
    c  = cL * cR
    v0 = replaceprime(_dns(v), 1 => 0)
    num_ali = real(scalar(dag(c) * v0))

    psi_d = MPS([_dns(psi[j]) for j in 1:N])
    PHd = ProjMPO(H); ITensorMPS.set_nsite!(PHd, 2); ITensorMPS.position!(PHd, psi_d, b)
    phid = psi_d[b] * psi_d[b+1]
    Hphid = ITensorMPS.product(PHd, phid)
    num_ref  = real(scalar(dag(phid) * Hphid))
    norm_ref = real(scalar(dag(phid) * phid))
    cc       = real(scalar(dag(c) * c))

    @printf("(B) ⟨c|v_o'⟩=% .8f  vs dense ⟨φ|H_eff|φ⟩=% .8f  rel=%.2e  %s\n",
            num_ali, num_ref, abs(num_ali-num_ref)/max(abs(num_ref),eps()),
            abs(num_ali-num_ref)/max(abs(num_ref),eps()) < 1e-8 ? "PASS ✓" : "FAIL ✗")
    @printf("(C) ⟨φ|φ⟩/⟨c|c⟩ = %.6f   log2 = %.4f\n", norm_ref/cc, log2(norm_ref/cc))
end
nothing
