# coreprojmpo_test.jl — STEP 3b: CoreProjMPO reproduces the validated one-shot matvec.
# Asserts product(cpm, φ) == the hand-rolled Lenv·φ·PH[sites]·Renv:
#   (A) channel-free, (B) ⟨c|v_o'⟩ == dense ⟨φ|P†HP|φ⟩, on the two-site window.

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

let
    N = 12; b = div(N, 2)
    H, psi, P, sites = build_setup_pxp(N)
    ITensorMPS.orthogonalize!(psi, b)

    cpm = ITensorMPS.CoreProjMPO(H, P; nsite=2)
    ITensorMPS.position!(cpm, psi, b)
    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    v   = ITensorMPS.product(cpm, phi)          # channel-free core (noprime'd)

    # (A) channel-free
    pfsm_ids = Set(ITensors.id(l) for l in ITensorMPS.linkinds(P))
    leftover = [i for i in inds(v) if ITensors.id(i) in pfsm_ids]
    @printf("(A) product(cpm,φ) inds (%d): %s\n", length(inds(v)),
            join([@sprintf("[d%d p%d %s]", dim(i), plev(i), string(tags(i))) for i in inds(v)], " "))
    @printf("    channel-free = %s   (leftover P-FSM idx: %d)\n",
            isempty(leftover) ? "YES ✓" : "NO ✗", length(leftover))

    # (B) action vs dense P†HP
    cL = SparseBackends.read_core(ITensors.get_external_storage(psi[b]))
    cR = SparseBackends.read_core(ITensors.get_external_storage(psi[b+1]))
    c  = cL * cR
    num_ali = real(scalar(dag(c) * _dns(v)))     # product already noprime'd → sites/core @0

    psi_d = MPS([_dns(psi[j]) for j in 1:N])
    PHd = ProjMPO(H); ITensorMPS.set_nsite!(PHd, 2); ITensorMPS.position!(PHd, psi_d, b)
    phid = psi_d[b] * psi_d[b+1]
    Hphid = ITensorMPS.product(PHd, phid)
    num_ref = real(scalar(dag(phid) * Hphid))
    rel = abs(num_ali - num_ref) / max(abs(num_ref), eps())
    @printf("(B) ⟨c|product(cpm,φ)⟩=% .8f  vs dense=% .8f  rel=%.2e  %s\n",
            num_ali, num_ref, rel, rel < 1e-8 ? "PASS ✓" : "FAIL ✗")
end
nothing
