# env_channel_audit.jl — AUDIT (read-only) for the factor-core matvec design.
#
# Two questions the scheme hinges on:
#  (1) SEPARABILITY: in the envs that feed the matvec (built from aliased ψ +
#      BARE H via position!), is the P-FSM "channel" a SEPARATE contractible
#      index from the core-link, or is it fused/merged with it?
#  (2) IDENTITY: is the env's channel Index the SAME Index object as P[b]'s outer
#      FSM bond (→ P·H contracts it automatically), or only dim-equal (needs
#      replaceind)?  The channel index traces P → contract → ψ → position! → env.
#
# Prints the raw index anatomy of P[b], ψ[b], φ, Lenv, Renv and does === identity
# checks between env link Indices and P[b]'s FSM Link Indices.

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

showinds(tag, T) = begin
    println("  $tag:")
    for i in inds(T)
        @printf("     dim=%-4d plev=%d  tags=%-28s  id=%s\n",
                ITensors.dim(i), ITensors.plev(i), string(ITensors.tags(i)), string(ITensors.id(i)%100000))
    end
end

let
    N = 12
    H, psi, P, sites = build_setup_pxp(N)
    b = div(N,2)
    println("===== env-channel audit  PXP N=$N  bond b=$b =====\n")

    # P[b] link (FSM) indices
    Plinks = [i for i in inds(P[b]) if ITensors.hastags(i, "Link")]
    println("P[b] FSM structure:"); showinds("P[$b]", P[b])

    # aliased ψ[b] — does its link retain P's FSM Index id?
    showinds("psi[$b] (aliased P·core)", psi[b])

    ITensorMPS.orthogonalize!(psi, b)
    println("\n(after orthogonalize! to b)"); showinds("psi[$b]", psi[b])

    PH = ProjMPO(H); ITensorMPS.set_nsite!(PH, 2); ITensorMPS.position!(PH, psi, b)
    Lenv = ITensorMPS.lproj(PH); Renv = ITensorMPS.rproj(PH)
    println("\n--- ENVIRONMENTS (bare H, aliased ψ) ---")
    Lenv === nothing ? println("  Lenv: (none — b at left edge)") : showinds("Lenv", Lenv)
    Renv === nothing ? println("  Renv: (none)") : showinds("Renv", Renv)

    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    println("\n--- φ = ψ[b]·ψ[b+1] ---"); showinds("phi", phi)

    # ── AUDIT 2: identity check — does any env Index === a P[b]/P[b+1] FSM link? ──
    println("\n--- AUDIT 2: Index-identity  env vs P FSM links ---")
    Pall = ITensors.Index[]
    for s in (b-1, b, b+1, b+2)
        1 <= s <= N || continue
        for i in inds(P[s]); ITensors.hastags(i,"Link") && push!(Pall, i); end
    end
    envinds = ITensors.Index[]
    Lenv !== nothing && append!(envinds, collect(inds(Lenv)))
    Renv !== nothing && append!(envinds, collect(inds(Renv)))
    for ei in envinds
        matches = [pi for pi in Pall if ei === pi]
        !isempty(matches) && @printf("  env index id=%s (dim=%d, %s) === P FSM link  ✓\n",
            string(ITensors.id(ei)%100000), ITensors.dim(ei), string(ITensors.tags(ei)))
    end
    anymatch = any(ei -> any(pi -> ei === pi, Pall), envinds)
    println(anymatch ? "  → at least one env↔P identity match (contraction may be automatic)"
                     : "  → NO env index is identical to any P FSM link (would need replaceind wiring)")
end
nothing
