# test_factor_core/twosite_matvec_dump.jl — STEP 3 scoping (READ-ONLY).
#
# Before writing output_P, understand the TWO-SITE block layout at each stage:
#   φ  = ψ[b]·ψ[b+1]        (clean P·core center)
#   Hφ = product(PH, φ)     (de-duped, needs output_P)
# Dump prefix(P)/dense(N2) split, keys (which prefix axis is physical), alias_ids,
# n_templates, dedup — so the physical-grouping for the collapse is grounded in fact.

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

function dump(tag, T)
    println("\n── $tag ──")
    println("  inds: ", [(ITensors.dim(i), string(ITensors.tags(i)), ITensors.plev(i)) for i in inds(T)])
    if ITensors.has_external_storage(T) && ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse
        w = ITensors.get_external_storage(T); a = w.aliased
        Pn = length(a.keys)>0 ? length(a.keys[1]) : (length(a.dims)-length(SparseBackends.dense_inds(w)))
        site_pos = [i for i in 1:Pn if ITensors.hastags(w.inds[i], "Site")]
        @printf("  ALIASED dims=%s blksize=%d n_templates=%d nkeys=%d  P(prefix)=%d  physical-prefix-axes=%s\n",
                a.dims, a.blksize, a.n_templates, length(a.keys), Pn, site_pos)
        println("  keys      = ", a.keys)
        println("  alias_ids = ", Int.(a.alias_ids))
        # group keys by their physical coords (the site prefix axes) → templates seen
        if !isempty(site_pos)
            groups = Dict{Any,Set{Int}}()
            for (i,k) in enumerate(a.keys)
                ph = Tuple(k[p] for p in site_pos)
                push!(get!(groups, ph, Set{Int}()), Int(a.alias_ids[i]))
            end
            println("  physical-coord → {template ids}:")
            for (ph,ts) in sort(collect(groups); by=x->x[1]); println("      ", ph, " → ", sort(collect(ts))); end
        end
    else
        println("  storage: ", ITensors.has_external_storage(T) ? typeof(ITensors.get_external_storage(T)) : "dense")
    end
end

let
    N = 12
    H, psi = build_setup_pxp(N)
    b = div(N,2)
    ITensorMPS.orthogonalize!(psi, b)
    PH = ProjMPO(H); ITensorMPS.set_nsite!(PH, 2); ITensorMPS.position!(PH, psi, b)
    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    Hphi = product(PH, phi; roofline=false, run_label="dump")
    println("===== STEP 3 scoping: two-site φ vs H·φ layout (PXP N=$N, b=$b) =====")
    dump("φ = ψ[b]·ψ[b+1]  (clean P·core center)", phi)
    dump("H·φ = product(PH, φ)  (needs output_P)", Hphi)
    println("\n→ for output_P: collapse H·φ's templates so each physical-coord maps to ONE template")
    println("  (like φ does), asserting all channels of a physical agree after ÷scalar.")
end
nothing
