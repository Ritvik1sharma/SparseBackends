# test_factor_core/roundtrip_slice_map.jl — STEP 1 (foundation, READ-ONLY on production).
#
# Goal: confirm the aliased ψ = P·core stores each template as ONE clean core
# slice (no s-scramble), so `slice_to_template` (core-slice → template id) is a
# well-defined bijection and `core` is recoverable from ψ's templates alone.
#
# This run is EXPLORATORY first: dump the aliased storage layout vs core's dims,
# and derive the slice→template map by matching each template to a core slice
# (up to scalar) — asserting each template matches exactly one slice. Then a
# round-trip sanity check. No production struct touched yet.

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

let
    N = 4
    Random.seed!(42)
    sites = siteinds("S=1", N)
    P = NotEqlsLoop_R1(sites)
    core = random_mps(sites; linkdims=3)                       # dense core
    psi = replaceprime(contract(P, copy(core), :coo, :aliased; denseLinksB=0), 1=>0)
    println("===== STEP 1: template ↔ core-slice layout (PXP, N=$N) =====")
    for b in 1:N
        T = psi[b]
        st = ITensors.has_external_storage(T) ? typeof(ITensors.get_external_storage(T)) : "dense"
        println("\n── site $b ──")
        println("  core[$b] inds: ", [(ITensors.dim(i), string(ITensors.tags(i))) for i in inds(core[b])])
        println("  ψ[$b]   inds: ", [(ITensors.dim(i), string(ITensors.tags(i))) for i in inds(T)])
        println("  ψ[$b]   storage: ", st)
        if ITensors.has_external_storage(T) && ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse
            a = ITensors.get_external_storage(T).aliased
            println("    dims=$(a.dims)  blksize=$(a.blksize)  n_templates=$(a.n_templates)  nkeys=$(length(a.keys))")
            println("    keys       = ", a.keys)
            println("    alias_ids  = ", a.alias_ids)
            println("    scalars    = ", round.(a.scalars; digits=4))
            # per-block norm to see structure
            bs = a.blksize
            tnorm(t) = (off=(t-1)*bs; sqrt(sum(abs2, @view a.templates[off+1:off+bs])))
            println("    template norms = ", [round(tnorm(t); digits=4) for t in 1:a.n_templates])
        end
    end
end
nothing
