# test_factor_core/core_canonical_factorize_test.jl — STEP 2.
#
# The aliased factorize with core_canonical=true must:
#   (A) reconstruct φ exactly:  L·R == φ   (it's still an exact SVD, just unwhitened)
#   (B) produce a CORE-canonical L: read_core(L) is left-isometric ⇒ ⟨core|core⟩ = I
#       (this is the metric-I gauge — no M). The whitened (default) L is NOT.

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
_dns(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x

# left-isometry defect of the CORE factor: contract read_core(L) with its dagger
# over (physical + left links), leaving the core mult bond shared with read_core(R);
# result should be I. (The shared core link — not commonind(L,R) which also has the
# FSM channel link that read_core excludes.)
function iso_defect(L, R)
    CL = SparseBackends.read_core(ITensors.get_external_storage(L))
    CR = SparseBackends.read_core(ITensors.get_external_storage(R))
    shared = commoninds(CL, CR)
    isempty(shared) && return NaN
    G = CL * dag(prime(CL, shared))
    d = prod(ITensors.dim.(shared))
    Gd = reshape(Array(G, shared..., prime.(shared)...), d, d)
    return norm(Gd - Matrix{Float64}(I, d, d)) / max(sqrt(d), 1)
end

let
    N = 6
    Random.seed!(3)
    sites = siteinds("S=1", N)
    P = NotEqlsLoop_R1(sites)
    core = random_mps(sites; linkdims=4)
    psi = replaceprime(contract(P, copy(core), :coo, :aliased; denseLinksB=0), 1=>0)
    b = 3
    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    println("===== STEP 2: core_canonical factorize (PXP N=$N, bond b=$b) =====")
    for cc in (false, true)
        L, R, _ = SparseBackends.itensor_aliased_factorize(phi, psi[b], psi[b+1];
                      ortho="left", maxdim=100, mindim=1, cutoff=1e-14, core_canonical=cc)
        recon = norm(_dns(*(L, R; preserve_bs_output=true)) - _dns(phi)) / max(norm(_dns(phi)), eps())
        idef = iso_defect(L, R)
        @printf("  core_canonical=%-5s : reconstruct ‖L·R−φ‖=%.2e   left-iso defect ‖CᵀC−I‖=%.2e  %s\n",
                cc, recon, idef, (cc && recon<1e-10 && idef<1e-10) ? "CORE-CANONICAL ✓" :
                                 (!cc ? "(whitened baseline)" : ""))
    end
    println("\n→ core_canonical=true should give reconstruct≈0 AND iso≈0 (metric I);")
    println("  core_canonical=false (whitened) reconstructs but is NOT core-iso.")
end
nothing
