# test_factor_core/read_write_core_test.jl — STEP 1 (functional): read_core/write_core!.
#
# Asserts:
#   (A) read_core(ψ[b]) == core[b] exactly (recover the underlying core MPS).
#   (B) write_core!(ψ[b], c′) then read_core == c′ (round-trip idempotence).
#   (C) after write_core!, the stored ψ changed accordingly (templates mutated,
#       P's keys/scalars untouched).

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

# relabel `t` (on ψ's phys+dense inds) onto core[b]'s indices by (tags,dim) match.
function relabel_to_core(t, coreb)
    ti = collect(inds(t)); ci = collect(inds(coreb)); pairs = Pair{Index,Index}[]
    for a in ti
        j = findfirst(c -> ITensors.dim(c)==ITensors.dim(a) && ITensors.tags(c)==ITensors.tags(a) &&
                            ITensors.plev(c)==ITensors.plev(a), ci)
        @assert j !== nothing "no core index matches $a"
        push!(pairs, a => ci[j]); ci[j] = Index(-1)   # consume
    end
    return replaceinds(t, pairs...)
end

let
    N = 6
    Random.seed!(7)
    sites = siteinds("S=1", N)
    P = NotEqlsLoop_R1(sites)
    core = random_mps(sites; linkdims=4)
    psi = replaceprime(contract(P, copy(core), :coo, :aliased; denseLinksB=0), 1=>0)
    println("===== STEP 1 read_core / write_core! (PXP N=$N) =====")
    maxA = 0.0; maxB = 0.0; okC = true; okD = true
    for b in 1:N
        w = ITensors.get_external_storage(psi[b])
        # (D) eager populate: field empty → populated → matches derived, read still exact
        okD &= isempty(w.aliased.slice_to_template)
        SparseBackends.populate_slice_map!(w)
        okD &= (!isempty(w.aliased.slice_to_template) &&
                w.aliased.slice_to_template == SparseBackends.slice_to_template(w))
        # (A) recover
        rc = SparseBackends.read_core(w)
        eA = norm(relabel_to_core(rc, core[b]) - core[b]) / max(norm(core[b]), eps())
        maxA = max(maxA, eA)
        # (B) write a scaled/perturbed core, read back
        cprime = core[b] * 1.5 + core[b] * 0.0            # simple deterministic perturb
        # build cprime on ψ's indices (same layout as read_core output)
        cprime_psi = relabel_to_core(rc, core[b])         # rc relabeled to core inds
        cprime_psi = cprime_psi * 1.5
        # write it back (write_core! expects core on ψ's phys+dense inds → relabel back)
        # easier: perturb rc directly (already on ψ inds), write, reread
        rc_scaled = rc * 1.5
        keys_before = copy(w.aliased.keys); sc_before = copy(w.aliased.scalars)
        SparseBackends.write_core!(w, rc_scaled)
        rc2 = SparseBackends.read_core(w)
        eB = norm(rc2 - rc_scaled) / max(norm(rc_scaled), eps())
        maxB = max(maxB, eB)
        # (C) keys/scalars untouched
        okC &= (w.aliased.keys == keys_before && w.aliased.scalars == sc_before)
        SparseBackends.write_core!(w, rc)                 # restore
        @printf("  site %d: read err=%.2e  write→read err=%.2e  keys/scalars fixed=%s\n", b, eA, eB, okC)
    end
    println("\n=====================================================")
    @printf("(A) read_core recovers core:      max err %.2e  %s\n", maxA, maxA<1e-12 ? "PASS ✓" : "FAIL ✗")
    @printf("(B) write→read idempotent:        max err %.2e  %s\n", maxB, maxB<1e-12 ? "PASS ✓" : "FAIL ✗")
    @printf("(C) keys/scalars untouched:       %s\n", okC ? "PASS ✓" : "FAIL ✗")
    @printf("(D) populate_slice_map! field:    %s\n", okD ? "PASS ✓" : "FAIL ✗")
end
nothing
