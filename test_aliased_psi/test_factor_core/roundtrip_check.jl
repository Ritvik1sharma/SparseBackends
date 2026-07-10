# test_factor_core/roundtrip_check.jl — STEP 1 assertion (READ-ONLY on production).
#
# Proves, for ψ = P·core (PXP, diagonal P):
#   (1) NO SCRAMBLE: every ψ-key sharing a physical coord shares one template
#       ⇒ slice_to_template[s] is well-defined and derivable from ψ alone.
#   (2) EXACT: template[slice_to_template[s]] == core[:,s,:] (each template is a
#       clean core slice) ⇒ read_core recovers core exactly, no s-mixing.
# If both hold at every site, the `slice_to_template` field + read_core are sound.

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

# derive slice_to_template from ψ[b] alone; assert no scramble.
function derive_map(T)
    a = ITensors.get_external_storage(T).aliased
    sinds = ITensors.get_external_storage(T).inds            # stored index order
    Pn = length(a.keys[1]); N2 = length(a.dims) - Pn
    phys_pos = findfirst(i -> ITensors.hastags(i, "Site"), collect(sinds)[1:Pn])
    @assert phys_pos !== nothing "no Site axis in prefix"
    phys_dim = a.dims[phys_pos]
    s2t = fill(-1, phys_dim)                                  # slice_to_template
    scramble = false
    for (i, k) in enumerate(a.keys)
        s = k[phys_pos]; tid = Int(a.alias_ids[i])
        if s2t[s] == -1
            s2t[s] = tid
        elseif s2t[s] != tid
            scramble = true                                  # same physical → two templates!
        end
    end
    return (; a, sinds, Pn, N2, phys_pos, phys_dim, s2t, scramble)
end

let
    N = 6
    Random.seed!(42)
    sites = siteinds("S=1", N)
    P = NotEqlsLoop_R1(sites)
    core = random_mps(sites; linkdims=4)
    psi = replaceprime(contract(P, copy(core), :coo, :aliased; denseLinksB=0), 1=>0)
    println("===== STEP 1 round-trip: templates ↔ core slices (PXP, N=$N) =====")
    worst_err = 0.0; any_scramble = false
    for b in 1:N
        T = psi[b]
        m = derive_map(T)
        a = m.a
        bs = a.blksize
        dense_inds = collect(m.sinds)[m.Pn+1:end]            # ψ dense tail = core's links
        phys_ind = collect(m.sinds)[m.phys_pos]
        # match ψ dense inds to core[b]'s link inds (by tags+dim)
        core_inds = collect(inds(core[b]))
        matched = ITensors.Index[]
        for di in dense_inds
            j = findfirst(ci -> ITensors.dim(ci)==ITensors.dim(di) && ITensors.tags(ci)==ITensors.tags(di), core_inds)
            @assert j !== nothing "cannot match ψ dense ind $di to a core link"
            push!(matched, core_inds[j])
        end
        core_arr = Array(core[b], phys_ind, matched...)      # [physical, link…]
        site_err = 0.0
        for s in 1:m.phys_dim
            m.s2t[s] == -1 && continue                       # physical s not present (P forbids)
            tid = m.s2t[s]
            tblk = reshape(a.templates[(tid-1)*bs+1 : tid*bs], Tuple(a.dims[m.Pn+1:end]))
            cslice = selectdim(core_arr, 1, s)               # core[:, s, :…] over the link axes
            e = norm(vec(tblk) .- vec(cslice)) / max(norm(vec(cslice)), eps())
            site_err = max(site_err, e)
        end
        worst_err = max(worst_err, site_err); any_scramble |= m.scramble
        @printf("  site %d: scramble=%s  slice_to_template=%s  max |template−core_slice|/|core_slice| = %.2e\n",
                b, m.scramble, m.s2t, site_err)
    end
    println("\n=====================================================")
    @printf("NO SCRAMBLE: %s   |   round-trip exact (max rel err %.2e): %s\n",
            !any_scramble, worst_err, worst_err < 1e-12 ? "YES ✓" : "NO ✗")
    println("→ if both pass: slice_to_template is well-defined + read_core recovers core exactly (no s-mixing).")
end
nothing
