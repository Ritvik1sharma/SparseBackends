# core_writeback_test.jl — STEP 4b: write the ground core back into aliased ψ.
# eigsolve → cg → SVD(cg) → build ψ[b],ψ[b+1] = P·core via contract_aliased_itensor.
# Asserts: (A) ψ[b]ψ[b+1] reconstructs P·cg, (B) ψ[b],ψ[b+1] are clean aliased P·core
# (read_core works). Prototypes core_writeback! before promoting to factor_core_dmrg.jl.

using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra
using KrylovKit: eigsolve

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
    return H, psi, P
end
_dns(T) = ITensors.has_external_storage(T) ?
    (s=ITensors.get_external_storage(T); s isa SparseBackends.WrappedAliasedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...) :
     s isa SparseBackends.WrappedBlockSparse ?
        ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...) : T) : T
isaliased(T) = ITensors.has_external_storage(T) &&
    ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse

let
    N = 12; b = div(N, 2)
    H, psi, P = build_setup_pxp(N)
    ITensorMPS.orthogonalize!(psi, b)
    cpm = ITensorMPS.CoreProjMPO(H, P; nsite=2)
    ITensorMPS.position!(cpm, psi, b)

    cL0 = SparseBackends.read_core(ITensors.get_external_storage(psi[b]))
    cR0 = SparseBackends.read_core(ITensors.get_external_storage(psi[b+1]))
    core0 = cL0 * cR0
    form_phi(core) = replaceprime(P[b] * P[b+1] * core, 1 => 0)
    f(core) = ITensorMPS.product(cpm, form_phi(core))
    vals, vecs = eigsolve(f, core0, 1, :SR; ishermitian=true, tol=1e-12, krylovdim=30)
    cg = vecs[1]
    @printf("eigsolve λ=% .10f\n", real(vals[1]))

    # ── rebuild_core: resize-capable write_core! — drop a dense single-site core into
    #    ψ[j]'s aliased template slots, keeping P structure (keys/alias_ids/scalars/
    #    channels) fixed; the core-link Index/dim/blksize come from `core`. ──
    function rebuild_core(t::ITensor, core::ITensor)
        w = ITensors.get_external_storage(t); a = w.aliased
        Pn = SparseBackends._abs_head_len(w)
        prefix_inds = w.inds[1:Pn]
        phys_pos = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), 1:Pn)
        phys_ind = w.inds[phys_pos]
        new_dense_inds = [i for i in inds(core) if !ITensors.hastags(i, "Site")]
        new_dense_dims = Tuple(ITensors.dim(i) for i in new_dense_inds)
        core_arr = Array(core, phys_ind, new_dense_inds...)          # [phys, dense…]
        new_bs = prod(new_dense_dims)
        s2t = SparseBackends.slice_to_template(w)                    # physical → template (fixed)
        new_templates = Vector{eltype(a.templates)}(undef, a.n_templates * new_bs)
        tail_ci = CartesianIndices(new_dense_dims)
        seen = falses(a.n_templates)
        @inbounds for s in 1:a.dims[phys_pos]
            tid = s2t[s]; tid == 0 && continue
            seen[tid] && error("template $tid shared by 2 slices"); seen[tid] = true
            off = (tid - 1) * new_bs
            for (lin, ci) in enumerate(tail_ci)
                new_templates[off + lin] = core_arr[s, Tuple(ci)...]
            end
        end
        new_dims = (ntuple(i -> a.dims[i], Pn)..., new_dense_dims...)
        new_ali  = typeof(a)(new_dims, new_bs, new_templates, a.n_templates,
                             copy(a.keys), copy(a.alias_ids), copy(a.scalars))
        return ITensors._itensor_from_external_storage(typeof(w)(new_ali, (prefix_inds..., new_dense_inds...)))
    end

    # ── write-back = svd(cg) → two rebuild_core (per-site write_core!) ──
    left_inds = commoninds(cg, cL0)                    # site-b: (s_b, left core-link)
    U, S, V = svd(cg, left_inds...; lefttags="Link,l=$b", maxdim=40, cutoff=1e-12)
    core_b, core_b1 = U, S * V                         # left-iso split (L→R sweep)
    ψb  = rebuild_core(psi[b],   core_b)
    ψb1 = rebuild_core(psi[b+1], core_b1)

    ref   = _dns(form_phi(cg))                         # = P·cg (dense)
    recon = _dns(*(ψb, ψb1; preserve_bs_output=true))
    rel   = norm(array(recon, inds(ref)...) .- array(ref)) / max(norm(array(ref)), eps())
    rcb   = try; SparseBackends.read_core(ITensors.get_external_storage(ψb));  true; catch; false; end
    rcb1  = try; SparseBackends.read_core(ITensors.get_external_storage(ψb1)); true; catch; false; end
    @printf("\n(A) ψ[b],ψ[b+1] aliased=%s,%s  recon ‖ψbψb1 − P·cg‖/‖·‖=%.2e  %s\n",
            isaliased(ψb), isaliased(ψb1), rel, rel < 1e-10 ? "PASS ✓" : "FAIL ✗")
    @printf("(B) read_core(ψb)=%s read_core(ψb1)=%s   center-bond dim=%d (was %d)\n",
            rcb ? "ok" : "FAIL", rcb1 ? "ok" : "FAIL",
            dim(commonind(core_b, core_b1)), dim(commonind(cL0, cR0)))
end
nothing
