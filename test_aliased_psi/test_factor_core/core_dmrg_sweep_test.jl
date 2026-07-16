# core_dmrg_sweep_test.jl — STEP 5 prototype: full core-mode DMRG sweep on PXP.
# Krylov over dense cores; matvec = form_φ → CoreProjMPO.product; write-back = svd +
# 2× rebuild_core. ψ stays aliased P·core; energy = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ → target −14.77.
# Prototypes dmrg_core_php before promoting to ITensorMPS/src/factor_core_dmrg.jl.

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
_ext(t) = ITensors.get_external_storage(t)
_rc(t)  = SparseBackends.read_core(_ext(t))

# form φ = P·core over the window sites (dense; matvec densifies at step 1 anyway)
function form_phi(P, core, sites)
    Pop = P[first(sites)]
    for j in sites[2:end]; Pop = Pop * P[j]; end
    return replaceprime(Pop * core, 1 => 0)
end

# resize-capable write_core!: drop dense single-site `core` into ψ[j]'s aliased slots.
function rebuild_core(t::ITensor, core::ITensor)
    w = _ext(t); a = w.aliased
    Pn = SparseBackends._abs_head_len(w)
    prefix_inds = w.inds[1:Pn]
    phys_pos = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), 1:Pn)
    phys_ind = w.inds[phys_pos]
    new_dense_inds = [i for i in inds(core) if !ITensors.hastags(i, "Site")]
    new_dense_dims = Tuple(ITensors.dim(i) for i in new_dense_inds)
    core_arr = Array(core, phys_ind, new_dense_inds...)
    new_bs = prod(new_dense_dims)
    s2t = SparseBackends.slice_to_template(w)
    new_templates = Vector{eltype(a.templates)}(undef, a.n_templates * new_bs)
    tail_ci = CartesianIndices(new_dense_dims)
    seen = falses(a.n_templates)
    @inbounds for s in 1:a.dims[phys_pos]
        tid = s2t[s]; tid == 0 && continue
        seen[tid] && error("template $tid shared"); seen[tid] = true
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

read_window(psi, sites) = (c = _rc(psi[first(sites)]); for j in sites[2:end]; c = c * _rc(psi[j]); end; c)

# svd(cg) + 2× rebuild_core. ha=1 (L→R): left-iso at b; ha=2 (R→L): right-iso at b+1.
function core_writeback!(psi, b, cg, ha; maxdim, cutoff)
    cL = _rc(psi[b])
    left_inds = commoninds(cg, cL)
    F = svd(cg, left_inds...; lefttags="Link,l=$b", maxdim=maxdim, cutoff=cutoff)
    U, S, V = F.U, F.S, F.V
    cb, cb1 = ha == 1 ? (U, S * V) : (U * S, V)
    psi[b]   = rebuild_core(psi[b],   cb)
    psi[b+1] = rebuild_core(psi[b+1], cb1)
    return psi
end

# initial core RIGHT-canonicalization (center → 1), no eigensolve, no truncation.
function core_canonicalize!(psi)
    N = length(psi)
    for b in (N-1):-1:1
        cg = read_window(psi, b:b+1)
        core_writeback!(psi, b, cg, 2; maxdim=typemax(Int), cutoff=0.0)  # ha=2 → right-iso at b+1
    end
    return psi
end

phys_energy(psi, H) = (pd = MPS([_dns(psi[j]) for j in 1:length(psi)]);
                       real(inner(pd', H, pd) / inner(pd, pd)))

function dmrg_core_php(H, P, psi0; nsweeps, maxdim, cutoff=1e-12, kdim=30, tol=1e-12)
    N = length(psi0); psi = copy(psi0)
    cpm = ITensorMPS.CoreProjMPO(H, P; nsite=2)
    core_canonicalize!(psi)
    @printf("init  E=%.8f\n", phys_energy(psi, H))
    local E
    for sw in 1:nsweeps
        for (b, ha) in ITensorMPS.sweepnext(N)
            ITensorMPS.position!(cpm, psi, b)
            core0 = read_window(psi, b:b+1)
            f(core) = ITensorMPS.product(cpm, form_phi(P, core, b:b+1))
            vals, vecs = eigsolve(f, core0, 1, :SR; ishermitian=true, krylovdim=kdim, tol=tol)
            core_writeback!(psi, b, vecs[1], ha; maxdim=maxdim, cutoff=cutoff)
        end
        E = phys_energy(psi, H)
        @printf("sweep %2d  E=%.8f\n", sw, E)
    end
    return E, psi
end

let
    N = 12
    H, psi, P = build_setup_pxp(N)
    E, _ = dmrg_core_php(H, P, psi; nsweeps=10, maxdim=40)
    @printf("\nfinal core-mode E = %.8f   (target ≈ −14.77)\n", E)
end
nothing
