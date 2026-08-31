# sparse_helpers.jl -- shared helpers for SparseBackends-based tests.
#
# Standalone file, NOT included into any module: include it after
# `using ITensors, ITensorMPS, SparseBackends` so every name resolves in the
# caller's scope. Sits beside utils.jl at the environment root.
#
# These were previously copy-pasted per test. Live non-comment duplicate counts
# before this file existed: 21 sites of the attach_P pattern, and a local
# `mulMPO` defined in every sparse test driver.

using ITensors, ITensorMPS, SparseBackends, Printf

# --- storage introspection --------------------------------------------------

storage_kind(T::ITensor) =
    !ITensors.has_external_storage(T) ? :dense :
    ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse ? :aliased :
    ITensors.get_external_storage(T) isa SparseBackends.WrappedBlockSparse ? :blocksparse : :other

storage_kinds(psi) = [storage_kind(psi[i]) for i in 1:length(psi)]

densify(T::ITensor) = ITensors.has_external_storage(T) ?
    SparseBackends.to_dense_itensors_unfused(T) : T
densify(psi::MPS) = MPS([densify(psi[j]) for j in 1:length(psi)])
densify(H::MPO)   = MPO([densify(H[j])   for j in 1:length(H)])

# Product of ALL shared indices (channel x multiplicity). `dim(commonind)` returns
# only one shared index and understates the real rank.
honest_linkdims(psi) = [(c = commoninds(psi[i], psi[i+1]); isempty(c) ? 0 : prod(ITensors.dim, c))
                        for i in 1:(length(psi)-1)]

reported_linkdims(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:(length(psi)-1)]

# Prefix / dense-tail split, plus how many Site axes are in the prefix. The aliased
# kernels require every Site axis to live in the prefix.
function prefix_split(T::ITensor)
    storage_kind(T) === :aliased || return (0, 0, 0)
    w = ITensors.get_external_storage(T)
    Pn = SparseBackends._abs_head_len(w)
    return (Pn, length(w.aliased.dims) - Pn,
            count(i -> ITensors.hastags(w.inds[i], "Site"), 1:Pn))
end

function footprint(psi)
    tmpl = meta = bs = dense = nb = nt = 0
    for T in psi
        k = storage_kind(T)
        if k === :aliased
            a = ITensors.get_external_storage(T).aliased
            tmpl  += Base.summarysize(a.templates)
            meta  += Base.summarysize(a.keys) + Base.summarysize(a.alias_ids) + Base.summarysize(a.scalars)
            bs    += length(a.keys) * a.blksize * sizeof(eltype(a.templates))
            dense += prod(a.dims) * sizeof(eltype(a.templates))
            nb += length(a.keys); nt += a.n_templates
        elseif k === :blocksparse
            b = ITensors.get_external_storage(T).blocksparse
            tmpl += Base.summarysize(b.data); bs += Base.summarysize(b.data)
            meta += Base.summarysize(b.keys) + Base.summarysize(b.ids)
            dense += prod(b.dims) * sizeof(eltype(b.data)); nb += length(b.keys)
        else
            n = Base.summarysize(ITensors.array(T)); tmpl += n; bs += n; dense += n
        end
    end
    return (; tmpl, meta, bs, dense, nb, nt)
end

"""(blocks, templates, dedup) -- compact enough to print every cycle, so dedup can
be tracked as a TRAJECTORY rather than only at the endpoints."""
function dedup_stats(psi)
    f = footprint(psi)
    return (nb = f.nb, nt = f.nt, ratio = f.nb / max(f.nt, 1))
end

function report_storage(label, psi)
    f = footprint(psi); k = storage_kinds(psi)
    @printf("[%s] aliased=%d BS=%d dense=%d  blocks=%d templates=%d dedup=%.2fx\n",
            label, count(==(:aliased), k), count(==(:blocksparse), k), count(==(:dense), k),
            f.nb, f.nt, f.nb / max(f.nt, 1))
    @printf("[%s] templates=%.1fKiB schema=%.1fKiB  vs_dense=%.2fx vs_BS=%.2fx  bonds=%s\n",
            label, f.tmpl/1024, f.meta/1024, f.dense/max(f.tmpl,1), f.bs/max(f.tmpl,1),
            honest_linkdims(psi))
    return f
end

# --- alias schema (encodes P; must stay invariant) ---------------------------

function schema_fingerprint(psi)
    map(1:length(psi)) do i
        storage_kind(psi[i]) === :aliased || return nothing
        a = ITensors.get_external_storage(psi[i]).aliased
        g = Dict{Int,Vector{Int}}()
        for (j, id) in enumerate(a.alias_ids); push!(get!(g, Int(id), Int[]), j); end
        (nkeys = length(a.keys), nt = a.n_templates, keys = Set(a.keys),
         part = Set(Set(a.keys[j] for j in v) for v in values(g)),
         sc = sort([(round(real(z), digits=10), round(imag(z), digits=10)) for z in a.scalars]))
    end
end

function schema_drift(a, b)
    d = String[]
    for i in eachindex(a)
        ai, bi = a[i], b[i]
        if ai === nothing || bi === nothing
            ai === bi || push!(d, "site $i storage kind changed"); continue
        end
        ai.nkeys != bi.nkeys && push!(d, "site $i nkeys $(ai.nkeys)->$(bi.nkeys)")
        ai.nt    != bi.nt    && push!(d, "site $i ntemplates $(ai.nt)->$(bi.nt)")
        ai.keys  != bi.keys  && push!(d, "site $i keys changed")
        ai.part  != bi.part  && push!(d, "site $i key->template partition changed")
        ai.sc    != bi.sc    && push!(d, "site $i scalars changed")
    end
    return d
end

# --- projector construction and application ---------------------------------

"""MPO product through the COO backend. `prime(B,"Site")` lifts B to levels 1/2 so
A's output legs meet B's input legs; `replaceprime(2=>1)` restores the 0/1 MPO
convention. `:coo,:coo` keeps the product sparse -- without it, building
P = prod_j Pi_j densifies at the first multiply."""
mul_coo(A::MPO, B::MPO) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)

function mul_coo(v::Vector{MPO})
    P = v[1]
    for j in 2:length(v); P = mul_coo(P, v[j]); end
    return P
end

"""psi = P * psi0 in a sparse format.

`psi0` IS dense, so it is declared `:dense` and the output backend is named
explicitly. Do NOT use the legacy `:coo, :aliased; denseLinksB=0` form: it
declares a dense operand as aliased purely so top_level_contract will INFER
Cbackend -- see the "old contract_and_fuse_links_aliased semantics" note in
SparseBackends/src/tensor_contraction.jl."""
attach_P(P::MPO, psi0::MPS; backend::Symbol = :aliased) =
    replaceprime(contract(P, copy(psi0), :coo, :dense; Cbackend = backend), 1 => 0)

# --- layout ------------------------------------------------------------------

"""Move Site axes to the END of the sparse prefix, leaving the dense tail alone.

A contraction against a dense gate emits the new Site axis last in the prefix (that
is where the fission detector needs it), so `psi0` straight out of `attach_P` -- Site
FIRST -- has a different layout from every later step. Same tensor either way; this
only makes the two schemas comparable.

Within-region permute, so prefix/tail membership and the alias schema are untouched
(same argument as `reorder_aliased_for_op`, tensor_wrappers_aliased.jl:1015)."""
function site_last(T::ITensor)
    storage_kind(T) === :aliased || return T
    w  = ITensors.get_external_storage(T)
    Pn = SparseBackends._abs_head_len(w)
    ix = collect(ITensors.inds(T)); Nc = length(ix)
    issite(I) = ITensors.hastags(I, "Site")
    target = vcat(ITensors.Index[ix[i] for i in 1:Pn if !issite(ix[i])],
                  ITensors.Index[ix[i] for i in 1:Pn if  issite(ix[i])],
                  ix[(Pn+1):Nc])
    ix == target && return T
    perm = Int[findfirst(==(target[i]), ix) for i in 1:Nc]
    new_w = SparseBackends.WrappedAliasedBlockSparse{eltype(w), Nc, Nc - Pn, Pn}(
                SparseBackends.permutedims(w.aliased, perm), Tuple(target))
    return ITensors._itensor_from_external_storage(new_w)
end

site_last(psi::MPS) = MPS([site_last(psi[i]) for i in 1:length(psi)])

# --- sparse MPO-layer evolution ---------------------------------------------
# No equivalent exists elsewhere: tebd_utils.jl's tebd_mpo_step / apply_layer_mpo
# are dense-only (ITensors.contract(Algorithm"naive"(), M, psi)).

"""Aliased-preserving ITensor product, following the convention DMRG already uses
(`_mul_preserve_aliased`, ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl:46).

The hint is the ALIASED operand's dense TAIL only, so the dense operand's site/link
axes are pulled into the sparse PREFIX rather than demoting the prefix into the tail.
`output_inds_hint` inverts semantics: it is the complete dense-tail set, everything
else forced sparse.

This is why the DMRG PHP path keeps its template sharing and our MPO*MPS path does
not: contract(MPO, psi, ...) goes through top_level_contract/contract_and_fuse_links,
which accept NO hint, while DMRG calls wrapped_contract_aliased directly."""
function mul_aliased(A::ITensor, B::ITensor; in_position::Bool = false)
    a_ali = storage_kind(A) === :aliased
    b_ali = storage_kind(B) === :aliased
    (a_ali || b_ali) || return A * B
    Aw = a_ali ? ITensors.get_external_storage(A) : SparseBackends.wrap_itensor(A; backend = :dense)
    Bw = b_ali ? ITensors.get_external_storage(B) : SparseBackends.wrap_itensor(B; backend = :dense)
    hint = Set{Index}()
    a_ali && union!(hint, SparseBackends.dense_inds(Aw))
    b_ali && union!(hint, SparseBackends.dense_inds(Bw))
    Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw;
            preserve_bs_output = true, output_inds_hint = hint, in_position = in_position)
    return Cw isa ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
end

"""One Trotter layer MPO onto a sparse psi.

The CONTRACTION is exact. `orthogonalize!` on an aliased pair is NOT a dense SVD
sweep: abstractmps.jl routes it to `itensor_aliased_factorize`, which inherits keys,
alias_ids, scalars and the channel axis VERBATIM and refactorizes the MULTIPLICITY
bond only. So it cannot mix channels -- but it does rewrite the template values, and
merging AFTER it then has to find proportionality the refactorization destroyed.

Two consequences, both measured on the D4 ring:

  * ORDER. Merge before the sweep, not after -- `merge = true` is now the DEFAULT and is
    done inside this function. `itensor_aliased_factorize` takes its schema from the INPUT
    tensors, so a merged input carries its small schema through. Nm=3, one layer: 45
    templates merging after vs 24 merging before (24 is also what merging with no sweep at
    all reaches). Nm=4: 71 vs 29. Storage over 3 cycles: 130 KB vs 46 KB. Pass
    `merge = false` to recover the old behaviour.

  * CUTOFF. `orthogonalize!` hardcoded cutoff = 0.0 in every external-storage branch,
    so the sweep could never drop a zero-weight multiplicity direction -- which is the
    axis the gate's virtual bond accumulates on. Nm=3 after one layer: nominal
    multiplicity [2,4,4,4,2] against an effective [1,1,1,1,1]. It is now a keyword,
    defaulting to 0.0 so nothing else changes; pass `cutoff` here to use it.

`maxdim <= 0` means EXACT: skip orthogonalize! entirely. Bonds then grow by the layer
bond dimension every layer, so that is for validation and short runs.

`merge` and `ortho_frame` are the two knobs the ordering result above rests on, and both
are now selectable so an accuracy sweep can measure them rather than assume them:

  * `merge` -- `:before` (default, the measured template win), `:after` (the old
    behaviour), `:both`, or `:none`. `true`/`false` still work and map to
    `:before`/`:none`, so existing callers are unchanged.
  * `ortho_frame` -- `:left` (default, centre at site 1) or `:right` (centre at site N).

Template counts were previously found identical across all four combinations, but their
FIDELITY was never compared: `itensor_aliased_factorize` discards on the P-weighted
blocks, so which multiplicity directions it drops can depend on the sweep direction even
when the resulting schema does not."""
function apply_layer_sparse(psi::MPS, M::MPO; maxdim::Int, cutoff::Real = 0.0,
                            merge::Union{Bool,Symbol} = :before,
                            ortho_frame::Symbol = :left,
                            alias_hint::Union{Nothing,Symbol} = :preserve_prefix)
    mode = merge isa Bool ? (merge ? :before : :none) : merge
    mode in (:none, :before, :after, :both) ||
        error("apply_layer_sparse: merge must be :none/:before/:after/:both (or a Bool); got $merge")
    ortho_frame in (:left, :right) ||
        error("apply_layer_sparse: ortho_frame must be :left or :right; got $ortho_frame")

    bk = storage_kind(psi[1])
    phi = replaceprime(contract(M, psi, :dense, bk; Cbackend = bk,
                                alias_hint = alias_hint), 1 => 0)
    # MERGE FIRST. The output-stationary kernel emits one template per output block
    # (dedup 1.00x by construction), and itensor_aliased_factorize inherits its schema
    # from the INPUT tensors -- so merging here hands the sweep the small schema, whereas
    # merging afterwards has to find proportionality the refactorization destroyed.
    (mode === :before || mode === :both) && merge_aliased_templates!(phi)
    if maxdim > 0
        j = ortho_frame === :left ? 1 : length(phi)
        # `cutoff` is forwarded ONLY when nonzero. The kwarg is a LOCAL patch to
        # SparseBackends/ITensorMPS.jl; upstream -- and the copy deployed on Sherlock --
        # has `orthogonalize!(M, j; maxdim, normalize)` and nothing else, so passing
        # cutoff unconditionally kills every remote run with an unsupported-keyword
        # MethodError, after setup and JIT, i.e. minutes in and long past any load
        # check. cutoff=0.0 is the default here and was measured inert on this model
        # anyway (relerr 0.000e+00 against 1e-14), so the default path needs no patched
        # library and the sweep does not have to ship one.
        if Float64(cutoff) > 0
            orthogonalize!(phi, j; maxdim = maxdim, cutoff = Float64(cutoff))
        else
            orthogonalize!(phi, j; maxdim = maxdim)
        end
    end
    (mode === :after || mode === :both) && merge_aliased_templates!(phi)
    return phi
end

"""Projective template merge on every aliased site of `psi`, in place.

Factored out of `apply_layer_sparse` because `merge = :both` needs it twice and the
sweep driver needs it standalone; it was previously an inline loop with no name."""
function merge_aliased_templates!(psi::MPS)
    for i in 1:length(psi)
        storage_kind(psi[i]) === :aliased || continue
        SparseBackends.compress_aliased_templates!(psi[i]; projective = true)
    end
    return psi
end

step_sparse(psi::MPS, layers::Vector{MPO}; maxdim::Int, cutoff::Real = 0.0,
            merge::Union{Bool,Symbol} = :before,
            ortho_frame::Symbol = :left,
            alias_hint::Union{Nothing,Symbol} = :preserve_prefix) =
    foldl((p, M) -> apply_layer_sparse(p, M; maxdim = maxdim, cutoff = cutoff,
                                       merge = merge, ortho_frame = ortho_frame,
                                       alias_hint = alias_hint),
          layers; init = psi)

# --- observables ------------------------------------------------------------

expect_mpos(psi::MPS, ops::Vector{MPO}) =
    (n2 = real(inner(psi, psi)); [real(inner(psi', O, psi)) / n2 for O in ops])
