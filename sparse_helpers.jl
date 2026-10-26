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
isdefined(@__MODULE__, :multiply_mpos) ||
    include(joinpath(@__DIR__, "..", "experiments", "utils", "cotenn_utils.jl"))

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

"""psi = P * psi0 in a sparse format.

`psi0` IS dense, so it is declared `:dense` and the output backend is named
explicitly. Do NOT use the legacy `:coo, :aliased; denseLinksB=0` form: it
declares a dense operand as aliased purely so top_level_contract will INFER
Cbackend -- see the note on an ALIASED input backend in top_level_contract,
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
(permuting inside the key or inside the dense block only relabels axes)."""
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

The output layout is `SparseBackends.dense_from_aliased_operand`: the output's dense
block holds only the ALIASED operand's dense indices, so the dense operand's site/link
indices go into the key rather than demoting the key into the dense block."""
function mul_aliased(A::ITensor, B::ITensor; in_position::Bool = false)
    a_ali = storage_kind(A) === :aliased
    b_ali = storage_kind(B) === :aliased
    (a_ali || b_ali) || return A * B
    Aw = a_ali ? ITensors.get_external_storage(A) : SparseBackends.wrap_itensor(A; backend = :dense)
    Bw = b_ali ? ITensors.get_external_storage(B) : SparseBackends.wrap_itensor(B; backend = :dense)
    Cw = SparseBackends.contract(Aw, Bw;
            preserve_bs_output = true, in_position = in_position,
            layout = SparseBackends.dense_from_aliased_operand(Aw, Bw))
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

`maxdim <= 0` means EXACT: skip the truncation entirely. Bonds then grow by the layer
bond dimension every layer, so that is for validation and short runs.

The truncation is stock `truncate!` (orthogonalize to site N, then a truncating sweep
N -> 1; the centre ends at site 1). There is one direction.

`merge` is selectable so an accuracy sweep can measure it rather than assume it:

  * `merge` -- `:before` (default, the measured template win), `:after` (the old
    behaviour), `:both`, or `:none`. `true`/`false` still work and map to
    `:before`/`:none`, so existing callers are unchanged.

`merge` defaults to `:before` for an MPO layer and to `:none` for a constrained layer `(gate, gauges)`:
the gauged kernel returns psi's own keys, template classes and scalars (one template per class, the
scalars kept), so there is nothing left to merge, and a numeric merge would only divide by computed
coefficients (`:before`/`:both`/`:after` still run it if asked for).

`M` may also be a constrained layer `(gate, gauges)` of `build_tebd_mpo_layers(...; P,
datastructure = :aliased)` for an aliased psi = P*core: the same `contract` call then applies
the gate and the bond gauges in one contraction (see the ITensorMPS fork's
`contract(A, psi, Abackend, Bbackend)`). Its default layout puts each bond's core link next
to the gate link it fuses with, (m_l, g_l, m_r, g_r), so the bond fusion after the contraction
needs no template permute (fused value m + Dm*(g-1), m fastest, as for an MPO layer)."""
function apply_layer_sparse(psi::MPS, M::Union{MPO,NamedTuple{(:gate, :gauges)}};
                            maxdim::Int, cutoff::Real = 0.0,
                            merge::Union{Bool,Symbol} = M isa MPO ? :before : :none,
                            layout = SparseBackends.OutputLayout(
                                sparse = [(:B,:l,1), (:B,:r,1), (:A,:sp)],
                                dense  = M isa MPO ? [(:B,:l,2), (:B,:r,2), (:A,:r), (:A,:l)] :
                                                     [(:B,:l,2), (:A,:l), (:B,:r,2), (:A,:r)]))
    mode = merge isa Bool ? (merge ? :before : :none) : merge
    mode in (:none, :before, :after, :both) ||
        error("apply_layer_sparse: merge must be :none/:before/:after/:both (or a Bool); got $merge")

    bk = storage_kind(psi[1])
    # Output of each site product gate[i] (A) * ψ[i] (B), for an aliased ψ = P·core:
    # key = P's links + the gate's output site s' (moved into the key); dense = core's
    # links + the gate links. Other storage kinds use the stock layout.
    phi = replaceprime(contract(M, psi, :dense, bk; Cbackend = bk,
                                layout = bk === :aliased ? layout : nothing), 1 => 0)
    # MERGE FIRST. The output-stationary kernel emits one template per output block
    # (dedup 1.00x by construction), and the single-site aliased factorize inherits its schema
    # from the INPUT tensors -- so merging here hands the sweep the small schema, whereas
    # merging afterwards has to find proportionality the refactorization destroyed.
    (mode === :before || mode === :both) && merge_aliased_templates!(phi)
    maxdim > 0 && truncate!(phi; maxdim = maxdim, cutoff = cutoff)
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

"""All layers of one Trotter cycle onto `psi`, each through `apply_layer_sparse`.
`merge = nothing` and `layout = nothing`: each layer gets `apply_layer_sparse`'s default for its type."""
function step_sparse(psi::MPS, layers::AbstractVector; maxdim::Int, cutoff::Real = 0.0,
                     merge::Union{Nothing,Bool,Symbol} = nothing,
                     layout = nothing)
    p = psi
    for M in layers
        p = apply_layer_sparse(p, M; maxdim = maxdim, cutoff = cutoff,
                               (merge === nothing ? (;) : (; merge = merge))...,
                               (layout === nothing ? (;) : (; layout = layout))...)
    end
    return p
end

# --- observables ------------------------------------------------------------

expect_mpos(psi::MPS, ops::Vector{MPO}) =
    (n2 = real(inner(psi, psi)); [real(inner(psi', O, psi)) / n2 for O in ops])
