# core_helpers/factor_core.jl — factor-core primitives for ψ = P·core (diagonal AND flip P).
#
# The aliased ψ stores each template as one pre-P core physical-slice `core[:, rv, :]`
# (verified: contract_aliased_coo_dense makes template = α·B[:,rv]). These let a
# DMRG/TDVP driver treat the *templates* as the variational `core` while P (the
# keys/alias_ids/scalars) stays fixed:
#   read_core(w)             → the dense core tensor at this site (physical + core links)
#   slice_to_template(w)     → rv-slice → template id (cached hint field, else derived)
#   window_write_map(φ,…)    → read the emitted (rv_b,rv_{b+1})→tid map off a 2-site φ
#   write_core_window!(φ,…)  → scatter a merged 2-site core back into φ's templates
# Round-trip is exact where each template is a single clean core slice; errors on
# scramble (a template shared across two physical slices).

# Position of the physical (Site-tagged) axis among the P prefix axes.
function _phys_prefix_pos(w::WrappedAliasedBlockSparse)
    a = w.aliased
    Pn = length(a.dims) - _n2(w)
    @inbounds for i in 1:Pn
        ITensors.hastags(w.inds[i], "Site") && return i
    end
    return 0
end
_n2(w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = N2

# core-slice → template id. Uses the cached hint if present, else derives it by
# grouping keys on the physical axis (and asserts no scramble).
function slice_to_template(w::WrappedAliasedBlockSparse; assume_diagonal_P::Bool = false)
    a = w.aliased
    isempty(a.keys) && return Int[]
    !isempty(a.slice_to_template) && return a.slice_to_template
    # NO SILENT FALLBACK. The derivation below reads the Site coordinate out of each
    # key, which is the POST-P index s'. That equals the pre-P slice rv only when P is
    # DIAGONAL. For an off-diagonal (flip) P it returns the right templates filed under
    # the wrong slices -- a permuted core -- and it cannot detect the difference,
    # because a relabelling produces no key collision for the `scramble` guard below.
    # P is not in scope here, so this function cannot check; the caller must assert it.
    assume_diagonal_P || error(
        "factor-core: slice_to_template is absent and cannot be derived safely. It is " *
        "written only by the COO-MPO x dense-MPS kernel (psi = P*core); any operation " *
        "that rebuilt this storage without carrying it forward loses it. Deriving it " *
        "from `keys` assumes a DIAGONAL P and silently returns a permuted core for a " *
        "flip P. If this call site knows P is diagonal, pass assume_diagonal_P=true.")
    phys = _phys_prefix_pos(w)
    phys == 0 && error("factor-core: no Site axis in prefix; cannot derive slice_to_template")
    s2t = zeros(Int, a.dims[phys])
    @inbounds for (i, k) in enumerate(a.keys)
        s = k[phys]; tid = Int(a.alias_ids[i])
        if s2t[s] == 0
            s2t[s] = tid
        elseif s2t[s] != tid
            error("factor-core: scramble at physical slice $s (templates $(s2t[s]) vs $tid) — not a clean P·core")
        end
    end
    return s2t
end

# Recover the dense core tensor at this site: core[:, s, :] = template[s2t[s]].
# Returned on (physical index, dense/core-link indices) — the aliased tensor's own.
function read_core(w::WrappedAliasedBlockSparse)
    a = w.aliased
    phys = _phys_prefix_pos(w)
    phys == 0 && error("factor-core: no Site axis")
    Pn = length(a.dims) - _n2(w)
    phys_ind   = w.inds[phys]
    dense_inds = w.inds[Pn+1:end]
    dense_dims = a.dims[Pn+1:end]
    bs = a.blksize
    s2t = slice_to_template(w)
    core_arr = zeros(eltype(a.templates), a.dims[phys], dense_dims...)
    tail_ci = CartesianIndices(Tuple(dense_dims))
    @inbounds for s in 1:a.dims[phys]
        tid = s2t[s]; tid == 0 && continue           # physical s absent (P forbids)
        off = (tid - 1) * bs
        for (lin, ci) in enumerate(tail_ci)
            core_arr[s, Tuple(ci)...] = a.templates[off + lin]
        end
    end
    return ITensors.ITensor(core_arr, phys_ind, dense_inds...)
end

# Which template pair built each template of a window φ = M_b·M_b1, read off the keys:
# a φ key plus a bond channel c gives the M_b key and the M_b1 key that met there, and
# their alias_ids are (a_L, a_R). Returns tid => (a_L, a_R) for every φ template.
# Needs only keys and alias_ids (no slice_to_template), so it works on any φ contracted
# from exactly these M_b, M_b1, including a copy. Errors if a φ key was built from two
# different template pairs (a demoted accumulator: φ is not a clean window) or if one
# template holds two pairs.
function window_pairs(wphi::WrappedAliasedBlockSparse,
                      wb::WrappedAliasedBlockSparse, wb1::WrappedAliasedBlockSparse)
    aphi, ab, ab1 = wphi.aliased, wb.aliased, wb1.aliased
    Pphi, Pb, Pb1 = _abs_head_len(wphi), _abs_head_len(wb), _abs_head_len(wb1)
    pre_b, pre_b1, pre_phi = wb.inds[1:Pb], wb1.inds[1:Pb1], wphi.inds[1:Pphi]
    ch = [I for I in pre_b if I in pre_b1]
    length(ch) == 1 || error("window_pairs: expected one shared bond channel in the " *
                             "prefixes, found $(length(ch))")
    cp_b, cp_b1 = findfirst(==(ch[1]), pre_b), findfirst(==(ch[1]), pre_b1)
    # φ prefix axis for every non-channel prefix axis of M_b / M_b1 (0 on the channel)
    src_b  = [p == cp_b  ? 0 : something(findfirst(==(pre_b[p]),  pre_phi), -1) for p in 1:Pb]
    src_b1 = [p == cp_b1 ? 0 : something(findfirst(==(pre_b1[p]), pre_phi), -1) for p in 1:Pb1]
    (any(==(-1), src_b) || any(==(-1), src_b1) || Pphi != Pb + Pb1 - 2) &&
        error("window_pairs: φ's prefix is not M_b's and M_b1's prefixes minus the bond channel")
    look_b  = Dict(map(Int, Tuple(k)) => i for (i, k) in enumerate(ab.keys))
    look_b1 = Dict(map(Int, Tuple(k)) => i for (i, k) in enumerate(ab1.keys))
    pairs = Dict{Int,NTuple{2,Int}}()
    kb, kb1 = zeros(Int, Pb), zeros(Int, Pb1)
    @inbounds for (i, K) in enumerate(aphi.keys)
        pair = (0, 0)
        for c in 1:ITensors.dim(ch[1])
            for p in 1:Pb;  kb[p]  = src_b[p]  == 0 ? c : Int(K[src_b[p]]);  end
            for p in 1:Pb1; kb1[p] = src_b1[p] == 0 ? c : Int(K[src_b1[p]]); end
            iL = get(look_b, Tuple(kb), 0);   iL == 0 && continue
            iR = get(look_b1, Tuple(kb1), 0); iR == 0 && continue
            pc = (Int(ab.alias_ids[iL]), Int(ab1.alias_ids[iR]))
            pair == (0, 0) || pair == pc ||
                error("window_pairs: φ key $K is built from template pairs $pair and $pc " *
                      "(a demoted accumulator); φ is not a clean window of M_b·M_b1")
            pair = pc
        end
        pair == (0, 0) && error("window_pairs: φ key $K has no source block in M_b, M_b1; " *
                                "φ was not contracted from these tensors")
        tid = Int(aphi.alias_ids[i])
        prev = get(pairs, tid, (0, 0))
        prev == (0, 0) || prev == pair ||
            error("window_pairs: φ template $tid holds template pairs $prev and $pair")
        pairs[tid] = pair
    end
    return pairs
end

# The (rv_b, rv_{b+1}) → φ template-id map for a window φ = ψ[b]·ψ[b+1], derived from the
# tensors: window_pairs gives tid => (a_L, a_R), and each site's slice_to_template,
# inverted, gives a_L → rv_b and a_R → rv_{b+1}. Returns (rvmap, [axis_b, axis_bp1]) with
# φ's Site axes ordered (b, b+1) so core_arr[rv_b, rv_{b+1}] lines up. Identity for
# diagonal P (rv == output), the flip permutation for KL.
function window_write_map(wphi::WrappedAliasedBlockSparse,
                          wb::WrappedAliasedBlockSparse, wbp1::WrappedAliasedBlockSparse)
    a = wphi.aliased
    t2rv(w) = begin
        inv = Dict{Int,Int}()
        for (rv, t) in enumerate(slice_to_template(w))
            t == 0 && continue
            haskey(inv, t) && error("window_write_map: template $t holds two core slices")
            inv[t] = rv
        end
        inv
    end
    rv_b, rv_bp1 = t2rv(wb), t2rv(wbp1)
    rvmap = Dict{NTuple{2,Int},Int}()
    for (tid, (aL, aR)) in window_pairs(wphi, wb, wbp1)
        (haskey(rv_b, aL) && haskey(rv_bp1, aR)) ||
            error("window_write_map: φ template $tid comes from templates ($aL, $aR), " *
                  "which hold no core slice (slice_to_template)")
        rvp = (rv_b[aL], rv_bp1[aR])
        haskey(rvmap, rvp) &&
            error("window_write_map: core entry $rvp maps to templates $(rvmap[rvp]) and $tid")
        rvmap[rvp] = tid
    end
    Pn = length(a.dims) - _n2(wphi)
    sp = [i for i in 1:Pn if ITensors.hastags(wphi.inds[i], "Site")]
    length(sp) == 2 || error("window_write_map: expected exactly 2 Site axes (got $(length(sp)))")
    sib   = wb.inds[_phys_prefix_pos(wb)]
    sibp1 = wbp1.inds[_phys_prefix_pos(wbp1)]
    jb   = findfirst(p -> ITensors.id(wphi.inds[p]) == ITensors.id(sib),   sp)
    jbp1 = findfirst(p -> ITensors.id(wphi.inds[p]) == ITensors.id(sibp1), sp)
    (jb === nothing || jbp1 === nothing) && error("window_write_map: φ Site axes don't match ψ[b]/ψ[b+1] ids")
    return rvmap, [sp[jb], sp[jbp1]]
end

# 2-site (window) counterpart of write_core!: scatter a dense MERGED 2-site core into
# the templates of an aliased window φ = ψ[b]·ψ[b+1], IN PLACE (keys/alias_ids/scalars
# untouched), using a PRECOMPUTED map from window_write_map (write does NOT re-derive).
# Used by the factor-core matvec to re-attach the ket-P to the evolving Krylov core each
# step: `write_core_window!(φ, core, wmap); product(...)`. `core` lives on φ's Site +
# dense (core-link) indices.
function write_core_window!(w::WrappedAliasedBlockSparse, core::ITensors.ITensor,
                            wmap::Tuple{<:AbstractDict,<:AbstractVector})
    tup2tid, sp = wmap
    a = w.aliased
    Pn = length(a.dims) - _n2(w)
    dense_inds = w.inds[Pn+1:end]
    dense_dims = a.dims[Pn+1:end]
    bs = a.blksize
    site_inds = [w.inds[p] for p in sp]
    core_arr = Array(core, site_inds..., dense_inds...)   # [phys-tuple…, dense…] in φ's order
    tail_ci = CartesianIndices(Tuple(dense_dims))
    @inbounds for (key, tid) in tup2tid
        off = (tid - 1) * bs
        for (lin, ci) in enumerate(tail_ci)
            a.templates[off + lin] = core_arr[key..., Tuple(ci)...]
        end
    end
    return w
end

# Drop a dense single-site `core` into t's aliased template slots. Keeps the P
# structure (keys/alias_ids/scalars, P-FSM channels) FIXED; the core-link Index/dim/
# blksize come from `core`, so the bond may grow or shrink. Returns a fresh aliased
# ITensor.
function core_rebuild(t::ITensors.ITensor, core::ITensors.ITensor)
    w = ITensors.get_external_storage(t)::WrappedAliasedBlockSparse; a = w.aliased
    Pn = _abs_head_len(w)
    prefix_inds = w.inds[1:Pn]
    phys_pos = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), 1:Pn)
    phys_ind = w.inds[phys_pos]
    new_dense_inds = [i for i in ITensors.inds(core) if !ITensors.hastags(i, "Site")]
    new_dense_dims = Tuple(ITensors.dim(i) for i in new_dense_inds)
    core_arr = Array(core, phys_ind, new_dense_inds...)
    new_bs = prod(new_dense_dims)
    s2t = slice_to_template(w)
    new_templates = Vector{eltype(a.templates)}(undef, a.n_templates * new_bs)
    tail_ci = CartesianIndices(new_dense_dims)
    seen = falses(a.n_templates)
    @inbounds for s in 1:a.dims[phys_pos]
        tid = s2t[s]; tid == 0 && continue
        seen[tid] && error("factor-core: template $tid shared by 2 slices"); seen[tid] = true
        off = (tid - 1) * new_bs
        for (lin, ci) in enumerate(tail_ci)
            new_templates[off + lin] = core_arr[s, Tuple(ci)...]
        end
    end
    new_dims = (ntuple(i -> a.dims[i], Pn)..., new_dense_dims...)
    new_ali  = typeof(a)(new_dims, new_bs, new_templates, a.n_templates,
                         copy(a.keys), copy(a.alias_ids), copy(a.scalars))
    # preserve the pre-P routing map (P structure unchanged) so read_core stays general
    # (off-diagonal P) across the sweep instead of re-deriving by output-site grouping.
    isempty(a.slice_to_template) || (new_ali.slice_to_template = copy(a.slice_to_template))
    return ITensors._itensor_from_external_storage(typeof(w)(new_ali, (prefix_inds..., new_dense_inds...)))
end

# Split a dense 2-site core `cg` and write the factors into the structure of M_b, M_b1
# with P fixed. ortho="left": L = U, R = S V (left-iso at b); "right": L = U S, R = V.
# Truncates on the core bond. The factor-core two-site write-back (dense core in); the
# direct aliased φ split is the separate itensor_aliased_factorize (aliased/factorize.jl).
function core_split(M_b::ITensors.ITensor, M_b1::ITensors.ITensor, cg::ITensors.ITensor;
                    ortho::String = "left", maxdim::Int = typemax(Int), mindim::Int = 1,
                    cutoff::Real = 0.0, tags = nothing)
    ortho in ("left", "right") || error("core_split: unknown ortho=$ortho")
    core_b  = read_core(ITensors.get_external_storage(M_b))
    core_b1 = read_core(ITensors.get_external_storage(M_b1))
    left_inds = ITensors.commoninds(cg, core_b)                # (s_b, left core-link)
    lefttags = tags === nothing ? ITensors.tags(only(ITensors.commoninds(core_b, core_b1))) : tags
    F = ITensors.svd(cg, left_inds...; lefttags, maxdim, mindim, cutoff)
    cb, cb1 = ortho == "left" ? (F.U, F.S * F.V) : (F.U * F.S, F.V)
    return core_rebuild(M_b, cb), core_rebuild(M_b1, cb1), F.spec
end
