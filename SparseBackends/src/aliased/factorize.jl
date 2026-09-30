# aliased/factorize.jl
#
# Aliased factorizations. P (keys, alias_ids, scalars, channel axes) stays fixed;
# only templates and the multiplicity (core-link) axis change.
#   itensor_aliased_factorize        two-site, direct: split an aliased φ in template space
#   _external_factorize_storage      single-site: the `factorize` hook (orthogonalize!, TEBD)
# The factor-core two-site split (dense core in, core_split) is a separate method in
# core_helpers/factor_core.jl; replacebond! picks between the two by the input's type.

import ITensors
using LinearAlgebra: svd, Diagonal, norm

"""
    itensor_aliased_factorize(phi, M_b, M_b1; ortho="left", maxdim, mindim=1,
                              cutoff=0.0, tags=nothing) -> (L, R, spec)

Direct two-site split of an aliased window φ back into the structure of `M_b`, `M_b1`
(the tensors φ was contracted from, i.e. the current ψ[b], ψ[b+1]), working on φ's
templates; no core is formed.

Placement comes from the tensors: `window_pairs` reads, from the keys and alias_ids, the
template pair (a_L, a_R) that built each φ template. φ's template is written, unscaled,
into the (a_L, a_R) block of
    M_red[(a_L, L_tail), (a_R, R_tail)],
which is SVD'd; U's row blocks become M_b's new templates and V's column blocks M_b1's.
Keys, alias_ids, scalars and the channel axes of both sides are kept. No per-channel
division and no averaging: a block written twice must agree, otherwise it errors.
"""
function itensor_aliased_factorize(
    phi  :: ITensors.ITensor,
    M_b  :: ITensors.ITensor,
    M_b1 :: ITensors.ITensor;
    ortho   :: String  = "left",
    maxdim  :: Int     = typemax(Int),
    mindim  :: Int     = 1,
    cutoff  :: Float64 = 0.0,
    tags    = nothing,
    kwargs...,
)
    ortho in ("left", "right") || error("itensor_aliased_factorize: unknown ortho=$ortho")
    phi_w = ITensors.get_external_storage(phi)::WrappedAliasedBlockSparse
    Bw    = ITensors.get_external_storage(M_b)::WrappedAliasedBlockSparse
    B1w   = ITensors.get_external_storage(M_b1)::WrappedAliasedBlockSparse
    # TODO: key-space SVD + re-merge for an off-manifold φ (one window_pairs rejects);
    # no exact split into the fixed P exists for it.
    pairs = window_pairs(phi_w, Bw, B1w)
    aphi = phi_w.aliased
    Ab, Ab1 = Bw.aliased, B1w.aliased
    Tel = promote_type(eltype(aphi.templates), eltype(Ab.templates), eltype(Ab1.templates))

    # ---- tails: M_b = (L_other..., mult), M_b1 = (mult, R_other...) ------------
    Pb, Pb1, Pphi = _abs_head_len(Bw), _abs_head_len(B1w), _abs_head_len(phi_w)
    tail_b  = collect(Bw.inds[Pb+1:end]);   tail_b1 = collect(B1w.inds[Pb1+1:end])
    mu = [I for I in tail_b if I in tail_b1]
    length(mu) == 1 || error("itensor_aliased_factorize: expected one shared multiplicity " *
                             "index in the dense tails, found $(length(mu))")
    mult_old = only(mu)
    jb, jb1 = findfirst(==(mult_old), tail_b), findfirst(==(mult_old), tail_b1)
    Lo = [I for I in tail_b  if I != mult_old]
    Ro = [I for I in tail_b1 if I != mult_old]
    nLo = prod(ITensors.dim.(Lo); init = 1); nRo = prod(ITensors.dim.(Ro); init = 1)
    tail_phi = collect(phi_w.inds[Pphi+1:end])
    (Set(tail_phi) == Set([Lo; Ro]) && length(tail_phi) == length(Lo) + length(Ro)) ||
        error("itensor_aliased_factorize: φ's dense tail is not (M_b tail, M_b1 tail) " *
              "minus the shared multiplicity index")
    perm_phi = [findfirst(==(I), tail_phi) for I in [Lo; Ro]]
    dims_phi = Tuple(ITensors.dim.(tail_phi))

    # ---- M_red: φ templates placed by (a_L, a_R), unscaled ---------------------
    M_red  = zeros(Tel, Ab.n_templates * nLo, Ab1.n_templates * nRo)
    filled = falses(Ab.n_templates, Ab1.n_templates)
    for (tid, (a_L, a_R)) in pairs
        blk = aphi.templates[(tid - 1) * aphi.blksize + 1 : tid * aphi.blksize]
        Tm  = reshape(permutedims(reshape(blk, dims_phi...), perm_phi), nLo, nRo)
        rows = (a_L - 1) * nLo + 1 : a_L * nLo
        cols = (a_R - 1) * nRo + 1 : a_R * nRo
        if filled[a_L, a_R]
            isapprox(M_red[rows, cols], Tm; rtol = 1e-12, atol = 1e-14 * max(norm(Tm), 1)) ||
                error("itensor_aliased_factorize: templates ($a_L, $a_R) receive two " *
                      "different φ blocks; φ does not have M_b·M_b1's structure")
        else
            M_red[rows, cols] .= Tm; filled[a_L, a_R] = true
        end
    end

    # ---- SVD + truncation -----------------------------------------------------
    F = svd(M_red)
    sv = F.S
    svmax = isempty(sv) ? zero(eltype(sv)) : sv[1]
    n_rank = svmax > 0 ? count(s -> s > 1e-12 * svmax, sv) : length(sv)
    n_keep = min(length(sv), maxdim, n_rank)
    if cutoff > 0
        total = sum(s -> s * s, sv); running = 0.0
        for k in length(sv):-1:1
            running += sv[k]^2
            if running > cutoff * total
                n_keep = min(n_keep, k); break
            end
        end
    end
    n_keep = clamp(n_keep, min(mindim, length(sv)), length(sv))
    k = max(n_keep, 1)
    Uk, Vtk, Sk = F.U[:, 1:k], F.V[:, 1:k]', sv[1:k]
    Lmat, Rmat = ortho == "left" ? (Uk, Diagonal(Sk) * Vtk) : (Uk * Diagonal(Sk), Vtk)

    # ---- write back into each side's templates --------------------------------
    new_tail_b  = Int[ITensors.dim(I) for I in tail_b];  new_tail_b[jb]   = k
    new_tail_b1 = Int[ITensors.dim(I) for I in tail_b1]; new_tail_b1[jb1] = k
    perm_b  = [setdiff(1:length(tail_b), jb); jb]          # (L_other..., m)
    perm_b1 = [jb1; setdiff(1:length(tail_b1), jb1)]       # (m, R_other...)
    blk_b, blk_b1 = prod(new_tail_b), prod(new_tail_b1)
    tmpl_b  = zeros(Tel, Ab.n_templates  * blk_b)
    tmpl_b1 = zeros(Tel, Ab1.n_templates * blk_b1)
    for a in 1:Ab.n_templates
        arr = permutedims(reshape(Lmat[(a - 1) * nLo + 1 : a * nLo, :], new_tail_b[perm_b]...),
                          invperm(perm_b))
        tmpl_b[(a - 1) * blk_b + 1 : a * blk_b] .= vec(arr)
    end
    for a in 1:Ab1.n_templates
        arr = permutedims(reshape(Rmat[:, (a - 1) * nRo + 1 : a * nRo], new_tail_b1[perm_b1]...),
                          invperm(perm_b1))
        tmpl_b1[(a - 1) * blk_b1 + 1 : a * blk_b1] .= vec(arr)
    end
    new_mult = ITensors.Index(k; tags = something(tags, ITensors.tags(mult_old)))
    L = _build_aliased_frozen_schema(Bw,  tmpl_b,  new_tail_b,  Pb + jb,   k, true, Tel, new_mult)
    R = _build_aliased_frozen_schema(B1w, tmpl_b1, new_tail_b1, Pb1 + jb1, k, true, Tel, new_mult)

    _sv_tot = sum(s -> s * s, sv)
    truncerr = (k >= length(sv) || _sv_tot <= 0) ? 0.0 : sum(s -> s * s, sv[k + 1:end]) / _sv_tot
    return L, R, ITensors.Spectrum(abs2.(Sk), truncerr)
end

# Construct a new aliased ITensor whose schema (keys, alias_ids, scalars,
# n_templates, channel-bond axis) is verbatim from M_w. The bond-multiplicity
# axis dim (if present) is replaced by mult_new; templates is the freshly
# computed numeric payload.
function _build_aliased_frozen_schema(
    M_w           :: WrappedAliasedBlockSparse,
    templates_new :: Vector{T},
    new_tail_dims :: Vector{Int},
    mu_pos        :: Int,
    mult_new      :: Int,
    has_mu        :: Bool,
    ::Type{T},
    shared_new_mult_ind :: Union{Nothing, ITensors.Index} = nothing,
) where {T}
    am = M_w.aliased
    P  = _abs_head_len(M_w)
    N  = ndims(am)
    N2 = N - P
    new_dims = ntuple(i -> i <= P ? am.dims[i] : new_tail_dims[i - P], Val(N))
    new_blksize = isempty(new_tail_dims) ? 1 : prod(new_tail_dims)
    @assert length(templates_new) == am.n_templates * new_blksize "templates length mismatch ($(length(templates_new)) vs $(am.n_templates) * $(new_blksize))"

    Kt = eltype(eltype(am.keys))
    ali_new = AliasedBlockSparse{T, N, N2, P, Kt}(
        new_dims, new_blksize,
        templates_new, am.n_templates,
        copy(am.keys), copy(am.alias_ids), T.(am.scalars),
    )
    # FROZEN schema, as the name says: keys, alias_ids and n_templates all come
    # across verbatim and only the dense tail (the multiplicity bond) is rebuilt.
    # Template ids therefore still denote the same slots, so the rv -> tid routing
    # survives and must be carried; leaving it empty made read_core guess, which is
    # a silent permutation for a flip P. Each of L and R is built by its own call
    # with its own M_w, so each correctly inherits its own input's map.
    isempty(am.slice_to_template) || (ali_new.slice_to_template = copy(am.slice_to_template))
    # Update the bond-multiplicity ind. Use the caller-provided shared Index
    # so L and R share the same id on the new bond.
    new_inds = collect(M_w.inds)
    if has_mu
        if shared_new_mult_ind !== nothing
            new_inds[mu_pos] = shared_new_mult_ind
        else
            old = new_inds[mu_pos]
            new_inds[mu_pos] = ITensors.Index(mult_new; tags = ITensors.tags(old))
        end
    end
    w = WrappedAliasedBlockSparse{T, N, N2, P}(ali_new, Tuple(new_inds))
    return ITensors._itensor_from_external_storage(w)
end

# Single-site factorize of an aliased tensor (the `factorize` hook, used by
# orthogonalize! and so by TEBD). Only the multiplicity axis is refactorized.
#
# Every block is scalars[i] * templates[alias_ids[i]], so one matrix acting on the
# multiplicity axis the same way for every channel is the only change the frozen
# schema can represent. With X = A:
#     W = vcat_a sqrt(w_a) * T_a      (rows: template a × other tail, cols: m)
#     w_a = Σ_{i : alias_ids[i] = a} |scalars[i]|²    so  W'W = Σ_blocks B'B
# svd(W) = U S V'. L is X with template a replaced by U_a / sqrt(w_a)
# (weighted-isometric); R = S V' is a dense (new m, old m) matrix for the caller to
# contract into the neighbour (`M[b+1] *= R`). L·R == A exactly whenever
# k ≥ rank(W): a direction with W v = 0 has T_a v = 0 for every referenced template.
# L keeps A's keys, alias_ids, scalars, n_templates, slice_to_template and the
# bond's channel index; only the multiplicity index is replaced.
function ITensors._external_factorize_storage(
    Aw :: WrappedAliasedBlockSparse,
    A  :: ITensors.ITensor,
    Linds...;
    mindim = nothing, maxdim = nothing, cutoff = nothing, ortho = nothing,
    eigen_perturbation = nothing, kwargs...,
)
    something(ortho, "left") == "left" ||
        error("aliased factorize: only ortho=\"left\" is supported (got $ortho)")
    eigen_perturbation === nothing ||
        error("aliased factorize: eigen_perturbation is not supported")
    maxdim = something(maxdim, typemax(Int))
    mindim = something(mindim, 1)
    cutoff = Float64(something(cutoff, 0.0))

    X  = Aw.aliased
    Px = _abs_head_len(Aw)
    Lis = ITensors.commoninds(A, ITensors.indices(Linds...))
    mus = [p for p in (Px + 1):ndims(X) if !(Aw.inds[p] in Lis)]
    length(mus) == 1 || error("aliased factorize: expected exactly one multiplicity " *
        "(dense-tail) index outside Linds, found $(length(mus))")
    mx = mus[1]
    old_mult = Aw.inds[mx]
    Tel = eltype(X.templates)

    # ---- stack weighted templates as (template × other tail) × m ------------
    tail_x = Int[X.dims[i] for i in (Px + 1):ndims(X)]
    jx = mx - Px
    m_old = tail_x[jx]
    nrest = X.blksize ÷ m_old
    perm_x = [setdiff(1:length(tail_x), jx); jx]          # m axis last
    w = zeros(real(Tel), X.n_templates)
    @inbounds for i in eachindex(X.alias_ids)
        w[X.alias_ids[i]] += abs2(X.scalars[i])
    end
    W = zeros(Tel, X.n_templates * nrest, m_old)
    for a in 1:X.n_templates
        Ta = reshape(permutedims(reshape(X.templates[(a - 1) * X.blksize + 1 : a * X.blksize],
                                         tail_x...), perm_x), nrest, m_old)
        W[(a - 1) * nrest + 1 : a * nrest, :] .= sqrt(w[a]) .* Ta
    end

    # ---- SVD + truncation ----------------------------------------------------
    F = svd(W)
    sv = F.S
    # Same rule as ITensors' truncate! (route C, dense): cutoff = 0 drops only exact
    # zeros; anything else is removed by maxdim or a nonzero cutoff below.
    n_rank = count(s -> s > 0, sv)
    n_keep = min(length(sv), maxdim, n_rank)
    if cutoff > 0
        total = sum(s -> s * s, sv); running = 0.0
        for k in length(sv):-1:1
            running += sv[k]^2
            if running > cutoff * total
                n_keep = min(n_keep, k); break
            end
        end
    end
    n_keep = clamp(n_keep, min(mindim, length(sv)), length(sv))
    k_new = max(n_keep, 1)

    # ---- L: templates U_a / sqrt(w_a), m axis -> k_new ------------------------
    new_tail_x = copy(tail_x); new_tail_x[jx] = k_new
    blk_x = prod(new_tail_x)
    tmpl_x = zeros(Tel, X.n_templates * blk_x)
    iperm_x = invperm(perm_x)
    for a in 1:X.n_templates
        w[a] > 0 || continue                               # unreferenced / zero-scalar
        Ua = F.U[(a - 1) * nrest + 1 : a * nrest, 1:k_new] ./ sqrt(w[a])
        arr = permutedims(reshape(Ua, new_tail_x[perm_x]...), iperm_x)
        tmpl_x[(a - 1) * blk_x + 1 : a * blk_x] .= vec(arr)
    end
    new_mult = ITensors.Index(k_new; tags = ITensors.tags(old_mult))
    L = _build_aliased_frozen_schema(Aw, tmpl_x, new_tail_x, mx, k_new, true, Tel, new_mult)

    # ---- R = S V' over (new m, old m) ----------------------------------------
    SVt = Matrix{Tel}(Diagonal(sv[1:k_new]) * F.V[:, 1:k_new]')
    R = ITensors.ITensor(SVt, new_mult, old_mult)

    _sv_tot = sum(s -> s * s, sv)
    truncerr_val = (k_new >= length(sv) || _sv_tot <= 0) ? 0.0 :
                   sum(s -> s * s, sv[k_new + 1:end]) / _sv_tot
    spec = ITensors.Spectrum(abs2.(sv[1:k_new]), truncerr_val)
    return L, R, spec, new_mult
end

# TODO: add a single-site block-sparse factorize once the aliased one is validated.
function ITensors._external_factorize_storage(Aw::WrappedBlockSparse, A::ITensors.ITensor,
                                              Linds...; kwargs...)
    error("factorize: block-sparse (non-aliased) storage is not supported yet")
end
