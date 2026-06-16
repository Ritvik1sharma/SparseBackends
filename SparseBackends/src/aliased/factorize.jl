# aliased/factorize.jl
#
# Tier-2 aliased factorize via alias-reduced SVD.
#
# Algebra: with (keys, alias_ids, scalars) frozen on both sides, each phi
# block at (i_L, i_R) pair satisfies
#     phi[i_phi] / (scalars_L[i_L] * scalars_R[i_R])
#       = templates_L[a_L(i_L)] @ templates_R[a_R(i_R)]    (contracted on bond)
# The right side depends only on (a_L, a_R, L_other_tail_coords,
# R_other_tail_coords). So we accumulate the "reduced" phi data into a small
# matrix indexed by (a_L × L_other_tail) × (a_R × R_other_tail), SVD it, and
# read off templates_L / templates_R. L's and R's keys, alias_ids, scalars,
# and channel-bond axis are all inherited from M_b / M_b1 verbatim; only the
# multiplicity-bond dim and the templates change.

import ITensors
using LinearAlgebra: svd, Diagonal

# SB_WHITEN_DIAG instrumentation: budgeted per-call print + running max of the
# relative c-spread of w[a,c]² (the whitening c-constant assumption check).
const _WHITEN_BUDGET = Ref{Int}(30)
const _WHITEN_MAXDEV = Ref{Float64}(0.0)

function itensor_aliased_factorize(
    phi  :: ITensors.ITensor,
    M_b  :: ITensors.ITensor,
    M_b1 :: ITensors.ITensor;
    ortho   :: String  = "left",
    maxdim  :: Int     = typemax(Int),
    mindim  :: Int     = 1,
    cutoff  :: Float64 = 0.0,
    kwargs...,
)
    @assert ITensors.has_external_storage(phi)
    @assert ITensors.has_external_storage(M_b)
    @assert ITensors.has_external_storage(M_b1)
    phi_w  = ITensors.get_external_storage(phi)::WrappedAliasedBlockSparse
    M_b_w  = ITensors.get_external_storage(M_b)::WrappedAliasedBlockSparse
    M_b1_w = ITensors.get_external_storage(M_b1)::WrappedAliasedBlockSparse
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        println("[SB_ALIASED_TRACE itensor_aliased_factorize] entered  ortho=$ortho")
    end
    L_R_spec = _aliased_alias_reduced_factorize(
        phi_w, M_b_w, M_b1_w;
        ortho, maxdim, mindim, cutoff)
    if get(ENV, "SB_ALIASED_TRACE", "0") == "1"
        L_, R_, _ = L_R_spec
        println("[SB_ALIASED_TRACE itensor_aliased_factorize] L storage=", typeof(L_.tensor.data),
                "  R storage=", typeof(R_.tensor.data))
    end
    return L_R_spec
end

# Identify shared inds between M_b and M_b1, classified as :channel (sparse
# prefix on M_b's side) or :multiplicity (dense tail). Returns Vector of
# (pos_in_M_b, pos_in_M_b1, kind).
function _identify_bond_axes(M_b_w::WrappedAliasedBlockSparse,
                              M_b1_w::WrappedAliasedBlockSparse)
    P_b = _abs_head_len(M_b_w)
    out = Tuple{Int,Int,Symbol}[]
    for (i, Ib) in enumerate(M_b_w.inds)
        for (j, Ib1) in enumerate(M_b1_w.inds)
            if Ib == Ib1
                kind = (i <= P_b) ? :channel : :multiplicity
                push!(out, (i, j, kind))
                break
            end
        end
    end
    return out
end

# Column-major linear index over coords with given dims; 1-based.
function _lin_col_major(coords::AbstractVector{Int}, dims::AbstractVector{Int})
    isempty(coords) && return 1
    idx = coords[1]
    stride = 1
    for j in 2:length(coords)
        stride *= dims[j-1]
        idx += (coords[j] - 1) * stride
    end
    return idx
end

# Decode a column-major linear index back into coords.
function _decode_col_major!(coords::AbstractVector{Int}, lin::Int, dims::AbstractVector{Int})
    rem = lin - 1
    for j in 1:length(dims)
        coords[j] = (rem % dims[j]) + 1
        rem ÷= dims[j]
    end
    return coords
end

function _aliased_alias_reduced_factorize(
    phi_w  :: WrappedAliasedBlockSparse,
    M_b_w  :: WrappedAliasedBlockSparse,
    M_b1_w :: WrappedAliasedBlockSparse;
    ortho  :: String,
    maxdim :: Int,
    mindim :: Int,
    cutoff :: Float64,
)
    am_b   = M_b_w.aliased
    am_b1  = M_b1_w.aliased
    am_phi = phi_w.aliased
    Tel = promote_type(eltype(am_b.templates), eltype(am_b1.templates), eltype(am_phi.templates))

    # ---- bond axes ----------------------------------------------------------
    shared = _identify_bond_axes(M_b_w, M_b1_w)
    @assert !isempty(shared) "M_b and M_b1 share no inds"
    ch_pos_b  = [p for (p,_,k) in shared if k === :channel]
    ch_pos_b1 = [q for (_,q,k) in shared if k === :channel]
    mu_pos_b  = [p for (p,_,k) in shared if k === :multiplicity]
    mu_pos_b1 = [q for (_,q,k) in shared if k === :multiplicity]
    @assert length(ch_pos_b) == 1  "expected exactly one shared channel axis (sparse prefix)"
    @assert length(mu_pos_b) <= 1 "expected at most one shared multiplicity axis (dense tail)"

    cp_b  = ch_pos_b[1]
    cp_b1 = ch_pos_b1[1]
    has_mu = !isempty(mu_pos_b)
    mp_b  = has_mu ? mu_pos_b[1]  : 0
    mp_b1 = has_mu ? mu_pos_b1[1] : 0

    P_b   = _abs_head_len(M_b_w);  N_b   = ndims(am_b);   N2_b  = N_b  - P_b
    P_b1  = _abs_head_len(M_b1_w); N_b1  = ndims(am_b1);  N2_b1 = N_b1 - P_b1
    P_phi = _abs_head_len(phi_w);  N_phi = ndims(am_phi); N2_phi = N_phi - P_phi

    # ---- L_other / R_other classification on the dense tail ----------------
    # L's tail (within M_b) minus the shared multiplicity axis (if present).
    L_other_pos_b  = [P_b  + i for i in 1:N2_b  if (has_mu ? (P_b  + i != mp_b)  : true)]
    R_other_pos_b1 = [P_b1 + i for i in 1:N2_b1 if (has_mu ? (P_b1 + i != mp_b1) : true)]
    L_other_dims = Int[am_b.dims[p]  for p in L_other_pos_b]
    R_other_dims = Int[am_b1.dims[p] for p in R_other_pos_b1]
    n_L_other = prod(L_other_dims; init=1)
    n_R_other = prod(R_other_dims; init=1)

    bond_mult_old = has_mu ? am_b.dims[mp_b] : 1
    bond_ch_dim   = am_b.dims[cp_b]

    L_other_inds = ITensors.Index[M_b_w.inds[p]  for p in L_other_pos_b]
    R_other_inds = ITensors.Index[M_b1_w.inds[p] for p in R_other_pos_b1]

    # ---- phi tail position → (side, local index) ---------------------------
    phi_tail_origin = Vector{Tuple{Symbol,Int}}(undef, N2_phi)
    for j in 1:N2_phi
        I = phi_w.inds[P_phi + j]
        li = findfirst(==(I), L_other_inds)
        if li !== nothing
            phi_tail_origin[j] = (:L, li); continue
        end
        ri = findfirst(==(I), R_other_inds)
        if ri !== nothing
            phi_tail_origin[j] = (:R, ri); continue
        end
        error("phi tail ind $I not in L_other or R_other inds")
    end
    phi_tail_dims = Int[am_phi.dims[P_phi + j] for j in 1:N2_phi]

    # ---- phi prefix position → (side, M_b or M_b1 axis position) -----------
    nonshared_pos_b  = [i for i in 1:P_b  if i != cp_b]
    nonshared_pos_b1 = [j for j in 1:P_b1 if j != cp_b1]
    nonshared_inds_b  = [M_b_w.inds[p]  for p in nonshared_pos_b]
    nonshared_inds_b1 = [M_b1_w.inds[p] for p in nonshared_pos_b1]
    phi_prefix_origin = Vector{Tuple{Symbol,Int}}(undef, P_phi)
    for i in 1:P_phi
        I = phi_w.inds[i]
        bi = findfirst(==(I), nonshared_inds_b)
        if bi !== nothing
            phi_prefix_origin[i] = (:b, nonshared_pos_b[bi]); continue
        end
        b1i = findfirst(==(I), nonshared_inds_b1)
        if b1i !== nothing
            phi_prefix_origin[i] = (:b1, nonshared_pos_b1[b1i]); continue
        end
        error("phi prefix ind $I not in M_b or M_b1 non-shared prefix")
    end

    # Pre-compute per-phi-prefix-axis: which (b/b1) and which axis position.
    # Build look-up vectors keyed by M_b's axis positions → phi prefix axis index.
    phi_idx_for_b_pos  = Dict{Int,Int}()
    phi_idx_for_b1_pos = Dict{Int,Int}()
    for (i_phi, (side, pos)) in enumerate(phi_prefix_origin)
        if side === :b;  phi_idx_for_b_pos[pos]  = i_phi; end
        if side === :b1; phi_idx_for_b1_pos[pos] = i_phi; end
    end

    # ---- key lookups -------------------------------------------------------
    M_b_lookup  = Dict(Tuple(k) => i for (i, k) in enumerate(am_b.keys))
    M_b1_lookup = Dict(Tuple(k) => i for (i, k) in enumerate(am_b1.keys))

    # ---- accumulate alias-reduced matrix M_red ------------------------------
    # Rows indexed by (a_L, L_other_lin) → row = (a_L-1)*n_L_other + L_other_lin
    # Cols indexed by (a_R, R_other_lin) → col = (a_R-1)*n_R_other + R_other_lin
    n_tL = am_b.n_templates
    n_tR = am_b1.n_templates
    nrows = n_tL * n_L_other * bond_mult_old
    ncols = n_tR * n_R_other * bond_mult_old
    # The bond multiplicity dim was contracted away in phi; it does NOT
    # appear in M_red. M_red shape: (n_tL * n_L_other, n_tR * n_R_other).
    # (After SVD, mult_new replaces bond_mult_old.)
    M_red  = zeros(Tel, n_tL * n_L_other, n_tR * n_R_other)
    counts = zeros(Int,  n_tL * n_L_other, n_tR * n_R_other)
    # SB_MRED_DIAG: on-manifold check for the eigensolver's φ. If φ = P × psi_dense
    # (the aliased manifold), then for a fixed environment template-pair the
    # per-CHANNEL contribution vectors are exact scalar multiples (the projector P
    # factors out, leaving the shared psi_dense) → channel-pair cosine = ±1. If the
    # eigsolve drifted φ off-manifold, channels are no longer proportional → cosine
    # < 1, and re-aliasing must lose energy. We accumulate per-channel matrices
    # M_red_per_c[:, :, c] and report the pairwise channel cosine distribution.
    _mred_diag = get(ENV, "SB_MRED_DIAG", "0") == "1"
    M_red_per_c = _mred_diag ? zeros(Tel, size(M_red, 1), size(M_red, 2), bond_ch_dim) :
                               zeros(Tel, 0, 0, 0)

    L_other_buf = zeros(Int, length(L_other_dims))
    R_other_buf = zeros(Int, length(R_other_dims))
    phi_tail_buf = zeros(Int, N2_phi)

    for i_phi in 1:length(am_phi.keys)
        K_phi = am_phi.keys[i_phi]
        a_phi = am_phi.alias_ids[i_phi]
        sc_phi = am_phi.scalars[i_phi]
        tmpl_off_phi = (a_phi - 1) * am_phi.blksize

        for c in 1:bond_ch_dim
            # Build M_b key (channel = c, non-shared coords from K_phi).
            kb = Vector{Int}(undef, P_b)
            for pos in nonshared_pos_b
                iphi = phi_idx_for_b_pos[pos]
                kb[pos] = K_phi[iphi]
            end
            kb[cp_b] = c
            i_L = get(M_b_lookup, Tuple(kb), 0)
            i_L == 0 && continue

            kb1 = Vector{Int}(undef, P_b1)
            for pos in nonshared_pos_b1
                iphi = phi_idx_for_b1_pos[pos]
                kb1[pos] = K_phi[iphi]
            end
            kb1[cp_b1] = c
            i_R = get(M_b1_lookup, Tuple(kb1), 0)
            i_R == 0 && continue

            a_L  = am_b.alias_ids[i_L]
            a_R  = am_b1.alias_ids[i_R]
            sc_L = am_b.scalars[i_L]
            sc_R = am_b1.scalars[i_R]
            denom = sc_L * sc_R

            for phi_tail_lin in 1:max(am_phi.blksize, 1)
                _decode_col_major!(phi_tail_buf, phi_tail_lin, phi_tail_dims)
                fill!(L_other_buf, 1); fill!(R_other_buf, 1)
                for j in 1:N2_phi
                    side, local_idx = phi_tail_origin[j]
                    if side === :L
                        L_other_buf[local_idx] = phi_tail_buf[j]
                    else
                        R_other_buf[local_idx] = phi_tail_buf[j]
                    end
                end
                L_lin = _lin_col_major(L_other_buf, L_other_dims)
                R_lin = _lin_col_major(R_other_buf, R_other_dims)
                row = (a_L - 1) * n_L_other + L_lin
                col = (a_R - 1) * n_R_other + R_lin
                phi_val = sc_phi * am_phi.templates[tmpl_off_phi + phi_tail_lin]
                _vadd = phi_val / denom
                M_red[row, col] += _vadd
                _mred_diag && (M_red_per_c[row, col, c] += _vadd)
                counts[row, col] += 1
            end
        end
    end

    if _mred_diag
        # Per-channel ON-MANIFOLD check: cosine between channel matrices M_c, M_c'.
        # |cos| ≈ 1 for every populated pair ⇒ all channels proportional ⇒ φ lies on
        # the P×psi_dense manifold (loss only sign/scale). |cos| < 1 ⇒ φ drifted off
        # the manifold (the eigensolve produced channel-dependent multiplicity that a
        # shared psi_dense cannot hold) ⇒ re-aliasing is intrinsically lossy.
        nrm = zeros(real(Tel), bond_ch_dim)
        @inbounds for c in 1:bond_ch_dim
            s = 0.0
            for col in 1:size(M_red_per_c, 2), row in 1:size(M_red_per_c, 1)
                s += abs2(M_red_per_c[row, col, c])
            end
            nrm[c] = sqrt(s)
        end
        mincos = 1.0; meancos = 0.0; npair = 0; nmisaligned = 0
        minoverlap = 1.0   # min fraction of shared-support entries among pairs
        @inbounds for c1 in 1:bond_ch_dim, c2 in (c1+1):bond_ch_dim
            (nrm[c1] == 0 || nrm[c2] == 0) && continue
            ip = zero(Tel); nz1 = 0; nz2 = 0; nzboth = 0
            for col in 1:size(M_red_per_c, 2), row in 1:size(M_red_per_c, 1)
                a = M_red_per_c[row, col, c1]; b = M_red_per_c[row, col, c2]
                ip += conj(a) * b
                a != 0 && (nz1 += 1); b != 0 && (nz2 += 1)
                (a != 0 && b != 0) && (nzboth += 1)
            end
            cosv = abs(ip) / (nrm[c1] * nrm[c2])
            ov = min(nz1, nz2) > 0 ? nzboth / min(nz1, nz2) : 0.0  # overlap of supports
            npair += 1; meancos += cosv
            cosv < mincos && (mincos = cosv)
            ov < minoverlap && (minoverlap = ov)
            cosv < 0.99 && (nmisaligned += 1)
        end
        println("[MRED_DIAG] channels=", bond_ch_dim, "  pairs=", npair,
                "  min|cos|=", round(mincos; sigdigits=4),
                "  mean|cos|=", npair > 0 ? round(meancos/npair; sigdigits=4) : 1.0,
                "  frac(|cos|<0.99)=", npair > 0 ? round(nmisaligned/npair; digits=3) : 0.0,
                "  min-support-overlap=", npair > 0 ? round(minoverlap; sigdigits=3) : 1.0,
                "   (cos→1 with overlap→1 ⇒ genuinely proportional, not disjoint)")
        # Add-path tally since the previous factorize (≈ this bond's eigsolve adds).
        println("[ADD_DIAG] since-last: axpy_match=", _ADD_AXPY_MATCH[],
                "  plus_match=", _ADD_PLUS_MATCH[],
                "  plus_merge=", _ADD_PLUS_MERGE[], "  plus_dense=", _ADD_PLUS_DENSE[],
                "   (merge/dense are EXACT but drop compression; all paths value-exact)")
        _ADD_AXPY_MATCH[] = 0; _ADD_PLUS_MATCH[] = 0
        _ADD_PLUS_MERGE[] = 0; _ADD_PLUS_DENSE[] = 0
    end
    @inbounds for i in eachindex(counts)
        if counts[i] > 1
            M_red[i] /= counts[i]
        end
    end

    # ---- WHITEN: account for block-multiplicity weighting so L,R are iso ----
    # Iso condition on L: PER bond-channel value c,
    #   sum_a w[a, c]² · template[a] · template[a]† = I  (on mult_new axis)
    # which requires w[a, c]² to be c-independent (an inherent property of
    # the alias structure on projector-derived psi).  Then W_L[a] := w[a, c]
    # is the correct whitening factor.
    #
    # Compute w[a, c]² per (alias_id, channel) and verify it's c-constant;
    # use w[a, c=1] (or whatever the constant value is) as W_L[a].
    bond_ch_dim_b  = bond_ch_dim                # M_b's channel dim at the bond
    bond_ch_dim_b1 = am_b1.dims[cp_b1]          # M_b1's channel dim at the bond
    wL_per_c = zeros(real(Tel), n_tL, bond_ch_dim_b)
    @inbounds for i_L in eachindex(am_b.keys)
        a  = am_b.alias_ids[i_L]
        c  = am_b.keys[i_L][cp_b]
        wL_per_c[a, c] += abs2(am_b.scalars[i_L])
    end
    wL = [sqrt(maximum(@view wL_per_c[a, :])) for a in 1:n_tL]
    wR_per_c = zeros(real(Tel), n_tR, bond_ch_dim_b1)
    @inbounds for i_R in eachindex(am_b1.keys)
        a  = am_b1.alias_ids[i_R]
        c  = am_b1.keys[i_R][cp_b1]
        wR_per_c[a, c] += abs2(am_b1.scalars[i_R])
    end
    wR = [sqrt(maximum(@view wR_per_c[a, :])) for a in 1:n_tR]
    # SB_WHITEN_DIAG: verify the whitening assumption that w[a,c]² is c-independent.
    # The whitening (wL = sqrt(max_c w[a,c]²)) and the channel-averaged M_red are only
    # EXACT if, for each template a, w[a,c]² is the same across all channels c it appears
    # in. If FP drift (or approximate aliasing) makes it vary, the SVD truncates a
    # distorted M_red → suboptimal energy. Report the worst relative c-spread.
    if get(ENV, "SB_WHITEN_DIAG", "0") == "1" && _WHITEN_BUDGET[] != 0
        maxdev = 0.0
        @inbounds for a in 1:n_tL
            mn = Inf; mx = 0.0
            for c in 1:bond_ch_dim_b
                v = wL_per_c[a, c]
                v > 0 && (mn = min(mn, v); mx = max(mx, v))
            end
            mx > 0 && mn < Inf && (maxdev = max(maxdev, (mx - mn) / mx))
        end
        @inbounds for a in 1:n_tR
            mn = Inf; mx = 0.0
            for c in 1:bond_ch_dim_b1
                v = wR_per_c[a, c]
                v > 0 && (mn = min(mn, v); mx = max(mx, v))
            end
            mx > 0 && mn < Inf && (maxdev = max(maxdev, (mx - mn) / mx))
        end
        _WHITEN_MAXDEV[] = max(_WHITEN_MAXDEV[], maxdev)
        println("[WHITEN_DIAG] max rel c-spread of w[a,c]² this call = ", maxdev,
                "   (running max = ", _WHITEN_MAXDEV[], ")  0⇒c-constant (SVD exact); >>0⇒distorted")
        _WHITEN_BUDGET[] > 0 && (_WHITEN_BUDGET[] -= 1)
    end
    # Scale rows and columns of M_red by w_L and w_R (broadcast per alias group).
    @inbounds for a_L in 1:n_tL, lo in 1:n_L_other, a_R in 1:n_tR, ro in 1:n_R_other
        row = (a_L - 1) * n_L_other + lo
        col = (a_R - 1) * n_R_other + ro
        M_red[row, col] *= wL[a_L] * wR[a_R]
    end

    # ---- SVD + truncation ---------------------------------------------------
    F = svd(M_red)
    sv = real.(F.S)
    # Per-cM cap (mirror of the BS Path-B factorize, ops_factorize_qr.jl:471).
    # The new bond is channel × multiplicity (doubled-link convention), so the
    # HONEST bond dim = bond_ch_dim × mult_new. To bound it at `maxdim`, the
    # multiplicity (= n_keep) must be capped at fld(maxdim, bond_ch_dim), NOT at
    # `maxdim` — otherwise the channel dim multiplies through and the honest bond
    # inflates (e.g. 4×4 = 16 > maxdim=4). This matches the BS per-cM cap exactly;
    # L stays non-iso (structural) and Path-B's M⁻¹ corrects it, same as BS.
    # Gated SB_ALIASED_PERCM_CAP. DEFAULT OFF (="0") = UNCAPPED multiplicity
    # (mult = maxdim, honest_bd = channel×maxdim). This is the regime we study and
    # benchmark: uncapped aliased Pareto-dominates dense on memory at iso-energy.
    # The cap (="1") starves multiplicity (mult = fld(maxdim, channel)) → honest_bd
    # ≤ maxdim like BS, but gives much worse energy per maxdim. Hardened OFF
    # 2026-06 after the capped default silently mismatched the benchmarked
    # (uncapped) working size — do NOT flip back without updating every runner.
    mult_cap = get(ENV, "SB_ALIASED_PERCM_CAP", "0") == "1" ?   # default OFF = uncapped
               max(1, fld(maxdim, max(1, bond_ch_dim))) : maxdim
    n_keep = min(length(sv), mult_cap)
    if cutoff > 0
        total = sum(s -> s*s, sv)
        running = 0.0
        for k in length(sv):-1:1
            running += sv[k]^2
            if running > cutoff * total
                n_keep = min(n_keep, k); break
            end
        end
    end
    n_keep = clamp(n_keep, mindim, length(sv))
    mult_new = max(n_keep, 1)

    Uk = F.U[:, 1:mult_new]
    Vk = F.V[:, 1:mult_new]
    Sk = sv[1:mult_new]
    # ortho convention:
    #   left  : L = U,     R = Σ V†  (L is left-iso, SVs go right)
    #   right : L = U Σ,   R = V†    (R is right-iso, SVs go left)
    L_block, R_block = if ortho == "left"
        Uk, (Diagonal(complex.(Sk)) * Vk')
    else
        (Uk * Diagonal(complex.(Sk))), Vk'
    end
    # L_block: (nrows × mult_new); R_block: (mult_new × ncols)

    # ---- assemble new L's templates ----------------------------------------
    # L's new tail dims = M_b's tail dims, but with bond-multiplicity axis dim
    # replaced by mult_new. Other tail dims unchanged. Column-major layout.
    new_L_tail_dims = Vector{Int}(undef, N2_b)
    for j in 1:N2_b
        pos_in_M = P_b + j
        new_L_tail_dims[j] = (has_mu && pos_in_M == mp_b) ? mult_new : am_b.dims[pos_in_M]
    end
    L_blksize_new = isempty(new_L_tail_dims) ? 1 : prod(new_L_tail_dims)
    templates_L = zeros(Tel, n_tL * L_blksize_new)
    # For each (a_L, L_other_coords, m_new) read U value into the right slot.
    L_other_buf2 = zeros(Int, length(L_other_dims))
    L_tail_buf   = zeros(Int, N2_b)
    for a_L in 1:n_tL
        for L_other_lin in 1:max(n_L_other, 1)
            _decode_col_major!(L_other_buf2, L_other_lin, L_other_dims)
            row = (a_L - 1) * n_L_other + L_other_lin
            inv_wL = wL[a_L] > 0 ? one(Tel) / wL[a_L] : zero(Tel)
            for m in 1:mult_new
                val = L_block[row, m] * inv_wL   # un-whiten: divide by w_L[a_L]
                lo_cursor = 0
                for j in 1:N2_b
                    pos_in_M = P_b + j
                    if has_mu && pos_in_M == mp_b
                        L_tail_buf[j] = m
                    else
                        lo_cursor += 1
                        L_tail_buf[j] = L_other_buf2[lo_cursor]
                    end
                end
                tlin = _lin_col_major(L_tail_buf, new_L_tail_dims)
                templates_L[(a_L - 1) * L_blksize_new + tlin] = val
            end
        end
    end

    # ---- assemble new R's templates ----------------------------------------
    new_R_tail_dims = Vector{Int}(undef, N2_b1)
    for j in 1:N2_b1
        pos_in_M = P_b1 + j
        new_R_tail_dims[j] = (has_mu && pos_in_M == mp_b1) ? mult_new : am_b1.dims[pos_in_M]
    end
    R_blksize_new = isempty(new_R_tail_dims) ? 1 : prod(new_R_tail_dims)
    templates_R = zeros(Tel, n_tR * R_blksize_new)
    R_other_buf2 = zeros(Int, length(R_other_dims))
    R_tail_buf   = zeros(Int, N2_b1)
    for a_R in 1:n_tR
        inv_wR = wR[a_R] > 0 ? one(Tel) / wR[a_R] : zero(Tel)
        for R_other_lin in 1:max(n_R_other, 1)
            _decode_col_major!(R_other_buf2, R_other_lin, R_other_dims)
            col = (a_R - 1) * n_R_other + R_other_lin
            for m in 1:mult_new
                val = R_block[m, col] * inv_wR    # un-whiten: divide by w_R[a_R]
                ro_cursor = 0
                for j in 1:N2_b1
                    pos_in_M = P_b1 + j
                    if has_mu && pos_in_M == mp_b1
                        R_tail_buf[j] = m
                    else
                        ro_cursor += 1
                        R_tail_buf[j] = R_other_buf2[ro_cursor]
                    end
                end
                tlin = _lin_col_major(R_tail_buf, new_R_tail_dims)
                templates_R[(a_R - 1) * R_blksize_new + tlin] = val
            end
        end
    end

    # ---- build new aliased ITensors -----------------------------------------
    # The new bond-multiplicity Index must be SHARED between L and R (same id),
    # otherwise the downstream Aliased×Aliased contract sees them as unrelated.
    new_mult_ind = has_mu ? ITensors.Index(mult_new; tags = ITensors.tags(M_b_w.inds[mp_b])) : nothing
    L_ali = _build_aliased_frozen_schema(M_b_w, templates_L, new_L_tail_dims, mp_b, mult_new, has_mu, Tel, new_mult_ind)
    R_ali = _build_aliased_frozen_schema(M_b1_w, templates_R, new_R_tail_dims, mp_b1, mult_new, has_mu, Tel, new_mult_ind)

    # SB_FACT_DIAG=1: report the NEW bond's channel/multiplicity placement on each
    # side. Canonical convention requires: channel in PREFIX (pos ≤ P), multiplicity
    # in DENSE tail (pos > P), and channel BEFORE multiplicity ("sparse precedes
    # dense"). Flag any violation — this is where a factorize would break the
    # convention and seed the downstream prefix/dense crossover.
    if get(ENV, "SB_FACT_DIAG", "0") == "1"
        _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
        _ch_ok_L = cp_b <= P_b
        _mu_ok_L = !has_mu || mp_b > P_b
        _ord_L   = !has_mu || cp_b < mp_b
        _ch_ok_R = cp_b1 <= P_b1
        _mu_ok_R = !has_mu || mp_b1 > P_b1
        _ord_R   = !has_mu || cp_b1 < mp_b1
        bad = !(_ch_ok_L && _mu_ok_L && _ord_L && _ch_ok_R && _mu_ok_R && _ord_R)
        println("[FACT_DIAG ortho=", ortho, " bond_ch=", _tg(M_b_w.inds[cp_b]),
                has_mu ? string("  mult=", _tg(M_b_w.inds[mp_b])) : "  (no mult yet)", "]")
        println("   L: P=$P_b  ch_pos=$cp_b(", _ch_ok_L ? "prefix✓" : "DENSE✗", ")  ",
                has_mu ? "mu_pos=$mp_b(" * (_mu_ok_L ? "dense✓" : "PREFIX✗") * ")  ch<mu:" * (_ord_L ? "✓" : "✗") : "no-mu")
        println("   R: P=$P_b1  ch_pos=$cp_b1(", _ch_ok_R ? "prefix✓" : "DENSE✗", ")  ",
                has_mu ? "mu_pos=$mp_b1(" * (_mu_ok_R ? "dense✓" : "PREFIX✗") * ")  ch<mu:" * (_ord_R ? "✓" : "✗") : "no-mu")
        bad && println("   ⚠ [FACT BREAKS CANON] this factorize emits a non-canonical bond classification")
        flush(stdout)
    end
    # Discarded weight as a FRACTION of total spectral weight (∑ discarded sv² / ∑ sv²),
    # so it is directly comparable to ITensors' (dense/BS) normalized truncerr. (Previously
    # this reported the absolute ∑ discarded sv², which is NOT comparable across backends.)
    _sv_tot = sum(s -> s*s, sv)
    truncerr_val = (mult_new >= length(sv) || _sv_tot <= 0) ? 0.0 :
                   sum(s -> s*s, sv[mult_new+1:end]) / _sv_tot
    spec = (truncerr = truncerr_val, truncation_error = truncerr_val, eigenvalues = Sk)
    return L_ali, R_ali, spec
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
