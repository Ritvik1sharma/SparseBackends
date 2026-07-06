using LinearAlgebra: qr, svd
using ITensors: Spectrum

# Returns (Q::NewBlockSparseSorted, R::Matrix, l_vals, new_bond_dim)
# A has sparse head = (l, r), dense tail = (s)  [N=3, N2=1, P=2]
# left_inds are the first P_left sparse head dims + all dense dims
# right_inds are the remaining P_right sparse head dims
function blocksparse_left_qr(A::NewBlockSparseSorted{T,3,1,2}) where T
    # collect nonzero l/r values from keys
    l_vals = sort(unique(k[1] for k in A.keys))
    r_vals = sort(unique(k[2] for k in A.keys))
    l_idx  = Dict(v => i for (i,v) in enumerate(l_vals))
    r_idx  = Dict(v => i for (i,v) in enumerate(r_vals))

    d_s = A.dims[3]
    n_l, n_r = length(l_vals), length(r_vals)

    # build dense matrix (n_l*d_s) × n_r
    M = zeros(T, n_l * d_s, n_r)
    for (key, id) in blocks_sorted(A)
        li, ri = l_idx[key[1]], r_idx[key[2]]
        M[(li-1)*d_s+1 : li*d_s, ri] .= _block_view(A, id)
    end

    # thin QR
    F = qr(M)
    k     = min(n_l * d_s, n_r)
    Q_mat = Matrix(F.Q)[:, 1:k]
    R_mat = Matrix(F.R)[1:k, :]

    # build Q: NewBlockSparseSorted{T,3,2,1}  sparse=(l,), dense=(s, new_bond)
    Q_dims = (A.dims[1], d_s, k)
    Q_bs   = NewBlockSparseSorted{T, 3, 2}(Q_dims)
    for (li, l_val) in enumerate(l_vals)
        blk = Q_mat[(li-1)*d_s+1 : li*d_s, :]  # d_s × k
        for b in 1:k, s in 1:d_s
            Q_bs[l_val, s, b] = blk[s, b]       # setindex! creates block on demand
        end
    end

    # R_mat is compact (n_r cols); caller needs to scatter back to full d_r
    return Q_bs, R_mat, l_vals, r_vals
end



function blocksparse_svd(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int = P  ÷ 2,   # # sparse dims sent to U
    n_left_dense::Int  = N2 ÷ 2,   # # dense  dims sent to U
    ortho::String      = "left",
    maxdim::Int        = typemax(Int),
    mindim::Int        = 1,
    cutoff::Float64    = 0.0,
) where {T, N, N2, P, K<:Integer}

    # ---- bipartition bookkeeping ------------------------------------------------
    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    # println("The sparse and dense dims are ", P, " ", N2, " ", n_left_sparse, " ", n_left_dense)
    @assert 0 ≤ nls ≤ P  "n_left_sparse must be in 0:$P, got $nls"
    @assert 0 ≤ nld ≤ N2 "n_left_dense must be in 0:$N2, got $nld"

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],            nls)
    right_sp_dims = ntuple(i -> dims[nls + i],      nrs)
    left_d_dims   = ntuple(i -> dims[P + i],        nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],  nrd)

    d_left  = prod(left_d_dims;  init = 1)
    d_right = prod(right_d_dims; init = 1)

    LKey = NTuple{nls, K}
    RKey = NTuple{nrs, K}

    # ---- bin existing blocks by left sparse key ---------------------------------
    bins = Dict{LKey, Vector{Tuple{RKey,Int}}}()
    for (key, id) in blocks_sorted(A)
        lk = ntuple(i -> key[i],       nls)
        rk = ntuple(i -> key[nls + i], nrs)
        push!(get!(bins, lk, Tuple{RKey,Int}[]), (rk, id))
    end
    l_vals = sort!(collect(keys(bins)))

    # ---- per-left-key mini-SVD --------------------------------------------------
    block_svds = Vector{Tuple{LKey, Matrix{T}, Vector{real(T)}, Matrix{T}, Vector{RKey}}}()
    for lk in l_vals
        entries = bins[lk]
        r_list  = sort!(unique!([rk for (rk, _) in entries]))
        n_Ri    = length(r_list)
        r_pos   = Dict(r => j for (j, r) in enumerate(r_list))

        # Rows: flattened left-dense.  Cols: (rj, flattened right-dense).
        M_i = zeros(T, d_left, n_Ri * d_right)
        for (rk, id) in entries
            rj  = r_pos[rk]
            blk = reshape(_block_view(A, id), d_left, d_right)
            M_i[:, (rj-1)*d_right + 1 : rj*d_right] .= blk
        end

        F = svd(M_i)
        push!(block_svds, (lk, F.U, F.S, F.Vt, r_list))
    end

    # ---- global truncation across all per-block spectra -------------------------
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < cutoff * max_sv
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)

    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    # ---- new bond legs: one sparse (sectors) + one dense (multiplicity) --------
    surviving    = [bi for bi in eachindex(block_svds) if k_kept[bi] > 0]
    n_new_sp     = length(surviving)
    n_new_d      = isempty(surviving) ? 0 : maximum(k_kept[bi] for bi in surviving)
    new_sp_of_bi = Dict(bi => j for (j, bi) in enumerate(surviving))

    # ---- assemble outputs -------------------------------------------------------
    # U  layout: (left_sp..., new_sp,  left_d...,  new_d)   sparse: nls+1, dense: nld+1
    # SV layout: (new_sp, right_sp..., new_d, right_d...)   sparse: 1+nrs, dense: 1+nrd
    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    # singular values laid out as (sector, multiplicity); zeros past k_kept[bi]
    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    for bi in surviving
        (lk, U_i, S_i, Vt_i, r_list) = block_svds[bi]
        ki = k_kept[bi]
        j  = new_sp_of_bi[bi]

        @inbounds for c in 1:ki
            svs_kept[j, c] = S_i[c]
        end

        # ---- U block at (lk..., j); dense shape (left_d..., n_new_d) ----
        id_u  = _ensure_block!(U_bs, (lk..., j))
        u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
        @inbounds for c in 1:ki
            sc = (ortho == "right") ? T(S_i[c]) : one(T)
            for s in 1:d_left
                u_mat[s, c] = U_i[s, c] * sc
            end
        end
        # cols ki+1:n_new_d remain 0

        # ---- SV blocks at (j, rk...) per right key; dense shape (n_new_d, right_d...) ----
        for (rj, rk) in enumerate(r_list)
            id_sv  = _ensure_block!(SV_bs, (j, rk...))
            sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
            base   = (rj - 1) * d_right
            @inbounds for c in 1:ki
                sc = (ortho == "left") ? T(S_i[c]) : one(T)
                for s in 1:d_right
                    sv_mat[c, s] = Vt_i[c, base + s] * sc
                end
            end
            # rows ki+1:n_new_d remain 0
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end


# ─────────────────────────────────────────────────────────────────────────────
# blocksparse_svd_right_binned
#
# Same I/O contract as `blocksparse_svd`, but bins blocks by the RIGHT sparse
# key instead of the left one. Produces a new bond whose sparse dimension equals
# the number of distinct right-sparse-key sectors with data, which matches the
# RIGHT side's original bond structure.
#
# Use this when you want the SVD to PRESERVE the right side's bond labeling
# (e.g., in `orthogonalize!` where the new bond should be isomorphic to the
# original right link of psi[b]).
# ─────────────────────────────────────────────────────────────────────────────
function blocksparse_svd_right_binned(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int     = P  ÷ 2,
    n_left_dense::Int      = N2 ÷ 2,
    n_right_bin_sparse::Int = -1,  # how many of right's sparse axes to bin by.
                                    # default (-1) = all of them (nrs). Pass a smaller
                                    # number to use only the first k right-sparse-axes
                                    # as the bond label (e.g., bin by Link only, with
                                    # remaining right-sparse axes folded into sub-matrix
                                    # columns). This keeps new bond's sparse dim equal
                                    # to the original sparse-link dim.
    target_n_new_sp::Int   = -1,    # if positive, pad new bond's sparse axis up to
                                    # this size (extra slots empty). Mirrors the
                                    # left-binned variant; used at the right boundary
                                    # to match the original M[b]↔M[b+1] sparse dim.
    ortho::String      = "left",
    maxdim::Int        = typemax(Int),
    mindim::Int        = 1,
    cutoff::Float64    = 0.0,
) where {T, N, N2, P, K<:Integer}

    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    @assert 0 ≤ nls ≤ P  "n_left_sparse must be in 0:$P, got $nls"
    @assert 0 ≤ nld ≤ N2 "n_left_dense must be in 0:$N2, got $nld"
    n_bin = n_right_bin_sparse < 0 ? nrs : n_right_bin_sparse
    @assert 0 ≤ n_bin ≤ nrs "n_right_bin_sparse must be in 0:$nrs, got $n_bin"
    n_col_sp = nrs - n_bin  # remaining right-sparse axes that fold into sub-matrix cols

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],                  nls)
    rbin_sp_dims  = ntuple(i -> dims[nls + i],            n_bin)
    rcol_sp_dims  = ntuple(i -> dims[nls + n_bin + i],    n_col_sp)
    right_sp_dims = ntuple(i -> dims[nls + i],            nrs)  # = (rbin..., rcol...)
    left_d_dims   = ntuple(i -> dims[P + i],              nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],        nrd)

    d_left   = prod(left_d_dims;   init = 1)
    d_right  = prod(right_d_dims;  init = 1)
    n_col_sp_dim = prod(rcol_sp_dims; init = 1)  # number of right-sparse-col combinations

    LKey   = NTuple{nls,      K}
    BinKey = NTuple{n_bin,    K}
    ColKey = NTuple{n_col_sp, K}

    # ---- bin existing blocks by the first n_bin right sparse axes --------------
    bins = Dict{BinKey, Vector{Tuple{LKey, ColKey, Int}}}()
    for (key, id) in blocks_sorted(A)
        lk    = ntuple(i -> key[i],                 nls)
        bin_k = ntuple(i -> key[nls + i],           n_bin)
        col_k = ntuple(i -> key[nls + n_bin + i],   n_col_sp)
        push!(get!(bins, bin_k, Tuple{LKey, ColKey, Int}[]), (lk, col_k, id))
    end
    bin_vals = sort!(collect(keys(bins)))

    # ---- per-bin-key SVD -------------------------------------------------------
    # For each bin_k: rows  = (lk_index × d_left)
    #                 cols  = (col_k_index × d_right)
    block_svds = Vector{Tuple{BinKey, Matrix{T}, Vector{real(T)}, Matrix{T}, Vector{LKey}, Vector{ColKey}}}()
    for bin_k in bin_vals
        entries  = bins[bin_k]
        l_list   = sort!(unique!([lk    for (lk, _, _)    in entries]))
        col_list = sort!(unique!([col_k for (_, col_k, _) in entries]))
        n_Li     = length(l_list)
        n_Ci     = length(col_list)
        l_pos    = Dict(l => i for (i, l) in enumerate(l_list))
        col_pos  = Dict(c => i for (i, c) in enumerate(col_list))

        M_i = zeros(T, n_Li * d_left, n_Ci * d_right)
        for (lk, col_k, id) in entries
            li = l_pos[lk]; ci = col_pos[col_k]
            blk = reshape(_block_view(A, id), d_left, d_right)
            M_i[(li-1)*d_left+1 : li*d_left, (ci-1)*d_right+1 : ci*d_right] .= blk
        end

        F = svd(M_i)
        push!(block_svds, (bin_k, F.U, F.S, F.Vt, l_list, col_list))
    end

    # ---- global truncation across all per-bin spectra --------------------------
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < cutoff * max_sv
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)

    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    surviving    = [bi for bi in eachindex(block_svds) if k_kept[bi] > 0]
    n_new_sp_nat = length(surviving)
    n_new_d_natural = isempty(surviving) ? 0 : maximum(k_kept[bi] for bi in surviving)
    # Per-channel mult cap: total effective bond ≤ maxdim.
    n_active_chan = n_new_sp_nat
    mult_cap = max(1, fld(maxdim, max(1, n_active_chan)))
    n_new_d = min(n_new_d_natural, mult_cap)
    dropped_by_cap = 0
    if n_new_d < n_new_d_natural
        for bi in surviving
            if k_kept[bi] > n_new_d
                S_i = block_svds[bi][3]
                for c in (n_new_d + 1):k_kept[bi]
                    if c <= length(S_i)
                        truncerr += S_i[c]^2
                    end
                end
                dropped_by_cap += (k_kept[bi] - n_new_d)
                k_kept[bi] = n_new_d
            end
        end
    end
    # SPARSE_SVD_CAP_DIAG debug print, disabled; flip to `true` + uncomment body to re-enable.
    if false
        # println(stdout, "[CAP_DIAG_RB] ortho=right maxdim=$maxdim n_active=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d eff_bond=$(n_active_chan*n_new_d) dropped_svs=$dropped_by_cap"); flush(stdout)
    end
    new_sp_of_bi = Dict(bi => j for (j, bi) in enumerate(surviving))
    # Pad if caller requested a specific sparse dim (boundary preservation).
    n_new_sp = target_n_new_sp > n_new_sp_nat ? target_n_new_sp : n_new_sp_nat
    # SPARSE_SVD_DIAG debug print, disabled; flip to `true` + uncomment body to re-enable.
    if false
        # Per-bin (rows, cols, nat_rank, kept) so we can see whether mult is
        # clamped by (a) tiny M_i, (b) cutoff, or (c) maxdim.
        # bin_info = String[]
        # for bi in eachindex(block_svds)
        #     (_, U_i, S_i, Vt_i, l_list, col_list) = block_svds[bi]
        #     push!(bin_info, "($(size(U_i,1))x$(size(Vt_i,2)) rk=$(length(S_i)) kept=$(k_kept[bi]) smax=$(isempty(S_i) ? 0.0 : round(S_i[1]; digits=4)) smin=$(isempty(S_i) ? 0.0 : round(S_i[end]; digits=4)))")
        # end
        # println(stdout, "[SVD_DIAG_RB] ortho=$ortho nls=$nls nrs=$nrs n_bin=$n_bin d_left=$d_left d_right=$d_right n_bins=$(length(block_svds)) n_new_sp_nat=$n_new_sp_nat n_new_d=$n_new_d keep_count=$keep_count maxdim=$maxdim total_svs=$(length(all_svs)) bins=[$(join(bin_info, ", "))]"); flush(stdout)
    end

    # Output format matches blocksparse_svd:
    # U  layout: (left_sp..., new_sp,  left_d...,  new_d)
    # SV layout: (new_sp, right_sp..., new_d, right_d...)
    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    for bi in surviving
        (bin_k, U_i, S_i, Vt_i, l_list, col_list) = block_svds[bi]
        ki = k_kept[bi]
        j  = new_sp_of_bi[bi]
        n_Li = length(l_list)
        n_Ci = length(col_list)

        @inbounds for c in 1:ki
            svs_kept[j, c] = S_i[c]
        end

        # U_i has shape (n_Li * d_left, ki). Slab li gives U-block at (lk..., j).
        for (li, lk) in enumerate(l_list)
            id_u  = _ensure_block!(U_bs, (lk..., j))
            u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
            row_off = (li - 1) * d_left
            @inbounds for c in 1:ki
                sc = (ortho == "right") ? T(S_i[c]) : one(T)
                for s in 1:d_left
                    u_mat[s, c] = U_i[row_off + s, c] * sc
                end
            end
        end

        # Vt_i has shape (ki, n_Ci * d_right). Slab ci goes to SV-block at (j, bin_k..., col_k...).
        for (ci, col_k) in enumerate(col_list)
            id_sv  = _ensure_block!(SV_bs, (j, bin_k..., col_k...))
            sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
            col_off = (ci - 1) * d_right
            @inbounds for c in 1:ki
                sc = (ortho == "left") ? T(S_i[c]) : one(T)
                for s in 1:d_right
                    sv_mat[c, s] = Vt_i[c, col_off + s] * sc
                end
            end
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end



function blocksparse_svd_channel_aware_fixed(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int,
    n_left_dense::Int,
    left_template::Vector{<:Tuple},      # entries: (lk_tuple, channel_value)
    right_template::Vector{<:Tuple},     # entries: (channel_value, rk_tuple)
    bond_sparse_dim::Int,                # = number of allowed channels (axis dim)
    bond_factor_dims::Vector{Int} = Int[bond_sparse_dim],
        # Decomposition of bond_sparse_dim into (fA, fB) factor dims. For a
        # monolithic bond (single (I+C) factor) pass [bond_sparse_dim]; for a
        # bond at the overlap of two (I+C) factors pass [fA_dim, fB_dim] with
        # fA_dim * fB_dim == bond_sparse_dim. Within each fA group (for
        # ortho="left") or fB group (for ortho="right") we run a joint SVD so
        # cross-channel rows / cols are globally orthonormal.
    ortho::String = "left",
    maxdim::Int   = typemax(Int),
    mindim::Int   = 1,
    cutoff::Float64 = 0.0,
) where {T, N, N2, P, K<:Integer}
    @assert prod(bond_factor_dims; init=1) == bond_sparse_dim ||
            (length(bond_factor_dims) == 1 && bond_factor_dims[1] == bond_sparse_dim) ||
            isempty(bond_factor_dims) "bond_factor_dims product must equal bond_sparse_dim"
    if length(bond_factor_dims) != 2
        # Pad to length 2 for uniform handling: [fA_dim, fB_dim].
        # For monolithic bonds use [bond_sparse_dim, 1] so fA = channel, fB = 1.
        bond_factor_dims = Int[bond_sparse_dim, 1]
    end
    fA_dim, fB_dim = bond_factor_dims[1], bond_factor_dims[2]
    # Channel encoding: k = (fB - 1) * fA_dim + fA   (k ∈ 1..bond_sparse_dim)
    # Inverse: fA = ((k - 1) mod fA_dim) + 1;  fB = ((k - 1) ÷ fA_dim) + 1
    fA_of(k::Integer) = ((Int(k) - 1) % fA_dim) + 1
    fB_of(k::Integer) = ((Int(k) - 1) ÷ fA_dim) + 1

    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    @assert 0 ≤ nls ≤ P  "n_left_sparse must be in 0:$P, got $nls"
    @assert 0 ≤ nld ≤ N2 "n_left_dense must be in 0:$N2, got $nld"

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],            nls)
    right_sp_dims = ntuple(i -> dims[nls + i],      nrs)
    left_d_dims   = ntuple(i -> dims[P + i],        nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],  nrd)

    d_left  = prod(left_d_dims;  init = 1)
    d_right = prod(right_d_dims; init = 1)

    LKey = NTuple{nls, K}
    RKey = NTuple{nrs, K}

    # ---- Build lookups from templates ---------------------------------------
    lk_to_channels = Dict{LKey, Set{K}}()
    for entry in left_template
        lk_raw, k = entry
        lk = NTuple{nls,K}(lk_raw)
        push!(get!(lk_to_channels, lk, Set{K}()), K(k))
    end
    rk_to_channels = Dict{RKey, Set{K}}()
    for entry in right_template
        k, rk_raw = entry
        rk = NTuple{nrs,K}(rk_raw)
        push!(get!(rk_to_channels, rk, Set{K}()), K(k))
    end

    # ---- Group phi blocks by channel (must be unique per block) -------------
    by_channel = Dict{K, Vector{Tuple{LKey, RKey, Int}}}()
    dropped = 0
    dropped_norm2 = 0.0
    kept_norm2    = 0.0
    ambiguous = 0
    for (key, id) in blocks_sorted(A)
        lk = ntuple(i -> key[i],       nls)
        rk = ntuple(i -> key[nls + i], nrs)
        c_lk = get(lk_to_channels, lk, Set{K}())
        c_rk = get(rk_to_channels, rk, Set{K}())
        common = intersect(c_lk, c_rk)
        if length(common) == 0
            dropped += 1
            blk = _block_view(A, id)
            dropped_norm2 += sum(abs2, blk)
            continue
        end
        if length(common) > 1
            ambiguous += 1
        end
        k = first(common)
        push!(get!(by_channel, k, Tuple{LKey,RKey,Int}[]), (lk, rk, id))
        blk = _block_view(A, id)
        kept_norm2 += sum(abs2, blk)
    end
    if dropped > 0 || ambiguous > 0
        @warn "blocksparse_svd_channel_aware: dropped=$dropped (‖²=$dropped_norm2)  kept_‖²=$kept_norm2  ambiguous=$ambiguous"
        println(stdout, "[SVD_DIAG_CA] dropped=$dropped (‖²=$dropped_norm2)  kept_‖²=$kept_norm2  ambiguous=$ambiguous")
    end

    # ---- Grouped joint SVD with column-partition (cross-c_m iso for free).
    # Group channels by fA value (ortho="left") or fB (ortho="right"). All
    # channels in same group share lk row support (for the (I±C) projector).
    # Per group: one joint SVD over M_G = [M_{c_1} | M_{c_2} | …]. Then
    # partition U_G's columns among c_m's in the group by argmax of
    # ‖V†[j, col_range(c_m)]‖². Each U_G column is assigned to exactly ONE
    # c_m → L's c_m blocks live in DISJOINT subspaces of U_G's column
    # space → cross-c_m iso is automatic. Reconstruction L*R = phi is
    # lossless because the partition covers all kept columns exactly once.
    channel_keys = sort!(collect(keys(by_channel)))
    block_svds = Vector{Tuple{K, Matrix{T}, Vector{real(T)}, Matrix{T}, Vector{LKey}, Vector{RKey}}}()

    group_of_k(k) = (ortho == "left") ? fA_of(k) : fB_of(k)
    groups_dict = Dict{Int, Vector{K}}()
    for k in channel_keys
        push!(get!(groups_dict, group_of_k(k), K[]), k)
    end
    sorted_group_ids = sort!(collect(keys(groups_dict)))

    for g in sorted_group_ids
        ks_in_g = sort!(groups_dict[g])
        l_list_g_set = Set{LKey}()
        for k in ks_in_g
            for (lk, _, _) in by_channel[k]
                push!(l_list_g_set, lk)
            end
        end
        l_list_g = sort!(collect(l_list_g_set))
        l_pos_g  = Dict(lk => i for (i, lk) in enumerate(l_list_g))
        n_Lg     = length(l_list_g)

        r_list_of_k     = Dict{K, Vector{RKey}}()
        col_offset_of_k = Dict{K, Int}()
        total_cols = 0
        for k in ks_in_g
            rl = sort!(unique!([rk for (_, rk, _) in by_channel[k]]))
            r_list_of_k[k]     = rl
            col_offset_of_k[k] = total_cols
            total_cols += length(rl) * d_right
        end

        M_G = zeros(T, n_Lg * d_left, total_cols)
        for k in ks_in_g
            rl = r_list_of_k[k]
            r_pos_k = Dict(rk => i for (i, rk) in enumerate(rl))
            for (lk, rk, id) in by_channel[k]
                li = l_pos_g[lk]; rj = r_pos_k[rk]
                blk = reshape(_block_view(A, id), d_left, d_right)
                r0 = (li - 1) * d_left
                c0 = col_offset_of_k[k] + (rj - 1) * d_right
                M_G[r0+1 : r0+d_left, c0+1 : c0+d_right] .= blk
            end
        end

        F = svd(M_G)
        U_G  = F.U
        S_G  = F.S
        Vt_G = F.Vt
        r_total  = length(S_G)
        n_U_cols = size(U_G, 2)

        # Compute full per-column-per-channel norm matrix W[j, ki] = ‖V†[j, col_range(ks_in_g[ki])]‖²
        n_chans_g = length(ks_in_g)
        W = zeros(real(T), n_U_cols, n_chans_g)
        for j in 1:n_U_cols
            for (ki, k) in enumerate(ks_in_g)
                nr_k = length(r_list_of_k[k])
                c_lo = col_offset_of_k[k] + 1
                c_hi = col_offset_of_k[k] + nr_k * d_right
                sn = 0.0
                @inbounds for c in c_lo:c_hi
                    sn += abs2(Vt_G[j, c])
                end
                W[j, ki] = sn
            end
        end

        # Partition: argmax over channels (current behavior). Round-robin
        # alternative (preserves all template channels even if V† is
        # concentrated) deleted 2026-06 — this whole function
        # (blocksparse_svd_channel_aware_fixed) has zero callers anywhere in
        # the tree, so the flag was already unreachable.
        rr_mode = false
        col_to_ki = Vector{Int}(undef, n_U_cols)
        for j in 1:n_U_cols
            if rr_mode
                col_to_ki[j] = ((j - 1) % n_chans_g) + 1
            else
                best_norm = -1.0
                best_idx  = 1
                for ki in 1:n_chans_g
                    if W[j, ki] > best_norm
                        best_norm = W[j, ki]
                        best_idx  = ki
                    end
                end
                col_to_ki[j] = best_idx
            end
        end

        # SB_GROUP_DIAG deleted 2026-06 (was print-only V†-concentration/partition
        # report) — this whole function has zero callers, so it was unreachable.
        # if get(ENV, "SB_GROUP_DIAG", "0") == "1"
        #     top5 = string([round(Float64(S_G[k]); sigdigits=3) for k in 1:min(5, r_total)])
        #     # How concentrated is V† mass per column? Report mean(max_ki W[j,ki] / sum_ki W[j,ki])
        #     conc = 0.0; cnt = 0
        #     for j in 1:min(r_total, n_U_cols)
        #         tot = sum(W[j, :])
        #         if tot > 0
        #             conc += maximum(view(W, j, :)) / tot
        #             cnt += 1
        #         end
        #     end
        #     conc_avg = cnt > 0 ? conc / cnt : 0.0
        #     # Per-channel: # cols assigned and total SV² assigned
        #     per_chan = String[]
        #     for (ki, k) in enumerate(ks_in_g)
        #         ass = count(==(ki), col_to_ki)
        #         sv2 = sum(j -> col_to_ki[j] == ki && j <= r_total ? Float64(S_G[j])^2 : 0.0, 1:n_U_cols)
        #         push!(per_chan, "c=$k:cols=$ass,sv²=$(round(sv2; sigdigits=3))")
        #     end
        #     partition_str = join(per_chan, " | ")
        #     println(stdout, "[GROUP_DIAG] ortho=$ortho g=$g ks=$ks_in_g m_g=$(n_Lg * d_left) cols=$total_cols r=$r_total topSV=$top5 V†_conc_max/sum=$(round(conc_avg; digits=2)) partition: $partition_str"); flush(stdout)
        # end

        for (ki, k) in enumerate(ks_in_g)
            rl    = r_list_of_k[k]
            nr_k  = length(rl)
            c_lo  = col_offset_of_k[k] + 1
            c_hi  = col_offset_of_k[k] + nr_k * d_right
            assigned = Int[j for j in 1:n_U_cols if col_to_ki[j] == ki]
            if isempty(assigned)
                push!(block_svds,
                      (k,
                       zeros(T, n_Lg * d_left, 0),
                       real(T)[],
                       zeros(T, 0, nr_k * d_right),
                       l_list_g,
                       rl))
                continue
            end
            S_k  = real(T)[ (j <= r_total) ? real(S_G[j]) : zero(real(T)) for j in assigned ]
            U_k  = U_G[:, assigned]
            Vt_k = Vt_G[assigned, c_lo:c_hi]
            push!(block_svds, (k, U_k, S_k, Vt_k, l_list_g, rl))
        end
    end

    # ---- Global truncation across all channels ------------------------------
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    # Floor below `cutoff * max_sv` keeps natural rank; in addition we
    # always drop SVs that are at-floating-point-zero (from `svd(...; full=true)`
    # padding) so n_new_d reflects genuine rank rather than min(m,n).
    sv_floor = max(cutoff * max_sv, 1e-12 * max(max_sv, 1.0))
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < sv_floor
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)
    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    n_new_sp = bond_sparse_dim
    n_new_d_natural = isempty(k_kept) ? 0 : maximum(k_kept; init = 0)

    # NEW total-bond cap: keep effective bond ≈ maxdim by capping
    # mult-per-channel at fld(maxdim, n_active_chan). Replaces the older
    # iso_cap = min_i m_i / n_new_sp rule which gave effective bond ≈
    # n_chan × maxdim (too generous — sparse was using ~n_chan× the bond of
    # dense at the same maxdim). With this rule total bond ≤ maxdim.
    # Active-channel cap: mult ≤ fld(maxdim, n_active) where n_active counts
    # channels that received ANY data from phi (k_kept > 0). For the (I±C)
    # projector, phi at overlap bonds populates only 2 of the 4 schema
    # channels (the others are zero in the input itself, not dropped by SVD).
    # With n_active=2 and maxdim=40: mult_cap=20, real bond = 2×20 = 40
    # (matches dense expressive power). Schema bond = n_new_sp × mult_cap
    # may exceed maxdim due to padded empty channels, but DOWNSTREAM SVDs
    # see the same active-channel pattern so the padding doesn't compound.
    n_active_chan = count(>(0), k_kept)
    mult_cap = max(1, fld(maxdim, max(1, n_active_chan)))
    n_new_d_natural_only = n_new_d_natural
    n_new_d = min(n_new_d_natural, mult_cap)
    # Diagnostic accumulators (read by callers via TIMER/env). dropped_by_cap:
    # SVs that the global top-maxdim selection kept but the per-channel cap
    # then dropped. dropped_truncerr_cap: their squared-SV sum (already added
    # to truncerr below in the re-truncate block).
    dropped_by_cap = 0
    if n_new_d < n_new_d_natural
        for bi in eachindex(k_kept)
            if k_kept[bi] > n_new_d
                dropped_by_cap += (k_kept[bi] - n_new_d)
            end
        end
    end
    # SPARSE_SVD_CAP_DIAG / SPARSE_SVD_DIAG debug prints, disabled; this occurrence
    # is inside blocksparse_svd_channel_aware_fixed, which has zero callers anywhere
    # (already unreachable). Flip to `true` (and restore the two checks below) to
    # re-enable.
    if false
    end
    # if get(ENV, "SPARSE_SVD_CAP_DIAG", "0") == "1"
    #     n_chan_possible = n_new_sp
    #     println(stdout, "[CAP_DIAG] ortho=$ortho maxdim=$maxdim n_chan_possible=$n_chan_possible n_active_chan=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d eff_bond=$(n_active_chan * n_new_d) dropped_by_cap_svs=$dropped_by_cap"); flush(stdout)
    # end
    # if get(ENV, "SPARSE_SVD_DIAG", "0") == "1"
    #     cap_fired = mult_cap < n_new_d_natural
    #     println(stdout, "[SVD_DIAG] ortho=$ortho bond_sp=$bond_sparse_dim fac=$bond_factor_dims d_left=$d_left d_right=$d_right n_active_chan=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d cap_fired=$cap_fired maxdim=$maxdim keep_count=$keep_count eff_bond=$(n_active_chan * n_new_d)"); flush(stdout)
    #     # Per-channel breakdown: shows where the aggregation asymmetry comes from.
    #     # For each new-bond channel c_new: rows × cols of its SVD matrix, the
    #     # number of (c_L, c_R) pairs aggregated into that channel, the natural
    #     # rank, and how many SVs were kept post-truncation.
    #     for bi in eachindex(block_svds)
    #         c_new = block_svds[bi][1]
    #         U_i   = block_svds[bi][2]
    #         S_i   = block_svds[bi][3]
    #         Vt_i  = block_svds[bi][4]
    #         l_lst = block_svds[bi][5]
    #         r_lst = block_svds[bi][6]
    #         n_combos_left  = length(l_lst)
    #         n_combos_right = length(r_lst)
    #         m_c = size(U_i, 1)
    #         n_c = size(Vt_i, 2)
    #         natural_rank = min(m_c, n_c)
    #         max_sv_chan = isempty(S_i) ? 0.0 : Float64(maximum(S_i))
    #         min_sv_chan = isempty(S_i) ? 0.0 : Float64(minimum(S_i))
    #         println(stdout, "  [SVD_DIAG]   c_new=$c_new  m_c=$m_c (= $n_combos_left lk × $d_left)  n_c=$n_c (= $n_combos_right rk × $d_right)  natural_rank=$natural_rank  k_kept=$(k_kept[bi])  SV_range=[$(round(min_sv_chan,sigdigits=3)), $(round(max_sv_chan,sigdigits=3))]"); flush(stdout)
    #     end
    # end
    # Re-truncate per-channel kept count so each k_kept[bi] ≤ n_new_d.
    if n_new_d < n_new_d_natural
        for bi in eachindex(k_kept)
            if k_kept[bi] > n_new_d
                # Recompute truncerr contribution from the now-dropped SVs.
                S_i = block_svds[bi][3]
                for c in (n_new_d + 1):k_kept[bi]
                    if c <= length(S_i)
                        truncerr += S_i[c]^2
                    end
                end
                k_kept[bi] = n_new_d
            end
        end
    end

    # SB_SV_REPORT debug print, disabled; this occurrence is inside
    # blocksparse_svd_channel_aware_fixed, which has zero callers anywhere (already
    # unreachable). Flip to `true` (and restore the check below) to re-enable.
    if false
    end
    # if get(ENV, "SB_SV_REPORT", "0") == "1"
    #     println(stdout, "[SV_REPORT_GROUPED] ortho=$ortho  maxdim=$maxdim  n_new_d=$n_new_d  mult_cap=$mult_cap  bond_sparse_dim=$bond_sparse_dim  n_active_chan=$n_active_chan")
    #     for bi in eachindex(block_svds)
    #         c_new = block_svds[bi][1]
    #         S_i   = block_svds[bi][3]
    #         k     = k_kept[bi]
    #         kept_str    = (k > 0 && k <= length(S_i)) ? string(S_i[k])   : "-"
    #         dropped_str = (k+1 <= length(S_i))        ? string(S_i[k+1]) : "-"
    #         println(stdout, "  cM=$c_new: kept $k/$(length(S_i))  smallest_kept=$kept_str  largest_dropped=$dropped_str")
    #     end
    #     flush(stdout)
    # end

    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    # Per-channel scatter: emit L blocks at (lk, channel_natural) and R blocks
    # at (channel_natural, rk) for each surviving channel. EXACT factorization
    # L · R = phi. Block-key set EXACTLY matches M_b / M_b1.
    #
    # Known limitation: per-channel V/U vectors are orthonormal WITHIN each
    # channel but not necessarily ACROSS channels that share rk/lk space (at
    # boundary or factor-overlap bonds). This makes psi[i] non-canonical in the
    # strict sense after orthogonalize. Fix is via generalized eigsolve with
    # an explicit Gram-matrix metric M in DMRG — see the plan file
    # `~/.claude/plans/all-channel-mult-slots-glittery-mccarthy.md`.
    surviving_bis = [bi for bi in eachindex(block_svds) if k_kept[bi] > 0]

    for bi in surviving_bis
        (k, U_i, S_i, Vt_i, l_list, r_list) = block_svds[bi]
        ki = k_kept[bi]
        ki == 0 && continue

        umax = min(n_new_d, size(U_i, 2))
        vmax = min(n_new_d, size(Vt_i, 1))

        @inbounds for c in 1:ki
            svs_kept[k, c] = S_i[c]
        end

        for (li, lk) in enumerate(l_list)
            id_u  = _ensure_block!(U_bs, (lk..., k))
            u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
            row_off = (li - 1) * d_left
            @inbounds for c in 1:umax
                S_c = c <= length(S_i) ? real(S_i[c]) : zero(real(T))
                sc  = (ortho == "right") ? T(S_c) : one(T)
                for s in 1:d_left
                    u_mat[s, c] = U_i[row_off + s, c] * sc
                end
            end
        end

        for (rj, rk) in enumerate(r_list)
            id_sv  = _ensure_block!(SV_bs, (k, rk...))
            sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
            col_off = (rj - 1) * d_right
            @inbounds for c in 1:vmax
                S_c = c <= length(S_i) ? real(S_i[c]) : zero(real(T))
                sc  = (ortho == "left") ? T(S_c) : one(T)
                for s in 1:d_right
                    sv_mat[c, s] = Vt_i[c, col_off + s] * sc
                end
            end
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end



# ─────────────────────────────────────────────────────────────────────────────
# blocksparse_svd_channel_aware
#
# Channel-aware block-sparse SVD. Given templates that enumerate the
# ALLOWED (left_sparse_keys, channel) and (channel, right_sparse_keys) tuples
# inherited from the OLD psi[b] / psi[b+1] block structure, this routine:
#   1. Determines a unique channel value for each phi block from the templates.
#   2. SVDs phi per channel (treating each channel's contribution independently).
#   3. Emits L blocks at exactly the (lk_tuple, channel) keys in left_template
#      and R blocks at exactly the (channel, rk_tuple) keys in right_template.
#
# Result: L and R block-key sets are subsets of the OLD psi[b], psi[b+1]
# block-key sets — the factorization "lives in" the allowed sparsity pattern,
# without inventing keys outside it. Cross-channel rows are disjoint by
# construction (each channel pairs with disjoint lk-sets), so L is globally
# isometric for ortho="left" and R for ortho="right".
# ─────────────────────────────────────────────────────────────────────────────
function blocksparse_svd_channel_aware(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int,
    n_left_dense::Int,
    left_template::Vector{<:Tuple},      # entries: (lk_tuple, channel_value)
    right_template::Vector{<:Tuple},     # entries: (channel_value, rk_tuple)
    bond_sparse_dim::Int,                # = number of allowed channels (axis dim)
    bond_factor_dims::Vector{Int} = Int[bond_sparse_dim],
        # Decomposition of bond_sparse_dim into (fA, fB) factor dims. For a
        # monolithic bond (single (I+C) factor) pass [bond_sparse_dim]; for a
        # bond at the overlap of two (I+C) factors pass [fA_dim, fB_dim] with
        # fA_dim * fB_dim == bond_sparse_dim. Within each fA group (for
        # ortho="left") or fB group (for ortho="right") we run a joint SVD so
        # cross-channel rows / cols are globally orthonormal.
    ortho::String = "left",
    maxdim::Int   = typemax(Int),
    mindim::Int   = 1,
    cutoff::Float64 = 0.0,
) where {T, N, N2, P, K<:Integer}
    @assert prod(bond_factor_dims; init=1) == bond_sparse_dim ||
            (length(bond_factor_dims) == 1 && bond_factor_dims[1] == bond_sparse_dim) ||
            isempty(bond_factor_dims) "bond_factor_dims product must equal bond_sparse_dim"
    if length(bond_factor_dims) != 2
        # Pad to length 2 for uniform handling: [fA_dim, fB_dim].
        # For monolithic bonds use [bond_sparse_dim, 1] so fA = channel, fB = 1.
        bond_factor_dims = Int[bond_sparse_dim, 1]
    end
    fA_dim, fB_dim = bond_factor_dims[1], bond_factor_dims[2]
    # Channel encoding: k = (fB - 1) * fA_dim + fA   (k ∈ 1..bond_sparse_dim)
    # Inverse: fA = ((k - 1) mod fA_dim) + 1;  fB = ((k - 1) ÷ fA_dim) + 1
    fA_of(k::Integer) = ((Int(k) - 1) % fA_dim) + 1
    fB_of(k::Integer) = ((Int(k) - 1) ÷ fA_dim) + 1

    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    @assert 0 ≤ nls ≤ P  "n_left_sparse must be in 0:$P, got $nls"
    @assert 0 ≤ nld ≤ N2 "n_left_dense must be in 0:$N2, got $nld"

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],            nls)
    right_sp_dims = ntuple(i -> dims[nls + i],      nrs)
    left_d_dims   = ntuple(i -> dims[P + i],        nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],  nrd)

    d_left  = prod(left_d_dims;  init = 1)
    d_right = prod(right_d_dims; init = 1)

    LKey = NTuple{nls, K}
    RKey = NTuple{nrs, K}

    # ---- Build lookups from templates ---------------------------------------
    lk_to_channels = Dict{LKey, Set{K}}()
    for entry in left_template
        lk_raw, k = entry
        lk = NTuple{nls,K}(lk_raw)
        push!(get!(lk_to_channels, lk, Set{K}()), K(k))
    end
    rk_to_channels = Dict{RKey, Set{K}}()
    for entry in right_template
        k, rk_raw = entry
        rk = NTuple{nrs,K}(rk_raw)
        push!(get!(rk_to_channels, rk, Set{K}()), K(k))
    end

    # ---- Group phi blocks by channel (must be unique per block) -------------
    by_channel = Dict{K, Vector{Tuple{LKey, RKey, Int}}}()
    dropped = 0
    dropped_norm2 = 0.0
    kept_norm2    = 0.0
    ambiguous = 0
    for (key, id) in blocks_sorted(A)
        lk = ntuple(i -> key[i],       nls)
        rk = ntuple(i -> key[nls + i], nrs)
        c_lk = get(lk_to_channels, lk, Set{K}())
        c_rk = get(rk_to_channels, rk, Set{K}())
        common = intersect(c_lk, c_rk)
        if length(common) == 0
            dropped += 1
            blk = _block_view(A, id)
            dropped_norm2 += sum(abs2, blk)
            continue
        end
        if length(common) > 1
            ambiguous += 1
        end
        k = first(common)
        push!(get!(by_channel, k, Tuple{LKey,RKey,Int}[]), (lk, rk, id))
        blk = _block_view(A, id)
        kept_norm2 += sum(abs2, blk)
    end
    if dropped > 0 || ambiguous > 0
        @warn "blocksparse_svd_channel_aware: dropped=$dropped (‖²=$dropped_norm2)  kept_‖²=$kept_norm2  ambiguous=$ambiguous"
    end

    # ---- SVD per channel (full SVD so we can extend padded slots with orthonormal basis) ----
    channel_keys = sort!(collect(keys(by_channel)))
    block_svds = Vector{Tuple{K, Matrix{T}, Vector{real(T)}, Matrix{T}, Vector{LKey}, Vector{RKey}}}()
    for k in channel_keys
        entries = by_channel[k]
        l_list = sort!(unique!([lk for (lk, _, _) in entries]))
        r_list = sort!(unique!([rk for (_, rk, _) in entries]))
        n_Li = length(l_list); n_Ri = length(r_list)
        l_pos = Dict(lk => i for (i, lk) in enumerate(l_list))
        r_pos = Dict(rk => j for (j, rk) in enumerate(r_list))

        M_i = zeros(T, n_Li * d_left, n_Ri * d_right)
        for (lk, rk, id) in entries
            li = l_pos[lk]; rj = r_pos[rk]
            blk = reshape(_block_view(A, id), d_left, d_right)
            M_i[(li-1)*d_left+1 : li*d_left, (rj-1)*d_right+1 : rj*d_right] .= blk
        end
        # Full SVD: U is m×m, Vt is n×n, S has min(m,n) singular values.
        # Truncation (below) uses F.S (natural ranks only). The extra
        # orthonormal columns of U / rows of Vt are used to pad the
        # uniform multiplicity axis with zero-SV orthonormal extensions
        # (preserves isometry of the iso side without contributing to L*R).
        F = svd(M_i; full = true)
        push!(block_svds, (k, F.U, F.S, F.Vt, l_list, r_list))
    end

    # ---- Global truncation across all channels ------------------------------
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    # Floor below `cutoff * max_sv` keeps natural rank; in addition we
    # always drop SVs that are at-floating-point-zero (from `svd(...; full=true)`
    # padding) so n_new_d reflects genuine rank rather than min(m,n).
    sv_floor = max(cutoff * max_sv, 1e-12 * max(max_sv, 1.0))
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < sv_floor
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)
    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    n_new_sp = bond_sparse_dim
    n_new_d_natural = isempty(k_kept) ? 0 : maximum(k_kept; init = 0)

    # NEW total-bond cap: keep effective bond ≈ maxdim by capping
    # mult-per-channel at fld(maxdim, n_active_chan). Replaces the older
    # iso_cap = min_i m_i / n_new_sp rule which gave effective bond ≈
    # n_chan × maxdim (too generous — sparse was using ~n_chan× the bond of
    # dense at the same maxdim). With this rule total bond ≤ maxdim.
    # Active-channel cap: mult ≤ fld(maxdim, n_active) where n_active counts
    # channels that received ANY data from phi (k_kept > 0). For the (I±C)
    # projector, phi at overlap bonds populates only 2 of the 4 schema
    # channels (the others are zero in the input itself, not dropped by SVD).
    # With n_active=2 and maxdim=40: mult_cap=20, real bond = 2×20 = 40
    # (matches dense expressive power). Schema bond = n_new_sp × mult_cap
    # may exceed maxdim due to padded empty channels, but DOWNSTREAM SVDs
    # see the same active-channel pattern so the padding doesn't compound.
    n_active_chan = count(>(0), k_kept)
    mult_cap = max(1, fld(maxdim, max(1, n_active_chan)))
    n_new_d_natural_only = n_new_d_natural
    n_new_d = min(n_new_d_natural, mult_cap)
    # Diagnostic accumulators (read by callers via TIMER/env). dropped_by_cap:
    # SVs that the global top-maxdim selection kept but the per-channel cap
    # then dropped. dropped_truncerr_cap: their squared-SV sum (already added
    # to truncerr below in the re-truncate block).
    dropped_by_cap = 0
    if n_new_d < n_new_d_natural
        for bi in eachindex(k_kept)
            if k_kept[bi] > n_new_d
                dropped_by_cap += (k_kept[bi] - n_new_d)
            end
        end
    end
    # SPARSE_SVD_CAP_DIAG / SPARSE_SVD_DIAG debug prints, disabled; flip to `true`
    # (and restore the two `if get(ENV,...)` checks below) to re-enable.
    if false
    end
    # if get(ENV, "SPARSE_SVD_CAP_DIAG", "0") == "1"
    #     n_chan_possible = n_new_sp
    #     println(stdout, "[CAP_DIAG] ortho=$ortho maxdim=$maxdim n_chan_possible=$n_chan_possible n_active_chan=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d eff_bond=$(n_active_chan * n_new_d) dropped_by_cap_svs=$dropped_by_cap"); flush(stdout)
    # end
    # if get(ENV, "SPARSE_SVD_DIAG", "0") == "1"
    #     cap_fired = mult_cap < n_new_d_natural
    #     println(stdout, "[SVD_DIAG] ortho=$ortho bond_sp=$bond_sparse_dim fac=$bond_factor_dims d_left=$d_left d_right=$d_right n_active_chan=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d cap_fired=$cap_fired maxdim=$maxdim keep_count=$keep_count eff_bond=$(n_active_chan * n_new_d)"); flush(stdout)
    #     # Per-channel breakdown: shows where the aggregation asymmetry comes from.
    #     # For each new-bond channel c_new: rows × cols of its SVD matrix, the
    #     # number of (c_L, c_R) pairs aggregated into that channel, the natural
    #     # rank, and how many SVs were kept post-truncation.
    #     for bi in eachindex(block_svds)
    #         c_new = block_svds[bi][1]
    #         U_i   = block_svds[bi][2]
    #         S_i   = block_svds[bi][3]
    #         Vt_i  = block_svds[bi][4]
    #         l_lst = block_svds[bi][5]
    #         r_lst = block_svds[bi][6]
    #         n_combos_left  = length(l_lst)
    #         n_combos_right = length(r_lst)
    #         m_c = size(U_i, 1)
    #         n_c = size(Vt_i, 2)
    #         natural_rank = min(m_c, n_c)
    #         max_sv_chan = isempty(S_i) ? 0.0 : Float64(maximum(S_i))
    #         min_sv_chan = isempty(S_i) ? 0.0 : Float64(minimum(S_i))
    #         println(stdout, "  [SVD_DIAG]   c_new=$c_new  m_c=$m_c (= $n_combos_left lk × $d_left)  n_c=$n_c (= $n_combos_right rk × $d_right)  natural_rank=$natural_rank  k_kept=$(k_kept[bi])  SV_range=[$(round(min_sv_chan,sigdigits=3)), $(round(max_sv_chan,sigdigits=3))]"); flush(stdout)
    #     end
    # end
    # Re-truncate per-channel kept count so each k_kept[bi] ≤ n_new_d.
    if n_new_d < n_new_d_natural
        for bi in eachindex(k_kept)
            if k_kept[bi] > n_new_d
                # Recompute truncerr contribution from the now-dropped SVs.
                S_i = block_svds[bi][3]
                for c in (n_new_d + 1):k_kept[bi]
                    if c <= length(S_i)
                        truncerr += S_i[c]^2
                    end
                end
                k_kept[bi] = n_new_d
            end
        end
    end

    # SB_SV_REPORT debug print, disabled; flip to `true` (and restore the check
    # below) to re-enable.
    if false
    end
    # if get(ENV, "SB_SV_REPORT", "0") == "1"
    #     println(stdout, "[SV_REPORT_GROUPED] ortho=$ortho  maxdim=$maxdim  n_new_d=$n_new_d  mult_cap=$mult_cap  bond_sparse_dim=$bond_sparse_dim  n_active_chan=$n_active_chan")
    #     for bi in eachindex(block_svds)
    #         c_new = block_svds[bi][1]
    #         S_i   = block_svds[bi][3]
    #         k     = k_kept[bi]
    #         kept_str    = (k > 0 && k <= length(S_i)) ? string(S_i[k])   : "-"
    #         dropped_str = (k+1 <= length(S_i))        ? string(S_i[k+1]) : "-"
    #         println(stdout, "  cM=$c_new: kept $k/$(length(S_i))  smallest_kept=$kept_str  largest_dropped=$dropped_str")
    #     end
    #     flush(stdout)
    # end

    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    # Per-channel scatter: emit L blocks at (lk, channel_natural) and R blocks
    # at (channel_natural, rk) for each surviving channel. EXACT factorization
    # L · R = phi. Block-key set EXACTLY matches M_b / M_b1.
    #
    # Known limitation: per-channel V/U vectors are orthonormal WITHIN each
    # channel but not necessarily ACROSS channels that share rk/lk space (at
    # boundary or factor-overlap bonds). This makes psi[i] non-canonical in the
    # strict sense after orthogonalize. Fix is via generalized eigsolve with
    # an explicit Gram-matrix metric M in DMRG — see the plan file
    # `~/.claude/plans/all-channel-mult-slots-glittery-mccarthy.md`.
    surviving_bis = [bi for bi in eachindex(block_svds) if k_kept[bi] > 0]

    for bi in surviving_bis
        (k, U_i, S_i, Vt_i, l_list, r_list) = block_svds[bi]
        ki = k_kept[bi]
        ki == 0 && continue

        umax = min(n_new_d, size(U_i, 2))
        vmax = min(n_new_d, size(Vt_i, 1))

        @inbounds for c in 1:ki
            svs_kept[k, c] = S_i[c]
        end

        for (li, lk) in enumerate(l_list)
            id_u  = _ensure_block!(U_bs, (lk..., k))
            u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
            row_off = (li - 1) * d_left
            @inbounds for c in 1:umax
                S_c = c <= length(S_i) ? real(S_i[c]) : zero(real(T))
                sc  = (ortho == "right") ? T(S_c) : one(T)
                for s in 1:d_left
                    u_mat[s, c] = U_i[row_off + s, c] * sc
                end
            end
        end

        for (rj, rk) in enumerate(r_list)
            id_sv  = _ensure_block!(SV_bs, (k, rk...))
            sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
            col_off = (rj - 1) * d_right
            @inbounds for c in 1:vmax
                S_c = c <= length(S_i) ? real(S_i[c]) : zero(real(T))
                sc  = (ortho == "left") ? T(S_c) : one(T)
                for s in 1:d_right
                    sv_mat[c, s] = Vt_i[c, col_off + s] * sc
                end
            end
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end


# ─────────────────────────────────────────────────────────────────────────────
# blocksparse_svd_left_binned
#
# Mirror of `blocksparse_svd_right_binned`: bins by the FIRST n_left_bin_sparse
# LEFT-side sparse axes (typically the LEFT LINK axes). Remaining left-sparse
# axes (e.g. site) fold into the SVD sub-matrix rows. Right-side sparse axes
# all fold into the sub-matrix columns.
#
# Use with ortho="left" to get a globally isometric U: each bin pins a distinct
# left-link key, which is part of the row index, so cross-bin row spaces are
# disjoint and cross-bin U columns are trivially orthogonal.
# ─────────────────────────────────────────────────────────────────────────────
function blocksparse_svd_left_binned(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int       = P ÷ 2,
    n_left_dense::Int        = N2 ÷ 2,
    n_left_bin_sparse::Int   = -1,   # default = nls. Pass smaller (e.g. number of
                                      # left-LINK sparse axes) to bin only by those
                                      # and fold the rest into matrix rows.
    target_n_new_sp::Int     = -1,   # if positive, pad the new bond's sparse axis
                                      # to at least this size. Used at boundary to
                                      # match the original M[b]↔M[b+1] sparse dim.
                                      # Extra sectors carry no data (empty blocks).
    ortho::String      = "left",
    maxdim::Int        = typemax(Int),
    mindim::Int        = 1,
    cutoff::Float64    = 0.0,
) where {T, N, N2, P, K<:Integer}

    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    @assert 0 ≤ nls ≤ P  "n_left_sparse must be in 0:$P, got $nls"
    @assert 0 ≤ nld ≤ N2 "n_left_dense must be in 0:$N2, got $nld"
    n_bin = n_left_bin_sparse < 0 ? nls : n_left_bin_sparse
    @assert 0 ≤ n_bin ≤ nls "n_left_bin_sparse must be in 0:$nls, got $n_bin"
    n_row_sp = nls - n_bin   # remaining left-sparse axes that fold into matrix rows

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],            nls)
    right_sp_dims = ntuple(i -> dims[nls + i],      nrs)
    left_d_dims   = ntuple(i -> dims[P + i],        nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],  nrd)

    d_left  = prod(left_d_dims;  init = 1)
    d_right = prod(right_d_dims; init = 1)

    BinKey = NTuple{n_bin,    K}
    RowKey = NTuple{n_row_sp, K}
    RKey   = NTuple{nrs,      K}

    # ---- bin existing blocks by the first n_bin left sparse axes -------------
    bins = Dict{BinKey, Vector{Tuple{RowKey, RKey, Int}}}()
    for (key, id) in blocks_sorted(A)
        bin_k = ntuple(i -> key[i],             n_bin)
        row_k = ntuple(i -> key[n_bin + i],     n_row_sp)
        rk    = ntuple(i -> key[nls + i],       nrs)
        push!(get!(bins, bin_k, Tuple{RowKey, RKey, Int}[]), (row_k, rk, id))
    end
    bin_vals = sort!(collect(keys(bins)))

    # ---- per-bin mini-SVD ----------------------------------------------------
    block_svds = Vector{Tuple{BinKey, Matrix{T}, Vector{real(T)}, Matrix{T}, Vector{RowKey}, Vector{RKey}}}()
    for bin_k in bin_vals
        entries  = bins[bin_k]
        row_list = sort!(unique!([row_k for (row_k, _, _) in entries]))
        r_list   = sort!(unique!([rk    for (_, rk, _)    in entries]))
        n_Li     = length(row_list)
        n_Ri     = length(r_list)
        row_pos  = Dict(r => i for (i, r) in enumerate(row_list))
        r_pos    = Dict(r => j for (j, r) in enumerate(r_list))

        M_i = zeros(T, n_Li * d_left, n_Ri * d_right)
        for (row_k, rk, id) in entries
            li = row_pos[row_k]; rj = r_pos[rk]
            blk = reshape(_block_view(A, id), d_left, d_right)
            M_i[(li-1)*d_left+1 : li*d_left, (rj-1)*d_right+1 : rj*d_right] .= blk
        end

        F = svd(M_i)
        push!(block_svds, (bin_k, F.U, F.S, F.Vt, row_list, r_list))
    end

    # ---- global truncation across all per-bin spectra ------------------------
    all_svs = Tuple{Float64, Int, Int}[]
    for (bi, t) in enumerate(block_svds)
        for (col, sv) in enumerate(t[3])
            push!(all_svs, (Float64(sv), bi, col))
        end
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < cutoff * max_sv
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)

    k_kept = zeros(Int, length(block_svds))
    for i in 1:keep_count
        k_kept[all_svs[i][2]] += 1
    end

    surviving    = [bi for bi in eachindex(block_svds) if k_kept[bi] > 0]
    n_new_sp_nat = length(surviving)
    n_new_d_natural = isempty(surviving) ? 0 : maximum(k_kept[bi] for bi in surviving)
    # Per-channel mult cap so total effective bond ≤ maxdim:
    # mult_cap = fld(maxdim, n_active_chan). When this fires (n_new_d_natural >
    # mult_cap) we drop the excess SVs in each over-quota channel and add their
    # squared norms to truncerr. Active here = surviving (≥1 SV after global
    # top-maxdim selection).
    n_active_chan = n_new_sp_nat
    mult_cap = max(1, fld(maxdim, max(1, n_active_chan)))
    n_new_d = min(n_new_d_natural, mult_cap)
    dropped_by_cap = 0
    if n_new_d < n_new_d_natural
        for bi in surviving
            if k_kept[bi] > n_new_d
                S_i = block_svds[bi][3]
                for c in (n_new_d + 1):k_kept[bi]
                    if c <= length(S_i)
                        truncerr += S_i[c]^2
                    end
                end
                dropped_by_cap += (k_kept[bi] - n_new_d)
                k_kept[bi] = n_new_d
            end
        end
    end
    # SPARSE_SVD_CAP_DIAG debug print, disabled; flip to `true` (and restore the
    # check below) to re-enable.
    if false
    end
    # if get(ENV, "SPARSE_SVD_CAP_DIAG", "0") == "1"
    #     println(stdout, "[CAP_DIAG_LB] ortho=left maxdim=$maxdim n_active=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d eff_bond=$(n_active_chan*n_new_d) dropped_svs=$dropped_by_cap"); flush(stdout)
    # end
    new_sp_of_bi = Dict(bi => j for (j, bi) in enumerate(surviving))
    # Pad the new bond's sparse dim up to the caller's target if requested.
    # Extra slots have no blocks scattered into them — they exist on the axis
    # but carry zero data. This is used at boundaries to keep the new bond's
    # sparse dim equal to the original M[b]↔M[b+1] sparse dim.
    n_new_sp = target_n_new_sp > n_new_sp_nat ? target_n_new_sp : n_new_sp_nat

    # Output layout matches blocksparse_svd / right_binned:
    #   U  storage: (left_sp..., new_sp,  left_d...,  new_d)
    #   SV storage: (new_sp, right_sp..., new_d, right_d...)
    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    for bi in surviving
        (bin_k, U_i, S_i, Vt_i, row_list, r_list) = block_svds[bi]
        ki = k_kept[bi]
        j  = new_sp_of_bi[bi]

        @inbounds for c in 1:ki
            svs_kept[j, c] = S_i[c]
        end

        # U_i shape (n_Li * d_left, ki). Slab li → U block at (bin_k..., row_k..., j).
        for (li, row_k) in enumerate(row_list)
            id_u  = _ensure_block!(U_bs, (bin_k..., row_k..., j))
            u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
            row_off = (li - 1) * d_left
            @inbounds for c in 1:ki
                sc = (ortho == "right") ? T(S_i[c]) : one(T)
                for s in 1:d_left
                    u_mat[s, c] = U_i[row_off + s, c] * sc
                end
            end
        end

        # Vt_i shape (ki, n_Ri * d_right). Slab rj → SV block at (j, rk...).
        for (rj, rk) in enumerate(r_list)
            id_sv  = _ensure_block!(SV_bs, (j, rk...))
            sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
            col_off = (rj - 1) * d_right
            @inbounds for c in 1:ki
                sc = (ortho == "left") ? T(S_i[c]) : one(T)
                for s in 1:d_right
                    sv_mat[c, s] = Vt_i[c, col_off + s] * sc
                end
            end
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end


"""
    svd_two_site(Φ; left_edge=false, kwargs...)

Convenience wrapper for DMRG two-site SVDs.

`Φ` is the contraction of two site tensors, with index order
`(left_phys, right_phys, left_link_sp, right_link_sp, left_link_d, right_link_d)`
in the bulk, dropping the left link legs at the left edge and the right link
legs at the right edge (right edge doesn't need a flag — `Φ`'s shape tells the
function everything via `P` and `N2`).

Bulk:        Φ has 4 sparse + 2 dense → U(3 sp + 2 d) ⊗ SV(3 sp + 2 d)
Left edge:   Φ has 3 sparse + 1 dense → U(2 sp + 1 d) ⊗ SV(3 sp + 2 d)
Right edge:  Φ has 3 sparse + 1 dense → U(3 sp + 2 d) ⊗ SV(2 sp + 1 d)
"""
function svd_two_site(Φ; left_edge::Bool = false, kwargs...)
    nls = left_edge ? 1 : 2
    nld = left_edge ? 0 : 1
    return blocksparse_svd(Φ; n_left_sparse = nls, n_left_dense = nld, kwargs...)
end




# # Block-diagonal SVD for 2-site phi tensor.
# # A: NewBlockSparseSorted{T,4,2,2} — prefix=(L,R), dense=(s1,s2)
# # Groups non-zero blocks by L value, runs mini-SVD per L group,
# # assembles sparse U and SV factors without full dense materialization.
# # Returns (U_bs, SV_bs, svs_kept, spec).
# function blocksparse_svd(
#     A::NewBlockSparseSorted{T,N,N2,P,K<:Integer};
#     ortho::String="left",
#     maxdim::Int=typemax(Int),
#     mindim::Int=1,
#     cutoff::Float64=0.0,
# ) where {T, N, N2, P, K}
#     n_L, n_R, d_s1, d_s2 = A.dims

#     # Group keys by first prefix dim (L_i)
#     l_to_rlist = Dict{Int, Vector{Int}}()
#     for (key, _) in blocks_sorted(A)
#         push!(get!(l_to_rlist, key[1], Int[]), key[2])
#     end
#     l_vals = sort(collect(keys(l_to_rlist)))

#     # Per-L mini-SVD
#     block_svds = []  # (L_i, U_i, S_i, Vt_i, r_list)
#     for L_i in l_vals
#         r_list = sort(l_to_rlist[L_i])
#         n_Ri   = length(r_list)
#         r_pos  = Dict(r => j for (j, r) in enumerate(r_list))
#         # Mini-matrix: rows=s1, cols=(R_j, s2) stacked
#         M_i = zeros(T, d_s1, n_Ri * d_s2)
#         for (key, id) in blocks_sorted(A)
#             key[1] == L_i || continue
#             rj  = r_pos[key[2]]
#             blk = reshape(_block_view(A, id), d_s1, d_s2)
#             M_i[:, (rj-1)*d_s2+1 : rj*d_s2] .= blk
#         end
#         F = svd(M_i)   # U: d_s1×k, S: k (decreasing), Vt: k×(n_Ri*d_s2)
#         push!(block_svds, (L_i, F.U, F.S, F.Vt, r_list))
#     end

#     # Collect all (sv, block_idx, col_within_block) and apply global truncation
#     all_svs = Tuple{Float64, Int, Int}[]
#     for (bi, (_, _, S_i, _, _)) in enumerate(block_svds)
#         for (col, sv) in enumerate(S_i)
#             push!(all_svs, (Float64(sv), bi, col))
#         end
#     end
#     sort!(all_svs; rev=true, by=x -> x[1])

#     max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
#     keep_count = min(maxdim, length(all_svs))
#     while keep_count > mindim && all_svs[keep_count][1] < cutoff * max_sv
#         keep_count -= 1
#     end
#     truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init=0.0)

#     # Per-block kept count
#     k_kept = zeros(Int, length(block_svds))
#     for i in 1:keep_count
#         k_kept[all_svs[i][2]] += 1
#     end
#     new_bond_dim = sum(k_kept)

#     # Contiguous offsets in the new bond dimension per block
#     offsets = Vector{Int}(undef, length(block_svds))
#     cur = 0
#     for bi in eachindex(block_svds)
#         offsets[bi] = cur
#         cur += k_kept[bi]
#     end

#     U_bs  = NewBlockSparseSorted{T,3,1}((n_L, new_bond_dim, d_s1))
#     SV_bs = NewBlockSparseSorted{T,3,1}((new_bond_dim, n_R, d_s2))
#     svs_kept = zeros(T, new_bond_dim)

#     for (bi, (L_i, U_i, S_i, Vt_i, r_list)) in enumerate(block_svds)
#         ki  = k_kept[bi]
#         ki == 0 && continue
#         off  = offsets[bi]
#         for col in 1:ki
#             nb = off + col
#             svs_kept[nb] = S_i[col]
#             # U block: (L_i, nb) → dense size d_s1
#             id_u = _ensure_block!(U_bs, (L_i, nb))
#             bv_u = _block_view(U_bs, id_u)
#             scale_u = (ortho == "right") ? T(S_i[col]) : one(T)
#             @inbounds for s in 1:d_s1
#                 bv_u[s] = U_i[s, col] * scale_u
#             end
#             # SV blocks: (nb, R_j) for each R_j
#             for (rj, R_j) in enumerate(r_list)
#                 id_sv = _ensure_block!(SV_bs, (nb, R_j))
#                 bv_sv = _block_view(SV_bs, id_sv)
#                 base  = (rj - 1) * d_s2
#                 scale_sv = (ortho == "left") ? T(S_i[col]) : one(T)
#                 @inbounds for s in 1:d_s2
#                     bv_sv[s] = Vt_i[col, base + s] * scale_sv
#                 end
#             end
#         end
#     end

#     spec = Spectrum(svs_kept .^ 2, truncerr)
#     return U_bs, SV_bs, svs_kept, spec
# end