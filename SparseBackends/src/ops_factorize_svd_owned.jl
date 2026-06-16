# Channel-aware per-cM SVD with primary-ownership and NO Gram-Schmidt.
#
# Designed for Path B (BMF_ISO_PATH=0): geneigsolve absorbs cross-channel
# non-orthogonality via the metric M = Lgram ⊗ Rgram, so we do not need
# cross-cM iso. The only correctness requirement is exact reconstruction
# L · R = phi.
#
# Primary-ownership: each (lk, rk) cell is owned by exactly one cM in its
# row group. Each cM's per-channel SVD sees phi restricted to ITS owned
# cells; non-owned cells are absent from that cM's local slab. On
# reconstruction, every (lk, rk) is covered by exactly one term, so the
# sum recovers phi exactly with no double-counting.
#
# vs `blocksparse_qr_channel_aware`:
#   - SAME: union-find row grouping, primary-ownership, per-cM SVD,
#           global SV truncation across channels, output layout.
#   - DIFFERENT (lighter):
#       * No Q_accum tracking.
#       * No cross_to_prev computation.
#       * No cross-term R-block scatter (L_blocks/R_blocks only carry
#         the (cM, rk)/(lk, cM) diagonal keys, no (prev_cM, rk) pairs).
#       * SVD acts directly on M_local instead of phi_resid.
#
# Result: L'L ≠ I across channels at shared lk's (relaxed iso). Path B's
# M-correction in geneigsolve handles the slack.
function blocksparse_svd_owned_channel_aware(
    A::NewBlockSparseSorted{T,N,N2,P,K};
    n_left_sparse::Int,
    n_left_dense::Int,
    left_template::Vector{<:Tuple},
    right_template::Vector{<:Tuple},
    bond_sparse_dim::Int,
    ortho::String     = "left",
    maxdim::Int       = typemax(Int),
    mindim::Int       = 1,
    cutoff::Float64   = 0.0,
    verbose::Bool     = false,
) where {T, N, N2, P, K<:Integer}
    nls, nrs = n_left_sparse, P  - n_left_sparse
    nld, nrd = n_left_dense,  N2 - n_left_dense
    @assert 0 ≤ nls ≤ P
    @assert 0 ≤ nld ≤ N2

    dims = A.dims
    left_sp_dims  = ntuple(i -> dims[i],            nls)
    right_sp_dims = ntuple(i -> dims[nls + i],      nrs)
    left_d_dims   = ntuple(i -> dims[P + i],        nld)
    right_d_dims  = ntuple(i -> dims[P + nld + i],  nrd)
    d_left  = prod(left_d_dims;  init = 1)
    d_right = prod(right_d_dims; init = 1)

    LKey = NTuple{nls, K}
    RKey = NTuple{nrs, K}

    lk_to_channels = Dict{LKey, Set{K}}()
    cM_to_cLs      = Dict{K, Set{LKey}}()
    for entry in left_template
        lk_raw, k = entry
        lk = NTuple{nls,K}(lk_raw)
        push!(get!(lk_to_channels, lk, Set{K}()), K(k))
        push!(get!(cM_to_cLs, K(k), Set{LKey}()), lk)
    end
    rk_to_channels = Dict{RKey, Set{K}}()
    cM_to_cRs      = Dict{K, Set{RKey}}()
    for entry in right_template
        k, rk_raw = entry
        rk = NTuple{nrs,K}(rk_raw)
        push!(get!(rk_to_channels, rk, Set{K}()), K(k))
        push!(get!(cM_to_cRs, K(k), Set{RKey}()), rk)
    end

    if ortho == "left"
        cM_to_support = cM_to_cLs
    elseif ortho == "right"
        cM_to_support = cM_to_cRs
    else
        error("unknown ortho=$ortho")
    end

    # Row-group construction by connected components of the lk↔cM bipartite
    # graph (identical to QR variant — needed for primary-ownership scope).
    cM_keys = collect(keys(cM_to_support))
    parent  = Dict{K, K}(c => c for c in cM_keys)
    function _find(x::K)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    function _union(a::K, b::K)
        ra = _find(a); rb = _find(b)
        ra == rb && return
        parent[ra] = rb
    end
    lk_to_cMs_local = ortho == "left" ? lk_to_channels : rk_to_channels
    for (_, cMs) in lk_to_cMs_local
        cMs_vec = collect(cMs)
        for i in 2:length(cMs_vec)
            _union(K(cMs_vec[1]), K(cMs_vec[i]))
        end
    end
    groups_by_root = Dict{K, Vector{K}}()
    for c in cM_keys
        push!(get!(groups_by_root, _find(c), K[]), c)
    end
    row_groups = Dict{Set,Vector{K}}()
    for (_, cMs) in groups_by_root
        merged_support = Set()
        for c in cMs
            union!(merged_support, cM_to_support[c])
        end
        row_groups[merged_support] = sort(cMs)
    end

    per_cM_cap = max(1, fld(maxdim, max(1, bond_sparse_dim)))

    adaptive_rank = get(ENV, "SB_ADAPTIVE_RANK", "0") == "1"
    adaptive_rel  = let s = get(ENV, "SB_ADAPTIVE_REL", "")
        isempty(s) ? 1e-8 : parse(Float64, s)
    end
    percM_sv_rel = adaptive_rank ? adaptive_rel : 1e-12

    phi_lookup = Dict{Tuple{LKey, RKey}, Int}()
    for (key, id) in blocks_sorted(A)
        lk = ntuple(i -> key[i],       nls)
        rk = ntuple(i -> key[nls + i], nrs)
        phi_lookup[(lk, rk)] = id
    end

    L_blocks = Dict{Tuple{LKey, K}, Matrix{T}}()
    R_blocks = Dict{Tuple{K, RKey}, Matrix{T}}()
    cM_chi   = Dict{K, Int}()
    cM_S     = Dict{K, Vector{real(T)}}()

    for (support_set, cM_list_in_group) in row_groups
        sort!(cM_list_in_group)

        if ortho == "left"
            l_list = sort(collect(support_set))
            l_pos  = Dict(lk => i for (i, lk) in enumerate(l_list))
            n_rows_group = length(l_list) * d_left

            # Primary-ownership of rk cells across cM's in this row group.
            balanced_owner = get(ENV, "SB_BALANCED_OWNERSHIP", "0") == "1"
            cM_owned_rks   = Dict{K, Vector{RKey}}(cM => RKey[] for cM in cM_list_in_group)
            if balanced_owner
                rk_candidates = Dict{RKey, Vector{K}}()
                for cM in cM_list_in_group
                    for rk in cM_to_cRs[cM]
                        push!(get!(rk_candidates, rk, K[]), cM)
                    end
                end
                for rk in sort(collect(keys(rk_candidates)))
                    cands = rk_candidates[rk]
                    best  = cands[1]; best_n = length(cM_owned_rks[best])
                    for c in cands
                        n = length(cM_owned_rks[c])
                        if n < best_n || (n == best_n && c < best)
                            best = c; best_n = n
                        end
                    end
                    push!(cM_owned_rks[best], rk)
                end
                for cM in cM_list_in_group; sort!(cM_owned_rks[cM]); end
            else
                assigned_rks_in_group = Set{RKey}()
                for cM in cM_list_in_group
                    full_rk_list = sort(collect(cM_to_cRs[cM]))
                    for rk in full_rk_list
                        if !(rk in assigned_rks_in_group)
                            push!(cM_owned_rks[cM], rk)
                            push!(assigned_rks_in_group, rk)
                        end
                    end
                end
            end

            for cM in cM_list_in_group
                rk_list_M = cM_owned_rks[cM]
                n_local_cols = length(rk_list_M) * d_right
                M_local = zeros(T, n_rows_group, n_local_cols)
                for (rk_idx, rk) in enumerate(rk_list_M)
                    for lk in l_list
                        if haskey(phi_lookup, (lk, rk))
                            id  = phi_lookup[(lk, rk)]
                            blk = reshape(_block_view(A, id), d_left, d_right)
                            li  = l_pos[lk]
                            r0  = (li - 1) * d_left
                            c0  = (rk_idx - 1) * d_right
                            M_local[r0+1 : r0+d_left, c0+1 : c0+d_right] .= blk
                        end
                    end
                end

                # Direct SVD on M_local — no GS, no Q_accum, no cross-terms.
                # Non-iso across cM's at shared lk's is expected and absorbed
                # by Path B's geneigsolve metric.
                if isempty(M_local) || all(size(M_local) .== 0) ||
                   size(M_local, 2) == 0
                    U_M     = Matrix{T}(undef, n_rows_group, 0)
                    S_r     = real(T)[]
                    SVt_M   = Matrix{T}(undef, 0, n_local_cols)
                else
                    Fr = svd(M_local)
                    sv_max = isempty(Fr.S) ? 1.0 : Fr.S[1]
                    sv_tol = max(1e-14, percM_sv_rel * sv_max)
                    chi_rank = count(>(sv_tol), Fr.S)
                    chi_cap = min(chi_rank, per_cM_cap)
                    U_M     = Fr.U[:, 1:chi_cap]
                    S_r     = Fr.S[1:chi_cap]
                    SVt_M   = Diagonal(S_r) * Fr.Vt[1:chi_cap, :]
                end
                chi_M = size(U_M, 2)
                cM_chi[cM] = chi_M
                cM_S[cM]   = S_r

                for (li, lk) in enumerate(l_list)
                    r0 = (li - 1) * d_left
                    L_blocks[(lk, cM)] = U_M[r0+1 : r0+d_left, :]
                end
                for (rk_idx, rk) in enumerate(rk_list_M)
                    c0 = (rk_idx - 1) * d_right
                    R_blocks[(cM, rk)] = SVt_M[:, c0+1 : c0+d_right]
                end
            end

        else  # ortho == "right" — mirror over cols
            r_list_support = sort(collect(support_set))
            r_pos_support  = Dict(rk => i for (i, rk) in enumerate(r_list_support))
            n_cols_group = length(r_list_support) * d_right

            balanced_owner = get(ENV, "SB_BALANCED_OWNERSHIP", "0") == "1"
            cM_owned_lks   = Dict{K, Vector{LKey}}(cM => LKey[] for cM in cM_list_in_group)
            if balanced_owner
                lk_candidates = Dict{LKey, Vector{K}}()
                for cM in cM_list_in_group
                    for lk in cM_to_cLs[cM]
                        push!(get!(lk_candidates, lk, K[]), cM)
                    end
                end
                for lk in sort(collect(keys(lk_candidates)))
                    cands = lk_candidates[lk]
                    best  = cands[1]; best_n = length(cM_owned_lks[best])
                    for c in cands
                        n = length(cM_owned_lks[c])
                        if n < best_n || (n == best_n && c < best)
                            best = c; best_n = n
                        end
                    end
                    push!(cM_owned_lks[best], lk)
                end
                for cM in cM_list_in_group; sort!(cM_owned_lks[cM]); end
            else
                assigned_lks_in_group = Set{LKey}()
                for cM in cM_list_in_group
                    full_lk_list = sort(collect(cM_to_cLs[cM]))
                    for lk in full_lk_list
                        if !(lk in assigned_lks_in_group)
                            push!(cM_owned_lks[cM], lk)
                            push!(assigned_lks_in_group, lk)
                        end
                    end
                end
            end

            for cM in cM_list_in_group
                lk_list_M = cM_owned_lks[cM]
                n_local_rows = length(lk_list_M) * d_left
                M_local = zeros(T, n_local_rows, n_cols_group)
                for (lk_idx, lk) in enumerate(lk_list_M)
                    for rk in r_list_support
                        if haskey(phi_lookup, (lk, rk))
                            id  = phi_lookup[(lk, rk)]
                            blk = reshape(_block_view(A, id), d_left, d_right)
                            rj  = r_pos_support[rk]
                            r0  = (lk_idx - 1) * d_left
                            c0  = (rj - 1) * d_right
                            M_local[r0+1 : r0+d_left, c0+1 : c0+d_right] .= blk
                        end
                    end
                end

                if isempty(M_local) || all(size(M_local) .== 0) ||
                   size(M_local, 1) == 0
                    Vt_M   = Matrix{T}(undef, 0, n_cols_group)
                    S_r    = real(T)[]
                    US_M   = Matrix{T}(undef, n_local_rows, 0)
                else
                    Fr = svd(M_local)
                    sv_max = isempty(Fr.S) ? 1.0 : Fr.S[1]
                    sv_tol = max(1e-14, percM_sv_rel * sv_max)
                    chi_rank = count(>(sv_tol), Fr.S)
                    chi_cap = min(chi_rank, per_cM_cap)
                    Vt_M   = Fr.Vt[1:chi_cap, :]
                    S_r    = Fr.S[1:chi_cap]
                    US_M   = Fr.U[:, 1:chi_cap] * Diagonal(S_r)
                end
                chi_M = size(Vt_M, 1)
                cM_chi[cM] = chi_M
                cM_S[cM]   = S_r

                for (rj, rk) in enumerate(r_list_support)
                    c0 = (rj - 1) * d_right
                    R_blocks[(cM, rk)] = Vt_M[:, c0+1 : c0+d_right]
                end
                for (li, lk) in enumerate(lk_list_M)
                    r0 = (li - 1) * d_left
                    L_blocks[(lk, cM)] = US_M[r0+1 : r0+d_left, :]
                end
            end
        end
    end

    # Global SV truncation across channels (identical to QR variant).
    all_svs = Tuple{Float64, K, Int}[]
    for (cM, S) in cM_S, (col, sv) in enumerate(S)
        push!(all_svs, (Float64(sv), cM, col))
    end
    sort!(all_svs; rev = true, by = x -> x[1])

    max_sv     = isempty(all_svs) ? 1.0 : all_svs[1][1]
    sv_floor   = max(cutoff * max_sv, 1e-12 * max(max_sv, 1.0))
    keep_count = min(maxdim, length(all_svs))
    while keep_count > mindim && all_svs[keep_count][1] < sv_floor
        keep_count -= 1
    end
    truncerr = sum(x[1]^2 for x in all_svs[keep_count+1:end]; init = 0.0)

    k_kept = Dict{K, Int}()
    for cM in keys(cM_chi); k_kept[cM] = 0; end
    for i in 1:keep_count
        cM = all_svs[i][2]
        k_kept[cM] += 1
    end

    # Without cross-term R blocks the floor logic from the QR variant is
    # unnecessary — k_kept already matches the natural per-cM count.
    chi_pre_max     = isempty(cM_chi) ? 0 : maximum(values(cM_chi); init = 0)
    n_new_d_natural = max(chi_pre_max, isempty(k_kept) ? 0 : maximum(values(k_kept); init = 0))
    n_active_chan   = count(>(0), values(k_kept))
    mult_cap        = max(1, fld(maxdim, max(1, bond_sparse_dim)))
    n_new_d         = min(n_new_d_natural, mult_cap)

    if n_new_d < n_new_d_natural
        for (cM, kk) in k_kept
            if kk > n_new_d
                S_i = cM_S[cM]
                for c in (n_new_d + 1):kk
                    c <= length(S_i) && (truncerr += S_i[c]^2)
                end
                k_kept[cM] = n_new_d
            end
        end
    end
    if verbose
        rg_info = String[]
        for (support_set, cM_list) in row_groups
            chis = ["$cM=>$(get(cM_chi, cM, 0))" for cM in sort(cM_list)]
            push!(rg_info, "[" * join(chis, ",") * "]")
        end
        rg_info_str = join(rg_info, "|")
        println(stdout, "[SVD_OWNED_TRUNC] maxdim=$maxdim keep_count=$keep_count n_active=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d  k_kept=$(sort(collect(k_kept)))  pre_trunc_chi_by_row_group=$rg_info_str  bond=$(bond_sparse_dim)x$(n_new_d)=$(bond_sparse_dim*n_new_d)")
    end

    # Build output BS tensors.
    n_new_sp = bond_sparse_dim
    NU  = (nls + 1) + (nld + 1)
    NSV = (1 + nrs) + (1 + nrd)
    U_bs  = NewBlockSparseSorted{T, NU,  nld + 1}(
        (left_sp_dims...,  n_new_sp, left_d_dims...,  n_new_d))
    SV_bs = NewBlockSparseSorted{T, NSV, nrd + 1}(
        (n_new_sp, right_sp_dims..., n_new_d, right_d_dims...))

    svs_kept = zeros(real(T), n_new_sp, n_new_d)

    for cM in keys(cM_chi)
        ki = k_kept[cM]
        ki == 0 && continue
        S_r = cM_S[cM]
        @inbounds for c in 1:ki
            svs_kept[cM, c] = (c <= length(S_r)) ? S_r[c] : zero(real(T))
        end
    end

    for ((lk, cM), Q_block) in L_blocks
        ki = k_kept[cM]
        ki == 0 && continue
        umax = min(n_new_d, size(Q_block, 2))
        id_u = _ensure_block!(U_bs, (lk..., cM))
        u_mat = reshape(_block_view(U_bs, id_u), d_left, n_new_d)
        @inbounds for c in 1:umax
            for s in 1:d_left
                u_mat[s, c] = Q_block[s, c]
            end
        end
    end

    for ((cM, rk), R_block) in R_blocks
        ki = k_kept[cM]
        ki == 0 && continue
        vmax = min(n_new_d, size(R_block, 1))
        id_sv  = _ensure_block!(SV_bs, (cM, rk...))
        sv_mat = reshape(_block_view(SV_bs, id_sv), n_new_d, d_right)
        @inbounds for c in 1:vmax
            for s in 1:d_right
                sv_mat[c, s] = R_block[c, s]
            end
        end
    end

    spec = Spectrum([all_svs[i][1]^2 for i in 1:keep_count], truncerr)
    return U_bs, SV_bs, svs_kept, spec
end
