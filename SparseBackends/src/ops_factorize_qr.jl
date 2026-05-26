# Channel-aware QR factorization with Gram-Schmidt-within-row-group.
#
# Bond patterns (for ortho="left"):
#   max_cMs_per_cL == 1  →  CLEAN: each c_L routes to one c_M.
#     - Stack rows from multiple c_L's per c_M (if any).
#     - Single QR per c_M. Cross-c_M iso automatic (disjoint row supports
#       across c_M values from different row groups).
#   max_cMs_per_cL  > 1  →  (2,4,2): one c_L fans out to multiple c_M's.
#     - Group c_M values by their row support (set of c_L's they route from).
#     - Within each row group, process c_M's sequentially with Gram-Schmidt:
#       project out accumulated Q before QR-ing each new c_M's data.
#       Cross-terms get absorbed as NEW block keys in R: (c_M_prev, c_R) where
#       c_R is in the later c_M's c_R-list.
#
# This unified algorithm handles both cases — when row group has size 1 the
# inner GS step degenerates to a single QR (no projection step), recovering
# the CLEAN behavior. Exact iso + exact reconstruction in both cases.
function blocksparse_qr_channel_aware(
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
    relax_iso_cap::Bool = false,
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

    # Templates → maps.
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

    max_cMs_per_cL = isempty(lk_to_channels) ? 0 : maximum(length, values(lk_to_channels))
    max_cLs_per_cM = isempty(cM_to_cLs)      ? 0 : maximum(length, values(cM_to_cLs))
    max_cMs_per_cR = isempty(rk_to_channels) ? 0 : maximum(length, values(rk_to_channels))
    if verbose
        println(stdout, "[QR_CLASSIFY] ortho=$ortho  n_L=$(length(lk_to_channels))  n_M=$bond_sparse_dim  n_R=$(length(rk_to_channels))  max_cMs/cL=$max_cMs_per_cL  max_cLs/cM=$max_cLs_per_cM  max_cMs/cR=$max_cMs_per_cR")
    end

    # For ortho="left": group by SHARED ROW SUPPORT (= set of c_L's routing to a c_M).
    # For ortho="right": group by SHARED COL SUPPORT (= set of c_R's routing from a c_M).
    if ortho == "left"
        cM_to_support = cM_to_cLs
    elseif ortho == "right"
        cM_to_support = cM_to_cRs
    else
        error("unknown ortho=$ortho")
    end

    # Row-group construction by connected components of the lk↔cM bipartite
    # graph: any two cM's whose supports overlap on even one lk must share a
    # row group, otherwise phi[lk, rk] at the shared lk gets reconstructed
    # independently by each row group's QR and double-counted at recon.
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
    # For each lk that has multiple cM's, union them all.
    lk_to_cMs_local = ortho == "left" ? lk_to_channels : rk_to_channels
    for (_, cMs) in lk_to_cMs_local
        cMs_vec = collect(cMs)
        for i in 2:length(cMs_vec)
            _union(K(cMs_vec[1]), K(cMs_vec[i]))
        end
    end
    # Bucket cM's by their root.
    groups_by_root = Dict{K, Vector{K}}()
    for c in cM_keys
        push!(get!(groups_by_root, _find(c), K[]), c)
    end
    # row_groups maps support_set => cM list. For unioned groups, the support
    # is the UNION of member supports.
    row_groups = Dict{Set,Vector{K}}()
    for (_, cMs) in groups_by_root
        merged_support = Set()
        for c in cMs
            union!(merged_support, cM_to_support[c])
        end
        row_groups[merged_support] = sort(cMs)
    end

    # Per-c_M cap to force balanced channel contributions. Without this, the
    # first c_M in each row group's GS iteration absorbs the entire row space
    # and subsequent c_M's contribute 0. With this cap, each c_M contributes
    # at most per_cM_cap cols, leaving room for later c_M's.
    per_cM_cap = max(1, fld(maxdim, max(1, bond_sparse_dim)))

    # Pre-build phi block lookup
    phi_lookup = Dict{Tuple{LKey, RKey}, Int}()
    for (key, id) in blocks_sorted(A)
        lk = ntuple(i -> key[i],       nls)
        rk = ntuple(i -> key[nls + i], nrs)
        phi_lookup[(lk, rk)] = id
    end

    # Output containers: store per (c_L, c_M) for L, per (c_M, c_R) for R.
    L_blocks = Dict{Tuple{LKey, K}, Matrix{T}}()
    R_blocks = Dict{Tuple{K, RKey}, Matrix{T}}()
    cM_chi   = Dict{K, Int}()
    cM_S     = Dict{K, Vector{real(T)}}()

    # Process each row/col group with sequential Gram-Schmidt.
    for (support_set, cM_list_in_group) in row_groups
        sort!(cM_list_in_group)

        if ortho == "left"
            l_list = sort(collect(support_set))
            l_pos  = Dict(lk => i for (i, lk) in enumerate(l_list))
            n_rows_group = length(l_list) * d_left
        else  # ortho == "right"
            r_list_support = sort(collect(support_set))
            r_pos_support  = Dict(rk => i for (i, rk) in enumerate(r_list_support))
            n_cols_group = length(r_list_support) * d_right
        end

        if ortho == "left"
            Q_accum = zeros(T, n_rows_group, 0)
            cM_col_range = Dict{K, UnitRange{Int}}()
            # Primary-ownership: each rk is owned by exactly one cM in the
            # row group, so phi[lk, rk] is never duplicated across M_local's
            # (would otherwise double-count at recon).
            #
            # Two strategies:
            #   default — smallest cM (by sort order) claims; produces
            #             monochannel bonds when one cM's rk_list is a superset.
            #   SB_BALANCED_OWNERSHIP=1 — for each rk, assign to the candidate cM
            #             with the fewest claims so far, ties broken by smallest cM.
            #             Spreads multiplicity across channels for better variational
            #             coverage in DMRG.
            balanced_owner = get(ENV, "SB_BALANCED_OWNERSHIP", "0") == "1"
            cM_owned_rks   = Dict{K, Vector{RKey}}(cM => RKey[] for cM in cM_list_in_group)
            if balanced_owner
                # Build rk → candidate-cM list (cM's whose rk_list contains rk).
                rk_candidates = Dict{RKey, Vector{K}}()
                for cM in cM_list_in_group
                    for rk in cM_to_cRs[cM]
                        push!(get!(rk_candidates, rk, K[]), cM)
                    end
                end
                # Assign in deterministic rk-sorted order, picking least-loaded candidate.
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

                # Gram-Schmidt: project out previously-accumulated directions.
                if size(Q_accum, 2) == 0
                    phi_resid     = M_local
                    cross_to_prev = nothing
                else
                    cross_to_prev = adjoint(Q_accum) * M_local
                    phi_resid     = M_local - Q_accum * cross_to_prev
                end

                # Direct SVD on phi_resid: U cols are an orthonormal basis for
                # phi_resid's column space (= orthogonal complement of Q_accum
                # after GS, intersected with phi_resid's range). True singular
                # values give exact numerical rank — avoids QR's basis-completion
                # cols that would break cross-c_M iso when phi_resid is
                # rank-deficient.
                if isempty(phi_resid) || all(size(phi_resid) .== 0) ||
                   size(phi_resid, 2) == 0
                    Q_M_new = Matrix{T}(undef, n_rows_group, 0)
                    S_r     = real(T)[]
                    R_M_new = Matrix{T}(undef, 0, n_local_cols)
                else
                    Fr = svd(phi_resid)
                    sv_max = isempty(Fr.S) ? 1.0 : Fr.S[1]
                    sv_tol = max(1e-12, 1e-12 * sv_max)
                    chi_rank = count(>(sv_tol), Fr.S)
                    chi_cap = min(chi_rank, per_cM_cap)
                    Q_M_new = Fr.U[:, 1:chi_cap]
                    S_r     = Fr.S[1:chi_cap]
                    R_M_new = Diagonal(S_r) * Fr.Vt[1:chi_cap, :]
                end
                chi_M = size(Q_M_new, 2)
                cM_chi[cM] = chi_M
                cM_S[cM]   = S_r

                # Scatter Q_M_new per c_L.
                for (li, lk) in enumerate(l_list)
                    r0 = (li - 1) * d_left
                    L_blocks[(lk, cM)] = Q_M_new[r0+1 : r0+d_left, :]
                end
                # Scatter R_M_new per c_R for the regular (c_M, c_R) keys.
                for (rk_idx, rk) in enumerate(rk_list_M)
                    c0 = (rk_idx - 1) * d_right
                    R_blocks[(cM, rk)] = R_M_new[:, c0+1 : c0+d_right]
                end

                # Cross-term absorption: extra block keys at (prev_cM, rk_in_current_cM).
                # cross_to_prev (sum_of_prev_chis × n_local_cols) was computed before
                # any truncation of CURRENT c_M (U_r only acts on current), so
                # cross_to_prev[prev_range, :] is still the right gauge for prev_cM.
                if cross_to_prev !== nothing
                    for prev_cM in cM_list_in_group
                        prev_cM == cM && break  # only process predecessors
                        prev_range = cM_col_range[prev_cM]
                        for (rk_idx, rk) in enumerate(rk_list_M)
                            c0  = (rk_idx - 1) * d_right
                            new = cross_to_prev[prev_range, c0+1 : c0+d_right]
                            key = (prev_cM, rk)
                            if haskey(R_blocks, key)
                                # Diagonal R already placed by prev_cM's own pass
                                # at this rk — both contributions to phi sum here.
                                R_blocks[key] = R_blocks[key] .+ new
                            else
                                R_blocks[key] = new
                            end
                        end
                    end
                end

                col_start = size(Q_accum, 2) + 1
                Q_accum   = hcat(Q_accum, Q_M_new)
                cM_col_range[cM] = col_start : (col_start + chi_M - 1)
            end

        else  # ortho == "right" — mirror over cols
            Qt_accum = zeros(T, 0, n_cols_group)  # we accumulate iso ROWS here for right ortho
            cM_row_range = Dict{K, UnitRange{Int}}()
            # Primary-ownership over the lk dimension (mirror of ortho=left).
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

                if size(Qt_accum, 1) == 0
                    phi_resid     = M_local
                    cross_to_prev = nothing
                else
                    cross_to_prev = M_local * adjoint(Qt_accum)
                    phi_resid     = M_local - cross_to_prev * Qt_accum
                end

                # Direct SVD on phi_resid (mirror of ortho=left branch).
                # Vt rows are an orthonormal basis for phi_resid's row space.
                if isempty(phi_resid) || all(size(phi_resid) .== 0) ||
                   size(phi_resid, 1) == 0
                    Qt_M_new = Matrix{T}(undef, 0, n_cols_group)
                    S_r      = real(T)[]
                    L_M_new  = Matrix{T}(undef, n_local_rows, 0)
                else
                    Fr = svd(phi_resid)
                    sv_max = isempty(Fr.S) ? 1.0 : Fr.S[1]
                    sv_tol = max(1e-12, 1e-12 * sv_max)
                    chi_rank = count(>(sv_tol), Fr.S)
                    chi_cap = min(chi_rank, per_cM_cap)
                    Qt_M_new = Fr.Vt[1:chi_cap, :]
                    S_r      = Fr.S[1:chi_cap]
                    L_M_new  = Fr.U[:, 1:chi_cap] * Diagonal(S_r)
                end
                chi_M = size(Qt_M_new, 1)
                cM_chi[cM] = chi_M
                cM_S[cM]   = S_r

                # Scatter Q^T per c_R: R_blocks[(cM, rk)] = Qt_M_new[:, cols_for_rk]
                for (rj, rk) in enumerate(r_list_support)
                    c0 = (rj - 1) * d_right
                    R_blocks[(cM, rk)] = Qt_M_new[:, c0+1 : c0+d_right]
                end
                # Scatter L (the non-iso side) per c_L for THIS cM.
                for (li, lk) in enumerate(lk_list_M)
                    r0 = (li - 1) * d_left
                    L_blocks[(lk, cM)] = L_M_new[r0+1 : r0+d_left, :]
                end

                if cross_to_prev !== nothing
                    for prev_cM in cM_list_in_group
                        prev_cM == cM && break
                        prev_range = cM_row_range[prev_cM]
                        for (li, lk) in enumerate(lk_list_M)
                            r0  = (li - 1) * d_left
                            new = cross_to_prev[r0+1 : r0+d_left, prev_range]
                            key = (lk, prev_cM)
                            if haskey(L_blocks, key)
                                L_blocks[key] = L_blocks[key] .+ new
                            else
                                L_blocks[key] = new
                            end
                        end
                    end
                end

                row_start = size(Qt_accum, 1) + 1
                Qt_accum  = vcat(Qt_accum, Qt_M_new)
                cM_row_range[cM] = row_start : (row_start + chi_M - 1)
            end
        end
    end

    # ---- Global truncation across all channels ------------------------------
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

    # The cross-term R blocks (stored at new keys like (prev_cM, rk_in_cur_M))
    # need ALL chi_prev cols of Q_prev_M_new to reconstruct phi correctly. If
    # global SV truncation drops below chi_prev for any prev_cM that has
    # downstream cross-terms, those cross-terms become unrepresentable. So
    # n_new_d must be at least the MAX pre-truncation chi across c_M's.
    chi_pre_max = isempty(cM_chi) ? 0 : maximum(values(cM_chi); init = 0)
    n_new_d_natural = max(chi_pre_max, isempty(k_kept) ? 0 : maximum(values(k_kept); init = 0))
    # Ensure each c_M's k_kept >= its pre-trunc chi (don't drop cross-term cols).
    for cM in keys(cM_chi)
        k_kept[cM] = max(k_kept[cM], cM_chi[cM])
    end
    n_active_chan   = count(>(0), values(k_kept))
    mult_cap        = max(1, fld(maxdim, max(1, bond_sparse_dim)))
    n_new_d         = relax_iso_cap ? n_new_d_natural : max(chi_pre_max, min(n_new_d_natural, mult_cap))

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
        # Also enumerate row groups and per-c_M chi *before* global truncation
        # so we can see whether GS itself produced multi-channel output or only one.
        rg_info = String[]
        for (support_set, cM_list) in row_groups
            chis = ["$cM=>$(get(cM_chi, cM, 0))" for cM in sort(cM_list)]
            push!(rg_info, "[" * join(chis, ",") * "]")
        end
        rg_info_str = join(rg_info, "|")
        println(stdout, "[QR_TRUNC] maxdim=$maxdim keep_count=$keep_count n_active=$n_active_chan mult_cap=$mult_cap n_new_d_natural=$n_new_d_natural n_new_d=$n_new_d  k_kept=$(sort(collect(k_kept)))  pre_trunc_chi_by_row_group=$rg_info_str  bond=$(bond_sparse_dim)x$(n_new_d)=$(bond_sparse_dim*n_new_d)")
    end

    # ---- Build output BS tensors ---------------------------------------------
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

    # Scatter L_blocks → U_bs. For my kernel, S is ALREADY baked into the R-side
    # blocks (R_M_new = Sigma * Vt; cross blocks are gauge coefficients with no
    # explicit S to apply). So the write loop uses sc=1 throughout — no second
    # multiplication. (Different convention from blocksparse_svd_channel_aware_fixed,
    # which stores Vt and multiplies by S at write time.)
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
