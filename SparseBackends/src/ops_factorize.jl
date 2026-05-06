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