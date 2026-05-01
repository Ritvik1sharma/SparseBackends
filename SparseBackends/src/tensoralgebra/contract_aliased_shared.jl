# tensoralgebra/contract_aliased_shared.jl
#
# Multi-label contraction kernels for AliasedBlockSparse tensors.
# These are the AliasedBlockSparse analogues of contract_shared! in
# contract_bs_dense.jl (AD case) and contract_bs_bs.jl (AA case).
#
# ─ Kernels provided ──────────────────────────────────────────────────────────
#
#   contract_shared!(C::AliasedBS, ..., A::AliasedBS, ..., B::Dense, ...)
#     Multiple shared (reduced) labels between AliasedBS A and dense B.
#     Shared labels partition into:
#       shared_prefix  → labels in A's sparse prefix; index B slices.
#       shared_dense   → labels in A's dense tail; form GEMM reduction axes.
#
#     Combined template key: (tidA, sp_lin) where sp_lin is the column-major
#     linear index of the shared-prefix values into B's permuted layout.
#     When n_sp == 0 the key reduces to (tidA, 1), giving at most n_tmplA
#     combined templates (perfect aliasing preservation, same as the
#     single-rlab dense kernel).
#
#   contract_shared!(C::AliasedBS, ..., A::AliasedBS, ..., B::AliasedBS, ...)
#     Multiple shared (reduced) labels between two AliasedBS tensors.
#     Key efficiency property: the combined template
#         GEMM(template_A[tidA], template_B[tidB])
#     is independent of the join-group (sp_vals), so the combined template map
#     is keyed on (tidA, tidB) — at most n_tmplA × n_tmplB templates regardless
#     of the number of distinct join groups.
#
# Both kernels reuse _aliased_contribute! and _commit_aliased_dicts! from
# contract_aliased.jl for the scalar-accumulation / aliasing bookkeeping.

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Permutation-plan helpers for AliasedBlockSparse
# (Identical bodies to the NewBlockSparseSorted overloads in contract_bs_bs.jl;
#  only the dispatch tag differs so the existing helpers can be reused here.)
# ─────────────────────────────────────────────────────────────────────────────

@inline function _find_perm_for_A_join_and_dense_order(
    :: AliasedBlockSparse{T,NA,N2A,PA},
    labelsA      :: AbstractVector,
    mapA         :: Dict,
    shared_prefix :: AbstractVector,
    desired_keepA :: AbstractVector,
    red_dense     :: AbstractVector,
) where {T,NA,N2A,PA}
    pref_axes        = collect(1:PA)
    shared_pref_axes = Int[mapA[lab] for lab in shared_prefix]
    shared_pref_set  = Set(shared_pref_axes)
    keep_pref_axes   = [ax for ax in pref_axes if !(ax in shared_pref_set)]
    perm_prefix      = vcat(keep_pref_axes, shared_pref_axes)

    keep_axes        = Int[(mapA[lab] - PA) for lab in desired_keepA]
    red_axes         = Int[(mapA[lab] - PA) for lab in red_dense]
    @assert length(keep_axes) + length(red_axes) == N2A
    perm_dense_local = vcat(keep_axes, red_axes)
    return vcat(perm_prefix, PA .+ perm_dense_local)
end

@inline function _find_perm_for_B_join_and_dense_redfirst_order(
    :: AliasedBlockSparse{T,NB,N2B,PB},
    labelsB      :: AbstractVector,
    mapB         :: Dict,
    shared_prefix :: AbstractVector,
    red_dense     :: AbstractVector,
    desired_keepB :: AbstractVector,
) where {T,NB,N2B,PB}
    pref_axes        = collect(1:PB)
    shared_pref_axes = Int[mapB[lab] for lab in shared_prefix]
    shared_pref_set  = Set(shared_pref_axes)
    keep_pref_axes   = [ax for ax in pref_axes if !(ax in shared_pref_set)]
    perm_prefix      = vcat(keep_pref_axes, shared_pref_axes)

    red_axes         = Int[(mapB[lab] - PB) for lab in red_dense]
    keep_axes        = Int[(mapB[lab] - PB) for lab in desired_keepB]
    @assert length(red_axes) + length(keep_axes) == N2B
    perm_dense_local = vcat(red_axes, keep_axes)
    return vcat(perm_prefix, PB .+ perm_dense_local)
end


# ─────────────────────────────────────────────────────────────────────────────
# contract_shared! : AliasedBlockSparse × Dense  →  AliasedBlockSparse
# ─────────────────────────────────────────────────────────────────────────────

"""
    contract_shared!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared_labels)

Multi-label contraction `C = A ⊗_{shared_labels} B` where `A` is
`AliasedBlockSparse`, `B` is a dense array, and `C` is `AliasedBlockSparse`.

All labels in `shared_labels` must be reduced (absent from `labelsC`).
They are partitioned by where they appear in `A`:
- `shared_prefix` — in A's sparse prefix → these index into B slices
- `shared_dense`  — in A's dense tail    → GEMM reduction axes

Combined template key: `(tidA, sp_lin)` where `sp_lin` is the 1-based
column-major linear index into B's shared-prefix dimensions.  When there are
no shared-prefix labels `sp_lin == 1` always, giving at most `n_A_templates`
combined templates (perfect aliasing preservation).
"""
function contract_shared!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC       :: AbstractVector,
    A             :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA       :: AbstractVector,
    B             :: AbstractArray{TB,NB},
    labelsB       :: AbstractVector,
    mapA          :: Dict,
    mapB          :: Dict,
    shared_labels :: Vector,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

    # ── 0) Preconditions ──────────────────────────────────────────────────────
    @inbounds for lab in shared_labels
        @assert !(lab in labelsC) "shared label $lab must be reduced (not in labelsC)"
    end

    # Clear output
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    # ── 1) Classify shared labels ─────────────────────────────────────────────
    shared_prefix = eltype(shared_labels)[]
    shared_dense  = eltype(shared_labels)[]
    @inbounds for lab in shared_labels
        @assert haskey(mapA, lab) "shared label $lab must exist in A"
        @assert haskey(mapB, lab) "shared label $lab must exist in B"
        if mapA[lab] <= PA
            push!(shared_prefix, lab)
        else
            push!(shared_dense, lab)
        end
    end

    # ── 2) Keep / reduction sets ──────────────────────────────────────────────
    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB
    redset  = Set(shared_dense)

    red_dense = [lab for lab in Adense0 if lab in redset]
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]
    keepB0    = [lab for lab in Bdense0 if !(lab in Set(shared_labels))]

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved &&
        error("Cdense interleaves A/B kept dims; unsupported — reorder C labels so A-kept and B-kept form contiguous groups")

    @assert length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0)
    @assert length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)

    # ── 3) Permute A: prefix=[keep_pref..., shared_pref...], dense=[keepA..., red...] ──
    permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)
    if permA != collect(1:NA)
        A       = permutedims(A, permA)
        labelsA = labelsA[permA]
        mapA    = Dict(l => i for (i, l) in enumerate(labelsA))
    end

    # ── 4) Permute B to layout (shared_prefix..., red_dense..., keepB...) ─────
    sp_axes_B = Int[mapB[lab] for lab in shared_prefix]
    rd_axes_B = Int[mapB[lab] for lab in red_dense]
    kb_axes_B = Int[mapB[lab] for lab in desired_keepB]
    permB     = vcat(sp_axes_B, rd_axes_B, kb_axes_B)
    @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"
    Bp = (permB == collect(1:NB)) ? B : permutedims(B, permB)

    n_sp    = length(shared_prefix)
    n_red   = length(red_dense)
    n_keepA = length(desired_keepA)
    n_keepB = length(desired_keepB)

    # ── 5) Dense contraction shapes ───────────────────────────────────────────
    dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))    # [keepA..., red...]  after permA
    dimsB_p     = size(Bp)                               # (sp..., red..., keepB...)

    M = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
    K = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+1:end])
    N = (n_keepB == 0) ? 1 : prod(dimsB_p[(n_sp + n_red + 1):end])

    if n_red > 0
        @assert K == prod(dimsB_p[(n_sp + 1):(n_sp + n_red)]) "Reduction extent mismatch between A dense tail and B"
    end
    @assert C.blksize == M * N "C.blksize must equal M*N (M=$M, N=$N, got C.blksize=$(C.blksize))"

    # ── 6) C prefix sourcing (from A sparse prefix, excluding shared_prefix) ──
    c_src_axes   = Vector{Int}(undef, PC)
    shared_p_set = Set(shared_prefix)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        @assert haskey(mapA, lab) "C prefix label $lab must exist in A"
        apos = mapA[lab]
        @assert apos <= PA "C prefix label $lab must come from A sparse prefix"
        @assert !(lab in shared_p_set) "C prefix label $lab cannot be a reduced shared-prefix label"
        c_src_axes[j] = apos
    end
    join_posA = Int[mapA[lab] for lab in shared_prefix]   # positions of shared_prefix in A.key

    # Precompute strides into B's shared-prefix dimensions for sp_lin computation.
    # sp_lin is the 1-based col-major linear index in (dimsB_p[1], ..., dimsB_p[n_sp]).
    sp_strides = Vector{Int}(undef, max(n_sp, 1))
    if n_sp > 0
        sp_strides[1] = 1
        for t in 2:n_sp
            sp_strides[t] = sp_strides[t-1] * dimsB_p[t-1]
        end
    end

    can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)

    # Combined template deduplication: (tidA, sp_lin) → combined_tid in C.
    # For n_sp == 0 sp_lin is always 1, so the map is keyed effectively on tidA.
    combined_tid_map = Dict{Tuple{Int,Int}, Int}()

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # ── 7) Main loop: runs of A sharing the same shared-prefix values ─────────
    iA = firstindex(A.keys); nA = lastindex(A.keys)

    @inbounds while iA <= nA
        iA2   = _advance_run(A.keys, iA, nA, join_posA)   # end-of-run (exclusive)
        akey0 = A.keys[iA]

        # Column-major linear index of this run's shared-prefix values in B
        sp_lin = 1
        for t in 1:n_sp
            sp_lin += (akey0[join_posA[t]] - 1) * sp_strides[t]
        end

        # Slice Bp along its leading n_sp dimensions
        if n_sp == 0
            Bsub = Bp
        else
            sp_vals = ntuple(t -> akey0[join_posA[t]], n_sp)
            idx     = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
            @views Bsub = Bp[idx...]
        end
        Bmat = reshape(Bsub, K, N)    # (K, N): red then keepB

        for ii in iA:(iA2-1)
            akey = A.keys[ii]
            tidA = A.alias_ids[ii]
            αA   = convert(TC, A.scalars[ii])

            # Get or compute combined template for (tidA, sp_lin)
            ct_key       = (tidA, sp_lin)
            combined_tid = get(combined_tid_map, ct_key, 0)
            if combined_tid == 0
                C.n_templates += 1
                combined_tid = C.n_templates
                combined_tid_map[ct_key] = combined_tid

                tmpl_A   = _aliased_template_view(A, tidA)    # length M*K
                Amat     = reshape(tmpl_A, M, K)               # (M, K): keepA then red
                new_tmpl = Vector{TC}(undef, C.blksize)

                if mode == :AthenB
                    # C (M, N) = Amat (M, K) @ Bmat (K, N)
                    Cmat = reshape(new_tmpl, M, N)
                    if can_blas
                        mul!(Cmat,
                             convert(Matrix{TC}, Amat),
                             convert(Matrix{TC}, Bmat))
                    else
                        fill!(new_tmpl, zero(TC))
                        for k in 1:K
                            _rank1_add_generic!(new_tmpl, one(TC),
                                                @view(Amat[:, k]), @view(Bmat[k, :]))
                        end
                    end
                else  # :BthenA  →  C (N, M) = Bmat^T (N, K) @ Amat^T (K, M)
                    Cmat = reshape(new_tmpl, N, M)
                    if can_blas
                        mul!(Cmat,
                             convert(Matrix{TC}, transpose(Bmat)),
                             convert(Matrix{TC}, transpose(Amat)))
                    else
                        fill!(new_tmpl, zero(TC))
                        for k in 1:K
                            _rank1_add_generic!(new_tmpl, one(TC),
                                                @view(Bmat[k, :]), @view(Amat[:, k]))
                        end
                    end
                end

                append!(C.templates, new_tmpl)
            end   # combined template

            ckey = ntuple(j -> akey[c_src_axes[j]], Val(PC))
            _aliased_contribute!(key_to_alias, key_to_accum, C.templates,
                                 ckey, combined_tid, αA, C.blksize)
        end   # ii loop

        iA = iA2
    end   # main while

    _commit_aliased_dicts!(C, key_to_alias, key_to_accum)
    return C
end


# ─────────────────────────────────────────────────────────────────────────────
# contract_shared! : AliasedBlockSparse × AliasedBlockSparse  →  AliasedBlockSparse
# ─────────────────────────────────────────────────────────────────────────────

"""
    contract_shared!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared_labels)

Multi-label contraction `C = A ⊗_{shared_labels} B` where both `A` and `B`
are `AliasedBlockSparse` and `C` is `AliasedBlockSparse`.

Shared labels are partitioned by position in A and B:
- `shared_prefix` — in the sparse prefix of BOTH A and B (join axes; merge-join)
- `shared_dense`  — in the dense tail of BOTH A and B (GEMM reduction axes)
  (cross prefix/dense is not supported and raises an error.)

Key efficiency: `GEMM(template_A[tidA], template_B[tidB])` does not depend on
the join-group that matched the two blocks, so the combined template map is
keyed on `(tidA, tidB)` — at most `n_A_templates × n_B_templates` unique
templates regardless of how many join-group matchings there are.
"""
function contract_shared!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC       :: AbstractVector,
    A             :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA       :: AbstractVector,
    B             :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB       :: AbstractVector,
    mapA          :: Dict,
    mapB          :: Dict,
    shared_labels :: Vector,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB,N2B,PB}

    # ── 0) Preconditions ──────────────────────────────────────────────────────
    @inbounds for lab in shared_labels
        @assert !(lab in labelsC) "shared label $lab must be reduced (not in labelsC)"
    end

    # Clear output
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    # ── 1) Classify shared labels ─────────────────────────────────────────────
    shared_prefix = eltype(shared_labels)[]
    shared_dense  = eltype(shared_labels)[]
    @inbounds for lab in shared_labels
        @assert haskey(mapA, lab) && haskey(mapB, lab) "shared label $lab must exist in both A and B"
        a_pos = mapA[lab]; b_pos = mapB[lab]
        a_pref = a_pos <= PA; b_pref = b_pos <= PB
        if a_pref && b_pref
            push!(shared_prefix, lab)
        elseif (!a_pref) && (!b_pref)
            push!(shared_dense, lab)
        else
            error("shared label $lab crosses prefix/dense boundary " *
                  "(A pos=$a_pos PA=$PA, B pos=$b_pos PB=$PB); not supported")
        end
    end

    # ── 2) Keep / reduction sets ──────────────────────────────────────────────
    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB[PB+1:NB]
    redset  = Set(shared_dense)

    red_dense = [lab for lab in Adense0 if lab in redset]
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]
    keepB0    = [lab for lab in Bdense0 if !(lab in redset)]

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved &&
        error("Cdense interleaves A/B kept dims; unsupported — reorder C labels so A-kept and B-kept form contiguous groups")

    @assert length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0)
    @assert length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)

    # ── 3) Permute A and B ────────────────────────────────────────────────────
    # A: prefix=[keep_pref..., shared_pref...], dense=[keepA..., red...]
    # B: prefix=[keep_pref..., shared_pref...], dense=[red...,   keepB...]
    permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)
    permB = _find_perm_for_B_join_and_dense_redfirst_order(B, labelsB, mapB, shared_prefix, red_dense, desired_keepB)

    A       = permutedims(A, permA)
    B       = permutedims(B, permB)
    labelsA = labelsA[permA]
    labelsB = labelsB[permB]
    mapA    = Dict(l => i for (i, l) in enumerate(labelsA))
    mapB    = Dict(l => i for (i, l) in enumerate(labelsB))

    n_keepA = length(desired_keepA)
    n_keepB = length(desired_keepB)
    n_red   = length(red_dense)

    # ── 4) Dense contraction shapes ───────────────────────────────────────────
    dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))   # [keepA..., red...]
    dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2B))   # [red...,   keepB...]

    M = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
    K = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+1:end])
    N = (n_keepB == 0) ? 1 : prod(dimsB_dense[n_red+1:end])

    if n_red > 0
        @assert K == prod(dimsB_dense[1:n_red]) "Reduction extent mismatch between A and B dense tails"
    end
    @assert C.blksize == M * N "C.blksize must equal M*N (M=$M, N=$N, got C.blksize=$(C.blksize))"

    # ── 5) C prefix sourcing ──────────────────────────────────────────────────
    # src[j] > 0  →  A.key[src[j]]
    # src[j] < 0  →  B.key[-src[j]]
    src = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab) && mapA[lab] <= PA
            src[j] = mapA[lab]
        elseif haskey(mapB, lab) && mapB[lab] <= PB
            src[j] = -mapB[lab]
        else
            error("C prefix label $lab must come from A or B sparse prefix (excluding shared_prefix)")
        end
    end
    join_posA = Int[mapA[lab] for lab in shared_prefix]
    join_posB = Int[mapB[lab] for lab in shared_prefix]

    can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)

    # Combined template deduplication: (tidA, tidB) → combined_tid in C.
    # This map is independent of the join group because templates carry only
    # dense content — the scalar factors (αA, αB) account for the sparse structure.
    combined_tid_map = Dict{Tuple{Int,Int}, Int}()

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # ── 6) Merge-join loop ────────────────────────────────────────────────────
    iA = firstindex(A.keys); nA = lastindex(A.keys)
    iB = firstindex(B.keys); nB = lastindex(B.keys)

    @inbounds while iA <= nA && iB <= nB
        cmp = _cmp_join_tuple(A.keys[iA], B.keys[iB], join_posA, join_posB)
        if cmp < 0
            iA = _advance_run(A.keys, iA, nA, join_posA); continue
        elseif cmp > 0
            iB = _advance_run(B.keys, iB, nB, join_posB); continue
        end

        # Matching join group: find run extents
        iA2 = _advance_run(A.keys, iA, nA, join_posA)
        iB2 = _advance_run(B.keys, iB, nB, join_posB)

        # Cross-product of all A blocks and all B blocks in the matching runs
        for ii in iA:(iA2-1)
            akey  = A.keys[ii]
            tidA  = A.alias_ids[ii]
            αA    = convert(TC, A.scalars[ii])
            # Reshape template view once per A block (cheap: pointer + size metadata)
            tmpl_A_mat = reshape(_aliased_template_view(A, tidA), M, K)   # (M, K)

            for jj in iB:(iB2-1)
                bkey = B.keys[jj]
                tidB = B.alias_ids[jj]
                αB   = convert(TC, B.scalars[jj])
                αC   = αA * αB

                # Get or compute combined template for (tidA, tidB)
                ct_key       = (tidA, tidB)
                combined_tid = get(combined_tid_map, ct_key, 0)
                if combined_tid == 0
                    C.n_templates += 1
                    combined_tid = C.n_templates
                    combined_tid_map[ct_key] = combined_tid

                    tmpl_B_mat = reshape(_aliased_template_view(B, tidB), K, N)   # (K, N)
                    new_tmpl   = Vector{TC}(undef, C.blksize)

                    if mode == :AthenB
                        # C (M, N) = Amat (M, K) @ Bmat (K, N)
                        Cmat = reshape(new_tmpl, M, N)
                        if can_blas
                            mul!(Cmat,
                                 convert(Matrix{TC}, tmpl_A_mat),
                                 convert(Matrix{TC}, tmpl_B_mat))
                        else
                            fill!(new_tmpl, zero(TC))
                            for k in 1:K
                                _rank1_add_generic!(new_tmpl, one(TC),
                                                    @view(tmpl_A_mat[:, k]),
                                                    @view(tmpl_B_mat[k, :]))
                            end
                        end
                    else  # :BthenA  →  C (N, M) = Bmat^T (N, K) @ Amat^T (K, M)
                        Cmat = reshape(new_tmpl, N, M)
                        if can_blas
                            mul!(Cmat,
                                 convert(Matrix{TC}, transpose(tmpl_B_mat)),
                                 convert(Matrix{TC}, transpose(tmpl_A_mat)))
                        else
                            fill!(new_tmpl, zero(TC))
                            for k in 1:K
                                _rank1_add_generic!(new_tmpl, one(TC),
                                                    @view(tmpl_B_mat[k, :]),
                                                    @view(tmpl_A_mat[:, k]))
                            end
                        end
                    end

                    append!(C.templates, new_tmpl)
                end   # combined template computed

                ckey = ntuple(j -> (src[j] > 0 ? akey[src[j]] : bkey[-src[j]]), Val(PC))
                _aliased_contribute!(key_to_alias, key_to_accum, C.templates,
                                     ckey, combined_tid, αC, C.blksize)
            end   # jj loop
        end   # ii loop

        iA = iA2; iB = iB2
    end   # merge-join

    _commit_aliased_dicts!(C, key_to_alias, key_to_accum)
    return C
end


# ─────────────────────────────────────────────────────────────────────────────
# contract! dispatch overloads for AliasedBlockSparse
#
# These let the top-level contract! in contract.jl route single-rlab calls
# to contract_aliased! (defined in contract_aliased.jl).  Multi-label calls
# go through contract_shared! above via the existing dispatch in contract.jl.
# ─────────────────────────────────────────────────────────────────────────────

# AliasedBS × Dense → AliasedBS  (single shared label)
function contract!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: AbstractVector,
    B        :: AbstractArray{TB,NB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
    return contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
end

# AliasedBS × AliasedBS → AliasedBS  (single shared label)
function contract!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: AbstractVector,
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB,N2B,PB}
    return contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
end
