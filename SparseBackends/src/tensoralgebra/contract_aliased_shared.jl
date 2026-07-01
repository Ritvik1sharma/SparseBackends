# tensoralgebra/contract_aliased_shared.jl
#
# Multi-label contraction helpers and AliasedBlockSparse × AliasedBlockSparse kernel.
# The AliasedBlockSparse × Dense kernel lives in contract_aliased_dense_shared.jl.

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
# Fission fallback helper (used by both Dense and AliasedBS overloads)
# ─────────────────────────────────────────────────────────────────────────────

# Fission fallback: when output_inds_hint pushes B labels into C's prefix
# (which the aliased kernel doesn't yet support natively), delegate the
# contract to the BS path (which DOES implement hint+fission), then
# trivially re-aliasify the BS result (one template per block).
#
# Loses alias compression for this single contract step but preserves C's
# requested axis classification, so subsequent contracts in the matvec
# chain don't see cross-region label mismatches.
function _aliased_shared_via_bs_fission!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC       :: AbstractVector,
    A             :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA       :: AbstractVector,
    B,
    labelsB       :: AbstractVector,
    mapA          :: Dict,
    mapB          :: Dict,
    shared_labels :: Vector;
    output_inds_hint=nothing,
    allowed_keys_C=nothing,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA}
    A_bs = to_blocksparse(A)
    # If B is also aliased, demote it too.
    B_bs = (B isa AliasedBlockSparse) ? to_blocksparse(B) : B
    Kt = eltype(eltype(A_bs.keys))
    C_bs = NewBlockSparseSorted{TC,NC,N2C,PC,Kt}(C.dims, C.blksize,
        NTuple{PC,Kt}[], Int[], TC[])
    contract!(C_bs, labelsC, A_bs, labelsA, B_bs, labelsB;
        output_inds_hint=output_inds_hint, allowed_keys_C=allowed_keys_C)
    # Re-aliasify: one template per non-empty block (trivial alias, no
    # compression). Compression can be recovered by a later optimization.
    nb = length(C_bs.keys)
    C.dims    = C_bs.dims
    C.blksize = C_bs.blksize
    empty!(C.templates); append!(C.templates, C_bs.data)
    C.n_templates = nb
    empty!(C.keys);      append!(C.keys, C_bs.keys)
    empty!(C.alias_ids); append!(C.alias_ids, _alias_id_range(eltype(C.alias_ids), nb))
    empty!(C.scalars);   append!(C.scalars, ones(TC, nb))
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
    shared_labels :: Vector;
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    allowed_keys_C=nothing,
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

    # Fission fallback (same as Aliased × Dense branch above) — delegate to
    # BS contract when hint moves labels into C's prefix.
    has_hint = output_inds_hint !== nothing || allowed_keys_C !== nothing
    actually_needs_fission = has_hint &&
        (length(desired_keepA) != length(keepA0) ||
         length(desired_keepB) != length(keepB0) ||
         allowed_keys_C !== nothing)
    if actually_needs_fission
        return _aliased_shared_via_bs_fission!(C, labelsC, A, labelsA, B, labelsB,
            mapA, mapB, shared_labels;
            output_inds_hint=output_inds_hint, allowed_keys_C=allowed_keys_C)
    end

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

    # Combined template deduplication: (tidA, tidB) → pending_tid (scratch).
    combined_tid_map = Dict{Tuple{Int,Int}, Int}()
    pending   = TC[]
    n_pending = 0

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

                # Get or compute combined template for (tidA, tidB) — lazy.
                ct_key       = (tidA, tidB)
                combined_tid = get(combined_tid_map, ct_key, 0)
                if combined_tid == 0
                    n_pending += 1
                    combined_tid = n_pending
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

                    @timeit TIMER "cas.append_pending" append!(pending, new_tmpl)
                end   # combined template computed

                ckey = ntuple(j -> (src[j] > 0 ? akey[src[j]] : bkey[-src[j]]), Val(PC))
                _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                     ckey, combined_tid, αC, C.blksize)
            end   # jj loop
        end   # ii loop

        iA = iA2; iB = iB2
    end   # merge-join

    _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
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

# Dense × AliasedBS → AliasedBS  (single shared label) — delegate by swapping args.
function contract!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: AbstractArray{TA,NA},
    labelsA  :: AbstractVector,
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2B,PB}
    return contract_aliased!(C, labelsC, B, labelsB, A, labelsA, mapB, mapA, rlab)
end

# Dense × AliasedBS → AliasedBS  (multi shared label) — delegate by swapping args.
function contract_shared!(
    C             :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC       :: AbstractVector,
    A             :: AbstractArray{TA,NA},
    labelsA       :: AbstractVector,
    B             :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB       :: AbstractVector,
    mapA          :: Dict,
    mapB          :: Dict,
    shared_labels :: Vector;
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    allowed_keys_C=nothing,
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2B,PB}
    return contract_shared!(C, labelsC, B, labelsB, A, labelsA, mapB, mapA, shared_labels;
                            output_inds_hint=output_inds_hint, allowed_keys_C=allowed_keys_C)
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
