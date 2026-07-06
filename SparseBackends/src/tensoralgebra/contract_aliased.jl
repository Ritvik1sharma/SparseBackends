# tensoralgebra/contract_aliased.jl
#
# Contraction kernels that produce or consume AliasedBlockSparse tensors.
# The fundamental observation is that because each block is  α * template[t],
# any linear contraction (GEMM, outer product) distributes over the scalar:
#
#   GEMM(α * template[t], B) = α * GEMM(template[t], B)
#
# So the combined template only needs to be computed ONCE per unique template-pair
# (or template × B-slice pair), while the scalars are accumulated per-block.
#
# ─ Kernels provided ──────────────────────────────────────────────────────────
#
#   contract_aliased!(C::AliasedBS, ..., A::AliasedBS, ..., B::StridedArray, ...)
#     r in A sparse prefix  →  outer(template_A[t], B[:,rv])  per (t,rv) pair
#     r in A dense tail     →  GEMM(template_A[t], B)          per t
#
#   contract_aliased!(C::AliasedBS, ..., A::AliasedBS, ..., B::AliasedBS, ...)
#     r in prefix of both   →  outer(template_A[tA], template_B[tB])  per (tA,tB)
#     r in dense tail both  →  GEMM(template_A[tA], template_B[tB])   per (tA,tB)
#
# In all cases blocks whose C key accumulates contributions from multiple
# (distinct) combined templates are handled correctly by materialising the
# accumulated block as its own concrete template (scalar = 1).

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# Permute AliasedBlockSparse so that `rlab` is the last axis in the sparse prefix.
@inline function _aliased_r_to_last_prefix(
    A      :: AliasedBlockSparse{T,N,N2,P},
    labels :: AbstractVector,
    map    :: Dict,
    rlab,
) where {T,N,N2,P}
    axisr = map[rlab]
    @assert axisr <= P "rlab must be in sparse prefix"
    axisr == P && return A, labels, map
    perm = vcat([a for a in 1:P if a != axisr], axisr, collect(P+1:N))
    A2      = permutedims(A, perm)
    labels2 = labels[perm]
    map2    = Dict(lab => i for (i, lab) in enumerate(labels2))
    return A2, labels2, map2
end

# Permute AliasedBlockSparse so that `rlab` is the last axis in the dense tail.
@inline function _aliased_r_to_last_dense(
    A      :: AliasedBlockSparse{T,N,N2,P},
    labels :: AbstractVector,
    map    :: Dict,
    rlab,
) where {T,N,N2,P}
    axisr = map[rlab]
    @assert axisr > P "rlab must be in dense tail"
    axisr == N && return A, labels, map
    r_local = axisr - P                                     # 1-based dense-local
    perm = vcat(collect(1:P), P .+ vcat([d for d in 1:N2 if d != r_local], r_local))
    A2      = permutedims(A, perm)
    labels2 = labels[perm]
    map2    = Dict(lab => i for (i, lab) in enumerate(labels2))
    return A2, labels2, map2
end

# Record a contribution (combined_tid, scalar) to an output block at `ckey`.
#
# The function maintains two dicts:
#   key_to_alias[k] = (tid, α)    — block comes from exactly one template (aliased)
#   key_to_accum[k] = Vector      — block accumulated from ≥2 templates (concrete)
#
# If two contributions share the same combined_tid, only their scalars are summed
# (still aliased).  If they differ, the block is demoted to accumulator.
@inline function _aliased_contribute!(
    key_to_alias :: Dict{K, Tuple{Int,TC}},
    key_to_accum :: Dict{K, Vector{TC}},
    C_templates  :: Vector{TC},           # flat template storage (indexed, not view)
    ckey         :: K,
    combined_tid :: Int,
    scalar       :: TC,
    blksize      :: Int,
) where {K, TC}
    if haskey(key_to_accum, ckey)
        # Already accumulating: add scalar * combined template into running block
        acc     = key_to_accum[ckey]
        src_off = (combined_tid - 1) * blksize
        @inbounds @simd for j in 1:blksize
            acc[j] += scalar * C_templates[src_off + j]
        end

    elseif haskey(key_to_alias, ckey)
        (tid_prev, α_prev) = key_to_alias[ckey]
        if tid_prev == combined_tid
            # Same template → just add scalars (stays aliased)
            key_to_alias[ckey] = (tid_prev, α_prev + scalar)
        else
            if false  # SB_TRACE — flip to true here for debug output
              println("[SB_TRACE]   _aliased_contribute! demotion @ ckey=", ckey,
                      "  prev_tid=", tid_prev, "  new_tid=", combined_tid)
            end
            # Different template → demote to accumulator
            delete!(key_to_alias, ckey)
            acc      = Vector{TC}(undef, blksize)
            off_prev = (tid_prev    - 1) * blksize
            off_new  = (combined_tid - 1) * blksize
            @inbounds @simd for j in 1:blksize
                acc[j] = α_prev * C_templates[off_prev + j] +
                         scalar * C_templates[off_new  + j]
            end
            key_to_accum[ckey] = acc
        end

    else
        # First contribution to this key
        key_to_alias[ckey] = (combined_tid, scalar)
    end
    return nothing
end

# Flush the intermediate dicts into C, sort keys.
# Accumulator blocks are stored as new concrete templates (scalar = 1).
function _commit_aliased_dicts!(
    C            :: AliasedBlockSparse{TC,NC,N2,PC},
    key_to_alias :: Dict{NTuple{PC,Int}, Tuple{Int,TC}},
    key_to_accum :: Dict{NTuple{PC,Int}, Vector{TC}},
) where {TC,NC,N2,PC}
    AI = eltype(C.alias_ids)
    for (k, (tid, α)) in key_to_alias
        push!(C.keys, k); push!(C.alias_ids, _alias_id(AI, tid)); push!(C.scalars, α)
    end
    for (k, acc) in key_to_accum
        C.n_templates += 1
        append!(C.templates, acc)
        push!(C.keys,      k)
        push!(C.alias_ids, _alias_id(AI, C.n_templates))
        push!(C.scalars,   one(TC))
    end
    if !isempty(C.keys)
        pdims = ntuple(i -> C.dims[i], Val(PC))
        p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
        C.keys      = C.keys[p]
        C.alias_ids = C.alias_ids[p]
        C.scalars   = C.scalars[p]
    end
    return C
end

# Lazy commit: kernels register combined templates into a SCRATCH `pending`
# array (not C.templates).  At commit time we copy only the *referenced*
# pending templates into C.templates, rewriting `alias_ids` to match.  This
# guarantees `n_templates ≤ n_blocks` (every template is referenced by at
# least one block), eliminating orphan templates from demotion.
function _commit_aliased_dicts_lazy!(
    C            :: AliasedBlockSparse{TC,NC,N2,PC},
    key_to_alias :: Dict{NTuple{PC,Int}, Tuple{Int,TC}},
    key_to_accum :: Dict{NTuple{PC,Int}, Vector{TC}},
    pending      :: Vector{TC},
    n_pending    :: Int,
) where {TC,NC,N2,PC}
    blksize = C.blksize
    # 1) Mark which pending tids are referenced by surviving alias entries.
    ref = falses(n_pending)
    for (_, (ptid, _)) in key_to_alias
        ref[ptid] = true
    end
    # 2) Copy referenced templates into C.templates and build remap.
    remap = zeros(Int, n_pending)
    final_tid = 0
    @inbounds for p in 1:n_pending
        ref[p] || continue
        final_tid += 1
        remap[p] = final_tid
        off = (p - 1) * blksize
        for j in 1:blksize
            push!(C.templates, pending[off + j])
        end
    end
    C.n_templates = final_tid

    # 3) Push aliased blocks with remapped tids.
    AI = eltype(C.alias_ids)
    for (k, (ptid, α)) in key_to_alias
        push!(C.keys, k)
        push!(C.alias_ids, _alias_id(AI, remap[ptid]))
        push!(C.scalars, α)
    end
    # 4) Push accumulator blocks as new concrete templates.
    for (k, acc) in key_to_accum
        C.n_templates += 1
        append!(C.templates, acc)
        push!(C.keys, k)
        push!(C.alias_ids, _alias_id(AI, C.n_templates))
        push!(C.scalars, one(TC))
    end
    # 5) Sort by prefix col-major.
    if !isempty(C.keys)
        pdims = ntuple(i -> C.dims[i], Val(PC))
        p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
        C.keys      = C.keys[p]
        C.alias_ids = C.alias_ids[p]
        C.scalars   = C.scalars[p]
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# AliasedBlockSparse × Dense  →  AliasedBlockSparse
# ─────────────────────────────────────────────────────────────────────────────

# Sub-case: r is in A's sparse prefix (last prefix axis after permutation).
# Combined template: outer(template_A[t], B[:,rv])  per unique (t, rv).
function _contract_aliased_prefix_outer_ad!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: AbstractVector,
    B        :: StridedArray{TB,NB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
    @assert mapA[rlab] == PA "rlab must be the last prefix axis of A (call _aliased_r_to_last_prefix first)"
    @assert !(rlab in labelsC)

    # Permute B so rlab is last
    axisBr = mapB[rlab]
    dimsB  = size(B)
    R      = dimsB[axisBr]
    if axisBr != NB
        permB   = vcat([d for d in 1:NB if d != axisBr], axisBr)
        B       = permutedims(B, permB)
        labelsB = labelsB[permB]
        mapB    = Dict(lab => i for (i, lab) in enumerate(labelsB))
    end

    # ── FISSION DETECTION ─────────────────────────────────────────────────────
    # C's prefix may include axes that came from B (when the caller passed an
    # output_inds_hint forcing them into the sparse prefix). Partition B's
    # non-r dims into:
    #   - fission_pos_in_B : B dims that land in C's prefix (→ fissioned)
    #   - remaining_pos_in_B : B dims that stay in C's dense tail
    n_A_prefix_in_C = PA - 1  # A's prefix axes excluding rlab
    n_fission       = PC - n_A_prefix_in_C
    @assert n_fission >= 0 "PC=$PC less than A's non-r prefix count $n_A_prefix_in_C"
    Cpref_labels    = labelsC[1:PC]
    fission_labels  = Set(Cpref_labels[i] for i in (n_A_prefix_in_C+1):PC)
    fission_pos_in_B   = Int[]
    remaining_pos_in_B = Int[]
    @inbounds for i in 1:NB-1   # iterate over B's non-r axes (after permutation)
        if labelsB[i] in fission_labels
            push!(fission_pos_in_B, i)
        else
            push!(remaining_pos_in_B, i)
        end
    end
    @assert length(fission_pos_in_B) == n_fission "fission axes count mismatch ($(length(fission_pos_in_B)) vs $n_fission)"

    # If fission dims aren't already at the front of B, permute them there so
    # B is laid out as (fission_prod, chunkB_remaining, R).
    if !isempty(fission_pos_in_B) &&
       (fission_pos_in_B != collect(1:n_fission) || remaining_pos_in_B != collect(n_fission+1:NB-1))
        permB2 = vcat(fission_pos_in_B, remaining_pos_in_B, [NB])
        B = permutedims(B, permB2)
        labelsB = labelsB[permB2]
        mapB = Dict(lab => i for (i, lab) in enumerate(labelsB))
    end

    fission_dims     = ntuple(i -> size(B, i), n_fission)
    fission_prod     = Int(prod(fission_dims; init=1))
    chunkB_remaining = Int(prod(size(B, p) for p in (n_fission+1):(NB-1); init=1))
    @assert C.blksize == A.blksize * chunkB_remaining "C.blksize must equal A.blksize * chunkB_remaining (got $(C.blksize) vs $(A.blksize)*$(chunkB_remaining))"
    Bvec = vec(B)   # column-major: B[fkey, chunk, rv] at fkey + (chunk-1)*fission_prod + (rv-1)*fission_prod*chunkB_remaining

    # Dense-order check (C's tail)
    Adense = labelsA[PA+1:NA]
    Bwo_remaining = labelsB[n_fission+1 : NB-1]  # B's non-fission, non-r labels
    Cdense = labelsC[PC+1:NC]
    dense_order = Cdense == vcat(Adense, Bwo_remaining) ? :AB :
                  Cdense == vcat(Bwo_remaining, Adense) ? :BA :
                  error("C dense tail must be [A_dense..., B_remaining...] or [B_remaining..., A_dense...]; got Cdense=$Cdense, Adense=$Adense, Bwo_remaining=$Bwo_remaining")

    # C prefix sourcing: first n_A_prefix_in_C entries come from A's prefix;
    # the remaining n_fission entries come from B's fission dims.
    src_A = Vector{Int}(undef, n_A_prefix_in_C)
    @inbounds for j in 1:n_A_prefix_in_C
        lab = Cpref_labels[j]
        ax  = mapA[lab]
        @assert ax <= PA && lab != rlab "C prefix label $lab (pos $j) must come from A sparse prefix excluding rlab"
        src_A[j] = ax
    end
    src_B_fission_pos = Vector{Int}(undef, n_fission)  # which fission axis (1..n_fission) each Cpref slot at PC+1-n_fission..PC maps to
    @inbounds for j in 1:n_fission
        lab = Cpref_labels[n_A_prefix_in_C + j]
        ax  = mapB[lab]   # after permutation, fission labels are at 1..n_fission
        @assert 1 <= ax <= n_fission
        src_B_fission_pos[j] = ax
    end

    # Combined-template dedup: (tidA, rv, fkey_lin) → pending tid.
    combined_tid_map = Dict{Tuple{Int,Int,Int}, Int}()
    pending  = TC[]
    n_pending = 0

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # Strides for decoding fission key linear index → per-axis coords.
    fission_strides = ones(Int, max(n_fission, 1))
    @inbounds for d in 2:n_fission
        fission_strides[d] = fission_strides[d-1] * fission_dims[d-1]
    end

    @inbounds for i in eachindex(A.keys)
        akey = A.keys[i]
        rv   = akey[PA]
        (1 <= rv <= R) || continue
        tidA = A.alias_ids[i]
        α    = convert(TC, A.scalars[i])

        # Iterate fission keys (1..fission_prod). For n_fission==0, fission_prod=1
        # and the loop runs once with no fission contribution to ckey.
        for fkey_lin in 1:fission_prod
            # Decode fkey_lin → per-axis coords (1-based).
            # fkey_coords[d] = ((fkey_lin-1) ÷ fission_strides[d]) mod fission_dims[d] + 1
            ckey = ntuple(j -> begin
                if j <= n_A_prefix_in_C
                    akey[src_A[j]]
                else
                    fax = src_B_fission_pos[j - n_A_prefix_in_C]
                    Int(((fkey_lin - 1) ÷ fission_strides[fax]) % fission_dims[fax]) + 1
                end
            end, Val(PC))

            # Get / compute combined sub-template at (tidA, rv, fkey_lin).
            combined_tid = get(combined_tid_map, (tidA, rv, fkey_lin), 0)
            if combined_tid == 0
                n_pending += 1
                combined_tid = n_pending
                combined_tid_map[(tidA, rv, fkey_lin)] = combined_tid
                new_tmpl = zeros(TC, C.blksize)
                tmpl_A   = _aliased_template_view(A, tidA)
                # B slice at (fkey, :, rv) of length chunkB_remaining.
                # Linear positions: base + (chunk-1)*fission_prod + fkey_lin, for chunk = 1..chunkB_remaining.
                base = (rv - 1) * chunkB_remaining * fission_prod
                Bslice = Vector{TC}(undef, chunkB_remaining)
                for chunk in 1:chunkB_remaining
                    Bslice[chunk] = Bvec[base + (chunk - 1) * fission_prod + fkey_lin]
                end
                if dense_order == :AB
                    _outer_add!(new_tmpl, tmpl_A, Bslice)
                else
                    _outer_add!(new_tmpl, Bslice, tmpl_A)
                end
                append!(pending, new_tmpl)
            end

            _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                 ckey, combined_tid, α, C.blksize)
        end
    end

    _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end


# Sub-case: r is in A's dense tail (last dense axis after permutation).
# Combined template: GEMM(template_A[t], B)  per unique t  (one per template).
# Because C prefix = A prefix (no sparse structure changes), aliasing is
# perfectly preserved:  n_C_templates = n_A_templates.
function _contract_aliased_dense_ad!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: AbstractVector,
    B        :: StridedArray{TB,NB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
    @assert mapA[rlab] == NA "rlab must be the last axis of A (call _aliased_r_to_last_dense first)"
    @assert !(rlab in labelsC)
    @assert PC == PA "C sparse prefix must equal A sparse prefix for dense-tail reduction"

    # Permute B so rlab is last
    axisBr = mapB[rlab]
    if axisBr != NB
        permB   = vcat([d for d in 1:NB if d != axisBr], axisBr)
        B       = permutedims(B, permB)
        labelsB = labelsB[permB]
        mapB    = Dict(lab => i for (i, lab) in enumerate(labelsB))
    end
    dimsB  = size(B)
    R      = dimsB[end]
    chunkB = Int(prod(dimsB[1:end-1]; init=1))

    dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))
    @assert dimsA_dense[end] == R "Reduction size R mismatch between A dense tail and B"
    chunkA = Int(A.blksize ÷ R)
    @assert C.blksize == chunkA * chunkB

    # Dense-order check
    Awo    = labelsA[PA+1:NA-1]   # A dense labels without r
    Bwo    = labelsB[1:NB-1]      # B labels without r
    Cdense = labelsC[PC+1:NC]
    dense_order = Cdense == vcat(Awo, Bwo) ? :AB :
                  Cdense == vcat(Bwo, Awo) ? :BA :
                  error("C dense tail must be [A_dense_wo_r...,B_wo_r...] or [B_wo_r...,A_dense_wo_r...]")

    # C prefix sourcing (must be a reordering of A's prefix axes)
    src = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]; ax = mapA[lab]
        @assert ax <= PA "C prefix label must come from A sparse prefix"
        src[j] = ax
    end

    Bvec   = vec(B)                        # (chunkB, R) column-major → Bmat = reshape(Bvec, chunkB, R)
    Bmat   = reshape(Bvec, chunkB, R)      # B[:,r] is column r

    # Precompute one combined template per unique template in A.
    # combined_tids[t] = new template id in C for old template t.
    n_tmplA      = A.n_templates
    combined_tids = Vector{Int}(undef, n_tmplA)

    can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)

    for t in 1:n_tmplA
        C.n_templates += 1
        combined_tids[t] = C.n_templates

        tmpl_A = _aliased_template_view(A, t)            # length chunkA * R
        Amat   = reshape(tmpl_A, chunkA, R)              # (chunkA, R)

        new_tmpl = Vector{TC}(undef, C.blksize)

        if dense_order == :AB
            # Cmat (chunkA, chunkB) = Amat (chunkA, R) @ Bmat^T (R, chunkB)
            Cmat = reshape(new_tmpl, chunkA, chunkB)
            if can_blas
                mul!(Cmat, convert(Matrix{TC}, Amat), convert(Matrix{TC}, transpose(Bmat)))
            else
                fill!(new_tmpl, zero(TC))
                for r in 1:R
                    As = @view Amat[:, r]   # length chunkA
                    Bs = @view Bmat[:, r]   # length chunkB
                    _rank1_add_generic!(new_tmpl, one(TC), As, Bs)
                end
            end
        else  # :BA
            # Cmat (chunkB, chunkA) = Bmat (chunkB, R) @ Amat^T (R, chunkA)
            Cmat = reshape(new_tmpl, chunkB, chunkA)
            if can_blas
                mul!(Cmat, convert(Matrix{TC}, Bmat), convert(Matrix{TC}, transpose(Amat)))
            else
                fill!(new_tmpl, zero(TC))
                for r in 1:R
                    As = @view Amat[:, r]
                    Bs = @view Bmat[:, r]
                    _rank1_add_generic!(new_tmpl, one(TC), Bs, As)  # outer(Bs, As)
                end
            end
        end

        append!(C.templates, new_tmpl)
    end

    # Build C blocks: same keys (possibly reordered by src), same scalars,
    # but alias_ids now point to combined templates.
    # No accumulation: each A key is unique and maps injectively to a C key.
    nblocks = length(A.keys)
    resize!(C.keys,      nblocks)
    resize!(C.alias_ids, nblocks)
    resize!(C.scalars,   nblocks)
    @inbounds for i in 1:nblocks
        akey           = A.keys[i]
        C.keys[i]      = ntuple(j -> akey[src[j]], Val(PC))
        C.alias_ids[i] = combined_tids[A.alias_ids[i]]
        C.scalars[i]   = convert(TC, A.scalars[i])
    end

    # Sort keys (src may reorder prefix axes)
    if !isempty(C.keys)
        pdims = ntuple(i -> C.dims[i], Val(PC))
        p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
        C.keys      = C.keys[p]
        C.alias_ids = C.alias_ids[p]
        C.scalars   = C.scalars[p]
    end
    return C
end


"""
    contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

Compute `C = A ⊗_rlab B` where `A` is an `AliasedBlockSparse` and `B` is
dense, writing an `AliasedBlockSparse` result.

The aliasing structure is preserved: the scalar in each A block factors through
the contraction, so only one combined template per unique `(template_id, B_slice)`
pair (or per template when r is in the dense tail) is ever stored.

- r in A sparse prefix: outer-product kernel. Combined templates = outer(tA, B[:,rv]).
  Template count ≤ nA_templates × R.
- r in A dense tail: GEMM kernel.  Combined templates = GEMM(tA, B).
  Template count = nA_templates  (perfect preservation of aliasing).
"""
function contract_aliased!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: Vector{Label},
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: Vector{Label},
    B        :: StridedArray{TB,NB},
    labelsB  :: Vector{Label},
    mapA     :: Dict{Label,Int},
    mapB     :: Dict{Label,Int},
    rlab     :: Label,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
    @assert !(rlab in labelsC)

    # Clear output
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    axisAr = mapA[rlab]
    if axisAr <= PA
        A, labelsA, mapA = _aliased_r_to_last_prefix(A, labelsA, mapA, rlab)
        return _contract_aliased_prefix_outer_ad!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    else
        A, labelsA, mapA = _aliased_r_to_last_dense(A, labelsA, mapA, rlab)
        return _contract_aliased_dense_ad!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# AliasedBlockSparse × AliasedBlockSparse  →  AliasedBlockSparse
# ─────────────────────────────────────────────────────────────────────────────

# Sub-case: r is in the sparse prefix of BOTH A and B (last prefix axis each).
# Merge-join on rv.  Combined template: outer(template_A[tA], template_B[tB])
# per unique (tA, tB) pair.
function _contract_aliased_prefix_outer_aa!(
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
    @assert mapA[rlab] == PA && mapB[rlab] == PB
    @assert !(rlab in labelsC)
    @assert C.blksize == A.blksize * B.blksize

    Adense = labelsA[PA+1:NA]
    Bdense = labelsB[PB+1:NB]
    Cdense = labelsC[PC+1:NC]
    dense_order = Cdense == vcat(Adense, Bdense) ? 1 :
                  Cdense == vcat(Bdense, Adense) ? 2 :
                  error("C dense must be [A_dense...,B_dense...] or [B_dense...,A_dense...]")

    # C prefix sourcing:  src[j] > 0 → A.key[src[j]];  src[j] < 0 → B.key[-src[j]]
    src = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab) && mapA[lab] != mapA[rlab]
            ax = mapA[lab]; @assert ax <= PA
            src[j] = ax
        else
            @assert haskey(mapB, lab) "C prefix label $lab not found in A or B"
            ax = mapB[lab]; @assert ax <= PB && lab != rlab
            src[j] = -ax
        end
    end

    # Combined template deduplication: (tidA, tidB) -> combined_tid in C
    combined_tid_map = Dict{Tuple{Int,Int}, Int}()

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # Merge-join A and B on rv (r is last prefix axis, keys sorted column-major)
    iA = firstindex(A.keys); nA = lastindex(A.keys)
    iB = firstindex(B.keys); nB = lastindex(B.keys)

    @inbounds while iA <= nA && iB <= nB
        rvA = A.keys[iA][PA]
        rvB = B.keys[iB][PB]

        if rvA < rvB
            while iA <= nA && A.keys[iA][PA] == rvA; iA += 1; end
            continue
        elseif rvB < rvA
            while iB <= nB && B.keys[iB][PB] == rvB; iB += 1; end
            continue
        end

        # rv matches: find the B run
        b_lo = iB
        while iB <= nB && B.keys[iB][PB] == rvA; iB += 1; end
        b_hi = iB - 1

        # Cross-product of A blocks and B blocks with the same rv
        while iA <= nA && A.keys[iA][PA] == rvA
            akey = A.keys[iA]
            tidA = A.alias_ids[iA]
            αA   = convert(TC, A.scalars[iA])
            for iB2 in b_lo:b_hi
                bkey = B.keys[iB2]
                tidB = B.alias_ids[iB2]
                αB   = convert(TC, B.scalars[iB2])
                αC   = αA * αB

                combined_tid = get(combined_tid_map, (tidA, tidB), 0)
                if combined_tid == 0
                    C.n_templates += 1
                    combined_tid = C.n_templates
                    combined_tid_map[(tidA, tidB)] = combined_tid
                    new_tmpl = zeros(TC, C.blksize)
                    tmpl_A   = _aliased_template_view(A, tidA)
                    tmpl_B   = _aliased_template_view(B, tidB)
                    if dense_order == 1
                        _outer_add!(new_tmpl, tmpl_A, tmpl_B)   # rows=A_dense, cols=B_dense
                    else
                        _outer_add!(new_tmpl, tmpl_B, tmpl_A)   # rows=B_dense, cols=A_dense
                    end
                    append!(C.templates, new_tmpl)
                end

                ckey = ntuple(Val(PC)) do j
                    s = src[j]; s > 0 ? akey[s] : bkey[-s]
                end

                _aliased_contribute!(key_to_alias, key_to_accum, C.templates,
                                     ckey, combined_tid, αC, C.blksize)
            end
            iA += 1
        end
    end

    _commit_aliased_dicts!(C, key_to_alias, key_to_accum)
    return C
end


# Sub-case: r is in the dense tail of BOTH A and B (last dense axis each).
# For each pair of A-block and B-block contributing to the same C key:
#   C_block += (αA * αB) * GEMM(template_A[tA], template_B[tB])
# Only one GEMM per unique (tA, tB) pair.
function _contract_aliased_dense_aa!(
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
    @assert mapA[rlab] == NA && mapB[rlab] == NB
    @assert !(rlab in labelsC)

    dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))
    dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2B))
    R = dimsA_dense[end]
    @assert dimsB_dense[end] == R "Reduction size mismatch"
    chunkA = Int(A.blksize ÷ R)
    chunkB = Int(B.blksize ÷ R)
    @assert C.blksize == chunkA * chunkB

    Awo    = labelsA[PA+1:NA-1]
    Bwo    = labelsB[PB+1:NB-1]
    Cdense = labelsC[PC+1:NC]
    dense_order = Cdense == vcat(Awo, Bwo) ? :AB :
                  Cdense == vcat(Bwo, Awo) ? :BA :
                  error("C dense tail must be [A_dense_wo_r...,B_dense_wo_r...] or swapped")

    # C prefix sourcing
    srcA = zeros(Int, PC)
    srcB = zeros(Int, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab)
            ax = mapA[lab]; @assert ax <= PA; srcA[j] = ax
        else
            @assert haskey(mapB, lab)
            ax = mapB[lab]; @assert ax <= PB; srcB[j] = ax
        end
    end

    can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)

    # Combined template deduplication: (tidA, tidB) -> combined_tid in C
    combined_tid_map = Dict{Tuple{Int,Int}, Int}()

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    @inbounds for ai in eachindex(A.keys)
        akey = A.keys[ai]
        tidA = A.alias_ids[ai]
        αA   = convert(TC, A.scalars[ai])

        for bi in eachindex(B.keys)
            bkey = B.keys[bi]
            tidB = B.alias_ids[bi]
            αB   = convert(TC, B.scalars[bi])
            αC   = αA * αB

            combined_tid = get(combined_tid_map, (tidA, tidB), 0)
            if combined_tid == 0
                C.n_templates += 1
                combined_tid = C.n_templates
                combined_tid_map[(tidA, tidB)] = combined_tid

                tmpl_A = _aliased_template_view(A, tidA)   # length chunkA * R
                tmpl_B = _aliased_template_view(B, tidB)   # length chunkB * R
                Amat   = reshape(tmpl_A, chunkA, R)        # (chunkA, R)
                Bmat   = reshape(tmpl_B, chunkB, R)        # (chunkB, R)

                new_tmpl = Vector{TC}(undef, C.blksize)

                if dense_order == :AB
                    # Cmat (chunkA, chunkB) = Amat @ Bmat^T
                    Cmat = reshape(new_tmpl, chunkA, chunkB)
                    if can_blas
                        mul!(Cmat,
                             convert(Matrix{TC}, Amat),
                             convert(Matrix{TC}, transpose(Bmat)))
                    else
                        fill!(new_tmpl, zero(TC))
                        for r in 1:R
                            _rank1_add_generic!(new_tmpl, one(TC),
                                                @view(Amat[:,r]), @view(Bmat[:,r]))
                        end
                    end
                else  # :BA
                    # Cmat (chunkB, chunkA) = Bmat @ Amat^T
                    Cmat = reshape(new_tmpl, chunkB, chunkA)
                    if can_blas
                        mul!(Cmat,
                             convert(Matrix{TC}, Bmat),
                             convert(Matrix{TC}, transpose(Amat)))
                    else
                        fill!(new_tmpl, zero(TC))
                        for r in 1:R
                            _rank1_add_generic!(new_tmpl, one(TC),
                                                @view(Bmat[:,r]), @view(Amat[:,r]))
                        end
                    end
                end

                append!(C.templates, new_tmpl)
            end

            ckey = ntuple(j -> (srcA[j] != 0 ? akey[srcA[j]] : bkey[srcB[j]]), Val(PC))

            _aliased_contribute!(key_to_alias, key_to_accum, C.templates,
                                 ckey, combined_tid, αC, C.blksize)
        end
    end

    _commit_aliased_dicts!(C, key_to_alias, key_to_accum)
    return C
end


"""
    contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

Compute `C = A ⊗_rlab B` where both `A` and `B` are `AliasedBlockSparse`,
writing an `AliasedBlockSparse` result.

The aliasing structure is maintained by computing combined templates once per
unique `(tidA, tidB)` pair (outer product when r is in the prefix of both, or
GEMM when r is in the dense tail of both).

- r in sparse prefix of both: merge-join kernel. n_combined ≤ nA_tmpl × nB_tmpl.
- r in dense tail of both:    GEMM kernel.       n_combined ≤ nA_tmpl × nB_tmpl.
"""
function contract_aliased!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: Vector{Label},
    A        :: AliasedBlockSparse{TA,NA,N2A,PA},
    labelsA  :: Vector{Label},
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: Vector{Label},
    mapA     :: Dict{Label,Int},
    mapB     :: Dict{Label,Int},
    rlab     :: Label,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB,N2B,PB}
    @assert !(rlab in labelsC)

    # Clear output
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    axisBr = mapB[rlab]
    if axisBr <= PB
        # r in prefix of both
        A, labelsA, mapA = _aliased_r_to_last_prefix(A, labelsA, mapA, rlab)
        B, labelsB, mapB = _aliased_r_to_last_prefix(B, labelsB, mapB, rlab)
        return _contract_aliased_prefix_outer_aa!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    else
        # r in dense tail of both
        axisAr = mapA[rlab]
        @assert axisAr > PA "rlab must be in A dense tail when it is in B dense tail"
        A, labelsA, mapA = _aliased_r_to_last_dense(A, labelsA, mapA, rlab)
        B, labelsB, mapB = _aliased_r_to_last_dense(B, labelsB, mapB, rlab)
        return _contract_aliased_dense_aa!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
end