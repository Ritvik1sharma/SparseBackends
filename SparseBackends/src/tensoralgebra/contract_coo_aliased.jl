# tensoralgebra/contract_coo_aliased.jl
#
# Contraction kernel: COOTensor × AliasedBlockSparse  →  AliasedBlockSparse
# Mirrors contract_coo_bs.jl but writes into an AliasedBlockSparse,
# preserving (and amplifying) aliasing structure.
#
# Single shared label `rlab`. Two sub-cases on where r sits in B:
#   r in B sparse prefix → prefix kernel (merge-join on r-runs).
#   r in B dense tail    → dense kernel (one combined template per (tidB, rv)).

using LinearAlgebra


# Aliased-overload of _permute_r_to_last_prefix (the BS version lives in contract_bs_bs.jl).
@inline function _permute_r_to_last_prefix(
    Tns    :: AliasedBlockSparse{T,N,N2,P},
    labels :: AbstractVector,
    map    :: Dict,
    rlab,
) where {T,N,N2,P}
    axisr = map[rlab]
    @assert axisr <= P "rlab must be in prefix to use prefix kernel"
    axisr == P && return Tns, labels, map
    perm   = vcat([a for a in 1:P if a != axisr], axisr, collect(P+1:N))
    Tns    = permutedims(Tns, perm)
    labels = labels[perm]
    map    = Dict(lab => i for (i, lab) in enumerate(labels))
    return Tns, labels, map
end


function contract!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: COOTensor{TA,NA},
    labelsA  :: AbstractVector,
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2B,PB}
    axisBr = mapB[rlab]
    A, labelsA, mapA = ensure_rlab_last_and_sorted(A, labelsA, rlab)
    if axisBr <= PB
        B, labelsB, mapB = _permute_r_to_last_prefix(B, labelsB, mapB, rlab)
        return _contract_coo_aliased_prefix!(C, labelsC, A, labelsA, B, labelsB,
                                             mapA, mapB, rlab)
    else
        return _contract_coo_aliased_dense!(C, labelsC, A, labelsA, B, labelsB,
                                            mapA, mapB, rlab)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# r in B's sparse prefix.  C dense tail = B dense tail.
# Merge A's r-runs against B's r-keys; for each match, accumulate
# α_A * α_B * template_B[tidB] into C, keyed by combined_tid := tidB.
# ─────────────────────────────────────────────────────────────────────────────
function _contract_coo_aliased_prefix!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: COOTensor{TA,NA},
    labelsA  :: AbstractVector,
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2B,PB}
    @assert !(rlab in labelsC) "rlab must be reduced"
    @assert N2C == N2B  "C dense rank must match B dense rank when r is in B's prefix"
    @assert C.blksize == B.blksize  "C and B block sizes must match"
    axisAr = mapA[rlab]
    @assert axisAr == NA
    axisBr = mapB[rlab]
    @assert axisBr == PB

    # Source map for each C prefix axis.
    srcA = zeros(Int, PC)
    srcB = zeros(Int, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab) && lab != rlab
            srcA[j] = mapA[lab]
        else
            ax = mapB[lab]
            @assert ax <= PB && lab != rlab "C prefix label $lab must come from a non-r prefix axis"
            srcB[j] = ax
        end
    end

    # Clear C; lazy: stage all of B's templates in `pending` (indexed 1..n_pending
    # = 1..B.n_templates).  Only the referenced ones are copied at commit time.
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)
    pending = copy(B.templates)
    n_pending = B.n_templates

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    # Bucket B blocks by their r-value (last prefix axis).
    Rdim = B.dims[PB]
    Bbuckets = Vector{Vector{Int}}(undef, Rdim)
    for r in 1:Rdim; Bbuckets[r] = Int[]; end
    @inbounds for i in eachindex(B.keys)
        push!(Bbuckets[B.keys[i][PB]], i)
    end

    @inbounds for i in eachindex(A.keys)
        acoord = A.keys[i]
        rv     = acoord[axisAr]
        (1 <= rv <= Rdim) || continue
        α      = convert(TC, A.vals[i])
        bi_list = Bbuckets[rv]
        isempty(bi_list) && continue

        for bi in bi_list
            bkey   = B.keys[bi]
            tidB   = B.alias_ids[bi]
            αB     = convert(TC, B.scalars[bi])
            ckey   = ntuple(j -> (srcA[j] != 0 ? acoord[srcA[j]] : bkey[srcB[j]]), Val(PC))
            scalar = α * αB
            _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                 ckey, tidB, scalar, C.blksize)
        end
    end

    _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end


# ─────────────────────────────────────────────────────────────────────────────
# r in B's dense tail.  C prefix = (A non-r axes) ∪ (B's full prefix);
# C dense tail = B dense tail with r removed.
# Strategy: permute B so r is the last dense axis. Then for each
# unique (tidB, rv) we produce one combined template = α_B * template_B[tidB][:,rv]
# (just the rv-slice). Combined_tid is shared across all (acoord, bkey) pairs
# that hit the same (tidB, rv) — so aliasing is preserved up to ≤ n_templates_B × R.
# Per-block scalar = α_A * α_B  → scalar from A only (αB is folded into template).
# Wait — that double-counts αB.  We fold αB into the template only ONCE per
# (tidB, rv); subsequent blocks reusing the same combined_tid carry only α_A.
# So we keep a side dict (tidB, rv) → (combined_tid, αB) and the per-block
# scalar is α_A * αB_recorded_at_creation? No — αB depends on B's BLOCK, not on
# (tidB, rv).  Different B blocks can share a tidB but have different αB.
# So we cannot fold αB into the template; we keep template = template_B[tidB][:,rv]
# (scalar 1) and per-block scalar = α_A * α_B.
# ─────────────────────────────────────────────────────────────────────────────
function _contract_coo_aliased_dense!(
    C        :: AliasedBlockSparse{TC,NC,N2C,PC},
    labelsC  :: AbstractVector,
    A        :: COOTensor{TA,NA},
    labelsA  :: AbstractVector,
    B        :: AliasedBlockSparse{TB,NB,N2B,PB},
    labelsB  :: AbstractVector,
    mapA     :: Dict,
    mapB     :: Dict,
    rlab,
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2B,PB}
    @assert !(rlab in labelsC) "rlab must be reduced"
    @assert N2C == N2B - 1 "C dense rank must equal B dense rank − 1 when r is in B's dense tail"

    # Permute B so r is the last dense axis (chunk layout: (chunk, R)).
    redpos = mapB[rlab] - PB        # 1..N2B
    @assert 1 <= redpos <= N2B
    if redpos != N2B
        perm_dense = vcat([d for d in 1:N2B if d != redpos], redpos)
        perm_global = vcat(collect(1:PB), PB .+ perm_dense)
        B       = permutedims(B, perm_global)
        labelsB = labelsB[perm_global]
        mapB    = Dict(lab => i for (i, lab) in enumerate(labelsB))
    end
    @assert mapB[rlab] == NB
    dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2B))
    Rdim     = dimsB_dense[end]
    chunklen = prod(dimsB_dense[1:end-1]; init=1)
    @assert chunklen == C.blksize "C.blksize=$(C.blksize) must equal B-dense-without-r product=$chunklen"
    @assert B.blksize == chunklen * Rdim

    axisAr = mapA[rlab]
    @assert axisAr == NA

    # Source map for each C prefix axis: each label comes from A's non-r axes
    # OR from B's prefix.
    srcA = zeros(Int, PC)
    srcB = zeros(Int, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab) && lab != rlab
            srcA[j] = mapA[lab]
        else
            ax = mapB[lab]
            @assert ax <= PB && lab != rlab "C prefix label $lab must come from A (non-r) or B prefix"
            srcB[j] = ax
        end
    end

    # Clear C; lazy: collect combined templates in `pending`, only commit referenced ones.
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    combined_tid_map = Dict{Tuple{Int,Int}, Int}()
    pending = TC[]
    n_pending = 0

    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    @inbounds for i in eachindex(A.keys)
        acoord = A.keys[i]
        rv     = acoord[axisAr]
        (1 <= rv <= Rdim) || continue
        α      = convert(TC, A.vals[i])

        for j in eachindex(B.keys)
            bkey = B.keys[j]
            tidB = B.alias_ids[j]
            αB   = convert(TC, B.scalars[j])

            combined_tid = get(combined_tid_map, (tidB, rv), 0)
            if combined_tid == 0
                n_pending += 1
                combined_tid = n_pending
                combined_tid_map[(tidB, rv)] = combined_tid
                t_off = (tidB - 1) * B.blksize + (rv - 1) * chunklen
                @inbounds for k in 1:chunklen
                    push!(pending, convert(TC, B.templates[t_off + k]))
                end
            end

            ckey   = ntuple(k -> (srcA[k] != 0 ? acoord[srcA[k]] : bkey[srcB[k]]), Val(PC))
            scalar = α * αB
            _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                 ckey, combined_tid, scalar, C.blksize)
        end
    end

    _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end
