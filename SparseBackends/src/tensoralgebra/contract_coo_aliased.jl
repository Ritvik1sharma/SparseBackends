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



# Lay out B's dense block for a COO × aliased contraction, from labelsC: permute it to
# [C's dense indices in C's order..., moved..., tail...], where "moved" are B dense
# indices listed in C's key (split into it) and `tail` is the contracted index if it sits
# in B's dense block. Returns B, labelsB, mapB, the moved labels and their dims.
# No permute when B is already in that order (the stock layout).
function _coo_aliased_layout_B(B::AliasedBlockSparse{TB,NB,N2B,PB}, labelsB, labelsC,
                               PC::Int, tail::Vector, kernel::String) where {TB,NB,N2B,PB}
    Bdense = labelsB[PB+1:NB]
    Ckey   = labelsC[1:PC]
    Cdense = collect(labelsC[PC+1:end])
    keyM   = [lab for lab in Ckey if lab in Bdense]
    want   = vcat(Cdense, keyM, tail)
    (length(want) == N2B && Set(want) == Set(Bdense)) || _layout_error(kernel,
        "C's dense block and moved key indices must be exactly B's uncontracted dense " *
        "indices (an index of A or of B's key cannot go in the dense block)")
    if want != Bdense
        perm_dense  = [findfirst(==(lab), Bdense) for lab in want]
        perm_global = vcat(collect(1:PB), PB .+ perm_dense)
        B       = permutedims(B, perm_global)
        labelsB = labelsB[perm_global]
    end
    mapB  = Dict(lab => i for (i, lab) in enumerate(labelsB))
    mdims = Int[B.dims[mapB[lab]] for lab in keyM]
    return B, labelsB, mapB, keyM, mdims
end

# C key slot j comes from A's axis srcA[j], B's key axis srcB[j], or moved index srcM[j].
function _coo_aliased_key_sources(labelsC, PC::Int, mapA, mapB, PB::Int, keyM, rlab, kernel)
    srcA = zeros(Int, PC); srcB = zeros(Int, PC); srcM = zeros(Int, PC)
    @inbounds for j in 1:PC
        lab = labelsC[j]
        if haskey(mapA, lab) && lab != rlab
            srcA[j] = mapA[lab]
        elseif (im = findfirst(==(lab), keyM)) !== nothing
            srcM[j] = im
        else
            ax = get(mapB, lab, 0)
            (1 <= ax <= PB && lab != rlab) || _layout_error(kernel,
                "C's key may only list indices of A, B's key indices and B's dense indices")
            srcB[j] = ax
        end
    end
    return srcA, srcB, srcM
end

@inline _mdigit(m::Int, d::Int, mstrides, mdims) = ((m - 1) ÷ mstrides[d]) % mdims[d] + 1
function _mstrides(mdims)
    st = ones(Int, max(length(mdims), 1))
    for d in 2:length(mdims); st[d] = st[d-1] * mdims[d-1]; end
    return st
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
    K = "COO × aliased (contracted index in B's key)"
    # B's dense block as [C dense..., moved...]: template tidB's chunk for moved value m
    # is then contiguous, and is combined template (tidB - 1) * Mprod + m.
    B, labelsB, mapB, keyM, mdims = _coo_aliased_layout_B(B, labelsB, labelsC, PC, Any[], K)
    Mprod = prod(mdims; init=1)
    @assert C.blksize * Mprod == B.blksize  "C block size × moved dims must equal B's block size"
    axisAr = mapA[rlab]
    @assert axisAr == NA
    axisBr = mapB[rlab]
    @assert axisBr == PB

    srcA, srcB, srcM = _coo_aliased_key_sources(labelsC, PC, mapA, mapB, PB, keyM, rlab, K)
    mstrides = _mstrides(mdims)

    # Clear C; lazy: stage all of B's templates in `pending`, viewed as n_templates × Mprod
    # chunks of C.blksize (indexed 1..n_pending). Only the referenced ones are copied at
    # commit time.
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)
    pending = copy(B.templates)
    n_pending = B.n_templates * Mprod

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
            tidB   = Int(B.alias_ids[bi])   # alias ids may be compacted (UInt8)
            αB     = convert(TC, B.scalars[bi])
            scalar = α * αB
            for m in 1:Mprod
                ckey = ntuple(j -> srcA[j] != 0 ? acoord[srcA[j]] :
                                   srcB[j] != 0 ? bkey[srcB[j]] :
                                   _mdigit(m, srcM[j], mstrides, mdims), Val(PC))
                _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                     ckey, (tidB - 1) * Mprod + m, scalar, C.blksize)
            end
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
    K = "COO × aliased (contracted index in B's dense block)"
    # B's dense block as [C dense..., moved..., r]: for a moved value m and reduction value
    # rv, template tidB's chunk (m, rv) is contiguous and becomes one combined template.
    B, labelsB, mapB, keyM, mdims = _coo_aliased_layout_B(B, labelsB, labelsC, PC, Any[rlab], K)
    @assert mapB[rlab] == NB
    Mprod = prod(mdims; init=1)
    dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2B))
    Rdim     = dimsB_dense[end]
    chunklen = prod(dimsB_dense[1:end-1-length(keyM)]; init=1)
    @assert chunklen == C.blksize "C.blksize=$(C.blksize) must equal B-dense-without-r-and-moved product=$chunklen"
    @assert B.blksize == chunklen * Mprod * Rdim

    axisAr = mapA[rlab]
    @assert axisAr == NA

    srcA, srcB, srcM = _coo_aliased_key_sources(labelsC, PC, mapA, mapB, PB, keyM, rlab, K)
    mstrides = _mstrides(mdims)

    # Clear C; lazy: collect combined templates in `pending`, only commit referenced ones.
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    combined_tid_map = Dict{Tuple{Int,Int,Int}, Int}()
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
            tidB = Int(B.alias_ids[j])   # alias ids may be compacted (UInt8)
            αB   = convert(TC, B.scalars[j])

            scalar = α * αB
            for m in 1:Mprod
                combined_tid = get(combined_tid_map, (tidB, rv, m), 0)
                if combined_tid == 0
                    n_pending += 1
                    combined_tid = n_pending
                    combined_tid_map[(tidB, rv, m)] = combined_tid
                    t_off = (tidB - 1) * B.blksize + ((rv - 1) * Mprod + (m - 1)) * chunklen
                    @inbounds for k in 1:chunklen
                        push!(pending, convert(TC, B.templates[t_off + k]))
                    end
                end

                ckey = ntuple(k -> srcA[k] != 0 ? acoord[srcA[k]] :
                                   srcB[k] != 0 ? bkey[srcB[k]] :
                                   _mdigit(m, srcM[k], mstrides, mdims), Val(PC))
                _aliased_contribute!(key_to_alias, key_to_accum, pending,
                                     ckey, combined_tid, scalar, C.blksize)
            end
        end
    end

    _commit_aliased_dicts_lazy!(C, key_to_alias, key_to_accum, pending, n_pending)
    return C
end
