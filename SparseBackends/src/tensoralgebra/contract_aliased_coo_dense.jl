# tensoralgebra/contract_aliased_coo_dense.jl
#
# Contraction kernel: COOTensor × Dense  →  AliasedBlockSparse
#
# This is the primary constructor for AliasedBlockSparse.  When the COO tensor
# has at most one nonzero per prefix key the result blocks are  α * B[:,rv],
# so only the scalar needs to be stored per block — the dense slice is shared.
#
# Mirrors contract_coo_dense.jl but writes an AliasedBlockSparse instead of
# a NewBlockSparseSorted.

"""
    contract_aliased!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

Compute `C = A ⊗_rlab B` where `A` is a `COOTensor`, `B` is dense, and `C` is
an `AliasedBlockSparse`.

Exploits the aliasing structure:
- When a prefix key in A appears exactly **once**, the resulting block is
  `α * B[:,rv]` — a scalar multiple of a stored template.  No copy of the
  dense slice is made; only the scalar is recorded.
- Multiple COO entries with the **same** `rv` (reduction index) share a
  single stored template, regardless of their prefix keys.
- When a prefix key accumulates contributions from multiple rv values,
  the block is accumulated concretely and stored as its own template
  (scalar = 1) so correctness is always maintained.

Preconditions (same as the COO × Dense `contract!`):
  - `rlab` is the unique shared label, reduced (not in labelsC).
  - A prefix labels (except rlab) map to C's prefix labels.
  - B labels (except rlab) map to C's dense-tail labels.
  - `C.blksize == prod(B dims excluding rlab)`.
"""
function contract_aliased!(
    C        :: AliasedBlockSparse{TC,NC,N2,PC},
    labelsC  :: Vector{Label},
    A        :: COOTensor{TA,NA},
    labelsA  :: Vector{Label},
    B        :: StridedArray{TB,NB},
    labelsB  :: Vector{Label},
    mapA     :: Dict{Label,Int},
    mapB     :: Dict{Label,Int},
    rlab     :: Label,
) where {TC,NC,N2,PC,TA,NA,TB,NB}
    if false  # SB_TRACE — flip to true here for debug output
      println("[SB_TRACE] contract_aliased_coo_dense.contract_aliased!",
              "  COO(nnz=", length(A.keys), ", dims=", A.dims, ")",
              " × Dense(dims=", size(B), ")",
              "  → AliasedBS{NC=", NC, ",N2=", N2, ",PC=", PC, ", blksize=", C.blksize, "}")
    end

    # ── Align A so prefix order matches labelsC[1:PC], r is last ──
    A, labelsA, mapA = aligned_A_to_Cprefix(A, labelsA, mapA, labelsC, PC, rlab)

    # ── Align B so non-r dims match labelsC[PC+1:end], r is last ──
    desired_not_rlab = [lab for lab in labelsC[PC+1:end] if lab != rlab]
    rpos = findfirst(==(rlab), labelsB)
    @assert rpos !== nothing "rlab not found in labelsB"
    perm = Vector{Int}(undef, NB)
    for (k, lab) in enumerate(desired_not_rlab)
        pos = findfirst(==(lab), labelsB)
        @assert pos !== nothing "label $lab not found in labelsB"
        perm[k] = pos
    end
    perm[NB] = rpos
    if any(perm[i] != i for i in 1:NB)
        B       = permutedims(B, perm)
        labelsB = labelsB[perm]
    end

    dimsB   = size(B)
    R       = dimsB[end]
    blksize = prod(dimsB[1:end-1]; init=1)
    @assert C.blksize == blksize "C.blksize=$(C.blksize) must equal prod(B dims without r)=$blksize"
    Bvec = vec(B)   # column-major; slice for fixed rv is contiguous

    # ── Clear output ──
    empty!(C.templates); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    # Template deduplication: rv index -> template id
    rv_to_tid = Dict{Int,Int}()

    # Two intermediate maps (resolved into C arrays after all COO entries processed):
    #   key_to_alias[k] = (tid, α)    — block has a single contribution  (aliased)
    #   key_to_accum[k] = Vector{TC}  — block accumulated from multiple contributions
    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    axisAr = NA   # r is the last axis of A after alignment

    @inbounds for idx in eachindex(A.keys)
        acoord = A.keys[idx]
        α      = convert(TC, A.vals[idx])
        rv     = acoord[axisAr]
        (1 <= rv <= R) || continue
        ckey   = ntuple(j -> acoord[j], Val(PC))

        if haskey(key_to_accum, ckey)
            # ── Key already accumulating: add α * B[:,rv] to running block ──
            acc     = key_to_accum[ckey]
            src_off = (rv - 1) * blksize
            @simd for j in 1:blksize
                acc[j] += α * convert(TC, Bvec[src_off + j])
            end

        elseif haskey(key_to_alias, ckey)
            if false  # SB_TRACE — flip to true here for debug output
              println("[SB_TRACE]   demotion fired @ ckey=", ckey, " (rv=", rv, ")")
            end
            # ── Second contribution to this key: demote alias → accumulator ──
            (tid_prev, α_prev) = key_to_alias[ckey]
            delete!(key_to_alias, ckey)
            acc      = Vector{TC}(undef, blksize)
            tmpl_off = (tid_prev - 1) * blksize
            src_off  = (rv - 1) * blksize
            @simd for j in 1:blksize
                acc[j] = α_prev * C.templates[tmpl_off + j] +
                         α      * convert(TC, Bvec[src_off + j])
            end
            key_to_accum[ckey] = acc

        else
            # ── First contribution: register/reuse a template ──
            tid = get(rv_to_tid, rv, 0)
            if tid == 0
                # New template: copy B[:,rv] into C.templates
                C.n_templates += 1
                tid = C.n_templates
                rv_to_tid[rv] = tid
                src_off = (rv - 1) * blksize
                for j in 1:blksize
                    push!(C.templates, convert(TC, Bvec[src_off + j]))
                end
            end
            key_to_alias[ckey] = (tid, α)
        end
    end

    # ── Collect alias blocks ──
    AI = eltype(C.alias_ids)
    for (k, (tid, α)) in key_to_alias
        push!(C.keys,      k)
        push!(C.alias_ids, _alias_id(AI, tid))
        push!(C.scalars,   α)
    end

    # ── Collect accumulated blocks: store each as its own template, scalar = 1 ──
    for (k, acc) in key_to_accum
        C.n_templates += 1
        append!(C.templates, acc)
        push!(C.keys,      k)
        push!(C.alias_ids, _alias_id(AI, C.n_templates))
        push!(C.scalars,   one(TC))
    end

    # ── Sort keys in column-major order ──
    pdims = ntuple(i -> C.dims[i], Val(PC))
    p = sortperm(C.keys; by = k -> _prefix_lin(k, pdims))
    C.keys      = C.keys[p]
    C.alias_ids = C.alias_ids[p]
    C.scalars   = C.scalars[p]

    return C
end