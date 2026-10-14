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

    # ── C's key: indices of A (COO), plus indices of B (dense) listed in it, which are
    #    moved into the key ("fission"). Read from labelsC; the key slots are filled by
    #    label, so moved indices may sit at any key position. ──
    Ckey = labelsC[1:PC]
    setA = Set(labelsA)
    keyA = [lab for lab in Ckey if lab in setA]
    keyM = [lab for lab in Ckey if !(lab in setA)]          # moved dense indices of B
    all(lab -> lab != rlab && lab in labelsB, keyM) || _layout_error("COO × dense → aliased",
        "C's key lists an index that is in neither operand")
    nM = length(keyM)

    # ── Align A so its axes are [keyA..., r] ──
    A, labelsA, mapA = aligned_A_to_Cprefix(A, labelsA, mapA, keyA, length(keyA), rlab)

    # ── Align B as [C dense..., moved..., r]: for a moved value m and reduction value rv
    #    the slice B[:, m, rv] is contiguous and becomes one template ──
    desired_not_rlab = vcat([lab for lab in labelsC[PC+1:end] if lab != rlab], keyM)
    rpos = findfirst(==(rlab), labelsB)
    @assert rpos !== nothing "rlab not found in labelsB"
    length(desired_not_rlab) == NB - 1 || _layout_error("COO × dense → aliased",
        "C's dense block and key must together hold every uncontracted index of the dense operand")
    perm = Vector{Int}(undef, NB)
    for (k, lab) in enumerate(desired_not_rlab)
        pos = findfirst(==(lab), labelsB)
        pos === nothing && _layout_error("COO × dense → aliased", "C's dense block lists " *
            "an index that is not in the dense operand (moving a COO index into the dense " *
            "block is not supported)")
        perm[k] = pos
    end
    perm[NB] = rpos
    if any(perm[i] != i for i in 1:NB)
        B       = permutedims(B, perm)
        labelsB = labelsB[perm]
    end

    dimsB   = size(B)
    R       = dimsB[end]
    mdims   = Int[dimsB[NB - 1 - nM + d] for d in 1:nM]
    Mprod   = prod(mdims; init=1)
    blksize = prod(dimsB[1:NB-1-nM]; init=1)
    @assert C.blksize == blksize "C.blksize=$(C.blksize) must equal prod(B dims without r and moved)=$blksize"
    Bvec = vec(B)   # column-major; slice (m, rv) starts at ((rv-1)*Mprod + (m-1))*blksize

    axisAr = NA   # r is the last axis of A after alignment

    # C key slot j comes from A's axis srcA[j] (> 0) or from moved index srcM[j] (> 0)
    srcA = zeros(Int, PC); srcM = zeros(Int, PC)
    @inbounds for j in 1:PC
        ia = findfirst(==(Ckey[j]), keyA)
        if ia !== nothing
            srcA[j] = ia
        else
            srcM[j] = findfirst(==(Ckey[j]), keyM)
        end
    end
    mstrides = ones(Int, max(nM, 1))
    @inbounds for d in 2:nM
        mstrides[d] = mstrides[d-1] * mdims[d-1]
    end
    @inline _mdigit(m, d) = ((m - 1) ÷ mstrides[d]) % mdims[d] + 1
    @inline _ckey(acoord, m) =
        ntuple(j -> srcA[j] != 0 ? acoord[srcA[j]] : _mdigit(m, srcM[j]), Val(PC))

    # ── Exact template count from A's integer keys alone (no dense data touched).
    #    Clean P (every A key hit once, e.g. P·core): one template per distinct (rv, m).
    #    Otherwise mirror the main loop: distinct rv among first contributions to an A
    #    key, plus one accumulated template per A key hit >= 2 times; each × Mprod (every
    #    moved value m repeats the A-key pattern). ──
    rvs   = Set{Int}()
    akeys = Set{NTuple{NA-1,Int}}()
    n_nz  = 0
    @inbounds for acoord in A.keys
        rv = acoord[axisAr]
        (1 <= rv <= R) || continue
        n_nz += 1
        push!(rvs, rv); push!(akeys, ntuple(j -> acoord[j], Val(NA-1)))
    end
    if length(akeys) == n_nz
        n_tmpl_final = length(rvs) * Mprod
    else
        akey_count = Dict{NTuple{NA-1,Int},Int}()
        first_rvs  = Set{Int}()
        @inbounds for acoord in A.keys
            rv = acoord[axisAr]
            (1 <= rv <= R) || continue
            akey = ntuple(j -> acoord[j], Val(NA-1))
            c = get(akey_count, akey, 0) + 1
            akey_count[akey] = c
            c == 1 && push!(first_rvs, rv)
        end
        n_tmpl_final = (length(first_rvs) + count(>=(2), values(akey_count))) * Mprod
    end

    # ── Clear output; templates allocated at their exact final size ──
    C.templates = Vector{TC}(undef, n_tmpl_final * blksize); C.n_templates = 0
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)

    # Template deduplication: (rv, m) -> template id
    rv_to_tid = Dict{Tuple{Int,Int},Int}()

    # Two intermediate maps (resolved into C arrays after all COO entries processed):
    #   key_to_alias[k] = (tid, α)    — block has a single contribution  (aliased)
    #   key_to_accum[k] = Vector{TC}  — block accumulated from multiple contributions
    key_to_alias = Dict{NTuple{PC,Int}, Tuple{Int,TC}}()
    key_to_accum = Dict{NTuple{PC,Int}, Vector{TC}}()

    @inbounds for idx in eachindex(A.keys)
        acoord = A.keys[idx]
        α      = convert(TC, A.vals[idx])
        rv     = acoord[axisAr]
        (1 <= rv <= R) || continue
        for m in 1:Mprod
        ckey    = _ckey(acoord, m)
        src_off = ((rv - 1) * Mprod + (m - 1)) * blksize

        if haskey(key_to_accum, ckey)
            # ── Key already accumulating: add α * B[:,m,rv] to running block ──
            acc     = key_to_accum[ckey]
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
            @simd for j in 1:blksize
                acc[j] = α_prev * C.templates[tmpl_off + j] +
                         α      * convert(TC, Bvec[src_off + j])
            end
            key_to_accum[ckey] = acc

        else
            # ── First contribution: register/reuse a template ──
            tid = get(rv_to_tid, (rv, m), 0)
            if tid == 0
                # New template: copy B[:,m,rv] into C.templates
                C.n_templates += 1
                tid = C.n_templates
                rv_to_tid[(rv, m)] = tid
                dst_off = (tid - 1) * blksize
                for j in 1:blksize
                    C.templates[dst_off + j] = convert(TC, Bvec[src_off + j])
                end
            end
            key_to_alias[ckey] = (tid, α)
        end
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
        copyto!(C.templates, (C.n_templates - 1) * blksize + 1, acc, 1, blksize)
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

    # ── factor-core: persist the PRE-P routing (rv → template id). rv = the reduction
    # index = the core (B) column = pre-P site slice. read_core/write_core! key the core
    # on rv via this map, so they work for OFF-DIAGONAL/flip P (KL), not just diagonal.
    # (Diagonal P has rv == output site, so this equals the group-by-s' derivation ⇒ PXP
    # stays bit-identical.) Accumulated (multi-rv) templates are absent from rv_to_tid —
    # a clean P·core has none. Unset rv slots stay 0 (that pre-P slice absent).
    # With moved indices one rv has Mprod templates, which this map cannot express: it is
    # left empty (absent), so the factor-core helpers refuse rather than guess.
    if Mprod == 1
        s2t = zeros(Int, R)
        for ((rv, _), tid) in rv_to_tid; s2t[rv] = tid; end
        C.slice_to_template = s2t
    else
        C.slice_to_template = Int[]
    end

    @assert C.n_templates == n_tmpl_final
    return C
end