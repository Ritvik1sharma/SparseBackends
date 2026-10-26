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
- The entries of one A key (one row, over the reduction index `rv`) give the block
  `Σ_rv a_rv B[:,rv]`. Rows whose entries are proportional (exactly, after dividing by the
  largest-magnitude entry) share ONE stored template, `Σ_rv (a_rv/λ) B[:,rv]`; each key keeps the
  scalar λ of its row. A row with a single entry is `α * B[:,rv]`: the template is the slice
  itself, shared by every row with that `rv`.
- No template is stored that no key references.

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

    # ── Rows of A: the entries of one A key (c_l, s', c_r) over the reduction index rv. Rows whose
    #    entries are proportional, compared exactly after dividing by the largest-magnitude entry
    #    (first of ties, in rv order), form one class. A class stores ONE template per moved value m,
    #    Σ_rv (a_rv / λ) B[:, m, rv], and each key of the class carries the scalar λ of its row.
    #    A row with one nonzero entry (a diagonal-type P) is the class {rv}: its template is
    #    B[:, m, rv] itself and its scalar the entry. Template ids: (class - 1) * Mprod + m, classes
    #    numbered by the first row that holds them. ──
    AK = NTuple{NA-1,Int}
    row_of = Dict{AK,Int}(); row_first = Int[]; row_entries = Vector{Vector{Int}}()
    @inbounds for idx in eachindex(A.keys)
        acoord = A.keys[idx]
        (1 <= acoord[axisAr] <= R) || continue
        ak = ntuple(j -> acoord[j], Val(NA-1))
        r = get!(row_of, ak, length(row_first) + 1)
        r > length(row_first) && (push!(row_first, idx); push!(row_entries, Int[]))
        push!(row_entries[r], idx)
    end
    nrow = length(row_first)
    class_of = Dict{Tuple{Vector{Int},Vector{TC}},Int}()
    class_rvs = Vector{Vector{Int}}(); class_ratio = Vector{Vector{TC}}()
    row_class = Vector{Int}(undef, nrow); row_scalar = Vector{TC}(undef, nrow)
    for r in 1:nrow
        ents = sort(row_entries[r]; by = i -> A.keys[i][axisAr])
        nz = [i for i in ents if !iszero(A.vals[i])]
        isempty(nz) && (nz = ents[1:1])                       # an explicit zero row: scalar 0
        vals = TC[convert(TC, A.vals[i]) for i in nz]
        lp = 1
        for q in 2:length(vals); abs(vals[q]) > abs(vals[lp]) && (lp = q); end
        λ = vals[lp]
        ratio = TC[q == lp ? one(TC) : vals[q] / λ + zero(TC) for q in eachindex(vals)]   # + 0: -0.0 -> 0.0
        rvs = Int[A.keys[i][axisAr] for i in nz]
        cls = get!(class_of, (rvs, ratio), length(class_rvs) + 1)
        cls > length(class_rvs) && (push!(class_rvs, rvs); push!(class_ratio, ratio))
        row_class[r] = cls; row_scalar[r] = λ
    end
    ncls = length(class_rvs)
    n_tmpl_final = ncls * Mprod

    # ── Clear output; templates allocated at their exact final size and written once ──
    C.templates = Vector{TC}(undef, n_tmpl_final * blksize); C.n_templates = n_tmpl_final
    empty!(C.keys); empty!(C.alias_ids); empty!(C.scalars)
    @inbounds for cls in 1:ncls, m in 1:Mprod
        dst = ((cls - 1) * Mprod + (m - 1)) * blksize
        rvs, rat = class_rvs[cls], class_ratio[cls]
        for q in eachindex(rvs)
            src = ((rvs[q] - 1) * Mprod + (m - 1)) * blksize
            if q == 1
                if isone(rat[1])
                    for j in 1:blksize; C.templates[dst + j] = convert(TC, Bvec[src + j]); end
                else
                    for j in 1:blksize; C.templates[dst + j] = rat[1] * convert(TC, Bvec[src + j]); end
                end
            else
                @simd for j in 1:blksize
                    C.templates[dst + j] += rat[q] * convert(TC, Bvec[src + j])
                end
            end
        end
    end

    AI = eltype(C.alias_ids)
    for r in 1:nrow, m in 1:Mprod
        push!(C.keys,      _ckey(A.keys[row_first[r]], m))
        push!(C.alias_ids, _alias_id(AI, (row_class[r] - 1) * Mprod + m))
        push!(C.scalars,   row_scalar[r])
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
    # It exists only when every class is a single rv (one nonzero per row: a clean P·core, whose
    # template is the core slice itself). A class of several rv (e.g. the parity gauge) is not a
    # core slice, and one rv with Mprod templates cannot be expressed: the map is left empty
    # (absent), so the factor-core helpers refuse rather than guess. Unset rv slots stay 0.
    if Mprod == 1 && all(c -> length(class_rvs[c]) == 1, 1:ncls)
        s2t = zeros(Int, R)
        for c in 1:ncls; s2t[class_rvs[c][1]] = c; end
        C.slice_to_template = s2t
    else
        C.slice_to_template = Int[]
    end

    @assert C.n_templates == n_tmpl_final
    return C
end