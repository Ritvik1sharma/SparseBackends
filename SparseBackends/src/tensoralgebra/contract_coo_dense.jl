const Label = NTuple{2,UInt64}


function aligned_A_to_Cprefix(A::COOTensor{TA,NA},
                              labelsA::Vector{Label},
                              mapA::Dict{Label,Int},
                              labelsC::Vector{Label},
                              PC::Int,
                              rlab::Label) where {TA,NA}
  desired = vcat(labelsC[1:PC], rlab)
  mapA = Dict(labelsA[i] => i for i in eachindex(labelsA))
  perm = [mapA[lab] for lab in desired]   # new axes -> old axes
  A2 = permutedims(A, perm)               # <-- NON-mutating version
  labelsA2 = copy(desired)
  sort!(A2)
  mapA2 = Dict(labelsA2[i] => i for i in eachindex(labelsA2))  # reconstruct map for permuted labels
  return A2, labelsA2, mapA2
end


@inline function _axpy_chunk!(
    Cvec::AbstractVector{TC},
    α::TC,
    Bvec::AbstractVector{TB},
    chunklen::Int,
    rv::Int,
) where {TC,TB}
    off = (rv - 1) * chunklen
    @inbounds @simd for t in 1:chunklen
        Cvec[t] += α * convert(TC, Bvec[off + t])
    end
    return nothing
end

"""
COO × Dense -> BlockSparse

Assumes:
  - Only shared label between A and B is rlab
  - rlab reduced: rlab ∉ labelsC
  - A's labels (except r) become C prefix labels (in the order given by labelsC prefix)
  - B's labels (except r) become C dense-tail labels (in the order given by labelsC dense tail)

Mutates:
  - A/labelsA/mapA via ensure_rlab_last_and_sorted!(...)
  - Does NOT mutate B; uses a permuted view/copy if needed
"""
function contract!(
    C::NewBlockSparseSorted{TC,NC,N2,PC},
    labelsC::Vector{Label},
    A::COOTensor{TA,NA},
    labelsA::Vector{Label},
    B::StridedArray{TB,NB},
    labelsB::Vector{Label},
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    rlab::Label;
) where {TC,NC,N2,PC,TA,NA,TB,NB}
  if get(ENV, "SB_TRACE", "0") == "1"
    println("[SB_TRACE] contract_coo_dense.contract!  COO(nnz=", length(A.keys),
            ", dims=", A.dims, ") × Dense(dims=", size(B), ")",
            "  → BS{NC=", NC, ",N2=", N2, ",PC=", PC, ", blksize=", C.blksize, "}")
  end
  time = @elapsed begin
    A, labelsA, mapA = aligned_A_to_Cprefix(A, labelsA, mapA, labelsC, PC, rlab)
    # align_A_to_Cprefix!(A, labelsA, mapA, labelsC, PC, rlab)
  end
  # println("Aligned A to C prefix in $time seconds")

  time = @elapsed begin
    @assert labelsA[end] == rlab
    axisAr = NA

    desired_not_rlab = [lab for lab in labelsC[PC+1:end] if lab != rlab]
    rpos = findfirst(==(rlab), labelsB)
    @assert rpos !== nothing "rlab not found in labelsB"
    # perm maps NEW axis order -> OLD axis positions
    perm = Vector{Int}(undef, NB)
    # Fill non-r axes according to desired order
    for (k, lab) in enumerate(desired_not_rlab)
      pos = findfirst(==(lab), labelsB)
      @assert pos !== nothing "label $lab not found in labelsB"
      perm[k] = pos
    end
    # Put rlab last
    perm[NB] = rpos
    # If this actually changes anything, permute + update labels/map
    if any(@inbounds perm[i] != i for i in 1:NB)
        B = permutedims(B, perm)  # or permutedims!(B, perm) if you have in-place
        labelsB = labelsB[perm]
        mapB = Dict(lab => i for (i, lab) in enumerate(labelsB))
    end

    @assert labelsB[end] == rlab
    @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
    expected_prefix = labelsA[1:end-1]          # all COO dims except r
    expected_dense  = labelsB[1:end-1]         # all dense dims except r
    @assert PC == length(expected_prefix) "PC mismatch: expected PC=$(length(expected_prefix))"
    @assert N2 == length(expected_dense)  "N2 mismatch: expected N2=$(length(expected_dense))"
    @assert NC == PC + N2                 "NC mismatch: expected NC=PC+N2"

    @assert labelsC[1:PC] == expected_prefix "C prefix labels must be A labels without r"
    @assert labelsC[PC+1:end] == expected_dense "C dense labels must be B labels without r"
    dimsB = size(B)                      # (d1,...,dN2, R)
    R = dimsB[end]
    chunklen = prod(dimsB[1:end-1]; init=1)
    @assert C.blksize == chunklen "C.blksize must equal prod(B dims without r)"
    empty!(C.keys); empty!(C.ids); empty!(C.data)
    Bvec = vec(B)   # column-major; slices for fixed rv are contiguous chunks
  end

  time = @elapsed begin
    @inbounds for idx in eachindex(A.keys)
      acoord = A.keys[idx]
      α = convert(TC, A.vals[idx])
      rv = acoord[axisAr]
      (1 <= rv <= R) || continue
      ckey = ntuple(j -> acoord[j], Val(PC))
      cid  = _ensure_block!(C, ckey)
      Cvec = _block_view(C, cid)
      _axpy_chunk!(Cvec, α, Bvec, chunklen, rv)
    end
  end
  return C
end