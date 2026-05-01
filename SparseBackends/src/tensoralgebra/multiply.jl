using LinearAlgebra
using SparseArrays

"""
    contract!(
        C, labelsC,
        A::COOTensor, labelsA,
        B::AbstractArray, labelsB
    )

Assumes:
  - Exactly one shared label r between A and B.
  - r is reduced: r ∉ labelsC.
  - C's prefix labels == labelsA without r (same order).
  - C's block labels   == labelsB without r (same order).
  - C has N2 == length(labelsB)-1 dense axes (last dims).
Overwrites C (clears it first).
"""
function contract!(
    C::NewBlockSparseSorted{TC,NC,N2},
    labelsC::AbstractVector,
    A::COOTensor{TA,NA},
    labelsA::AbstractVector,
    B::AbstractArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab
) where {TC,NC,N2,TA,NA,TB,NB}
  # Expected output label layout:
  preA = [lab for lab in labelsA if lab != rlab]
  blkB = [lab for lab in labelsB if lab != rlab]
  PC = length(preA)
  @assert N2 == length(blkB) "Need C.N2 == (rank(B)-1). Got N2=$N2 but expected $(length(blkB))"
  @assert NC == PC + N2
  @assert labelsC[1:PC] == preA "C prefix labels must equal labelsA without $rlab"
  @assert labelsC[PC+1:NC] == blkB "C block labels must equal labelsB without $rlab"
  # Dim checks
  @assert A.dims[mapA[rlab]] == size(B, mapB[rlab]) "Reduction dim mismatch for $rlab"
  # prefix dims in C match A (excluding r)
  @inbounds for j in 1:PC
    lab = labelsC[j]
    @assert C.dims[j] == A.dims[mapA[lab]] "C prefix dim mismatch for label $lab"
  end
  # block dims in C match B (excluding r)
  # We check in the order blkB appears (same as labelsC block part)
  @inbounds for t in 1:N2
    lab = labelsC[PC+t]
    ax = mapB[lab]
    @assert C.dims[PC+t] == size(B, ax) "C block dim mismatch for label $lab"
  end
  # Clear output (overwrite semantics)
  empty!(C.keys); empty!(C.ids); empty!(C.data)
  # Axis positions
  a_red = mapA[rlab]
  b_red = mapB[rlab]
  # Main loop:
  # For each COO nonzero at coordinate acoord with reduction index rv,
  # take slice = selectdim(B, b_red, rv) and add α * slice into the block keyed by A's prefix coords.
  @inbounds for (acoord, aval) in A.data
    rv = acoord[a_red]
    # prefix key for C comes from A's coordinates excluding the reduced axis
    ckey = ntuple(j -> begin
      lab = labelsC[j]      # these are exactly preA labels
      acoord[mapA[lab]]
    end, Val(PC))
    cid = _ensure_block!(C, ckey)
    Cvec = _block_view(C, cid)
    slice = selectdim(B, b_red, rv)   # rank NB-1 view/array
    @assert length(Cvec) == length(slice) == C.blksize
    _scale_add_block!(Cvec, convert(TC, aval), slice)
  end
  return C
end



# @inline function blocks_by_axis(B::NewBlockSparseSorted{TB,NB,N2}, axis::Int) where {TB,NB,N2}
#   PB = NB - N2
#   dim = B.dims[axis]
#   out = [Vector{Tuple{NTuple{PB,Int},Int}}() for _ in 1:dim]
#   @inbounds for i in eachindex(B.keys)
#     key = B.keys[i]
#     id  = B.ids[i]
#     v   = key[axis]
#     push!(out[v], (key, id))
#   end
#   return out
# end

# # Optional convenience: BlockSparseSorted × COOTensor (same math, just swap arguments)
# function contract!(
#     C::NewBlockSparseSorted{TC,NC,N2},
#     labelsC::AbstractVector,
#     B::NewBlockSparseSorted{TB,NB,N2},
#     labelsB::AbstractVector,
#     A::COOTensor{TA,NA},
#     labelsA::AbstractVector,
#     mapA::Dict,
#     mapB::Dict
# ) where {TC,NC,N2,TA,NA,TB,NB}
#   return contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB)
# end