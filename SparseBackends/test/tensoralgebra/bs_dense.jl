using Test
using SparseBackends

const NewBS = SparseBackends.NewBlockSparseSorted
const blocksparse_from_dense = SparseBackends.blocksparse_from_dense

function _blocks_by_key(T::NewBS)
  d = Dict{typeof(T.keys[1]), Vector{eltype(T.data)}}()
  @inbounds for idx in eachindex(T.keys)
    k  = T.keys[idx]
    id = T.ids[idx]
    off = (id - 1) * T.blksize
    d[k] = collect(@view T.data[off+1 : off+T.blksize])
  end
  return d
end

function _assert_blocksparse_equal(A::NewBS, B::NewBS)
  @test A.dims == B.dims
  @test A.blksize == B.blksize
  @test A.keys == B.keys  # if your canonical order is deterministic
  dA = _blocks_by_key(A)
  dB = _blocks_by_key(B)
  @test keys(dA) == keys(dB)
  for k in keys(dA)
    @test dA[k] == dB[k]
  end
end

function dense_ref_prefix_outer_matmul_bd(
    A_dense::AbstractArray{T,3}, labelsA::Vector{Symbol},
    B_dense::AbstractArray{T,3}, labelsB::Vector{Symbol},
    labelsC::Vector{Symbol},
    rlab::Symbol
) where {T}

  @assert rlab in labelsA
  @assert rlab in labelsB
  @assert !(rlab in labelsC)

  # Canonicalize A to (i, xa, r)
  A_ixar =
    if labelsA == [:i, :r, :xa]
      permutedims(A_dense, (1, 3, 2))   # (i,xa,r)
    elseif labelsA == [:r, :i, :xa]
      permutedims(A_dense, (2, 3, 1))   # (i,xa,r)
    elseif labelsA == [:i, :xa, :r]
      A_dense                           # already (i,xa,r)
    else
      error("unsupported labelsA: $labelsA")
    end

  # Canonicalize B to (j, xb, r)
  B_jxbr =
    if labelsB == [:j, :r, :xb]
      permutedims(B_dense, (1, 3, 2))   # (j,xb,r)
    elseif labelsB == [:r, :j, :xb]
      permutedims(B_dense, (2, 3, 1))   # (j,xb,r)
    elseif labelsB == [:j, :xb, :r]
      B_dense                           # already (j,xb,r)
    else
      error("unsupported labelsB: $labelsB")
    end

  I, XA, R  = size(A_ixar)
  J, XB, R2 = size(B_jxbr)
  @assert R == R2

  # Flatten non-r dims as rows, r as columns
  A2 = reshape(A_ixar, I * XA, R)   # (I*XA)×R
  B2 = reshape(B_jxbr, J * XB, R)   # (J*XB)×R

  # Sum_r outer => A2 * B2'
  M = A2 * transpose(B2)            # (I*XA) × (J*XB)

  # Unflatten to (i, xa, j, xb)
  C_ixajxb = reshape(M, I, XA, J, XB)

  # Canonical output (i, j, xa, xb)
  C_ijxaxb = permutedims(C_ixajxb, (1, 3, 2, 4))

  # Permute to match labelsC
  if labelsC == [:i, :j, :xa, :xb]
    return C_ijxaxb
  elseif labelsC == [:i, :j, :xb, :xa]
    return permutedims(C_ijxaxb, (1, 2, 4, 3))
  elseif labelsC == [:j, :i, :xa, :xb]
    return permutedims(C_ijxaxb, (2, 1, 3, 4))
  elseif labelsC == [:j, :i, :xb, :xa]
    return permutedims(C_ijxaxb, (2, 1, 4, 3))

  # NEW (needed for BS×Dense tests with N2C=3 layout):
  # (i, xa, j, xb) = permute (i, j, xa, xb) axes (1,3,2,4)
  elseif labelsC == [:i, :xa, :j, :xb]
    return permutedims(C_ijxaxb, (1, 3, 2, 4))

  else
    error("unsupported labelsC: $labelsC")
  end
end

@testset "contract! (blocksparse x dense, r in A prefix, outer-product dense)" begin
  I, J, R = 3, 2, 4
  XA, XB = 2, 3
  N2A = 1
  N2C = 3  # dense tail will be [:xa,:xb] or [:xb,:xa]

  A_canon = zeros(Float64, I, R, XA)  # (i,r,xa)
  B_canon = zeros(Float64, J, R, XB)  # (j,r,xb)

  A_canon[1,1,1] = 2.0
  A_canon[2,1,2] = -1.0
  A_canon[3,3,1] = 0.5
  A_canon[1,4,2] = 1.5

  B_canon[1,1,2] = 7.0
  B_canon[2,1,1] = -2.0
  B_canon[2,3,3] = 1.0
  B_canon[1,4,1] = 0.25

  labelsA_variants = [
    [:i, :r, :xa],
    [:r, :i, :xa],
  ]

  labelsB_variants = [
    [:j, :r, :xb],
    [:r, :j, :xb],
    [:j, :xb, :r],
  ]

  # IMPORTANT: C prefix must be only [:i] (A prefix excluding :r).
  # Dense tail must contain [:j, :xa, :xb] in an order that your kernel supports:
  #   either [A_dense..., B_wo_r...] = [:xa, :j, :xb]
  #   or     [B_wo_r..., A_dense...] = [:j, :xb, :xa]
  labelsC_variants = [
    [:i, :xa, :j, :xb],  # AB form
    [:i, :j, :xb, :xa],  # BA form
  ]

  for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
    @testset "labelsA=$(labelsA), labelsB=$(labelsB), labelsC=$(labelsC)" begin
      A_dense = if labelsA == [:i,:r,:xa]
        A_canon
      elseif labelsA == [:r,:i,:xa]
        permutedims(A_canon, (2,1,3))
      else
        error("unexpected labelsA")
      end

      B_dense = if labelsB == [:j,:r,:xb]
        B_canon
      elseif labelsB == [:r,:j,:xb]
        permutedims(B_canon, (2,1,3))
      elseif labelsB == [:j,:xb,:r]
        permutedims(B_canon, (1,3,2))
      else
        error("unexpected labelsB")
      end

      A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)

      C_ref_dense = dense_ref_prefix_outer_matmul_bd(
        A_dense, collect(labelsA),
        B_dense, collect(labelsB),
        collect(labelsC),
        :r
      )

      C = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2C); atol=0.0, rtol=0.0)

      mapA = Dict(l => i for (i,l) in enumerate(labelsA))
      mapB = Dict(l => i for (i,l) in enumerate(labelsB))

      SparseBackends.contract!(C, labelsC, A, labelsA, B_dense, labelsB, mapA, mapB, :r)

      C_ref = blocksparse_from_dense(C_ref_dense, Val(N2C); atol=0.0, rtol=0.0)
      _assert_blocksparse_equal(C, C_ref)
    end
  end
end

# @testset "contract! (blocksparse x dense, r in A prefix, outer-product dense)" begin
#   # Sizes (same style as BB test)
#   I, J, R = 3, 2, 4
#   XA, XB = 2, 3        # A.blksize = XA, B has XB as its tail
#   N2A = 1              # A has 1 dense axis (:xa)
#   N2C = 2              # C dense = (:xa, :xb) in some order

#   # Canonical dense tensors:
#   # A(i,r,xa), B(j,r,xb)
#   A_canon = zeros(Float64, I, R, XA)
#   B_canon = zeros(Float64, J, R, XB)

#   # Fill A: multiple i for same r
#   A_canon[1,1,1] = 2.0
#   A_canon[2,1,2] = -1.0
#   A_canon[3,3,1] = 0.5
#   A_canon[1,4,2] = 1.5

#   # Fill B: multiple j for same r
#   B_canon[1,1,2] = 7.0
#   B_canon[2,1,1] = -2.0
#   B_canon[2,3,3] = 1.0
#   B_canon[1,4,1] = 0.25

#   # A variants: only those where :r is in prefix (PA=NA-N2A=2)
#   labelsA_variants = [
#     [:i, :r, :xa],   # r last in prefix already
#     [:r, :i, :xa],   # r first in prefix -> forces permute in _permute_r_to_last_prefix
#     [:i, :xa, :r],   # r in dense tail -> NOT prefix, exclude
#   ]
#   labelsA_variants = filter(l -> (findfirst(==( :r), l) <= 2), labelsA_variants)

#   # B (dense) variants: we can allow r anywhere, since BD kernel permutes B to make r last
#   labelsB_variants = [
#     [:j, :r, :xb],
#     [:r, :j, :xb],
#     [:j, :xb, :r],
#   ]

#   labelsC_variants = [
#     [:i,:j,:xa,:xb],
#     [:i,:j,:xb,:xa],
#     [:j,:i,:xa,:xb],
#     [:j,:i,:xb,:xa],
#   ]

#   for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
#     @testset "labelsA=$(labelsA), labelsB=$(labelsB), labelsC=$(labelsC)" begin
#       # Build A_dense in requested order from canonical A(i,r,xa)
#       A_dense = if labelsA == [:i,:r,:xa]
#         A_canon
#       elseif labelsA == [:r,:i,:xa]
#         permutedims(A_canon, (2,1,3))
#       else
#         error("unexpected labelsA in loop")
#       end

#       # Build B_dense in requested order from canonical B(j,r,xb)
#       B_dense = if labelsB == [:j,:r,:xb]
#         B_canon
#       elseif labelsB == [:r,:j,:xb]
#         permutedims(B_canon, (2,1,3))
#       elseif labelsB == [:j,:xb,:r]
#         permutedims(B_canon, (1,3,2))
#       else
#         error("unexpected labelsB in loop")
#       end

#       # Convert A to blocksparse, keep B as dense
#       A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)

#       # Dense reference (same math as BB)
#       C_ref_dense = dense_ref_prefix_outer_matmul_bd(
#         A_dense, collect(labelsA),
#         B_dense, collect(labelsB),
#         collect(labelsC),
#         :r
#       )

#       # Allocate C blocksparse skeleton (zeros) with correct dims
#       C = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2C); atol=0.0, rtol=0.0)

#       mapA = Dict(l => i for (i,l) in enumerate(labelsA))
#       mapB = Dict(l => i for (i,l) in enumerate(labelsB))

#       # Call your BD top-level dispatcher
#       SparseBackends.contract!(C, labelsC, A, labelsA, B_dense, labelsB, mapA, mapB, :r)

#       # Compare against reference blocksparse
#       C_ref = blocksparse_from_dense(C_ref_dense, Val(N2C); atol=0.0, rtol=0.0)
#       _assert_blocksparse_equal(C, C_ref)
#     end
#   end
# end