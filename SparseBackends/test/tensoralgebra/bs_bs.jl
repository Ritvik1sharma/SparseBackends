using Test
using SparseBackends

const NewBS = SparseBackends.NewBlockSparseSorted
const blocksparse_from_dense = SparseBackends.blocksparse_from_dense

# --- Helper: stable comparison of blocksparse tensors by (key -> payload)
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


# function dense_ref_prefix_outer_matmul(
#     A_dense::AbstractArray{T,3}, labelsA::Vector{Symbol},
#     B_dense::AbstractArray{T,3}, labelsB::Vector{Symbol},
#     labelsC::Vector{Symbol}
# ) where {T}

#   # --- Put A into canonical layout (i, xa, r) so we can flatten (i,xa) as rows
#   A_ixar = begin
#     if labelsA == [:i, :r, :xa]
#       permutedims(A_dense, (1, 3, 2))   # (i,xa,r)
#     elseif labelsA == [:r, :i, :xa]
#       permutedims(A_dense, (2, 3, 1))   # (i,xa,r)
#     elseif labelsA == [:i, :xa, :r]
#       A_dense                             # already (i,xa,r)
#     else
#       error("unsupported labelsA: $labelsA")
#     end
#   end

#   # --- Put B into canonical layout (j, xb, r) so we can flatten (j,xb) as rows
#   B_jxbr = begin
#     if labelsB == [:j, :r, :xb]
#       permutedims(B_dense, (1, 3, 2))   # (j,xb,r)
#     elseif labelsB == [:r, :j, :xb]
#       permutedims(B_dense, (2, 3, 1))   # (j,xb,r)
#     elseif labelsB == [:j, :xb, :r]
#       B_dense                             # already (j,xb,r)
#     else
#       error("unsupported labelsB: $labelsB")
#     end
#   end
#   I, XA, R  = size(A_ixar)
#   J, XB, R2 = size(B_jxbr)
#   @assert R == R2
#   # Flatten non-r dims as rows, r as columns:
#   #   A2: (I*XA) × R
#   #   B2: (J*XB) × R
#   A2 = reshape(A_ixar, I * XA, R)
#   B2 = reshape(B_jxbr, J * XB, R)
#   # Matmul: (I*XA × R) * (R × J*XB) => (I*XA × J*XB)
#   M = A2 * transpose(B2)
#   # Reshape back into (i, xa, j, xb)
#   C_ixajxb = reshape(M, I, XA, J, XB)
#   # Canonical output we like is (i, j, xa, xb)
#   C_ijxaxb = permutedims(C_ixajxb, (1, 3, 2, 4))
#   # Finally permute to match labelsC (prefix order + dense order)
#   if labelsC == [:i, :j, :xa, :xb]
#     return C_ijxaxb
#   elseif labelsC == [:i, :j, :xb, :xa]
#     return permutedims(C_ijxaxb, (1, 2, 4, 3))
#   elseif labelsC == [:j, :i, :xa, :xb]
#     return permutedims(C_ijxaxb, (2, 1, 3, 4))
#   elseif labelsC == [:j, :i, :xb, :xa]
#     return permutedims(C_ijxaxb, (2, 1, 4, 3))
#   else
#     error("unsupported labelsC: $labelsC")
#   end
# end

function dense_ref_prefix_outer_matmul(
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
      A_dense                             # already (i,xa,r)
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
      B_dense                             # already (j,xb,r)
    else
      error("unsupported labelsB: $labelsB")
    end

  I, XA, R  = size(A_ixar)
  J, XB, R2 = size(B_jxbr)
  @assert R == R2

  # Flatten non-r dims as rows, r as columns
  A2 = reshape(A_ixar, I * XA, R)   # (I*XA)×R
  B2 = reshape(B_jxbr, J * XB, R)   # (J*XB)×R

  # (I*XA×R) * (R×J*XB) -> (I*XA×J*XB)
  M = A2 * transpose(B2)

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
  else
    error("unsupported labelsC: $labelsC")
  end
end

function dense_ref_dense_bb_simple(
    A_dense, labelsA::Vector{Symbol}, N2A::Int,
    B_dense, labelsB::Vector{Symbol}, N2B::Int,
    labelsC::Vector{Symbol}, rlab::Symbol,
)
  NA = length(labelsA); NB = length(labelsB)
  PA = NA - N2A;        PB = NB - N2B

  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
  axAr = findfirst(==(rlab), labelsA); @assert axAr !== nothing
  axBr = findfirst(==(rlab), labelsB); @assert axBr !== nothing
  @assert axAr > PA "rlab must be in A dense tail"
  @assert axBr > PB "rlab must be in B dense tail"

  # Canonicalize so r is LAST dense axis in both:
  A_prefix = labelsA[1:PA]
  B_prefix = labelsB[1:PB]
  A_dense_labs = labelsA[PA+1:NA]
  B_dense_labs = labelsB[PB+1:NB]
  Awo = [l for l in A_dense_labs if l != rlab]
  Bwo = [l for l in B_dense_labs if l != rlab]

  labelsA_can = vcat(A_prefix, Awo, [rlab])
  labelsB_can = vcat(B_prefix, Bwo, [rlab])

  permA = Tuple(map(l -> findfirst(==(l), labelsA), labelsA_can))
  permB = Tuple(map(l -> findfirst(==(l), labelsB), labelsB_can))

  A_can = permutedims(A_dense, permA)
  B_can = permutedims(B_dense, permB)

  dimsA = size(A_can)
  dimsB = size(B_can)
  R = dimsA[end]
  @assert dimsB[end] == R
  # Flatten all non-r dims into rows; r into columns
  rowsA = prod(dimsA[1:end-1]; init=1)
  rowsB = prod(dimsB[1:end-1]; init=1)
  A2 = reshape(A_can, rowsA, R)
  B2 = reshape(B_can, rowsB, R)
  M = A2 * transpose(B2)  # rowsA × rowsB
  # Reshape back to: [A_prefix..., Awo..., B_prefix..., Bwo...]
  dimsA_no_r = dimsA[1:end-1]
  dimsB_no_r = dimsB[1:end-1]
  C_tmp = reshape(M, dimsA_no_r..., dimsB_no_r...)
  labels_tmp = vcat(labelsA_can[1:end-1], labelsB_can[1:end-1])

  # Finally permute to match labelsC exactly
  permC = Tuple(map(l -> findfirst(==(l), labels_tmp), labelsC))
  @assert all(p -> p !== nothing, permC)
  return permutedims(C_tmp, permC)
end

@testset "contract_prefix_outer_bb! (blocksparse x blocksparse, r in prefix, outer-product dense)" begin
  # Sizes
  I, J, R = 3, 2, 4
  XA, XB = 2, 3   # dense tail sizes (A.blksize=XA, B.blksize=XB)
  N2A = 1
  N2B = 1
  N2C = N2A + N2B  # =2

  # Nonzero patterns: create multiple blocks per r on both sides.
  # A depends on (i,r,xa), B on (j,r,xb)
  A_canon = zeros(Float64, I, R, XA)  # (i,r,xa)
  B_canon = zeros(Float64, J, R, XB)  # (j,r,xb)

  # Fill A: multiple i for same r
  A_canon[1,1,1] = 2.0
  A_canon[2,1,2] = -1.0
  A_canon[3,3,1] = 0.5
  A_canon[1,4,2] = 1.5

  # Fill B: multiple j for same r
  B_canon[1,1,2] = 7.0
  B_canon[2,1,1] = -2.0
  B_canon[2,3,3] = 1.0
  B_canon[1,4,1] = 0.25

  # Generate test variants:
  labelsA_variants = [
    [:i, :r, :xa],   # r already in prefix, not last? (prefix is [:i,:r] so r is last already)
    [:r, :i, :xa],   # r first in prefix -> forces permute
    [:i, :xa, :r],   # r in dense tail (should NOT use prefix kernel; but this test suite is prefix-only)
  ]

  labelsB_variants = [
    [:j, :r, :xb],   # r last in prefix already
    [:r, :j, :xb],   # r first in prefix -> forces permute
    [:j, :xb, :r],   # r in dense tail (not prefix) -> not for this kernel
  ]

  # Only keep prefix-valid ones (r in prefix means position ≤ PB=NB-N2B = 2)
  labelsA_variants = filter(l -> (findfirst(==( :r), l) <= 2), labelsA_variants)
  labelsB_variants = filter(l -> (findfirst(==( :r), l) <= 2), labelsB_variants)

  labelsC_variants = [
    [:i,:j,:xa,:xb],  # dense order A then B
    [:i,:j,:xb,:xa],  # dense order B then A
    [:j,:i,:xa,:xb],  # prefix swapped
    [:j,:i,:xb,:xa],  # prefix swapped + dense swapped
  ]

  for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
    @testset "labelsA=$(labelsA), labelsB=$(labelsB), labelsC=$(labelsC)" begin
      # Make A_dense in the requested labels order from canonical (i,r,xa)
      A_dense = if labelsA == [:i,:r,:xa]
        A_canon
      elseif labelsA == [:r,:i,:xa]
        permutedims(A_canon, (2,1,3))
      else
        error("unexpected labelsA in loop")
      end

      # Make B_dense in requested labels order from canonical (j,r,xb)
      B_dense = if labelsB == [:j,:r,:xb]
        B_canon
      elseif labelsB == [:r,:j,:xb]
        permutedims(B_canon, (2,1,3))
      else
        error("unexpected labelsB in loop")
      end

      # Convert to blocksparse
      A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)
      B = blocksparse_from_dense(B_dense, Val(N2B); atol=0.0, rtol=0.0)

      # Output C skeleton (all zeros) with correct dims according to labelsC
      C_ref_dense = dense_ref_prefix_outer_matmul(
        A_dense, labelsA, B_dense, labelsB, labelsC, :r
      )
      C = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2C); atol=0.0, rtol=0.0)

      mapA = Dict(l => i for (i,l) in enumerate(labelsA))
      mapB = Dict(l => i for (i,l) in enumerate(labelsB))

      # Call kernel through your public contract! dispatcher (or call contract_prefix_outer_bb! directly)
      SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, :r)
      # Reference blocksparse
      C_ref = blocksparse_from_dense(C_ref_dense, Val(N2C); atol=0.0, rtol=0.0)
      _assert_blocksparse_equal(C, C_ref)
    end
  end


@testset "contract_dense_bb_simple! (BlockSparse×BlockSparse, r in dense tails)" begin
  rlab = :r
  # Helper: build output dims from labelsC and input dense arrays
  function _dims_from_labels(labels::Vector{Symbol}, A_dense, labelsA::Vector{Symbol}, B_dense, labelsB::Vector{Symbol})
    d = Dict{Symbol,Int}()
    for (i,l) in enumerate(labelsA); d[l] = size(A_dense, i); end
    for (i,l) in enumerate(labelsB); d[l] = size(B_dense, i); end
    return Tuple(d[l] for l in labels)
  end

  # Variants: r in dense tail of BOTH, but not necessarily last (forces permutes)
  labelsA_variants = [
    [:pa, :a1, :r],   # r last in dense tail
    [:pa, :r,  :a1],  # r not last in dense tail (forces dense permute)
  ]
  labelsB_variants = [
    [:pb, :b1, :r],
    [:pb, :r,  :b1],
  ]
  labelsC_variants = [
    [:pa, :pb, :a1, :b1],  # prefix AB, dense AB
    [:pa, :pb, :b1, :a1],  # prefix AB, dense BA
    [:pb, :pa, :a1, :b1],  # prefix BA, dense AB
    [:pb, :pa, :b1, :a1],  # prefix BA, dense BA
  ]
  # Sizes
  PA, PB = 2, 3        # prefix dims
  A1, B1 = 4, 5        # non-r dense dims
  R      = 3           # reduced dim
  N2A = 2
  N2B = 2
  N2C = (N2A - 1) + (N2B - 1)  # = 2

  for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
    @testset "labelsA=$(labelsA) labelsB=$(labelsB) labelsC=$(labelsC)" begin
      # Build canonical dense tensors in easy-to-think order first:
      # A_can: (pa, a1, r)
        # B_can: (pb, b1, r)
        A_can = zeros(Float64, PA, A1, R)
        B_can = zeros(Float64, PB, B1, R)

        # Put a few structured nonzeros so we get sparse-ish blocks, not full
        A_can[1, 1, 1] = 2.0
        A_can[2, 3, 2] = -1.0
        A_can[1, 4, 3] = 0.5

        B_can[1, 2, 1] = 7.0
        B_can[3, 5, 2] = -2.0
        B_can[2, 1, 3] = 1.0

        # Permute A_can/B_can into requested label orders
        # (A_can is [:pa,:a1,:r], B_can is [:pb,:b1,:r])
        function _perm_from(src_labels, dst_labels)
          return Tuple(map(l -> findfirst(==(l), src_labels), dst_labels))
        end

        permA = _perm_from([:pa,:a1,:r], labelsA)
        permB = _perm_from([:pb,:b1,:r], labelsB)
        A_dense = permutedims(A_can, permA)
        B_dense = permutedims(B_can, permB)

        # Convert to blocksparse (dense tail length = N2A/N2B)
        A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)
        B = blocksparse_from_dense(B_dense, Val(N2B); atol=0.0, rtol=0.0)

        # Build output skeleton C as all-zeros blocksparse with correct dims
        C_dims = _dims_from_labels(collect(labelsC), A_dense, collect(labelsA), B_dense, collect(labelsB))
        C0 = zeros(Float64, C_dims...)
        C  = blocksparse_from_dense(C0, Val(N2C); atol=0.0, rtol=0.0)

        mapA = Dict(l => i for (i,l) in enumerate(labelsA))
        mapB = Dict(l => i for (i,l) in enumerate(labelsB))

        # --- Run kernel (direct call)
        SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

        # --- Dense reference (ONLY dense permute/reshape/*) then convert to blocksparse
        Cref_dense = dense_ref_dense_bb_simple(
          A_dense, collect(labelsA), N2A,
          B_dense, collect(labelsB), N2B,
          collect(labelsC), rlab
        )
        C_ref = blocksparse_from_dense(Cref_dense, Val(N2C); atol=0.0, rtol=0.0)
        _assert_blocksparse_equal(C, C_ref)
      end
    end
  end
end
