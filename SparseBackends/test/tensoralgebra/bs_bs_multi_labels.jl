using Test
const NewBS = SparseBackends.NewBlockSparseSorted

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

# helper: permute canonical dense array to a target label order
function _permute_from_canon(X, canon_labels::Vector{Symbol}, target_labels::Vector{Symbol})
  pos = Dict(l => i for (i,l) in enumerate(canon_labels))
  perm = Tuple(pos[l] for l in target_labels)
  return permutedims(X, perm)
end

using LinearAlgebra

# --- grouped check: returns :AthenB, :BthenA, or :interleaved
@inline function _cdense_grouped_mode(Cdense::Vector{Symbol}, keepAset::Set{Symbol}, keepBset::Set{Symbol})
  originA = BitVector(undef, length(Cdense))
  @inbounds for i in 1:length(Cdense)
    lab = Cdense[i]
    if lab in keepAset
      originA[i] = true
    elseif lab in keepBset
      originA[i] = false
    else
      error("Cdense label $lab not in keepA/keepB")
    end
  end

  firstB = findfirst(!, originA)
  lastA  = findlast(identity, originA)
  firstA = findfirst(identity, originA)
  lastB  = findlast(!, originA)

  if firstB === nothing || firstA === nothing
    return :AthenB
  elseif lastA < firstB
    return :AthenB
  elseif lastB < firstA
    return :BthenA
  else
    return :interleaved
  end
end

"""
dense_ref_contract(A, labelsA, PA, B, labelsB, PB, labelsC, PC, shared_labels)

Reference that does:
  permutedims + reshape + one GEMM + reshape + permutedims

Semantics:
- shared_labels are always reduced (must not appear in labelsC)
- shared labels either both prefix or both dense (no cross)
- Cdense must be grouped: [Akeep..., Bkeep...] or [Bkeep..., Akeep...] (within-group perms ok)
"""
function dense_ref_contract(
    A::Array{T,NA}, labelsA::Vector{Symbol}, PA::Int,
    B::Array{T,NB}, labelsB::Vector{Symbol}, PB::Int,
    labelsC::Vector{Symbol}, PC::Int,
    shared_labels::Vector{Symbol}
) where {T,NA,NB}

  # 0) shared reduced
  @inbounds for lab in shared_labels
    (lab in labelsC) && error("shared label $lab must be reduced; found in labelsC")
  end

  posA = Dict(l => i for (i,l) in enumerate(labelsA))
  posB = Dict(l => i for (i,l) in enumerate(labelsB))

  # 1) classify shared
  shared_prefix = Symbol[]
  shared_dense  = Symbol[]
  @inbounds for lab in shared_labels
    ap = posA[lab] <= PA
    bp = posB[lab] <= PB
    ap == bp || error("shared label $lab crosses prefix/dense; not supported")
    (ap ? push!(shared_prefix, lab) : push!(shared_dense, lab))
  end
  spref = Set(shared_prefix)
  sden  = Set(shared_dense)

  # 2) dims per label + consistency
  dims = Dict{Symbol,Int}()
  @inbounds for (i,l) in enumerate(labelsA)
    dims[l] = size(A,i)
  end
  @inbounds for (i,l) in enumerate(labelsB)
    if haskey(dims,l) && dims[l] != size(B,i)
      error("dimension mismatch for label $l")
    end
    dims[l] = size(B,i)
  end

  # 3) kept prefix/dense labels
  A_pref_keep = [l for l in labelsA[1:PA] if !(l in spref)]
  B_pref_keep = [l for l in labelsB[1:PB] if !(l in spref)]

  A_dense_keep = [l for l in labelsA[PA+1:end] if !(l in sden)]
  B_dense_keep = [l for l in labelsB[PB+1:end] if !(l in sden)]

  keepAset = Set(A_dense_keep)
  keepBset = Set(B_dense_keep)

  Cprefix = labelsC[1:PC]
  Cdense  = labelsC[PC+1:end]

  # Sanity: Cprefix should be exactly A_pref_keep ∪ B_pref_keep (any order)
  expected_prefix = Set(vcat(A_pref_keep, B_pref_keep))
  Set(Cprefix) == expected_prefix || error("Cprefix must be kept prefix labels from A/B (shared prefix reduced)")

  # Cdense grouped check
  mode = _cdense_grouped_mode(collect(Cdense), keepAset, keepBset)
  mode == :interleaved && error("Cdense interleaves A/B kept dims; would require permuting C => unsupported")

  # Within-group desired orders from Cdense
  desired_keepA = [l for l in Cdense if l in keepAset]
  desired_keepB = [l for l in Cdense if l in keepBset]
  Set(desired_keepA) == keepAset || error("Cdense missing/duplicating A kept dims")
  Set(desired_keepB) == keepBset || error("Cdense missing/duplicating B kept dims")

  red_dense = [l for l in labelsA[PA+1:end] if l in sden]  # stable A-dense order
  A_order = vcat(A_pref_keep, desired_keepA, shared_prefix, red_dense)
  B_order = vcat(shared_prefix, red_dense, B_pref_keep, desired_keepB)

  permA = Tuple(posA[l] for l in A_order)
  permB = Tuple(posB[l] for l in B_order)

  A2 = permutedims(A, permA)
  B2 = permutedims(B, permB)

  M = prod(dims[l] for l in vcat(A_pref_keep, desired_keepA); init=1)
  K = prod(dims[l] for l in vcat(shared_prefix, red_dense); init=1)
  N = prod(dims[l] for l in vcat(B_pref_keep, desired_keepB); init=1)

  Amat = reshape(A2, M, K)
  Bmat = reshape(B2, K, N)

  Cmat = Matrix{T}(undef, M, N)
  mul!(Cmat, Amat, Bmat)
  produced_labels = vcat(A_pref_keep, desired_keepA, B_pref_keep, desired_keepB)
  produced_dims   = Tuple(dims[l] for l in produced_labels)
  Cprod = reshape(Cmat, produced_dims...)
  posP = Dict(l => i for (i,l) in enumerate(produced_labels))
  permC = Tuple(posP[l] for l in labelsC)
  return permutedims(Cprod, permC)
end


@testset "contract! reference via ONE dense GEMM (shared prefix + shared dense, all reduced)" begin
  # Sizes
  I, J  = 2, 3
  R1, R2 = 2, 2            # shared prefix (reduced)
  XA1, XA2 = 2, 2          # A kept dense
  K1, K2 = 2, 2            # shared dense (reduced)
  XB = 2                   # B kept dense

  # Canonical layouts:
  # A: (i, r1, r2, xa1, xa2, k1, k2)   PA=3, N2A=4
  # B: (j, r1, r2, k1, k2, xb)         PB=3, N2B=3
  labelsA_canon = [:i, :r1, :r2, :xa1, :xa2, :k1, :k2]
  labelsB_canon = [:j, :r1, :r2, :k1, :k2, :xb]

  N2A, N2B, N2C = 4, 3, 3
  PA = length(labelsA_canon) - N2A
  PB = length(labelsB_canon) - N2B
  PC = 2

  A_canon = zeros(Float64, I, R1, R2, XA1, XA2, K1, K2)
  B_canon = zeros(Float64, J, R1, R2, K1, K2, XB)

  # Fill with multiple (r1,r2) so shared-prefix reduction is exercised
  A_canon[1,1,1, 1,2, 1,2] =  2.0
  A_canon[1,2,1, 2,1, 2,1] = -1.0
  A_canon[2,1,2, 1,1, 2,2] =  0.5
  A_canon[2,2,2, 2,2, 1,1] =  1.5

  B_canon[1,1,1, 1,2, 2] =  7.0
  B_canon[1,2,1, 2,1, 1] = -2.0
  B_canon[3,1,2, 2,2, 2] =  1.0
  B_canon[2,2,2, 1,1, 1] =  0.25

  shared_labels = [:r1, :r2, :k1, :k2]  # ALL reduced

  # label variants (must keep r1,r2 in prefix; k1,k2 in dense)
  labelsA_variants = [
    [:i, :r1, :r2, :xa1, :xa2, :k1, :k2],
    [:r1, :i, :r2, :xa2, :xa1, :k2, :k1],
    [:r2, :r1, :i, :k1, :xa1, :xa2, :k2],
  ]
  labelsB_variants = [
    [:j, :r1, :r2, :k1, :k2, :xb],
    [:r2, :j, :r1, :k2, :xb, :k1],
    [:r1, :r2, :j, :xb, :k1, :k2],
  ]

  function _valid(lbls::Vector{Symbol}, P::Int, shared_pref::Vector{Symbol}, shared_den::Vector{Symbol})
    for lab in shared_pref
      p = findfirst(==(lab), lbls)
      (p !== nothing && p <= P) || return false
    end
    for lab in shared_den
      p = findfirst(==(lab), lbls)
      (p !== nothing && p > P) || return false
    end
    return true
  end
  shared_pref = [:r1, :r2]
  shared_den  = [:k1, :k2]
  labelsA_variants = filter(l -> _valid(l, PA, shared_pref, shared_den), labelsA_variants)
  labelsB_variants = filter(l -> _valid(l, PB, shared_pref, shared_den), labelsB_variants)

  # C variants: grouped dense tail, within-group perms ok, prefix order can swap
  labelsC_variants = [
    [:i, :j, :xa1, :xa2, :xb],   # AthenB
    [:i, :j, :xa2, :xa1, :xb],   # AthenB with A-keep perm
    [:j, :i, :xa2, :xa1, :xb],   # prefix swapped
    [:i, :j, :xb, :xa1, :xa2],   # BthenA
    [:j, :i, :xb, :xa2, :xa1],   # BthenA + A-keep perm + prefix swapped
  ]

  for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
    @testset "labelsA=$(labelsA), labelsB=$(labelsB), labelsC=$(labelsC)" begin
      A_dense = _permute_from_canon(A_canon, labelsA_canon, labelsA)
      B_dense = _permute_from_canon(B_canon, labelsB_canon, labelsB)

      A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)
      B = blocksparse_from_dense(B_dense, Val(N2B); atol=0.0, rtol=0.0)

      C_ref_dense = dense_ref_contract(
        A_dense, labelsA, PA,
        B_dense, labelsB, PB,
        labelsC, PC,
        shared_labels
      )

      C = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2C); atol=0.0, rtol=0.0)

      mapA = Dict(l => i for (i,l) in enumerate(labelsA))
      mapB = Dict(l => i for (i,l) in enumerate(labelsB))

      SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared_labels)

      C_ref = blocksparse_from_dense(C_ref_dense, Val(N2C); atol=0.0, rtol=0.0)
      _assert_blocksparse_equal(C, C_ref)
    end
  end
end

@testset "contract! errors when Cdense interleaves A/B kept dims" begin
  I,J,R1,R2,XA1,XA2,K1,K2,XB = 2,2,2,2,2,2,2,2,2
  labelsA = [:i,:r1,:r2,:xa1,:xa2,:k1,:k2]   # PA=3, N2A=4
  labelsB = [:j,:r1,:r2,:k1,:k2,:xb]         # PB=3, N2B=3
  N2A,N2B,N2C = 4,3,3
  PA = length(labelsA) - N2A
  PB = length(labelsB) - N2B
  PC = 2
  shared = [:r1,:r2,:k1,:k2]

  A_dense = zeros(Float64, I,R1,R2,XA1,XA2,K1,K2)
  B_dense = zeros(Float64, J,R1,R2,K1,K2,XB)
  A_dense[1,1,1,1,1,1,1] = 1.0
  B_dense[1,1,1,1,1,1]   = 2.0

  A = blocksparse_from_dense(A_dense, Val(N2A); atol=0.0, rtol=0.0)
  B = blocksparse_from_dense(B_dense, Val(N2B); atol=0.0, rtol=0.0)

  # Interleaved dense tail: Akeep, Bkeep, Akeep => must error
  labelsC_bad = [:i,:j,:xa1,:xb,:xa2]

  C0 = zeros(Float64, I,J,XA1,XB,XA2)
  C  = blocksparse_from_dense(C0, Val(N2C); atol=0.0, rtol=0.0)

  mapA = Dict(l => i for (i,l) in enumerate(labelsA))
  mapB = Dict(l => i for (i,l) in enumerate(labelsB))

  @test_throws ErrorException SparseBackends.contract!(C, labelsC_bad, A, labelsA, B, labelsB, mapA, mapB, shared)
  @test_throws ErrorException dense_ref_contract(A_dense, labelsA, PA, B_dense, labelsB, PB, labelsC_bad, PC, shared)
end
