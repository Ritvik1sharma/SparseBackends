using Test
using LinearAlgebra

# -----------------------------
# Dense gold for COO×dense -> BlockSparse
# C(prefix from A sans r, dense from B sans r) = sum_r A(prefix,r) * B(dense,r)
# Returned in labelsC order.
# -----------------------------
function dense_ref_coo_dense(
    A_dense::AbstractArray{T,NA}, labelsA::Vector{Symbol},
    B_dense::AbstractArray{T,NB}, labelsB::Vector{Symbol},
    labelsC::Vector{Symbol}, rlab::Symbol
) where {T,NA,NB}
  @assert rlab in labelsA && rlab in labelsB
  PC = NA - 1
  # println("bs_bs.jl ", "PC=$PC", " NA=$NA", " NB=$NB", " labelsA=$labelsA", " labelsB=$labelsB", " labelsC=$labelsC", " rlab=$rlab")
  @assert Set(labelsC[1:PC]) == Set(filter(!=(rlab), labelsA))
  @assert Set(labelsC[PC+1:end]) == Set(filter(!=(rlab), labelsB))
  # A -> (C_prefix..., r)
  mapA = Dict(l => i for (i,l) in enumerate(labelsA))
  permA = [mapA[l] for l in vcat(labelsC[1:PC], [rlab])]
  A_can = permutedims(A_dense, Tuple(permA))
  R = size(A_can, NA)
  prefix_dims = size(A_can)[1:PC]
  rows = prod(prefix_dims; init=1)
  A2 = reshape(A_can, rows, R)  # rows×R
  # B -> (C_dense..., r)
  mapB = Dict(l => i for (i,l) in enumerate(labelsB))
  permB = [mapB[l] for l in vcat(labelsC[PC+1:end], [rlab])]
  B_can = permutedims(B_dense, Tuple(permB))
  @assert size(B_can, NB) == R
  dense_dims = size(B_can)[1:NB-1]
  chunklen = prod(dense_dims; init=1)
  B2 = reshape(B_can, chunklen, R)  # chunklen×R
  # (rows×R) * (R×chunklen)
  M = A2 * transpose(B2)
  return reshape(M, prefix_dims..., dense_dims...)
end

# -----------------------------
# Compare BlockSparse results by key -> blockpayload mapping
# -----------------------------
function _blocksparse_to_dict(C)
  d = Dict{eltype(C.keys), Vector{eltype(C.data)}}()
  bs = C.blksize
  @inbounds for i in eachindex(C.keys)
    key = C.keys[i]
    id  = C.ids[i]
    lo = (id - 1) * bs + 1
    hi = id * bs
    d[key] = copy(C.data[lo:hi])
  end
  return d
end

function _assert_blocksparse_equal(C, Cref; atol=0.0, rtol=0.0)
  @test C.dims == Cref.dims
  @test C.blksize == Cref.blksize
  dC = _blocksparse_to_dict(C)
  dR = _blocksparse_to_dict(Cref)
  @test Set(keys(dC)) == Set(keys(dR))
  for k in keys(dR)
    v1 = dC[k]; v2 = dR[k]
    @test length(v1) == length(v2)
    @test all(isapprox.(v1, v2; atol=atol, rtol=rtol))
  end
end

@testset "contract!(COO × dense -> BlockSparse) (skip_zero_slices=true)" begin

  # ---- Config 1: NA=2, NB=2 with varying label orders
  @testset "Config1: A(i,r) with B(x,r) -> C(i,x)" begin
    I, R, X = 3, 4, 5
    rlab = :r
    A_ir = zeros(Float64, I, R)
    A_ir[1,1] = 2.0
    A_ir[3,2] = -1.0
    A_ir[2,4] = 0.5
    B_xr = zeros(Float64, X, R)
    B_xr[2,1] = 7.0
    B_xr[5,2] = -2.0
    B_xr[1,4] = 1.0

    labelsA_variants = [[:i,:r], [:r,:i]]
    labelsB_variants = [[:x,:r], [:r,:x]]
    labelsC_variants = [[:i,:x], [:i,:x]]

    for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
      println("Testing labelsA=$labelsA, labelsB=$labelsB, labelsC=$labelsC")
      A_dense = labelsA == [:i,:r] ? A_ir : permutedims(A_ir, (2,1))
      B_dense = labelsB == [:x,:r] ? B_xr : permutedims(B_xr, (2,1))
      A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
      C_ref_dense = dense_ref_coo_dense(A_dense, labelsA, B_dense, labelsB, labelsC, rlab)
      N2 = length(labelsB) - 1
      C  = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2); atol=0.0, rtol=0.0)
      mapA = Dict(l => i for (i,l) in enumerate(copy(labelsA)))
      mapB = Dict(l => i for (i,l) in enumerate(copy(labelsB)))
      SparseBackends.contract!(
        C, copy(labelsC),
        A, copy(labelsA),
        B_dense, copy(labelsB),
        mapA, mapB,
        rlab)
      C_ref = blocksparse_from_dense(C_ref_dense, Val(N2); atol=0.0, rtol=0.0)
      _assert_blocksparse_equal(C, C_ref; atol=0.0, rtol=0.0)
    end
  end

  # ---- Config 2: NA=3, NB=3, reorder prefix + dense label orders
  @testset "Config2: A(p,i,r) with B(y,x,r) -> C(prefix+dense reorder)" begin
    P, I, R = 2, 3, 4
    X, Y    = 2, 3
    rlab = :r

    A_pir = zeros(Float64, P, I, R)
    A_pir[1,1,1] = 2.0
    A_pir[2,3,2] = -1.0
    A_pir[1,2,4] = 0.5
    B_yxr = zeros(Float64, Y, X, R)
    B_yxr[1,2,1] = 7.0
    B_yxr[3,1,2] = -2.0
    B_yxr[2,2,4] = 1.0
    labelsA_variants = [[:p,:i,:r], [:i,:p,:r], [:r,:p,:i]]
    labelsB_variants = [[:y,:x,:r], [:x,:y,:r], [:r,:y,:x]]

    # PC=2, dense=2
    labelsC_variants = [
      [:p,:i,:y,:x],
      [:i,:p,:y,:x],
      [:p,:i,:x,:y],
      [:i,:p,:x,:y],
    ]

    makeA(labelsA) = labelsA == [:p,:i,:r] ? A_pir :
                     labelsA == [:i,:p,:r] ? permutedims(A_pir, (2,1,3)) :
                     labelsA == [:r,:p,:i] ? permutedims(A_pir, (3,1,2)) :
                     error("unexpected labelsA $labelsA")
    makeB(labelsB) = labelsB == [:y,:x,:r] ? B_yxr :
                     labelsB == [:x,:y,:r] ? permutedims(B_yxr, (2,1,3)) :
                     labelsB == [:r,:y,:x] ? permutedims(B_yxr, (3,1,2)) :
                     error("unexpected labelsB $labelsB")
    for labelsA in labelsA_variants, labelsB in labelsB_variants, labelsC in labelsC_variants
      # println("Testing labelsA=$labelsA, labelsB=$labelsB, labelsC=$labelsC")
      A_dense = makeA(labelsA)
      B_dense = makeB(labelsB)
      A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
      C_ref_dense = dense_ref_coo_dense(A_dense, labelsA, B_dense, labelsB, labelsC, rlab)
      N2 = length(labelsB) - 1
      C  = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2); atol=0.0, rtol=0.0)
      mapA = Dict(l => i for (i,l) in enumerate(copy(labelsA)))
      mapB = Dict(l => i for (i,l) in enumerate(copy(labelsB)))
      SparseBackends.contract!(
        C, copy(labelsC),
        A, copy(labelsA),
        B_dense, copy(labelsB),
        mapA, mapB,
        rlab)
      C_ref = blocksparse_from_dense(C_ref_dense, Val(N2); atol=0.0, rtol=0.0)
      _assert_blocksparse_equal(C, C_ref; atol=0.0, rtol=0.0)
    end
  end

  # ---- Config 3: ensure skip_zero_slices does not create any block when the only contributing rv slice is zero
  @testset "skip_zero_slices=true => no blocks if every contributing slice is all-zero" begin
    I, R, X = 3, 4, 5
    rlab = :r
    labelsA = [:i,:r]
    labelsB = [:x,:r]
    labelsC = [:i,:x]
    A_dense = zeros(Float64, I, R)
    A_dense[2,3] = 1.0               # only rv=3 would contribute
    B_dense = zeros(Float64, X, R)    # but B[:,3] is all-zero => output all-zero
    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    C_ref_dense = dense_ref_coo_dense(A_dense, labelsA, B_dense, labelsB, labelsC, rlab)
    @test all(iszero, C_ref_dense)
    N2 = length(labelsB) - 1
    C  = blocksparse_from_dense(zeros(Float64, size(C_ref_dense)...), Val(N2); atol=0.0, rtol=0.0)
    mapA = Dict(l => i for (i,l) in enumerate(copy(labelsA)))
    mapB = Dict(l => i for (i,l) in enumerate(copy(labelsB)))
    SparseBackends.contract!(
      C, copy(labelsC),
      A, copy(labelsA),
      B_dense, copy(labelsB),
      mapA, mapB,
      rlab)
    @test isempty(C.keys)
    @test isempty(C.ids)
    @test isempty(C.data)
  end
end
