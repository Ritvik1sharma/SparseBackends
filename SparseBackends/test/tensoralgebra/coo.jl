using Test
using SparseBackends

const COOTensor      = SparseBackends.COOTensor
const coo_from_dense = SparseBackends.coo_from_dense
const to_dense       = SparseBackends.to_dense

@testset "COO contract! vs dense reference (no dense_contract_ref)" begin
  @testset "2D x 2D (matmul shape): A(i,r), B(r,j) -> C(i,j)" begin
    labelsA = [:i, :r]
    labelsB = [:r, :j]
    labelsC = [:i, :j]
    rlab = :r
    A_dense = zeros(Float64, 3, 4)  # i=3,r=4
    B_dense = zeros(Float64, 4, 2)  # r=4,j=2
    A_dense[1,1] = 1.0
    A_dense[3,2] = -2.0
    A_dense[2,4] = 3.0
    B_dense[1,2] = 5.0
    B_dense[2,1] = -1.0
    B_dense[4,2] = 2.0
    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    B = coo_from_dense(B_dense; atol=0.0, rtol=0.0)
    C = COOTensor{Float64,2}((3,2))
    mapA = Dict(lab => i for (i,lab) in enumerate(labelsA))
    mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))
    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    println("CDATA is ", C.vals, C.keys)
    sort!(C)
    # Dense reference: here it is literally matrix multiplication
    C_ref = A_dense * B_dense
    # Compare against COO of dense ref (structure/values)
    C_ref_coo = coo_from_dense(C_ref; atol=0.0, rtol=0.0)
    @test length(C.keys) == length(C_ref_coo.keys)
    @test C.keys == C_ref_coo.keys
    @test C.vals == C_ref_coo.vals
    @test to_dense(C_ref_coo) == to_dense(C)  # should match exactly
  end

  @testset "3D x 3D: A(a,b,r), B(r,c,d) -> C(a,b,c,d)" begin
    labelsA = [:a, :b, :r]
    labelsB = [:r, :c, :d]
    labelsC = [:a, :b, :c, :d]
    rlab = :r
    A_dense = zeros(Float64, 2, 3, 4) # a,b,r
    B_dense = zeros(Float64, 4, 2, 2) # r,c,d
    A_dense[1,1,1] = 2.0
    A_dense[2,3,2] = -1.0
    A_dense[1,2,4] = 3.5
    B_dense[1,1,2] = 7.0
    B_dense[2,2,1] = -2.0
    B_dense[4,1,1] = 1.0
    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    B = coo_from_dense(B_dense; atol=0.0, rtol=0.0)
    C = COOTensor{Float64,4}((2,3,2,2))
    mapA = Dict(lab => i for (i,lab) in enumerate(labelsA))
    mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))
    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    sort!(C)
    # Dense reference for general position of reduced axes
    a, b, r = size(A_dense)
    r2, c, d = size(B_dense)
    @assert r == r2
    A_mat = reshape(A_dense, a*b, r)
    B_mat = reshape(B_dense, r, c*d)
    C_ref = reshape(A_mat * B_mat, a, b, c, d)
    @test to_dense(C) == C_ref
    C_ref_coo = coo_from_dense(C_ref; atol=0.0, rtol=0.0)
    # C_ref = dense_reduce_contract(A_dense, mapA[rlab], B_dense, mapB[rlab])
    # @test to_dense(C) == C_ref
    # C_ref_coo = coo_from_dense(C_ref; atol=0.0, rtol=0.0)
    @test length(C.keys) == length(C_ref_coo.keys)
    @test to_dense(C_ref_coo) == to_dense(C)
    @test C.keys == C_ref_coo.keys
    @test C.vals == C_ref_coo.vals
  end
end



# function dense_reduce_contract_matmul(A::AbstractArray, rposA::Int,
#                                       B::AbstractArray, rposB::Int)
#   @assert size(A, rposA) == size(B, rposB)
#   R = size(A, rposA)
#   # permute A so r is last: Aperm dims = (A_out..., R)
#   permA = [d for d in 1:ndims(A) if d != rposA]
#   push!(permA, rposA)
#   Aperm = permutedims(A, permA)
#   dimsA_out = size(Aperm)[1:end-1]
#   A_mat = reshape(Aperm, :, R)  # (prod(A_out) x R)
#   # permute B so r is first: Bperm dims = (R, B_out...)
#   permB = [rposB; [d for d in 1:ndims(B) if d != rposB]...]
#   Bperm = permutedims(B, permB)
#   dimsB_out = size(Bperm)[2:end]
#   B_mat = reshape(Bperm, R, :)  # (R x prod(B_out))
#   # matmul
#   C_mat = A_mat * B_mat  # (prod(A_out) x prod(B_out))
#   # reshape back to tensor with dims (A_out..., B_out...)
#   return reshape(C_mat, (dimsA_out..., dimsB_out...))
# end


#   # helper: dense contraction for A(..., r) and B(r, ...) with exactly one reduced dim r
#   function dense_reduce_contract(A::AbstractArray, rposA::Int, B::AbstractArray, rposB::Int)
#     @assert size(A, rposA) == size(B, rposB)
#     R = size(A, rposA)

#     # output dims = A dims without r + B dims without r
#     dimsA_out = [size(A,d) for d in 1:ndims(A) if d != rposA]
#     dimsB_out = [size(B,d) for d in 1:ndims(B) if d != rposB]
#     dimsC = Tuple(vcat(dimsA_out, dimsB_out))

#     C = zeros(promote_type(eltype(A), eltype(B)), dimsC...)

#     # We do this by explicit indexing (small sizes in tests => fine)
#     # Map output index -> A index and B index
#     for I in CartesianIndices(C)
#       idxC = Tuple(I)
#       # split output index into the A-part and B-part
#       idxA_out = idxC[1:length(dimsA_out)]
#       idxB_out = idxC[length(dimsA_out)+1:end]
#       acc = zero(eltype(C))
#       for r in 1:R
#         # build full indices for A/B
#         idxA = Vector{Int}(undef, ndims(A))
#         idxB = Vector{Int}(undef, ndims(B))
#         # fill A indices
#         ia = 1
#         for d in 1:ndims(A)
#           if d == rposA
#             idxA[d] = r
#           else
#             idxA[d] = idxA_out[ia]; ia += 1
#           end
#         end
#         # fill B indices
#         ib = 1
#         for d in 1:ndims(B)
#           if d == rposB
#             idxB[d] = r
#           else
#             idxB[d] = idxB_out[ib]; ib += 1
#           end
#         end
#         acc += A[idxA...] * B[idxB...]
#       end
#       C[idxC...] = acc
#     end
#     return C
#   end
