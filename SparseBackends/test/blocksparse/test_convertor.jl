using Test
using SparseBackends

const from_dense = SparseBackends.blocksparse_from_dense
const to_dense   = SparseBackends.to_dense
@testset "NewBlockSparseSorted dense<->sparse round-trip + footprint" begin
    @testset "2D: N=2, N2=1, P=1" begin
    dims = (6, 5)
    A = zeros(Float64, dims)
    A[2, :] .= (1, 0, 2, 0, 3)
    A[5, :] .= (-1, 4, 0, 0, 7)

    B = from_dense(A, Val(1); atol=0.0, rtol=0.0, dropzeros=true)
    A2 = to_dense(B)

    @test A2 == A
    @test B.dims == dims
    @test B.blksize == 5
    @test B.keys == [(2,), (5,)]
    @test length(B.data) == 2 * 5
    @test length(B.data) < length(A)
  end

  @testset "4D: N=4, N2=2, P=2" begin
    dims = (4, 3, 2, 5)
    A = zeros(Float64, dims)

    for k in 1:dims[3], l in 1:dims[4]
      A[1, 2, k, l] = 10 + (k - 1) * dims[4] + l
      A[3, 1, k, l] = -((k - 1) * dims[4] + l)
    end
    A[4, 3, 1, 1] = 7
    A[4, 3, 2, 5] = -9

    B = from_dense(A, Val(2); atol=0.0, rtol=0.0, dropzeros=true)
    A2 = to_dense(B)

    @test A2 == A
    @test B.dims == dims
    @test B.blksize == 10
    @test B.keys == [(3,1), (1,2), (4,3)] # Keys will be sorted by column major order
    # [(1,2), (3,1), (4,3)]
    @test length(B.data) == 3 * 10
    @test length(B.data) < length(A)
  end
end


#   @testset "2D: N=2, N2=1 (block-sparse over dim1, dense over dim2)" begin
#     # dims = (P=1 sparse prefix, N2=1 dense suffix)
#     dims = (6, 5)
#     A = zeros(Float64, dims)

#     # Choose 2 nonzero blocks out of 6 (each block is a full row of length 5)
#     A[2, :] .= [1, 0, 2, 0, 3]
#     A[5, :] .= [-1, 4, 0, 0, 7]

#     B = from_dense(A, Val(1); atol=0.0, rtol=0.0, dropzeros=true)
#     A2 = to_dense(B)

#     # Correctness
#     @test size(A2) == size(A)
#     @test A2 == A

#     # Footprint checks
#     @test B.dims == dims
#     @test B.blksize == prod((dims[2],)) == 5

#     # We expect exactly 2 stored blocks: prefixes (2) and (5)
#     @test length(B.keys) == 2
#     @test B.keys == [(2,), (5,)]  # lexicographic order
#     @test B.ids == collect(1:length(B.keys))

#     # Data length should match nblocks * blksize
#     @test length(B.data) == length(B.keys) * B.blksize

#     # "Compression" vs storing all 6 blocks densely as block storage
#     full_block_storage = dims[1] * B.blksize  # 6 * 5 = 30
#     @test length(B.data) == 2 * 5
#     @test length(B.data) < full_block_storage

#     # Optional stronger check: total stored entries in sparse repr is smaller than dense
#     # (keys+ids overhead is ignored here; it's fine since you asked "tensor memory footprint")
#     @test length(B.data) < length(A)
#   end


#   @testset "4D: N=4, N2=2 (block-sparse over dims1-2, dense over dims3-4)" begin
#     # dims = (P=2 sparse prefix, N2=2 dense suffix)
#     dims = (4, 3, 2, 5)
#     A = zeros(Float64, dims)

#     # Dense block size = 2*5 = 10 entries per (i,j) prefix
#     # Choose 3 nonzero blocks out of 4*3 = 12 possible blocks
#     # Block at prefix (1,2)
#     for k in 1:dims[3], l in 1:dims[4]
#       A[1, 2, k, l] = 10 + (k-1)*dims[4] + l   # 11..20
#     end
#     # Block at prefix (3,1)
#     for k in 1:dims[3], l in 1:dims[4]
#       A[3, 1, k, l] = -((k-1)*dims[4] + l)     # -1..-10
#     end
#     # Block at prefix (4,3) with some internal zeros (still nonzero block)
#     A[4, 3, 1, 1] = 7
#     A[4, 3, 2, 5] = -9

#     B = from_dense(A, Val(2); atol=0.0, rtol=0.0, dropzeros=true)
#     A2 = to_dense(B)

#     # Correctness
#     @test size(A2) == size(A)
#     @test A2 == A

#     # Footprint checks
#     @test B.dims == dims
#     @test B.blksize == dims[3] * dims[4] == 10

#     # We expect exactly 3 stored blocks; keys must be lexicographically sorted
#     @test length(B.keys) == 3
#     @test B.keys == [(1,2), (3,1), (4,3)]
#     @test B.ids == collect(1:length(B.keys))

#     @test length(B.data) == length(B.keys) * B.blksize  # 3 * 10 = 30

#     full_block_storage = (dims[1] * dims[2]) * B.blksize  # 12 * 10 = 120
#     @test length(B.data) == 30
#     @test length(B.data) < full_block_storage

#     @test length(B.data) < length(A)  # 30 < 4*3*2*5=120
#   end

# end