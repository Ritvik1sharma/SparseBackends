using Test
using SparseBackends

const from_dense = SparseBackends.blocksparse_from_dense
const to_dense   = SparseBackends.to_dense

@testset "blocksparse: getindex/setindex!/permutedims" begin

  @testset "getindex returns zeros for missing blocks" begin
    # N=4, N2=2 => P=2
    dims = (4, 3, 2, 5)
    A = SparseBackends.NewBlockSparseSorted{Float64,4,2,2}(dims, 10, NTuple{2,Int}[], Int[], Float64[])
    # no blocks allocated
    @test A[1,1,1,1] == 0.0
    @test A[4,3,2,5] == 0.0
  end

  @testset "getindex returns stored values for existing blocks" begin
    # N=4, N2=2 => P=2, blksize = 2*5 = 10
    dims = (4, 3, 2, 5)
    A = SparseBackends.NewBlockSparseSorted{Float64,4,2,2}(dims, 10, NTuple{2,Int}[], Int[], Float64[])
    # Create 1 block manually at prefix (3,1) with id=1
    push!(A.keys, (3,1))
    push!(A.ids, 1)
    append!(A.data, zeros(Float64, A.blksize))  # allocate payload for block 1
    # Write two entries inside the block payload directly
    # Block axes are dims 3 and 4, so (k,l) = (1,1) is linear index 1
    A.data[1] = 42.0
    # (k,l) = (2,5): linear index = 1 + (2-1)*1 + (5-1)*2 = 10 in column-major for (2,5)
    # (Because dim3=2 is first/fast axis, dim4=5 is second)
    A.data[10] = -7.0
    # Now check getindex sees them at the correct full indices
    @test A[3,1,1,1] == 42.0
    @test A[3,1,2,5] == -7.0
    # Other coordinates in same block should still be zero
    @test A[3,1,2,4] == 0.0
    # Different prefix (missing block) should be zero
    @test A[1,2,1,1] == 0.0
  end

  @testset "setindex! allocates blocks and round-trips via getindex" begin
    dims = (4, 3, 2, 5)  # P=2, N2=2, blksize=10
    A = SparseBackends.NewBlockSparseSorted{Float64,4,2,2}(dims, 10, NTuple{2,Int}[], Int[], Float64[])
    A[3,1,2,5] = 7.5
    A[3,1,1,1] = -2.0   # same block, different within-block coord
    println("A.keys after sets: ", A.keys)
    A[1,2,1,1] = 3.0    # different block
    println("A.keys after sets: ", A.keys)
    # Check values via getindex
    @test A[3,1,2,5] == 7.5
    @test A[3,1,1,1] == -2.0
    @test A[1,2,1,1] == 3.0
    # Missing block still reads as zero
    @test A[4,3,1,1] == 0.0
    # Keys should be sorted
    @test A.keys == [(3,1), (1,2)]
    # Data size: nblocks * blksize
    @test length(A.data) == length(A.keys) * A.blksize
  end

  @testset "from_dense keys are in column-major prefix order" begin
    dims = (4, 3, 2, 5) # P=2, N2=2
    A = zeros(Float64, dims)

    for k in 1:dims[3], l in 1:dims[4]
        A[1, 2, k, l] = 10 + (k - 1) * dims[4] + l
        A[3, 1, k, l] = -((k - 1) * dims[4] + l)
    end
    A[4, 3, 1, 1] = 7
    A[4, 3, 2, 5] = -9

    B = from_dense(A, Val(2); atol=0.0, rtol=0.0, dropzeros=true)

    # Exact expected order in column-major over prefix dims (4,3):
    @test B.keys == [(3,1), (1,2), (4,3)]

    # Also check it's consistent with column-major order generically:
    prefix_dims = (dims[1], dims[2])
    B.keys == [(3,1), (1,2), (4,3)]
  end

  @testset "permutedims matches dense permutedims for allowed perms (prefix-only, block-only)" begin
    dims = (4, 3, 2, 5) # P=2, N2=2
    D = zeros(Float64, dims)

    # Put a few blocks in
    for k in 1:dims[3], l in 1:dims[4]
      D[1,2,k,l] = 10 + (k-1)*dims[4] + l
      D[3,1,k,l] = -((k-1)*dims[4] + l)
    end
    D[4,3,1,1] = 7
    D[4,3,2,5] = -9

    A = from_dense(D, Val(2); atol=0.0, rtol=0.0, dropzeros=true)

    # Permute prefix dims (swap dims 1 and 2), keep block dims order
    perm1 = [2, 1, 3, 4]
    B1 = permutedims(A, perm1)
    @test to_dense(B1) == permutedims(D, perm1)

    # Permute block dims (swap dims 3 and 4), keep prefix order
    perm2 = [1, 2, 4, 3]
    B2 = permutedims(A, perm2)
    @test to_dense(B2) == permutedims(D, perm2)

    # Permute both within their groups
    perm3 = [2, 1, 4, 3]
    B3 = permutedims(A, perm3)
    @test to_dense(B3) == permutedims(D, perm3)
  end

  @testset "permutedims rejects crossing prefix/block axes" begin
    dims = (4, 3, 2, 5) # P=2, N2=2
    D = zeros(Float64, dims)
    D[1,2,1,1] = 1.0
    A = from_dense(D, Val(2); atol=0.0, rtol=0.0, dropzeros=true)

    # This crosses: puts a block axis (3) into prefix position 1
    badperm = [3, 2, 1, 4]
    @test_throws AssertionError permutedims(A, badperm)
  end
end