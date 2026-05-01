using Test
using SparseBackends

const COOTensor      = SparseBackends.COOTensor
const coo_from_dense = SparseBackends.coo_from_dense
const to_dense       = SparseBackends.to_dense

@testset "COO manual tests (exact key order)" begin

  @testset "2D: dense -> COO -> dense + exact column-major key order" begin
    dims = (4, 3)
    A = zeros(Float64, dims)
    # Nonzeros at:
    # (3,1), (1,2), (4,3)
    A[3,1] = 2.0
    A[1,2] = -5.0
    A[4,3] = 7.0
    S  = coo_from_dense(A; atol=0.0, rtol=0.0)
    A2 = to_dense(S)
    @test A2 == A
    # Column-major traversal for dims (4,3) visits (i,j) with i fastest:
    # (1,1),(2,1),(3,1),(4,1),(1,2),...,(4,3)
    # So the nonzeros appear in this exact order:
    @test S.keys == [(3,1), (1,2), (4,3)]
    @test S.vals == [2.0, -5.0, 7.0]
    # Compression sanity
    @test length(S.vals) == 3
    @test length(S.vals) < length(A)
  end

  @testset "4D: dense -> COO -> dense + exact column-major key order" begin
    dims = (2, 2, 2, 3)
    A = zeros(Float64, dims)
    # Choose 4 nonzeros:
    A[2,1,1,1] = 11.0
    A[1,2,2,1] = -3.0
    A[2,2,1,2] = 5.0
    A[1,1,2,3] = 7.0
    S  = coo_from_dense(A; atol=0.0, rtol=0.0)
    A2 = to_dense(S)
    @test S.keys == [(2,1,1,1), (1,2,2,1), (2,2,1,2), (1,1,2,3)]
    @test length(S.vals) == 4
    @test length(S.vals) < length(A)
  end

  @testset "setindex!: dirty refresh gives exact key order" begin
    dims = (4, 3)
    S = COOTensor{Float64,2}(dims)
    # Insert keys in arbitrary order
    setindex!(S, 7.0, 4, 3; dirty=true)
    setindex!(S, -5.0, 1, 2; dirty=true)
    setindex!(S, 2.0, 3, 1; dirty=true)
    # S[4,3] = 7.0
    # S[1,2] = -5.0
    # S[3,1] = 2.0
    # After refresh, keys_sorted should be in column-major order for dims (4,3)
    @test S.keys == [(4,3), (1,2), (3,1)]
    @test S.dirty == true
    # Delete one by setting to zero
    S[1,2] = 0.0
    @test S.keys == [(4,3), (3,1)]
    sort!(S)  # force refresh
    @test S.keys == [(3,1), (4,3)]
    @test S.dirty == false
  end

  @testset "permutedims matches dense permutedims + exact key order" begin
    dims = (3, 2, 4)
    A = zeros(Float64, dims)

    A[1,1,1] = 1.0
    A[3,2,4] = -2.0
    A[2,1,3] = 7.0

    S = coo_from_dense(A; atol=0.0, rtol=0.0)

    perm = [3, 1, 2]
    Sp = permutedims(S, perm)
    @test Sp.keys == [(1,1,1), (3,2,1), (4,3,2)]
    Ap = permutedims(A, perm)
    @test to_dense(Sp) == Ap
  end
end
