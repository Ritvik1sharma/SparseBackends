using Test
using SparseBackends

const coo_from_dense = SparseBackends.coo_from_dense
const blocksparse_from_dense = SparseBackends.blocksparse_from_dense
const NewBlockSparseSorted = SparseBackends.NewBlockSparseSorted

# Dense reference: A(..., r) with r last; B(r, ...) with r first
function dense_ref_r_last_A_r_first_B(A_dense::AbstractArray, B_dense::AbstractArray)
  rA = size(A_dense, ndims(A_dense))
  rB = size(B_dense, 1)
  @assert rA == rB
  A_mat = reshape(A_dense, :, rA)   # (prod(A_out) x r)
  B_mat = reshape(B_dense, rB, :)   # (r x prod(B_out))
  C_mat = A_mat * B_mat
  A_out = size(A_dense)[1:end-1]
  B_out = size(B_dense)[2:end]
  return reshape(C_mat, (A_out..., B_out...))
end

@testset "COO × BlockSparse kernel equals dense->blocksparse reference" begin
  @testset "A(i,r) × B(r,j,k) -> C(i,j,k) (N2=2)" begin
    labelsA = [:i, :r]
    labelsB = [:r, :j, :k]
    labelsC = [:i, :j, :k]
    rlab = :r
    N2 = 2
    I, R, J, K = 3, 4, 2, 3
    A_dense = zeros(Float64, I, R)
    B_dense = zeros(Float64, R, J, K)

    A_dense[1,1] = 2.0
    A_dense[3,2] = -1.0
    A_dense[2,4] = 0.5

    B_dense[1,1,2] = 7.0
    B_dense[2,2,1] = -2.0
    B_dense[4,1,3] = 1.0

    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    B = blocksparse_from_dense(B_dense, Val(2); atol=0.0, rtol=0.0)

    # Build empty/zero output container with correct dims
    C0_dense = zeros(Float64, I, J, K)
    C = blocksparse_from_dense(C0_dense, Val(2); atol=0.0, rtol=0.0)
    mapA = Dict(lab => i for (i,lab) in enumerate(labelsA))
    mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))
    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    # Dense reference then convert to blocksparse
    C_ref_dense = dense_ref_r_last_A_r_first_B(A_dense, B_dense)
    C_ref = blocksparse_from_dense(C_ref_dense, Val(2); atol=0.0, rtol=0.0)

    @test C.dims == C_ref.dims
    @test C.blksize == C_ref.blksize
    @test C.keys == C_ref.keys
    for i in 1:length(C.keys)
        key1, key2 = C.keys[i], C_ref.keys[i]
        @test key1 == key2
        id1, id2 = C.ids[i], C_ref.ids[i]
        blockdata1 = C.data[id1*C.blksize - (C.blksize - 1) : id1*C.blksize]
        blockdata2 = C_ref.data[id2*C_ref.blksize - (C_ref.blksize - 1) : id2*C_ref.blksize]
        @test blockdata1 == blockdata2
    end
    # @test C.data == C_ref.data
    # @test C.ids == C_ref.ids
  end

  @testset "A(r,i) × B(r,j,k) -> C(i,j,k) (forces A permute)" begin
    labelsA = [:r, :i]
    labelsB = [:r, :j, :k]
    labelsC = [:i, :j, :k]
    rlab = :r
    N2 = 2

    R, I, J, K = 4, 3, 2, 2
    A_dense = zeros(Float64, R, I)
    B_dense = zeros(Float64, R, J, K)

    A_dense[1,1] = 2.0
    A_dense[2,3] = -1.0
    A_dense[4,2] = 0.5

    B_dense[1,1,2] = 7.0
    B_dense[2,2,1] = -2.0
    B_dense[4,1,1] = 1.0

    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    B = blocksparse_from_dense(B_dense, Val(2); atol=0.0, rtol=0.0)

    C0_dense = zeros(Float64, I, J, K)
    C = blocksparse_from_dense(C0_dense, Val(2); atol=0.0, rtol=0.0)

    mapA = Dict(lab => i for (i,lab) in enumerate(labelsA))
    mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))

    println("==============")
    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    println("==============")

    # Reference: C(i,j,k) = sum_r A(r,i)*B(r,j,k)
    # Put A into (i,r) so "r last" holds for the matmul helper:
    A_ir = permutedims(A_dense, (2,1))  # (i,r)
    C_ref_dense = dense_ref_r_last_A_r_first_B(A_ir, B_dense)
    C_ref = blocksparse_from_dense(C_ref_dense, Val(2); atol=0.0, rtol=0.0)

    @test C.dims == C_ref.dims
    @test C.blksize == C_ref.blksize
    @test C.keys == C_ref.keys
    for i in 1:length(C.keys)
        key1, key2 = C.keys[i], C_ref.keys[i]
        @test key1 == key2
        id1, id2 = C.ids[i], C_ref.ids[i]
        blockdata1 = C.data[id1*C.blksize - (C.blksize - 1) : id1*C.blksize]
        blockdata2 = C_ref.data[id2*C_ref.blksize - (C_ref.blksize - 1) : id2*C_ref.blksize]
        @test blockdata1 == blockdata2
    end
    # @test C.data == C_ref.data
    # @test C.ids == C_ref.ids
  end

  @testset "A(i,p,r) × B(r,j,k) -> C(i,p,j,k) (bigger prefix)" begin
    labelsA = [:i, :p, :r]
    labelsB = [:r, :j, :k]
    labelsC = [:i, :p, :j, :k]
    rlab = :r
    N2 = 2

    I, P, R, J, K = 2, 3, 4, 2, 2
    A_dense = zeros(Float64, I, P, R)
    B_dense = zeros(Float64, R, J, K)

    A_dense[1,1,1] = 2.0
    A_dense[2,3,2] = -1.0
    A_dense[1,2,4] = 3.5

    B_dense[1,1,2] = 7.0
    B_dense[2,2,1] = -2.0
    B_dense[4,1,1] = 1.0

    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    B = blocksparse_from_dense(B_dense, Val(2); atol=0.0, rtol=0.0)

    C0_dense = zeros(Float64, I, P, J, K)
    C = blocksparse_from_dense(C0_dense, Val(2); atol=0.0, rtol=0.0)

    mapA = Dict(lab => i for (i,lab) in enumerate(labelsA))
    mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))

    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

    C_ref_dense = dense_ref_r_last_A_r_first_B(A_dense, B_dense)
    C_ref = blocksparse_from_dense(C_ref_dense, Val(2); atol=0.0, rtol=0.0)

    @test C.dims == C_ref.dims
    @test C.blksize == C_ref.blksize
    @test C.keys == C_ref.keys
    for i in 1:length(C.keys)
      key1, key2 = C.keys[i], C_ref.keys[i]
      @test key1 == key2
      id1, id2 = C.ids[i], C_ref.ids[i]
      blockdata1 = C.data[id1*C.blksize - (C.blksize - 1) : id1*C.blksize]
      blockdata2 = C_ref.data[id2*C_ref.blksize - (C_ref.blksize - 1) : id2*C_ref.blksize]
      @test blockdata1 == blockdata2
    end
  end

  @testset "B dense tail already (x,r): labelsC = [:i,:j,:x]" begin
    labelsA = [:i, :r]          # r last
    labelsB = [:j, :x, :r]      # r in dense tail, already last
    labelsC = [:i, :j, :x]      # dense tail of C is [:x]
    rlab = :r
    I, R, J, X = 3, 4, 2, 3
    A_dense = zeros(Float64, I, R)
    B_dense = zeros(Float64, J, X, R)   # (j, x, r)
    A_dense[1,1] = 2.0
    A_dense[3,2] = -1.0
    A_dense[2,4] = 0.5
    B_dense[1,1,1] = 7.0
    B_dense[2,2,2] = -2.0
    B_dense[1,3,4] = 1.0
    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    sort!(A)  # r-runs contiguous (r last)
    B = blocksparse_from_dense(B_dense, Val(2); atol=0.0, rtol=0.0)
    C0 = zeros(Float64, I, J, X)
    C  = blocksparse_from_dense(C0, Val(1); atol=0.0, rtol=0.0)
    mapA = Dict(l => i for (i,l) in enumerate(labelsA))
    mapB = Dict(l => i for (i,l) in enumerate(labelsB))
    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

    # ---- dense reference via matmul per j-block:
    # Put B_dense into layout (j, x, r) so r is last
    Bp = B_dense
    Bmat = reshape(Bp, :, size(A_dense,2))
    M = A_dense * transpose(Bmat)
    C_ref = reshape(M, size(A_dense,1), size(Bp,1), size(Bp,2))
    C_ref = blocksparse_from_dense(C_ref, Val(1); atol=0.0, rtol=0.0)
    @test C.keys == C_ref.keys
    @test C.blksize == C_ref.blksize
    for i in 1:length(C.keys)
      key1, key2 = C.keys[i], C_ref.keys[i]
      @test key1 == key2
      id1, id2 = C.ids[i], C_ref.ids[i]
      blockdata1 = C.data[id1*C.blksize - (C.blksize - 1) : id1*C.blksize]
      blockdata2 = C_ref.data[id2*C_ref.blksize - (C_ref.blksize - 1) : id2*C_ref.blksize]
      @test blockdata1 == blockdata2
    end
  end

  @testset "B dense tail is (r,x) so kernel must permute: labelsC = [:i,:j,:x]" begin
    labelsA = [:i, :r]
    labelsB = [:j, :r, :x]      # r in dense tail but NOT last
    labelsC = [:i, :j, :x]
    rlab = :r
    I, R, J, X = 2, 3, 3, 2
    A_dense = zeros(Float64, I, R)
    B_dense = zeros(Float64, J, R, X)   # (j, r, x)
    A_dense[1,1] = 1.0
    A_dense[2,3] = -2.0
    B_dense[1,1,1] = 4.0
    B_dense[2,2,2] = 5.0
    B_dense[3,3,1] = -1.0

    A = coo_from_dense(A_dense; atol=0.0, rtol=0.0)
    sort!(A)
    B = blocksparse_from_dense(B_dense, Val(2); atol=0.0, rtol=0.0)
    C0 = zeros(Float64, I, J, X)
    C  = blocksparse_from_dense(C0, Val(1); atol=0.0, rtol=0.0)
    mapA = Dict(l => i for (i,l) in enumerate(labelsA))
    mapB = Dict(l => i for (i,l) in enumerate(labelsB))

    SparseBackends.contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)

    # ---- dense reference: permute B_dense to (j, x, r) then matmul
    Bp = permutedims(B_dense, (1, 3, 2))   # (j, x, r)
    Bmat = reshape(Bp, :, size(A_dense,2))        # (J*X)×R
    M = A_dense * transpose(Bmat)                 # I×(J*X)
    C_ref = reshape(M, size(A_dense,1), size(Bp,1), size(Bp,2))  # I×J×X
    C_ref = blocksparse_from_dense(C_ref, Val(1); atol=0.0, rtol=0.0)
    @test C.keys == C_ref.keys
    @test C.blksize == C_ref.blksize
    @test C.keys == C_ref.keys
    for i in 1:length(C.keys)
      key1, key2 = C.keys[i], C_ref.keys[i]
      @test key1 == key2
      id1, id2 = C.ids[i], C_ref.ids[i]
      blockdata1 = C.data[id1*C.blksize - (C.blksize - 1) : id1*C.blksize]
      blockdata2 = C_ref.data[id2*C_ref.blksize - (C_ref.blksize - 1) : id2*C_ref.blksize]
      @test blockdata1 == blockdata2
    end
  end
end