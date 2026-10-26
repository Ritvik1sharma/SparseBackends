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

using ITensors, Random, LinearAlgebra
# Regression tests for the t26 block-sparse fixes (complex bs x dense, column-major key order,
# + / map! index identity, factorize grouping the labels a row key meets).
@testset "block-sparse fixes (t26)" begin

  SB = SparseBackends
  rng = MersenneTwister(26)
  wrapbs(bs, is) = ITensors._itensor_from_external_storage(SB.WrappedBlockSparse(bs, Tuple(is)))
  todense(T) = SB.to_dense_itensors_unfused(T)
  keys_colmajor(T) = (bsx = ITensors.get_external_storage(T).blocksparse;
                      pd = bsx.dims[1:SB._P(bsx)]; lin = [SB._prefix_lin(k, pd) for k in bsx.keys];
                      all(i -> lin[i] < lin[i+1], 1:length(lin)-1))
  # block-sparse ITensor over `is` (first P key, rest dense) holding the listed keys (default: all)
  function bsten(::Type{T}, is, P; keep = k -> true) where {T}
    N = length(is); dims = Tuple(ITensors.dim.(is)); blk = prod(dims[P+1:N]; init = 1)
    ks = [Tuple(c) for c in CartesianIndices(dims[1:P]) if keep(Tuple(c))]
    wrapbs(SB.NewBlockSparseSorted{T,N,N-P,P,Int}(dims, blk, ks, collect(1:length(ks)), randn(rng, T, blk * length(ks))), is)
  end
  a, b, c = Index(2, "a"), Index(2, "b"), Index(2, "c")
  d, e = Index(3, "d"), Index(2, "e")

  @testset "complex bs x dense is not conjugated (contract_dense_bd!)" begin
    for TB in (ComplexF64, Float64)
      A = bsten(ComplexF64, [a, d], 1; keep = k -> k[1] == 2)      # key a, dense d (contracted)
      B = ITensor(randn(rng, TB, 3, 2), d, e)
      C = ITensors.contract(A, B; preserve_bs_output = true)
      @test norm(todense(C) - todense(A) * B) < 1e-12
    end
  end

  @testset "bs x dense via contract_shared!: keys in column-major order" begin
    A = bsten(Float64, [a, b, c, d], 3; keep = k -> !(k[1] == 1 && k[2] == 1))
    for B in (ITensor(randn(rng, 2, 3, 2), c, d, e), ITensor(randn(rng, 2), e))   # 2 shared / outer product
      C = ITensors.contract(A, B; preserve_bs_output = true)
      @test keys_colmajor(C)
      @test norm(todense(C) - todense(A) * B) < 1e-12
      @test norm(todense(C + C) - 2 * todense(C)) < 1e-12
    end
  end

  @testset "fuse_sparse_axes on non-adjacent key axes keeps column-major keys" begin
    A = bsten(Float64, [a, b, c, d], 3; keep = k -> k != (1, 2, 2))
    F = SB.fuse_sparse_axes(ITensors.get_external_storage(A).blocksparse, 1, 3)
    pd = F.dims[1:2]; lin = [SB._prefix_lin(k, pd) for k in F.keys]
    @test all(diff(lin) .> 0)
    Fa = reshape(permutedims(Array(todense(A), a, b, c, d), (1, 3, 2, 4)), 4, 2, 3)   # fused (a,c) index: a fastest
    @test all(F[i, j, k] == Fa[i, j, k] for i in 1:4, j in 1:2, k in 1:3)
  end

  @testset "+ and map! need the same indices in the same order" begin
    A  = bsten(Float64, [a, b, d, e], 2)
    B  = bsten(Float64, [a, b, d, e], 2)
    Bp = SB.permute(B, b, a, e, d)
    @test norm(todense(A + B) - (todense(A) + todense(B))) < 1e-12
    C = copy(A); map!((x, y) -> x + y, C, C, B)
    @test norm(todense(C) - (todense(A) + todense(B))) < 1e-12
    # same set in another order, different index sets, or a different key/dense split: errors
    @test_throws ErrorException A + Bp
    @test_throws ErrorException (C1 = copy(A); map!((x, y) -> x + y, C1, C1, Bp))
    @test_throws ErrorException A + bsten(Float64, [a, b, d, Index(2, "z")], 2)
    Bx = bsten(Float64, [a, d, b, e], 2)
    @test_throws ErrorException A + Bx
    @test_throws ErrorException (C2 = copy(A); map!((x, y) -> x + y, C2, C2, Bx))
  end

  @testset "factorize groups the bond labels a row key meets" begin
    # keys (site s, label b), dense axis = multiplicity; L: (s, b*, new mult), R: (b*, b, new mult, mult)
    arr(B) = [B[Tuple(I)...] for I in CartesianIndices(B.dims)]
    LR(L, R) = (l = arr(L); r = arr(R);
                [sum(l[s, c, k] * r[c, b, k, m] for c in axes(l, 2), k in axes(l, 3))
                 for s in axes(l, 1), b in axes(r, 2), m in axes(r, 4)])
    # (1) every site value meets both labels: one group {1, 2}, b* = 1
    # (2) s=1 meets b=1,2, s=2 meets b=2, s=3 meets b=3: groups {1, 2} (rows s=1,2) and {3} (row s=3)
    for (dims, keys, groups) in (((2, 2, 2), [(1, 1), (2, 1), (1, 2), (2, 2)], [([1, 2], [1, 2])]),
                                 ((3, 3, 2), [(1, 1), (1, 2), (2, 2), (3, 3)], [([1, 2], [1, 2]), ([3], [3])]))
      A  = SB.NewBlockSparseSorted{Float64,3,1,2,Int}(dims, dims[3], keys, collect(1:length(keys)),
                                                      randn(rng, dims[3] * length(keys)))
      Ad = arr(A)
      # dense SVD per group of the matrix [s, (b, m)]
      Fs = [svd(reshape(Ad[ss, bs, :], length(ss), :)) for (ss, bs) in groups]
      for md in (typemax(Int), 1)
        L, R, spec = SB.blocksparse_factorize(A, [1]; maxdim = md, cutoff = 0.0)
        @test L.keys == sort([(s, first(bs)) for (ss, bs) in groups for s in ss])     # (s, b*)
        @test R.keys == [(first(bs), b) for (ss, bs) in groups for b in bs]           # (b*, b)
        Ld = arr(L)
        Lm = reshape(Ld, size(Ld, 1), :)                                               # [s, (b, new mult)]
        kk = [min(md, length(F.S)) for F in Fs]
        kb = zeros(Int, dims[2]); for ((ss, bs), k) in zip(groups, kk); kb[first(bs)] = k; end
        @test norm(Lm' * Lm - Diagonal(vec([mu <= kb[b] ? 1.0 : 0.0 for b in 1:dims[2], mu in 1:size(Ld, 3)]))) < 1e-12
        # L * R against the per-group dense truncated SVD (full rank: A itself)
        ref = zeros(size(Ad))
        for ((ss, bs), F, k) in zip(groups, Fs, kk)
          ref[ss, bs, :] .= reshape(F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :], length(ss), length(bs), :)
        end
        @test norm(LR(L, R) - ref) < 1e-12
        md == typemax(Int) && @test norm(LR(L, R) - Ad) < 1e-12
        sv = reduce(vcat, (F.S for F in Fs))
        @test spec.eigs ≈ sort(reduce(vcat, (abs2.(F.S[1:k]) for (F, k) in zip(Fs, kk))); rev = true)
        terr = sum(sum(abs2, F.S[(k + 1):end]; init = 0.0) for (F, k) in zip(Fs, kk)) / sum(abs2, sv)
        @test abs(spec.truncerr - terr) < 1e-14
      end
    end
    A2 = SB.NewBlockSparseSorted{Float64,3,1,2,Int}((2, 2, 2), 2, [(1, 1), (2, 2)], collect(1:2), randn(rng, 4))
    L, R, _ = SB.blocksparse_factorize(A2, [1]; maxdim = 4, cutoff = 0.0)   # label conserved: R diagonal
    @test L.keys == A2.keys && R.keys == [(1, 1), (2, 2)]
  end
end
