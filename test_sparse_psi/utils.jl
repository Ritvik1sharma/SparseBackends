using ITensors
using ITensorMPS
using SparseBackends

# Compare two MPO tensors that may have the same indices in different orders,
# and link indices with matching tags but different IDs.
# Matches indices by (tags, prime level) so "Link,l=1" pairs with "Link,l=1"
# across both MPOs even when their IDs differ.
# Returns (matches::Bool, max_abs_diff::Float64).
function compare_mpo_tensors(T1::ITensor, T2::ITensor; atol=1e-12)
  inds1 = collect(inds(T1))
  inds2 = collect(inds(T2))

  if length(inds1) != length(inds2)
    @warn "Tensor rank mismatch: $(length(inds1)) vs $(length(inds2))"
    return false, Inf
  end

  arr1 = SparseBackends.to_dense(T1)
  arr2 = SparseBackends.to_dense(T2)

  # Build permutation: perm[i] = j means output dim i comes from arr2's dim j.
  # Key = (tag string, prime level) uniquely identifies each index slot.
  index_key(idx) = (string(tags(idx)), plev(idx))

  perm = zeros(Int, length(inds1))
  used = falses(length(inds2))

  for (i, idx1) in enumerate(inds1)
    key1 = index_key(idx1)
    for (j, idx2) in enumerate(inds2)
      if !used[j] && index_key(idx2) == key1
        perm[i] = j
        used[j] = true
        break
      end
    end
  end

  if any(perm .== 0)
    @warn "Could not match all indices between tensors" inds1 inds2
    return false, Inf
  end

  arr2_perm = permutedims(arr2, perm)
  max_diff = maximum(abs, arr1 .- arr2_perm)
  result = isapprox(arr1, arr2_perm; atol=atol)
  return isapprox(arr1, arr2_perm; atol=atol), max_diff, (arr1, arr2_perm)
end



function clean!(op::MPO; tol=1e-12)
  for j in 1:length(op)
      T = op[j]
      A = array(T)  # convert to dense Julia array
      for i in eachindex(A)
          if abs(A[i]) < tol
              A[i] = 0.0
          elseif abs(A[i] - 1.0) < tol
              A[i] = 1.0
          elseif abs(A[i] + 1.0) < tol
              A[i] = -1.0
          elseif abs(A[i] - 0.5) < tol
              A[i] = 0.5
          elseif abs(A[i] + 0.5) < tol
              A[i] = -0.5
          elseif abs(A[i] - 1.0im) < tol
              A[i] = 1.0im
          elseif abs(A[i] + 1.0im) < tol
              A[i] = -1.0im
          elseif abs(A[i] - 0.5im) < tol
              A[i] = 0.5im
          elseif abs(A[i] + 0.5im) < tol
              A[i] = -0.5im
          elseif abs(A[i] + 2.0) < tol
              A[i] = -2.0
          elseif abs(A[i] - 2.0) < tol
              A[i] = 2.0
          end
      end
      op[j] = ITensor(A, inds(T)...)  # rebuild tensor with same indices
  end
  return op
end


function mpo_memory_bytes(H::MPO)
  total_bytes = 0
  for (i, W) in enumerate(H)
      bytes_i = Base.summarysize(W)
      total_bytes += bytes_i
  end
  println("Total MPO memory: $(total_bytes/1e6) MB")
  return total_bytes
end

function mps_memory_bytes(H::MPS)
  total_bytes = 0
  for (i, W) in enumerate(H)
      bytes_i = Base.summarysize(W)
      total_bytes += bytes_i
  end
  println("Total MPO memory: $(total_bytes/1e6) MB")
  return total_bytes
end



_mpsnorm(x::MPS) = norm(x)  # sqrt(real(inner(x', x)))

# Estimate operator norm ||P|| by power iteration on MPS space
function opnorm_estimate(P::MPO, sites::Vector{Index{Int64}}; iters::Int=8, linkdim::Int=2)
  s = sites
  v = randomMPS(s; linkdims=linkdim)
  for _ in 1:iters
    w = P * v
    n = _mpsnorm(w)
    n == 0 && return 0.0
    v = w / n
  end
  # Rayleigh-like estimate of ||P||
  return _mpsnorm(P * v) / _mpsnorm(v)
end


function isometric_check(P::MPO, sites::Vector{Index{Int64}}; trials::Int=2, power_iters::Int=6, tol::Float64=1e-10, linkdim::Int=2)
  s = sites
  dev_max = 0.0

  # Power iteration to estimate ||B|| where B := P'P - I
  for _ in 1:trials
    v = randomMPS(s; linkdims=linkdim)
    v = ITensorMPS.orthogonalize(v, 1)

    for _ in 1:power_iters
      w = dag(P) * (P * v) - v   # B*v
      n = _mpsnorm(w)
      n == 0 && break
      v = w / n
    end

    w = dag(P) * (P * v) - v
    dev = _mpsnorm(w) / max(_mpsnorm(v), 1e-300)
    dev_max = max(dev_max, dev)
  end

  pop = opnorm_estimate(P, sites; iters=power_iters, linkdim)

  if dev_max < tol
    println("✅ P is (approximately) isometric: ||P'P - I|| ≈ $dev_max  ≤ tol=$tol")
    println("   Approx operator norm ||P|| ≈ $pop")
    return true, dev_max, pop
  else
    println("❌ P is NOT isometric: ||P'P - I|| ≈ $dev_max  (tol=$tol)")
    println("   Approx operator norm ||P|| ≈ $pop")
    return false, dev_max, pop
  end
end

function _maxabs(T::ITensor)
  is = inds(T)
  if isempty(is)
      return abs(scalar(T))
  else
      return maximum(abs, Array(T, is...))
  end
end

maxabs(H::MPO) = maximum(_maxabs, H)

function normalize_by_max(H::MPO)
  m = maxabs(H)
  println("max value of the MPO is ", m)
  @assert m > 0 "All entries are zero; cannot normalize."
  return (1/m) * H
end
