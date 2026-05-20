# --- small helpers ---

@inline _prefix_len(::Val{N}, ::Val{N2}) where {N,N2} = N - N2

@inline function _full_index(prefix::NTuple{P,Int}, suffix::NTuple{N2,Int}, ::Val{N}) where {P,N2,N}
  # full[k] = prefix[k] for k<=P else suffix[k-P]
  return ntuple(k -> (k <= P ? prefix[k] : suffix[k-P]), Val(N))
end

# --- from_dense ---

"""
    blocksparse_from_dense(A, ::Val{N2}; atol=0, rtol=0, dropzeros=true)

Convert a dense N-D array `A` into `NewBlockSparseSorted{T,N,N2}` where the first P=N-N2
dimensions are block-sparse keys and the last N2 dimensions are stored as dense blocks.

A block is kept if its max(abs(.)) > tol, where tol = max(atol, rtol*global_maxabs).
Set `dropzeros=false` to keep all blocks.
"""
function blocksparse_from_dense(A::AbstractArray{T,N}, ::Val{N2};
                                atol::Real = 0,
                                rtol::Real = 0,
                                dropzeros::Bool = true) where {T,N,N2}

  @assert 0 ≤ N2 ≤ N
  dims = size(A)::NTuple{N,Int}
  prefix_dims = ntuple(i -> dims[i], Val(N-N2))
  suffix_dims = ntuple(i -> dims[N-N2+i], Val(N2))
  blksize = prod(suffix_dims)

  # Compute global max for rtol thresholding (only if needed)
  global_max = (rtol == 0) ? zero(real(T)) : maximum(abs, A)
  tol = max(atol, rtol * global_max)

  keys = Vector{NTuple{N-N2,Int}}()
  ids  = Int[]
  data = Vector{T}()
  sizehint!(data, blksize) # small default; caller can resize later if desired

  suffix_CI = CartesianIndices(suffix_dims)
  suffix_LI = LinearIndices(suffix_dims)

  # Reusable buffer for one block
  buf = Vector{T}(undef, blksize)

  # Iterate prefixes in lexicographic order -> keys are naturally sorted
  for pCI in CartesianIndices(prefix_dims)
    # println("pCI: ", pCI)
    prefix = Tuple(pCI)::NTuple{N-N2,Int}

    # Fill buf and compute block max in one pass
    block_max = zero(real(T))
    @inbounds for sCI in suffix_CI
      suffix = Tuple(sCI)::NTuple{N2,Int}
      full = _full_index(prefix, suffix, Val(N))
      v = A[full...]
      lin = suffix_LI[sCI]
      buf[lin] = v
      av = abs(v)
      if av > block_max
        block_max = av
      end
    end

    keep = !dropzeros || (block_max > tol)
    if keep
      push!(keys, prefix)
      push!(ids, length(keys))
      append!(data, buf)  # copies buf into the backing store
    end
  end

  return NewBlockSparseSorted{T,N,N2,N-N2,Int}(dims, blksize, keys, ids, data)
end

# --- to_dense ---
"""
    to_dense(B; init=zero(eltype(B)))

Materialize a dense Array with shape `B.dims`, filling missing blocks with `init`.
"""
function to_dense(B::NewBlockSparseSorted{T,N,N2,P,K};
                  init::T = zero(T),
                  merged_axes::Union{Nothing,Vector{Vector{Int}}} = nothing
                 ) where {T,N,N2,P,K}

  dims = B.dims
  suffix_dims = ntuple(i -> dims[P+i], Val(N2))

  # ---- build merge plan (gaps allowed), keep axis = min, delete axis = max ----
  plan = Tuple{Int,Int}[]  # (keep, other) in evolving axis numbering
  out_dims = dims

  if merged_axes !== nothing
    pairs0 = Vector{Tuple{Int,Int}}(undef, length(merged_axes))
    used = falses(N)

    @inbounds for k in eachindex(merged_axes)
      g = merged_axes[k]
      length(g) == 2 || error("Each merge group must have length 2, got $g")
      a = g[1]; b = g[2]
      1 ≤ a ≤ N || error("Axis out of range in $g (N=$N)")
      1 ≤ b ≤ N || error("Axis out of range in $g (N=$N)")
      a != b || error("Duplicate axes in merge group $g")

      keep  = ifelse(a < b, a, b)
      other = ifelse(a < b, b, a)

      (used[keep] || used[other]) && error("Axis appears in multiple merge groups: $g")
      used[keep] = true
      used[other] = true

      pairs0[k] = (keep, other)
    end

    # delete higher axes first so renumbering is stable
    sort!(pairs0; by = last, rev = true)

    dims_work = collect(dims)
    sizehint!(plan, length(pairs0))
    @inbounds for (keep, other) in pairs0
      dims_work[keep] *= dims_work[other]
      deleteat!(dims_work, other)
      push!(plan, (keep, other))
    end
    out_dims = Tuple(dims_work)
  end

  out = Array{T}(undef, out_dims)
  fill!(out, init)

  suffix_CI = CartesianIndices(suffix_dims)
  suffix_LI = LinearIndices(suffix_dims)

  blksize = B.blksize
  @assert blksize == prod(suffix_dims)
  @assert length(B.data) == length(B.keys) * blksize

  # Map oldfull (NTuple{N}) -> newfull (NTuple{N-k}) by applying plan.
  @inline function map_full_index(oldfull::NTuple{N,Int})
    merged_axes === nothing && return oldfull
    # use fixed-size buffers for speed (no allocs)
    idx = Vector{Int}(undef, N)
    dimw = Vector{Int}(undef, N)
    @inbounds for i in 1:N
      idx[i] = oldfull[i]
      dimw[i] = dims[i]
    end
    len = N

    @inbounds for (keep, other) in plan
      # fused coordinate: keep is minor (fast), other is major (slow)
      fused = idx[keep] + (idx[other] - 1) * dimw[keep]
      idx[keep] = fused
      # update dimw[keep] before shifting
      dimw[keep] *= dimw[other]
      # delete position `other` by shifting left
      for j in other:(len-1)
        idx[j] = idx[j+1]
        dimw[j] = dimw[j+1]
      end
      len -= 1
    end

    # build return tuple without allocating a new Vector
    if len == N
      return oldfull
    else
      return ntuple(i -> idx[i], Val(N - length(plan)))
    end
  end

  @inbounds for i in eachindex(B.keys)
    prefix = B.keys[i]
    bid    = B.ids[i]
    base   = (bid - 1) * blksize
    for sCI in suffix_CI
      suffix  = Tuple(sCI)::NTuple{N2,Int}
      oldfull = _full_index(prefix, suffix, Val(N))  # NTuple{N,Int}
      lin = suffix_LI[sCI]
      newfull = map_full_index(oldfull)
      out[newfull...] = B.data[base + lin]
    end
  end
  return out
end

"""
    to_dense!(out, B)

In-place densification into a pre-allocated buffer `out::Array{T,N}` of shape `B.dims`.
Equivalent to `out .= to_dense(B)` but allocates nothing. Only supports the
no-`merged_axes` case (the only one needed by recast_bs_to_template).
"""
function to_dense!(out::AbstractArray{T,N}, B::NewBlockSparseSorted{T,N,N2,P,K};
                   init::T = zero(T)) where {T,N,N2,P,K}
  size(out) == B.dims || error("to_dense! buffer size $(size(out)) ≠ B.dims $(B.dims)")
  fill!(out, init)
  blksize = B.blksize
  suffix_dims = ntuple(i -> B.dims[P+i], Val(N2))
  suffix_CI = CartesianIndices(suffix_dims)
  suffix_LI = LinearIndices(suffix_dims)
  @inbounds for i in eachindex(B.keys)
    prefix = B.keys[i]
    bid    = B.ids[i]
    base   = (bid - 1) * blksize
    for sCI in suffix_CI
      suffix = Tuple(sCI)::NTuple{N2,Int}
      full = _full_index(prefix, suffix, Val(N))
      lin = suffix_LI[sCI]
      out[full...] = B.data[base + lin]
    end
  end
  return out
end

# function to_dense(B::NewBlockSparseSorted{T,N,N2,P,K};
#                   init::T = zero(T),
#                   merged_axes::Union{Nothing,Vector{Int}} = nothing) where {T,N,N2,P}
#   dims = B.dims
#   suffix_dims = ntuple(i -> dims[P+i], Val(N2))

#   out = Array{T}(undef, dims)
#   fill!(out, init)
#   suffix_CI = CartesianIndices(suffix_dims)
#   suffix_LI = LinearIndices(suffix_dims)

#   blksize = B.blksize
#   @assert blksize == prod(suffix_dims)
#   @assert length(B.data) == length(B.keys) * blksize

#   @inbounds for i in eachindex(B.keys)
#     prefix = B.keys[i]
#     bid = B.ids[i]
#     base = (bid - 1) * blksize
#     for sCI in suffix_CI
#       suffix = Tuple(sCI)::NTuple{N2,Int}
#       full = _full_index(prefix, suffix, Val(N))
#       lin = suffix_LI[sCI]
#       out[full...] = B.data[base + lin]
#     end
#   end
#   return out
# end


@inline function _move_to_after!(v::Vector{Int}, from::Int, after::Int)
  # Move element at `from` to position `after+1`, stable.
  from == after + 1 && return v
  x = v[from]
  deleteat!(v, from)
  insert!(v, after + 1, x)
  return v
end

function fuse_sparse_axes(
  A::NewBlockSparseSorted{T,N,N2,P,K},
  ax1::Int, ax2::Int
) where {T,N,N2,P,K}
  (1 <= ax1 <= P) || throw(ArgumentError("ax1 must be in sparse head 1:$P"))
  (1 <= ax2 <= P) || throw(ArgumentError("ax2 must be in sparse head 1:$P"))
  ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
  a1, a2 = min(ax1, ax2), max(ax1, ax2)
  dims = A.dims
  d1, d2 = dims[a1], dims[a2]
  # New full dims (drop axis a2, fold into a1)
  newdims = ntuple(i -> begin
    if i < a2
      i == a1 ? d1 * d2 : dims[i]
    else
      dims[i + 1]
    end
  end, Val(N - 1))
  nblocks = length(A.keys)
  newkeys = Vector{NTuple{P-1,K}}(undef, nblocks)
  @inbounds for i in 1:nblocks
    k = A.keys[i]
    fused = (k[a2] - 1) * d1 + k[a1]  # 1..(d1*d2)
    # print(" starting and ending key value ", k, " fused to ", fused)
    newkeys[i] = ntuple(j -> begin
      if j < a2
        j == a1 ? fused : k[j]
      else
        k[j + 1]
      end
    end, Val(P - 1))
    # println(" new key is ", newkeys[i])
  end
  # Reuse ids/data; blksize unchanged; prefix length P -> P-1
  return NewBlockSparseSorted{T,N-1,N2,P-1,K}(newdims, A.blksize, newkeys, A.ids, A.data)
end

function fuse_dense_axes!(
  A::NewBlockSparseSorted{T,N,N2,P,K},
  ax1::Int, ax2::Int
) where {T,N,N2,P,K}
  (P + 1 <= ax1 <= N) || throw(ArgumentError("ax1 must be in dense tail $(P+1):$N"))
  (P + 1 <= ax2 <= N) || throw(ArgumentError("ax2 must be in dense tail $(P+1):$N"))
  ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
  a1, a2 = min(ax1, ax2), max(ax1, ax2)
  # Dense-tail positions in 1..N2
  p1 = a1 - P
  p2 = a2 - P
  p1, p2 = min(p1, p2), max(p1, p2)
  # If not adjacent, permute dense-tail so p2 moves to p1+1 (no cross boundary).
  if p2 != p1 + 1
    perm = collect(1:N)                # global axes permutation
    perm_block = collect(1:N2)         # dense-tail permutation 1..N2
    _move_to_after!(perm_block, p2, p1)
    @inbounds for i in 1:N2
      perm[P + i] = P + perm_block[i]
    end
    permutedims!(A, perm)              # uses your implementation
    a2 = a1 + 1                        # now adjacent in global axes
  end
  dims = A.dims
  newdims = ntuple(i -> begin
    if i < a2
      i == a1 ? dims[a1] * dims[a2] : dims[i]
    else
      dims[i + 1]
    end
  end, Val(N - 1))
  return NewBlockSparseSorted{T,N-1,N2-1,P,K}(newdims, A.blksize, A.keys, A.ids, A.data)
end

function fuse_two_axes!(
  A::NewBlockSparseSorted{T,N,N2,P,K},
  ax1::Int, ax2::Int
) where {T,N,N2,P,K}
  (1 <= ax1 <= N && 1 <= ax2 <= N) || throw(ArgumentError("axes must be in 1:$N"))
  ax1 == ax2 && throw(ArgumentError("axes must be distinct"))
  in_sparse1 = ax1 <= P
  in_sparse2 = ax2 <= P
  if in_sparse1 && in_sparse2
    return fuse_sparse_axes(A, ax1, ax2)
  elseif (!in_sparse1) && (!in_sparse2)
    return fuse_dense_axes!(A, ax1, ax2)
  else
    throw(ArgumentError("Cannot fuse across sparse-head (1:$P) and dense-tail ($(P+1):$N) axes"))
  end
end

# ─────────────────────────────────────────────────────────────────────────────
# rekey — convert prefix key integer type
# ─────────────────────────────────────────────────────────────────────────────

"""
    rekey(A::NewBlockSparseSorted{T,N,N2,P,K}, ::Type{K2}; check=true)
        → NewBlockSparseSorted{T,N,N2,P,K2}

Convert the prefix key integer type.  With `check=true` (default), errors if
any dimension size or existing key coordinate does not fit in K2.
"""
function rekey(A::NewBlockSparseSorted{T,N,N2,P,K}, ::Type{K2};
               check::Bool=true) where {T,N,N2,P,K,K2<:Integer}
    if check
        for i in 1:P
            (1 <= typemin(K2) || true) && A.dims[i] <= typemax(K2) ||
                error("Dimension $(A.dims[i]) at prefix axis $i cannot be " *
                      "indexed by $K2 (max=$(typemax(K2)))")
        end
        for key in A.keys, v in key
            (typemin(K2) <= v <= typemax(K2)) ||
                error("Key coordinate $v is out of range for $K2 " *
                      "($(typemin(K2))..$(typemax(K2)))")
        end
    end
    new_keys = [ntuple(j -> K2(key[j]), Val(P)) for key in A.keys]
    return NewBlockSparseSorted{T,N,N2,P,K2}(
        A.dims, A.blksize, new_keys, copy(A.ids), copy(A.data))
end

# # Dense conversion: storage -> Array
# function to_dense(A::RestrictedBlockSparse{T,N}) where {T,N}
#   out = zeros(T, A.dims)
#   P = N - 2
#   nps = _n_prefix_states(A.dims, A.reps)
#   rows, cols = A.dims[N-1], A.dims[N]

#   if P == 0
#     @inbounds for c in 1:cols, r in 1:rows
#         idx = (c-1)*rows + r
#         out[r, c] = A.data[idx]
#     end
#     return out
#   end

#   rep_dims = [A.dims[r] for r in A.reps]
#   rep_pos  = _rep_pos(A.reps, P)
#   prefix   = Vector{Int}(undef, P)
#   rep_vals = Vector{Int}(undef, length(rep_dims))

#   @inbounds for plin in 1:nps
#     if !isempty(rep_dims)
#       _rep_decode!(rep_vals, plin, rep_dims)
#       _prefix_from_rep!(prefix, A.rep_of, rep_pos, rep_vals)
#     else
#       fill!(prefix, 1)
#     end

#     off = (plin - 1) * A.blksize
#     for c in 1:cols, r in 1:rows
#       idx_in_block = (c - 1) * rows + r
#       out[CartesianIndex(prefix..., r, c)] = A.data[off + idx_in_block]
#     end
#   end
#   return out
# end

# # Array -> RestrictedCOO
# function from_dense_blocksparse(A::AbstractArray{T,N};
#                                 diag_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
#                                 atol::Real = 1e-12,
#                                 rtol::Real = 0) where {T,N}
#   @assert N >= 2
#   dims = ntuple(i -> size(A, i), Val(N))
#   @assert all(dims .>= 1)

#   P = N - 2
#   reps, rep_of = _build_rep_map(dims, diag_pairs)
#   nps = _n_prefix_states(dims, reps)

#   rows = dims[N-1]
#   cols = dims[N]
#   blksize = rows * cols

#   data = Vector{T}(undef, nps * blksize)

#   # Fast path: no prefix axes
#   if P == 0
#     @inbounds for c in 1:cols, r in 1:rows
#       idx = (c - 1) * rows + r
#       data[idx] = A[r, c]
#     end
#     return blocksparse{T,N}(dims, copy(diag_pairs), reps, rep_of, blksize, data)
#   end

#   rep_dims = [dims[r] for r in reps]
#   rep_pos  = _rep_pos(reps, P)
#   prefix   = Vector{Int}(undef, P)
#   rep_vals = Vector{Int}(undef, length(rep_dims))

#   @inbounds for plin in 1:nps
#     if !isempty(rep_dims)
#       _rep_decode!(rep_vals, plin, rep_dims)
#       _prefix_from_rep!(prefix, rep_of, rep_pos, rep_vals)
#     else
#       fill!(prefix, 1)
#     end

#     off = (plin - 1) * blksize
#     for c in 1:cols, r in 1:rows
#       idx_in_block = (c - 1) * rows + r
#       data[off + idx_in_block] = A[CartesianIndex(prefix..., r, c)]
#     end
#   end

#   return blocksparse{T,N}(dims, copy(diag_pairs), reps, rep_of, blksize, data)
# end
# function dense_itensor(A::blocksparse{T,N}, inds::Vararg{Index,N}) where {T,N}
#   return ITensor(Array(A), inds...)
# end

# function blocksparse_from_itensor(T::ITensor;
#                                   diag_pairs=Tuple{Int,Int}[],
#                                   atol=1e-12,
#                                   rtol=0.0)
#   indsT = inds(T)
#   A = Array(T, indsT...)
#   return blocksparse_from_dense(A; diag_pairs=diag_pairs, atol=atol, rtol=rtol)
# end

# Base.Array(A::blocksparse{T,N}) where {T,N} = to_dense(A)



# #   dims = ntuple(i -> size(A, i), Val(N))
# #   P    = N - 2
# #   rows = dims[N-1]
# #   cols = dims[N]

# #   reps, rep_of = _build_rep_map(dims, diag_pairs)
# #   nps = _n_prefix_states(dims, reps)

# #   data = [Vector{Tuple{Tuple{Int,Int},T}}() for _ in 1:nps]

# #   if P == 0
# #     @inbounds for r in 1:rows, c in 1:cols
# #       v = A[r,c]
# #       if abs(v) > atol
# #         push!(data[1], ((r,c), v))
# #       end
# #     end
# #     return RestrictedCOO{T}(dims, data; diag_pairs=diag_pairs)
# #   end

# #   rep_dims = [dims[r] for r in reps]
# #   rep_pos  = _rep_pos(reps, P)
# #   prefix   = Vector{Int}(undef, P)
# #   rep_vals = Vector{Int}(undef, length(rep_dims))

# #   @inbounds for plin in 1:nps
# #     if !isempty(rep_dims)
# #       _rep_decode!(rep_vals, plin, rep_dims)
# #       _prefix_from_rep!(prefix, rep_of, rep_pos, rep_vals)
# #     else
# #       fill!(prefix, 1)
# #     end

# #     for r in 1:rows, c in 1:cols
# #       v = A[CartesianIndex(prefix..., r, c)]
# #       if abs(v) > atol
# #         push!(data[plin], ((r,c), v))
# #       end
# #     end
# #   end

# #   return RestrictedCOO{T}(dims, data; diag_pairs=diag_pairs)
# # end
