using Base: OneTo

mutable struct NewBlockSparseSorted{T,N,N2,P,K<:Integer} <: SparseTensor{T,N}  # AbstractArray{T,N}
  dims::NTuple{N,Int}
  blksize::Int
  keys::Vector{NTuple{P,K}}   # P = N - N2, key type K
  ids::Vector{Int}
  data::Vector{T}
end

# Backwards-compatible outer constructor (K=Int default)
NewBlockSparseSorted{T,N,N2,P}(dims, blksize, keys::Vector{NTuple{P,Int}}, ids, data) where {T,N,N2,P} =
    NewBlockSparseSorted{T,N,N2,P,Int}(dims, blksize, keys, ids, data)

# Generalized _prefix_lin: accepts any Integer key type
@inline function _prefix_lin(k::NTuple{P,<:Integer}, pdims::NTuple{P,Int}) where {P}
  return LinearIndices(pdims)[CartesianIndex(map(Int, k))]
end

Base.eltype(::Type{NewBlockSparseSorted{T,N,N2,P,K}}) where {T,N,N2,P,K} = T
Base.eltype(A::NewBlockSparseSorted) = eltype(typeof(A))
Base.size(A::NewBlockSparseSorted{T,N,N2,P,K}) where {T,N,N2,P,K} = A.dims
Base.axes(A::NewBlockSparseSorted{T,N,N2,P,K}) where {T,N,N2,P,K} = ntuple(i -> OneTo(A.dims[i]), Val(N))
Base.IndexStyle(::Type{<:NewBlockSparseSorted}) = IndexCartesian()

@inline function _check_indexes!(dims::NTuple{N,Int}, I::NTuple{N,Int}) where {N}
  @inbounds for d in 1:N
    1 <= I[d] <= dims[d] || throw(BoundsError())
  end
  return nothing
end

@inline function _prefix_key(I::NTuple{N,Int}, ::Val{P}) where {N,P}
  ntuple(i -> I[i], Val(P))
end

@inline function _block_index(I::NTuple{N,Int}, dims::NTuple{N,Int}, ::Val{P}, ::Val{N2}) where {N,P,N2}
  lin = 1
  stride = 1
  @inbounds for j in 1:N2
    d = P + j
    lin += (I[d] - 1) * stride
    stride *= dims[d]
  end
  return lin
end

# Do a binary search for the prefix key (K-parameterized)
@inline function _find_key(A::NewBlockSparseSorted{T,N,N2,P,K}, k::NTuple{P,K}) where {T,N,N2,P,K}
  pdims  = ntuple(i -> A.dims[i], Val(P))
  target = _prefix_lin(k, pdims)
  lo = 1
  hi = length(A.keys) + 1
  @inbounds while lo < hi
    mid = (lo + hi) >>> 1
    midv = _prefix_lin(A.keys[mid], pdims)
    if midv < target
      lo = mid + 1
    else
      hi = mid
    end
  end
  found = (lo <= length(A.keys) && A.keys[lo] == k)
  return lo, found
end

@inline function _alloc_block!(A::NewBlockSparseSorted{T,N,N2,P,K}) where {T,N,N2,P,K}
  id = length(A.data) ÷ A.blksize + 1
  append!(A.data, fill(zero(T), A.blksize))
  return id
end

@inline function _ensure_block!(A::NewBlockSparseSorted{T,N,N2,P,K}, k::NTuple{P,K}) where {T,N,N2,P,K}
  i, found = _find_key(A, k)
  if found
    return A.ids[i]
  else
    id = _alloc_block!(A)
    insert!(A.keys, i, k)
    insert!(A.ids,  i, id)
    return id
  end
end

# Constructors (K=Int default)
function NewBlockSparseSorted{T,N,N2}(dims::NTuple{N,Int}) where {T,N,N2}
  @assert (N == 0 && N2 == 0) || (0 < N2 <= N)  # allow scalar (N=N2=0) as special case
  @assert all(dims .>= 1)
  P = N - N2
  blksize = prod(ntuple(i -> dims[P+i], Val(N2)))
  keys = Vector{NTuple{P,Int}}()
  return NewBlockSparseSorted{T,N,N2,P,Int}(dims, blksize, keys, Int[], T[])
end

# Iteration in sorted order is trivial now:
keys_sorted(A::NewBlockSparseSorted) = A.keys
blocks_sorted(A::NewBlockSparseSorted) = ((A.keys[i], A.ids[i]) for i in eachindex(A.keys))

# getindex: missing block => zero
function Base.getindex(A::NewBlockSparseSorted{T,N,N2,P,K}, I::Vararg{Int,N}) where {T,N,N2,P,K}
  It = ntuple(i -> I[i], Val(N))
  _check_indexes!(A.dims, It)
  k = ntuple(i -> K(It[i]), Val(P))   # convert to key type K
  i, found = _find_key(A, k)
  found || return zero(T)

  id = A.ids[i]
  bi = _block_index(It, A.dims, Val(P), Val(N2))
  off = (id - 1) * A.blksize
  return @inbounds A.data[off + bi]
end

# setindex!: creates block on demand
function Base.setindex!(A::NewBlockSparseSorted{T,N,N2,P,K}, v, I::Vararg{Int,N}) where {T,N,N2,P,K}
  It = ntuple(i -> I[i], Val(N))
  _check_indexes!(A.dims, It)
  k = ntuple(i -> K(It[i]), Val(P))   # convert to key type K
  id = _ensure_block!(A, k)

  bi = _block_index(It, A.dims, Val(P), Val(N2))
  off = (id - 1) * A.blksize
  @inbounds A.data[off + bi] = convert(T, v)
  return v
end

@inline function _check_no_cross_perm(perm::AbstractVector{Int}, N::Int, P::Int, N2::Int)
  @assert length(perm) == N
  @assert sort(collect(perm)) == collect(1:N) "perm must be a permutation of 1:N"
  @assert all(p -> p <= P, perm[1:P]) "No-cross reorder violated: output prefix axis uses old block axis"
  @assert all(p -> p >  P, perm[P+1:end]) "No-cross reorder violated: output block axis uses old prefix axis"
  return nothing
end

@inline function _map_prefix_key(oldk::NTuple{P,<:Integer}, perm_prefix::AbstractVector{Int}) where {P}
  return ntuple(j -> oldk[perm_prefix[j]], Val(P))
end

@inline function _block_view(A::NewBlockSparseSorted{T,N,N2,P,K}, id::Int) where {T,N,N2,P,K}
  off = (id - 1) * A.blksize
  return @view(A.data[off+1 : off + A.blksize])
end

# function Base.permutedims(A::NewBlockSparseSorted{T,N,N2,P}, perm::AbstractVector{Int}) where {T,N,N2,P}
#   _check_no_cross_perm(perm, N, P, N2)

#   dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
#   perm_prefix = perm[1:P]
#   perm_block_global = perm[P+1:end]
#   perm_block = [perm_block_global[i] - P for i in 1:N2]
#   is_identity_block = all(perm_block[i] == i for i in 1:N2)
#   blockDimsA = ntuple(i -> A.dims[P+i], Val(N2))
#   blockDimsB = ntuple(i -> dimsB[P+i], Val(N2))
#   blksizeB   = prod(blockDimsB)
#   nblocks = length(A.keys)
#   @assert length(A.ids) == nblocks
#   @assert length(A.data) == nblocks * A.blksize
#   # Compute permuted keys
#   newkeys_tmp = Vector{NTuple{P,Int}}(undef, nblocks)
#   @inbounds for i in 1:nblocks
#     newkeys_tmp[i] = _map_prefix_key(A.keys[i], perm_prefix)
#   end
#   # Sort blocks by new key in column major order
#   prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
#   p = sortperm(newkeys_tmp; by = k -> _prefix_lin(k, prefix_dimsB))
#   keysB = newkeys_tmp[p]
#   idsB  = collect(1:nblocks)
#   dataB = Vector{T}(undef, nblocks * blksizeB)

#   @inbounds for outi in 1:nblocks
#     src_i = p[outi]
#     id    = A.ids[src_i]
#     v     = _block_view(A, id)  # length A.blksize
#     offB = (outi - 1) * blksizeB
#     if is_identity_block
#       # Just copy block payload as-is
#       @assert length(v) == blksizeB
#       copyto!(dataB, offB + 1, v, 1, blksizeB)
#     else
#       blkA = reshape(v, blockDimsA...)
#       blkB = permutedims(blkA, perm_block)  # allocates one Array per block (unavoidable unless you write a strided permute)
#       copyto!(dataB, offB + 1, vec(blkB), 1, blksizeB)
#     end
#   end
#   return NewBlockSparseSorted{T,N,N2,P}(dimsB, blksizeB, keysB, idsB, dataB)
# end


function Base.permutedims(A::NewBlockSparseSorted{T,N,N2,P,K},
                          perm::AbstractVector{Int}) where {T,N,N2,P,K}
  _check_no_cross_perm(perm, N, P, N2)
  dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
  perm_prefix = perm[1:P]
  perm_block  = ntuple(i -> perm[P+i] - P, Val(N2))  # 1..N2
  blockDimsA = ntuple(i -> A.dims[P+i], Val(N2))
  blockDimsB = ntuple(i -> dimsB[P+i], Val(N2))
  blksizeB   = prod(blockDimsB)
  nblocks = length(A.keys)
  @assert length(A.ids)  == nblocks
  @assert length(A.data) == nblocks * A.blksize
  # Compute permuted prefix keys (unsorted)
  newkeys_tmp = Vector{NTuple{P,K}}(undef, nblocks)
  @inbounds for i in 1:nblocks
    newkeys_tmp[i] = _map_prefix_key(A.keys[i], perm_prefix)
  end
  # Sort blocks by new key in column-major order
  prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
  p = sortperm(newkeys_tmp; by = k -> _prefix_lin(k, prefix_dimsB))
  keysB = newkeys_tmp[p]
  idsB  = collect(1:nblocks)  # packed
  dataB = Vector{T}(undef, nblocks * blksizeB)
  # Fast path: if block perm is identity and blksizes match, just copy
  is_identity_block = @inbounds all(perm_block[i] == i for i in 1:N2)
  same_blksize = (A.blksize == blksizeB)
  # Precompute dense-tail linear map if needed (allocation-free per block)
  map_old = nothing
  scratch = nothing
  if !(is_identity_block && same_blksize)
    @assert prod(blockDimsA) == A.blksize
    @assert prod(blockDimsB) == blksizeB
    # Build mapping: new_lin -> old_lin for dense tail
    invp = invperm(collect(perm_block))  # tiny; OK once
    strides_old = Vector{Int}(undef, N2)
    s = 1
    @inbounds for d in 1:N2
      strides_old[d] = s
      s *= blockDimsA[d]
    end
    map_old = Vector{Int}(undef, blksizeB)
    idx_new = ones(Int, N2)
    @inbounds for new_lin in 1:blksizeB
      old_lin = 1
      for oldd in 1:N2
        old_lin += (idx_new[invp[oldd]] - 1) * strides_old[oldd]
      end
      map_old[new_lin] = old_lin
      for d in 1:N2
        idx_new[d] += 1
        if idx_new[d] <= blockDimsB[d]
          break
        else
          idx_new[d] = 1
        end
      end
    end
    scratch = Vector{T}(undef, blksizeB)
  end
  @inbounds for outi in 1:nblocks
    src_i = p[outi]
    id    = A.ids[src_i]          # may be non-packed in A
    v     = _block_view(A, id)    # length A.blksize
    offB = (outi - 1) * blksizeB
    if is_identity_block && same_blksize
      copyto!(dataB, offB + 1, v, 1, blksizeB)
    else
      # reorder dense tail into scratch using map_old then copy
      @assert map_old !== nothing && scratch !== nothing
      @inbounds for newi in 1:blksizeB
        scratch[newi] = v[map_old[newi]]
      end
      copyto!(dataB, offB + 1, scratch, 1, blksizeB)
    end
  end
  return NewBlockSparseSorted{T,N,N2,P,K}(dimsB, blksizeB, keysB, idsB, dataB)
end

# Convenience: accept NTuple perm as well
function Base.permutedims(A::NewBlockSparseSorted{T,N,N2,P,K}, perm::NTuple{N,Int}) where {T,N,N2,P,K}
  return permutedims(A, collect(perm))
end

function permutedims!(A::NewBlockSparseSorted{T,N,N2,P,K},
                      perm::AbstractVector{Int}) where {T,N,N2,P,K}
  _check_no_cross_perm(perm, N, P, N2)

  nblocks = length(A.keys)
  @assert length(A.ids)  == nblocks
  @assert length(A.data) == nblocks * A.blksize

  # IMPORTANT invariant: your block payload is stored packed by block-slot,
  # and _block_view(A, id) assumes id is that packed slot.
  @assert all(@inbounds A.ids[i] == i for i in 1:nblocks) "permutedims! requires packed ids (ids[i]==i)"

  # New dims
  dimsB = ntuple(j -> A.dims[perm[j]], Val(N))

  perm_prefix = perm[1:P]
  perm_block  = ntuple(i -> perm[P+i] - P, Val(N2))  # 1:N2

  prefix_id = @inbounds all(perm_prefix[i] == i for i in 1:P)
  block_id  = @inbounds all(perm_block[i]  == i for i in 1:N2)

  # -----------------------------
  # 1) Dense-tail-only permute (in place, no keys/sort)
  # -----------------------------
  if !block_id
    oldDims = ntuple(i -> A.dims[P+i], Val(N2))
    newDims = ntuple(i -> oldDims[perm_block[i]], Val(N2))
    @assert prod(oldDims) == A.blksize
    @assert prod(newDims) == A.blksize
    # invp[oldd] = position in the *new* index vector whose value maps to oldd
    invp = invperm(collect(perm_block))  # small; ok
    # Column-major strides for old layout
    strides_old = Vector{Int}(undef, N2)
    s = 1
    @inbounds for d in 1:N2
      strides_old[d] = s
      s *= oldDims[d]
    end
    # map_old[new_lin] = old_lin
    map_old = Vector{Int}(undef, A.blksize)
    idx_new = ones(Int, N2)  # multi-index in NEW dims, column-major
    @inbounds for new_lin in 1:A.blksize
      old_lin = 1
      for oldd in 1:N2
        old_lin += (idx_new[invp[oldd]] - 1) * strides_old[oldd]
      end
      map_old[new_lin] = old_lin
      # increment idx_new in column-major for newDims
      for d in 1:N2
        idx_new[d] += 1
        if idx_new[d] <= newDims[d]
          break
        else
          idx_new[d] = 1
        end
      end
    end
    scratch = Vector{T}(undef, A.blksize)
    @inbounds for bid in 1:nblocks
      off = (bid - 1) * A.blksize
      for newi in 1:A.blksize
        scratch[newi] = A.data[off + map_old[newi]]
      end
      copyto!(A.data, off + 1, scratch, 1, A.blksize)
    end
  end
  # If prefix unchanged, dense-tail-only case is done
  if prefix_id
    A.dims = dimsB
    return A
  end
  # -----------------------------
  # 2) Prefix permute + sort + in-place block reorder
  # -----------------------------
  newkeys = Vector{NTuple{P,K}}(undef, nblocks)
  @inbounds for i in 1:nblocks
    k = A.keys[i]
    newkeys[i] = ntuple(j -> k[perm_prefix[j]], Val(P))
  end
  prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
  # Column-major linearization for prefix key (1-based)
  @inline function lin(k::NTuple{P,Int})
    stride = 1
    idx = 1
    @inbounds for d in 1:P
      idx += (k[d] - 1) * stride
      stride *= prefix_dimsB[d]
    end
    return idx
  end

  p = sortperm(newkeys; by = lin)   # p[newpos] = oldpos
  q = invperm(p)                   # q[oldpos] = newpos
  block_scratch = Vector{T}(undef, A.blksize)

  @inline function swap_block!(i::Int, j::Int)
    i == j && return
    offi = (i - 1) * A.blksize
    offj = (j - 1) * A.blksize
    copyto!(block_scratch, 1, A.data, offi + 1, A.blksize)
    copyto!(A.data, offi + 1, A.data, offj + 1, A.blksize)
    copyto!(A.data, offj + 1, block_scratch, 1, A.blksize)
    return
  end
  # Start from "newkeys in old order", then permute both keys and blocks into sorted order.
  A.keys = newkeys
  @inbounds for i in 1:nblocks
    while q[i] != i
      j = q[i]
      A.keys[i], A.keys[j] = A.keys[j], A.keys[i]
      swap_block!(i, j)
      q[i], q[j] = q[j], q[i]
    end
  end
  # Normalize ids to packed order (required by _block_view)
  resize!(A.ids, nblocks)
  @inbounds for i in 1:nblocks
    A.ids[i] = i
  end
  A.dims = dimsB
  return A
end

# NTuple perm convenience
function permutedims!(A::NewBlockSparseSorted{T,N,N2,P,K},
                      perm::NTuple{N,Int}) where {T,N,N2,P,K}
  return permutedims!(A, collect(perm))
end

import Base: sort!


function sort!(C::NewBlockSparseSorted)
  # If you guarantee unique keys via _ensure_block!, you only need sorting:
  perm = sortperm(C.keys)  # relies on lexicographic ordering of NTuples
  C.keys = C.keys[perm]
  C.ids  = C.ids[perm]
  # C.data stays as-is because ids refer into C.data
  return C
end


function is_dense(A::NewBlockSparseSorted{T,N,N2,P,K}) where {T,N,N2,P,K}
  if P === 0
    @assert length(A.keys) <= 1 "P=0 means no prefix, so at most one block (the whole tensor) is allowed"
    @assert A.keys == [] || A.keys[1] == () "If there is a block, its key must be the empty tuple"
    return true
  else
    # println("Checking density: keys = ", length(A.keys), ", expected for dense: ", prod(ntuple(i -> A.dims[i], Val(P))))
    sparse_dims_prod = prod(ntuple(i -> A.dims[i], Val(P)))
    return length(A.keys) == sparse_dims_prod
    #   # If the number of blocks matches the number of prefix combinations, we just need to check that all keys are present.
    #   println("Number of blocks matches the number of prefix combinations. Checking keys...")
    #   println("Expected keys: ", Set(ntuple(i -> 1:A.dims[i], Val(P))))
    #   println("Actual keys: ", Set(A.keys))
    #   expected_keys = Set(ntuple(i -> 1:A.dims[i], Val(P)))
    #   actual_keys = Set(A.keys)
    #   println("result is ", expected_keys == actual_keys)
    #   return expected_keys == actual_keys
    # end
  end
  return false
end

# permutedims!(A::NewBlockSparseSorted{T,N,N2,P}, perm::NTuple{N,Int}) where {T,N,N2,P} =
#   permutedims!(A, collect(perm))

# --------------------------
# Helpers
# # --------------------------
# @inline function _swap_block!(data::Vector{T}, blksize::Int, i::Int, j::Int, scratch::Vector{T}) where {T}
#   i == j && return
#   offi = (i - 1) * blksize
#   offj = (j - 1) * blksize
#   @inbounds begin
#     copyto!(scratch, 1, data, offi + 1, blksize)
#     copyto!(data, offi + 1, data, offj + 1, blksize)
#     copyto!(data, offj + 1, scratch, 1, blksize)
#   end
#   return
# end

# # Precompute mapping: new_lin -> old_lin for permuting dense tail inside a block.
# function _dense_oldidx_for_newidx(
#     oldDims::NTuple{N2,Int},
#     perm_block::NTuple{N2,Int}
# ) where {N2}
#   # new axis j comes from old axis perm_block[j]
#   newDims = ntuple(j -> oldDims[perm_block[j]], Val(N2))
#   blksize = prod(newDims)
#   invp = invperm(collect(perm_block))  # old axis k comes from new axis invp[k]
#   map_old = Vector{Int}(undef, blksize)
#   # Column-major linear index:
#   # lin = 1 + Σ (idx[d]-1)*stride[d], stride[d]=Π_{t<d} dim[t]
#   strides_old = Vector{Int}(undef, N2)
#   s = 1
#   for d in 1:N2
#     strides_old[d] = s
#     s *= oldDims[d]
#   end
#   strides_new = Vector{Int}(undef, N2)
#   s = 1
#   for d in 1:N2
#     strides_new[d] = s
#     s *= newDims[d]
#   end
#   # Iterate over new indices (multi-index) in column-major order
#   # and compute corresponding old linear index
#   # (This is done once per perm; reused for every block.)
#   idx = ones(Int, N2)
#   @inbounds for new_lin in 1:blksize
#     # old_idx[k] = new_idx[invp[k]]
#     old_lin = 1
#     for k in 1:N2
#       old_lin += (idx[invp[k]] - 1) * strides_old[k]
#     end
#     map_old[new_lin] = old_lin
#     # increment idx in column-major (dimension 1 fastest)
#     for d in 1:N2
#       idx[d] += 1
#       if idx[d] <= newDims[d]
#         break
#       else
#         idx[d] = 1
#       end
#     end
#   end
#   return newDims, map_old
# end

# @inline function _permute_block_payload!(
#     data::Vector{T},
#     off::Int,                 # 0-based offset into data
#     blksize::Int,
#     map_old::Vector{Int},
#     scratch::Vector{T}
# ) where {T}
#   @inbounds for newi in 1:blksize
#     scratch[newi] = data[off + map_old[newi]]
#   end
#   @inbounds copyto!(data, off + 1, scratch, 1, blksize)
#   return
# end

# # --------------------------
# # The efficient permutedims!
# # --------------------------
# function permutedims!(
#     A::NewBlockSparseSorted{T,N,N2,P},
#     perm::AbstractVector{Int}
# ) where {T,N,N2,P}
#   _check_no_cross_perm(perm, N, P, N2)
#   # New dims
#   dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
#   perm_prefix = perm[1:P]          # all <= P
#   perm_block_global = perm[P+1:end] # all > P
#   perm_block = ntuple(i -> perm_block_global[i] - P, Val(N2))
#   prefix_identity = @inbounds all(perm_prefix[i] == i for i in 1:P)
#   block_identity  = @inbounds all(perm_block[i]  == i for i in 1:N2)
#   nblocks = length(A.keys)
#   @assert length(A.ids) == nblocks
#   @assert length(A.data) == nblocks * A.blksize
#   # ---- Case 1: dense tail only (prefix unchanged) => do NOT touch keys, do NOT sort
#   if prefix_identity && !block_identity
#     oldBlockDims = ntuple(i -> A.dims[P+i], Val(N2))
#     newBlockDims, map_old = _dense_oldidx_for_newidx(oldBlockDims, perm_block)
#     blksizeB = prod(newBlockDims)
#     @assert blksizeB == A.blksize  # permuting axes within tail doesn't change block size
#     scratch = Vector{T}(undef, A.blksize)
#     @inbounds for bid in 1:nblocks
#       off = (bid - 1) * A.blksize
#       _permute_block_payload!(A.data, off, A.blksize, map_old, scratch)
#     end
#     A.dims = dimsB
#     # A.blksize unchanged
#     return A
#   end
#   # ---- Case 2: prefix only (dense tail unchanged) => reorder blocks/keys, no payload permutes
#   # ---- Case 3: both => permute payload inside blocks first, then reorder blocks/keys
#   # If dense tail permutes, do it first in current block order.
#   if !block_identity
#     oldBlockDims = ntuple(i -> A.dims[P+i], Val(N2))
#     newBlockDims, map_old = _dense_oldidx_for_newidx(oldBlockDims, perm_block)
#     blksizeB = prod(newBlockDims)
#     @assert blksizeB == A.blksize  # still constant for within-tail perm
#     scratch = Vector{T}(undef, A.blksize)
#     @inbounds for bid in 1:nblocks
#       off = (bid - 1) * A.blksize
#       _permute_block_payload!(A.data, off, A.blksize, map_old, scratch)
#     end
#   end
#   # Now handle prefix remap + sort + in-place reorder
#   newkeys_tmp = Vector{NTuple{P,Int}}(undef, nblocks)
#   @inbounds for i in 1:nblocks
#     newkeys_tmp[i] = _map_prefix_key(A.keys[i], perm_prefix)
#   end
#   prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
#   p = sortperm(newkeys_tmp; by = k -> _prefix_lin(k, prefix_dimsB))  # newpos -> oldpos
#   # Apply permutation in-place via swaps.
#   # Convert to q: position -> where it should go (oldpos -> newpos)
#   q = invperm(p)
#   visited = falses(nblocks)
#   block_scratch = Vector{T}(undef, A.blksize)
#   # First, replace keys with their permuted version in the *current* order
#   A.keys = newkeys_tmp
#   @inbounds for i in 1:nblocks
#     while q[i] != i
#       j = q[i]
#       # swap keys
#       A.keys[i], A.keys[j] = A.keys[j], A.keys[i]
#       # swap payload blocks at positions i and j (since ids should correspond to packed blocks)
#       _swap_block!(A.data, A.blksize, i, j, block_scratch)
#       # keep q consistent
#       q[i], q[j] = q[j], q[i]
#     end
#   end
#   # Normalize ids to packed order 1:nblocks (required by _block_view).
#   resize!(A.ids, nblocks)
#   @inbounds for i in 1:nblocks
#     A.ids[i] = i
#   end
#   A.dims = dimsB
#   # A.blksize unchanged for within-tail perms; prefix-only also unchanged
#   return A
# end






# function permutedims!(A::NewBlockSparseSorted{T,N,N2,P},
#                       perm::AbstractVector{Int}) where {T,N,N2,P}
#   _check_no_cross_perm(perm, N, P, N2)
#   # New dims
#   dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
#   # Split perm into prefix and block parts
#   perm_prefix = perm[1:P]                 # global axes, but all <= P due to _check_no_cross_perm
#   perm_block_global = perm[P+1:end]       # global axes, all > P
#   perm_block = ntuple(i -> perm_block_global[i] - P, Val(N2))  # in 1:N2
#   is_identity_block = all(perm_block[i] == i for i in 1:N2)
#   # Block sizes
#   blockDimsA = ntuple(i -> A.dims[P+i], Val(N2))
#   blockDimsB = ntuple(i -> dimsB[P+i], Val(N2))
#   blksizeB   = prod(blockDimsB)
#   nblocks = length(A.keys)
#   @assert length(A.ids)  == nblocks
#   @assert length(A.data) == nblocks * A.blksize
#   # 1) Compute new prefix keys (unsorted)
#   newkeys_tmp = Vector{NTuple{P,Int}}(undef, nblocks)
#   @inbounds for i in 1:nblocks
#     newkeys_tmp[i] = _map_prefix_key(A.keys[i], perm_prefix)
#   end
#   # 2) Compute sort order for keys in column-major prefix linearization
#   prefix_dimsB = ntuple(i -> dimsB[i], Val(P))
#   p = sortperm(newkeys_tmp; by = k -> _prefix_lin(k, prefix_dimsB))
#   # 3) Allocate new payload buffer (needed if blksize changes or we must reorder blocks)
#   dataB = Vector{T}(undef, nblocks * blksizeB)

#   # 4) Re-pack blocks in sorted order; assign new ids = 1:nblocks
#   @inbounds for outi in 1:nblocks
#     src_i = p[outi]
#     src_id = A.ids[src_i]
#     v = _block_view(A, src_id)  # length A.blksize
#     offB = (outi - 1) * blksizeB
#     if is_identity_block
#       @assert length(v) == blksizeB
#       copyto!(dataB, offB + 1, v, 1, blksizeB)
#     else
#       blkA = reshape(v, blockDimsA...)
#       blkB = permutedims(blkA, Tuple(perm_block))  # allocates per block (ok for now)
#       copyto!(dataB, offB + 1, vec(blkB), 1, blksizeB)
#     end
#   end
#   # 5) Mutate A in-place
#   A.dims    = dimsB
#   A.blksize = blksizeB
#   # keys sorted
#   A.keys = newkeys_tmp[p]
#   # ids match packed order 1:nblocks
#   A.ids = collect(1:nblocks)
#   # replace payload
#   A.data = dataB
#   return A
# end



# mutable struct NewBlockSparseSorted{T,N,N2} <: AbstractArray{T,N}
#   dims::NTuple{N,Int}
#   blksize::Int
#   # Sorted prefix keys (P = N-N2)
#   keys::Vector{NTuple{N-N2,Int}}   # sorted unique
#   ids::Vector{Int}                  # same length as keys; block id (1..nblocks)
#   data::Vector{T}                   # concatenated dense blocks
# end
#
# function NewBlockSparseSorted{T,N,N2}(dims::NTuple{N,Int}) where {T,N,N2}
#   @assert 0 < N2 <= N
#   @assert all(dims .>= 1)
#   blksize = prod(dims[N-N2+1:N])
#   return NewBlockSparseSorted{T,N,N2}(dims, blksize, NTuple{N-N2,Int}[], Int[], T[])
# end
#
# function Base.permutedims(A::NewBlockSparseSorted{T,N,N2,P}, perm::AbstractVector{Int}) where {T,N,N2,P}
#   _check_no_cross_perm(perm, N, P, N2)
#   dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
#   perm_prefix = perm[1:P]                    # values in 1:P
#   perm_block_global = perm[P+1:end]          # values in P+1:N
#   perm_block = [perm_block_global[i] - P for i in 1:N2]  # local 1:N2
#   blockDimsA = ntuple(i -> A.dims[P+i], Val(N2))
#   blockDimsB = ntuple(i -> dimsB[P+i], Val(N2))
#   nblocks = length(A.keys)
#   @assert length(A.ids) == nblocks
#   @assert length(A.data) == nblocks * A.blksize

#   newkeys_tmp = Vector{NTuple{P,Int}}(undef, nblocks)
#   newpayloads = Vector{Vector{T}}(undef, nblocks)

#   @inbounds for i in 1:nblocks
#     oldk = A.keys[i]
#     id   = A.ids[i]
#     newk = _map_prefix_key(oldk, perm_prefix)
#     newkeys_tmp[i] = newk
#     v = _block_view(A, id)
#     if perm_block == collect(1:N2)
#       newpayloads[i] = collect(v)
#     else
#       blkA = reshape(v, blockDimsA...)  # N2-D array view
#       blkB = permutedims(blkA, perm_block)  # allocates an Array
#       newpayloads[i] = vec(blkB)
#     end
#   end

#   p = sortperm(newkeys_tmp)
#   keysB = newkeys_tmp[p]
#   blksizeB = prod(blockDimsB)  # equals A.blksize (since just a permutation), but compute anyway
#   dataB = Vector{T}(undef, nblocks * blksizeB)
#   idsB  = collect(1:nblocks)

#   @inbounds for outi in 1:nblocks
#     pay = newpayloads[p[outi]]
#     @assert length(pay) == blksizeB
#     off = (outi - 1) * blksizeB
#     copyto!(dataB, off + 1, pay, 1, blksizeB)
#   end
#   return NewBlockSparseSorted{T,N,N2}(dimsB, blksizeB, keysB, idsB, dataB)
# end
