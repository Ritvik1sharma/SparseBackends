include("../utils.jl")

Base.size(A::blocksparse{T,N}) where {T,N} = A.dims
Base.axes(A::blocksparse{T,N}) where {T,N} = ntuple(d -> Base.OneTo(A.dims[d]), N)
Base.IndexStyle(::Type{<:blocksparse}) = IndexCartesian()

function Base.getindex(A::blocksparse{T,N}, I::Vararg{Int,N}) where {T,N}
  It = ntuple(k -> I[k], N)
  @boundscheck checkbounds(A, It...)

  # Invalid (violates diag constraints) => empty/zero
  if !_valid_diag_prefix(A.rep_of, It)
    return zero(T)
  end

  plin = (N <= 2 || isempty(A.reps)) ? 1 : _plin_from_I(A.dims, A.reps, It)

  row = It[N-1]
  col = It[N]
  nrow = A.dims[N-1]                 # rows in last-2 block
  idx_in_block = (col - 1) * nrow + row  # col-major
  off = (plin - 1) * A.blksize

  return @inbounds A.data[off + idx_in_block]
end

function Base.setindex!(A::blocksparse{T,N}, v::T, I::Vararg{Int,N}) where {T,N}
  It = ntuple(k -> I[k], N)
  @boundscheck checkbounds(A, It...)

  if !_valid_diag_prefix(A.rep_of, It)
    throw(ArgumentError("Prefix indices violate diagonal constraints; cannot set value."))
  end

  plin = (N <= 2 || isempty(A.reps)) ? 1 : _plin_from_I(A.dims, A.reps, It)

  row = It[N-1]
  col = It[N]
  nrow = A.dims[N-1]
  idx_in_block = (col - 1) * nrow + row
  off = (plin - 1) * A.blksize

  @inbounds A.data[off + idx_in_block] = v
  return v
end


# similar (empty tensor with same diag_pairs)
function Base.similar(A::blocksparse{T,N}, ::Type{S}, dims::NTuple{N,Int}) where {T,N,S}
  reps, rep_of = _build_rep_map(dims, A.diag_pairs)
  nps = _n_prefix_states(dims, reps)
  return blocksparse{S,N}(dims, A.diag_pairs, reps, rep_of, zeros(Int, nps+1),
                           Tuple{Int,Int}[], S[])
end

Base.copy(A::blocksparse{T,N}) where {T,N} =
  blocksparse{T,N}(A.dims, copy(A.diag_pairs), copy(A.reps), copy(A.rep_of),
                     copy(A.coo_segment), copy(A.sel), copy(A.val))


function Base.permutedims(A::blocksparse{T,N}, perm::AbstractVector{Int}) where {T,N}
  @assert length(perm) == N
  @assert sort(collect(perm)) == collect(1:N)

  P = N - 2
  prefix_perm = collect(perm[1:P])

  last_ok = (perm[P+1] == N-1 && perm[P+2] == N)
  last_swap = (perm[P+1] == N && perm[P+2] == N-1)
  @assert last_ok || last_swap "Currently only supports keeping last-2 dims in place or swapping them."

  # New dims
  dims_new = ntuple(k -> A.dims[perm[k]], Val(N))

  # New diag_pairs live in prefix space only; update via invperm of prefix part
  invperm_full = invperm(collect(perm))
  invperm_prefix = invperm_full[1:P]
  diag_pairs_new = _permute_diag_pairs(A.diag_pairs, invperm_prefix)

  reps_new, rep_of_new = _build_rep_map(dims_new, diag_pairs_new)
  nps_new = _n_prefix_states(dims_new, reps_new)

  rows_old = A.dims[N-1]
  cols_old = A.dims[N]
  rows_new = dims_new[N-1]
  cols_new = dims_new[N]
  blksize_new = rows_new * cols_new

  @assert nps_new == _n_prefix_states(A.dims, A.reps)  # prefix permutation preserves nps
  data_new = Vector{T}(undef, nps_new * blksize_new)

  if last_ok
    # Just reorder blocks by prefix permutation; payload layout unchanged
    _prefix_perm_reorder_blocks!(data_new, A.data, A.blksize,
                                 dims_new,
                                 A.reps, A.rep_of,
                                 reps_new, rep_of_new,
                                 prefix_perm)
    return blocksparse{T,N}(dims_new, diag_pairs_new, reps_new, rep_of_new, A.blksize, data_new)
  else
    # Swap last two dims => transpose each block payload
    # First reorder blocks into a temp with old blocksize, then transpose into new blocksize
    tmp = Vector{T}(undef, nps_new * A.blksize)
    _prefix_perm_reorder_blocks!(tmp, A.data, A.blksize,
                                 dims_new,
                                 A.reps, A.rep_of,
                                 reps_new, rep_of_new,
                                 prefix_perm)

    @inbounds for plin in 1:nps_new
      src_off = (plin - 1) * A.blksize
      dst_off = (plin - 1) * blksize_new
      # new block is old block transposed: (r_new, c_new) = (c_old, r_old)
      for c_old in 1:cols_old, r_old in 1:rows_old
        v = tmp[src_off + (c_old - 1) * rows_old + r_old]
        # r_new = c_old, c_new = r_old
        data_new[dst_off + (r_old - 1) * rows_new + c_old] = v
      end
    end

    return blocksparse{T,N}(dims_new, diag_pairs_new, reps_new, rep_of_new, blksize_new, data_new)
  end
end