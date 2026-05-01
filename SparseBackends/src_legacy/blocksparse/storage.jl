include("../utils.jl")


struct RestrictedBlockSparse{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
  diag_pairs::Vector{Tuple{Int,Int}}
  reps::Vector{Int}
  rep_of::Vector{Int}
  blksize::Int                      # Equal to the sizer of the block equal to the dims[N] * dims[N-1]
  data::Vector{T}                   # concatenated dense block payloads (col-major within block)
end

Base.eltype(::Type{blocksparse{T, N}}) where {T, N} = T
Base.eltype(A::blocksparse) = eltype(typeof(A))

const blocksparse = RestrictedBlockSparse


function RestrictedBlockSparse{T}(dims::NTuple{N,Int},
                                  data::Vector{T};
                                  diag_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]) where {T,N}
  @assert N >= 2
  @assert all(dims .>= 1)
  reps, rep_of = _build_rep_map(dims, diag_pairs)
  nps = _n_prefix_states(dims, reps)
  blksize = dims[N] * dims[N-1]
  @assert length(data) == nps * blksize
  A = blocksparse{T,N}(dims, copy(diag_pairs), reps, rep_of, blksize, data)
  return A
end



# function _getblock!(B::blocksparseBuilder{T,N}, key::NTuple{N-2,Int}) where {T,N}
#   blk = get(B.blocks, key, nothing)
#   if blk === nothing
#     sh = _blkshape(B.dims, B.blockdims, key)
#     blk = zeros(T, sh)
#     B.blocks[key] = blk
#   end
#   return blk::Array{T,N}
# end

# function add!(B::blocksparseBuilder{T,N}, I::NTuple{N,Int}, v::T) where {T,N}
#   key = ntuple(a -> _blkcoord(I[a], B.blockdims[a]), Val(N))
#   localI = ntuple(a -> _inblk(I[a], B.blockdims[a]), Val(N))
#   blk = _getblock!(B, key)
#   # Guard against edge blocks where localI might exceed actual size:
#   @boundscheck begin
#     sh = size(blk)
#     for a in 1:N
#       1 <= localI[a] <= sh[a] || throw(BoundsError(blk, localI))
#     end
#   end
#   @inbounds blk[localI...] += v
#   return B
# end

# function finalize(B::blocksparseBuilder{T,N}) where {T,N}
#   keys = collect(keys(B.blocks))
#   sort!(keys)  # stable deterministic ordering

#   nblocks = length(keys)
#   blkkeys = Vector{NTuple{N,Int}}(undef, nblocks)
#   blkptr  = Vector{Int}(undef, nblocks + 1)
#   blkptr[1] = 0

#   # pass 1: sizes
#   for i in 1:nblocks
#     blkkeys[i] = keys[i]
#     blk = B.blocks[keys[i]]::Array{T,N}
#     blkptr[i+1] = blkptr[i] + length(blk)
#   end

#   data = Vector{T}(undef, blkptr[end])

#   # pass 2: pack
#   for i in 1:nblocks
#     blk = B.blocks[blkkeys[i]]::Array{T,N}
#     off0 = blkptr[i]
#     # pack in Julia column-major order
#     @inbounds for (t, val) in enumerate(vec(blk))
#       data[off0 + t] = val
#     end
#   end

#   key2id = Dict{NTuple{N,Int},Int}()
#   @inbounds for i in 1:nblocks
#     key2id[blkkeys[i]] = i
#   end

#   return blocksparse{T,N}(B.dims, B.blockdims, blkkeys, blkptr, data, key2id)
# end