# Base.size(A::COOTensor{T,N}) where {T,N} = A.dims
# Base.axes(A::COOTensor{T,N}) where {T,N} = ntuple(d -> Base.OneTo(A.dims[d]), Val(N))
# Base.IndexStyle(::Type{<:COOTensor}) = IndexCartesian()

# # getindex with linear scan inside segment (Step 1)
# @inline function Base.getindex(A::COOTensor{T,N}, I::Vararg{Int,N}) where {T,N}
#   @boundscheck begin
#     for d in 1:N
#       1 <= I[d] <= A.dims[d] || throw(BoundsError(A, I))
#     end
#   end



#   if P == 0
#     @inbounds for k in 1:length(A.sel)
#       if A.sel[k] == (row, col)
#         return A.val[k]
#       end
#     end
#     return zero(T)
#   end

#   prefix = ntuple(j -> I[j], Val(P))
#   plin = _prefix_to_plin(A.dims, A.reps, prefix)

#   lo = A.COOTensor_segment[plin] + 1
#   hi = A.COOTensor_segment[plin + 1]

#   @inbounds for k in lo:hi
#     if A.sel[k] == (row, col)
#       return A.val[k]
#     end
#   end
#   return zero(T)
# end

# function Base.setindex!(A::COOTensor{T,N}, v, I::Vararg{Int,N}) where {T,N}
#   P = N - 2
#   row = I[N-1]
#   col = I[N]

#   plin = if P == 0
#     1
#   else
#     prefix = ntuple(j -> I[j], Val(P))
#     _prefix_to_plin(A.dims, A.reps, prefix)
#   end

#   lo = A.COOTensor_segment[plin] + 1
#   hi = A.COOTensor_segment[plin + 1]

#   @inbounds for k in lo:hi
#     r = A.sel[k][1]
#     c = A.sel[k][2]
#     if r == row && c == col
#       A.val[k] = convert(T, v)
#       return A
#     elseif (r > row) || (r == row && c > col)
#       insert!(A.sel, k, (row, col))
#       insert!(A.val, k, convert(T, v))
#       @inbounds for j in (plin+1):length(A.COOTensor_segment)
#         A.coo_segment[j] += 1
#       end
#       return A
#     end
#   end

#   insert!(A.sel, hi + 1, (row, col))
#   insert!(A.val, hi + 1, convert(T, v))
#   @inbounds for j in (plin+1):length(A.coo_segment)
#     A.coo_segment[j] += 1
#   end
#   return A
# end

# # similar (empty tensor with same diag_pairs)
# function Base.similar(A::COO{T,N}, ::Type{S}, dims::NTuple{N,Int}) where {T,N,S}
#   reps, rep_of = _build_rep_map(dims, A.diag_pairs)
#   nps = _n_prefix_states(dims, reps)
#   return COO{S,N}(dims, A.diag_pairs, reps, rep_of, zeros(Int, nps+1),
#                            Tuple{Int,Int}[], S[])
# end

# Base.copy(A::COO{T,N}) where {T,N} =
#   COO{T,N}(A.dims, copy(A.diag_pairs), copy(A.reps), copy(A.rep_of),
#                      copy(A.coo_segment), copy(A.sel), copy(A.val))
