# using ITensors: ITensor, inds

# Dense conversion: storage -> Array
function to_dense(A::COOTensor{T,N,K}) where {T,N,K}
  out = zeros(T, A.dims...)
  @inbounds for (coord, v) in zip(A.keys, A.vals)
    out[coord...] = v
  end
  return out
end

# Array -> COO
function coo_from_dense(A::AbstractArray{T,N};
                        atol::Real = 1e-12,
                        rtol::Real = 0.0) where {T,N}
  dims = ntuple(i -> size(A,i), Val(N))
  keys, vals = NTuple{N,Int}[], T[]
  @inbounds for I in CartesianIndices(A)
    v = A[I]
    if abs(v) > atol + rtol * abs(v)
      push!(keys, Tuple(I))
      push!(vals, v)
    end
  end
  return COOTensor{T,N,Int}(dims, keys, vals, false)
end


Base.Array(A::COOTensor{T,N,K}) where {T,N,K} = to_dense(A)

# ─────────────────────────────────────────────────────────────────────────────
# rekey — convert key integer type
# ─────────────────────────────────────────────────────────────────────────────

"""
    rekey(A::COOTensor{T,N,K}, ::Type{K2}; check=true) → COOTensor{T,N,K2}

Convert the key integer type.  With `check=true` (default), errors if any
coordinate value does not fit in K2.
"""
function rekey(A::COOTensor{T,N,K}, ::Type{K2}; check::Bool=true) where {T,N,K,K2<:Integer}
    if check
        for key in A.keys
            for v in key
                (typemin(K2) <= v <= typemax(K2)) ||
                    error("Coordinate $v is out of range for $K2 " *
                          "($(typemin(K2))..$(typemax(K2)))")
            end
        end
    end
    new_keys = [ntuple(j -> K2(key[j]), Val(N)) for key in A.keys]
    return COOTensor{T,N,K2}(A.dims, new_keys, copy(A.vals), A.dirty)
end

# ─────────────────────────────────────────────────────────────────────────────
# retypeval — convert payload element type with lossless checking
# ─────────────────────────────────────────────────────────────────────────────

"""
    retypeval(A::COOTensor{T,N,K}, ::Type{T2}) → COOTensor{T2,N,K}

Convert the payload element type.  Errors if any stored value cannot be
exactly represented in T2 (checked via roundtrip: T(T2(v)) == v).
"""
function retypeval(A::COOTensor{T,N,K}, ::Type{T2}) where {T,N,K,T2}
    new_vals = Vector{T2}(undef, length(A.vals))
    for (i, v) in enumerate(A.vals)
        if T2 <: Integer && !isinteger(v)
            error("Value $v is non-integer; cannot convert to $T2 without loss")
        end
        c = try
            T2(v)
        catch e
            error("Value $v cannot be converted to $T2: $e")
        end
        T(c) == v ||
            error("Value $v → $c → $(T(c)): roundtrip failed; " *
                  "conversion to $T2 is lossy")
        new_vals[i] = c
    end
    return COOTensor{T2,N,K}(A.dims, copy(A.keys), new_vals, A.dirty)
end

# function fuse_axes_coo(A::COOTensor{T,N,K}, ax1::Int, ax2::Int) where {T,N,K}
#     ax1 == ax2 && throw(ArgumentError("ax1 and ax2 must be distinct"))
#     (1 <= ax1 <= N) || throw(BoundsError(A, ax1))
#     (1 <= ax2 <= N) || throw(BoundsError(A, ax2))

#     a1, a2 = min(ax1, ax2), max(ax1, ax2)

#     dims = A.dims
#     d1, d2 = dims[a1], dims[a2]

#     newdims = ntuple(i -> begin
#         if i < a2
#             i == a1 ? d1 * d2 : dims[i]
#         else
#             dims[i + 1]
#         end
#     end, Val(N - 1))

#     n = length(A.keys)
#     newkeys = Vector{NTuple{N-1,Int}}(undef, n)

#     @inbounds for k in 1:n
#         coord = A.keys[k]
#         x = coord[a1]
#         y = coord[a2]
#         f = (x - 1) * d2 + y

#         newkeys[k] = ntuple(j -> begin
#             if j < a2
#                 j == a1 ? f : coord[j]
#             else
#                 coord[j + 1]
#             end
#         end, Val(N - 1))
#     end
#     # values unchanged; safest to mark dirty since ordering likely changed
#     return COOTensor{T,N-1}(newdims, newkeys, copy(A.vals), true)
# end

# function fuse_axes_coo(A::COOTensor{T,N,K}, ax1::Int, ax2::Int) where {T,N,K}
#   ax1 == ax2 && throw(ArgumentError("ax1 and ax2 must be distinct"))
#   (1 <= ax1 <= N) || throw(BoundsError(A, ax1))
#   (1 <= ax2 <= N) || throw(BoundsError(A, ax2))

#   dims = A.dims
#   dkeep, ddrop = dims[ax1], dims[ax2]

#   # In the output, the kept axis shifts left by 1 if you removed an earlier axis.
#   ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)

#   # Map output axis j -> old axis index (skipping dropped axis).
#   @inline old_axis(j) = (j < ax2) ? j : (j + 1)

#   newdims = ntuple(j -> begin
#     oa = old_axis(j)
#     if j == ax1_out
#       dkeep * ddrop
#     else
#       dims[oa]
#     end
#   end, Val(N - 1))

#   n = length(A.keys)
#   entries = Vector{Tuple{NTuple{N-1,Int},T}}(undef, n)

#   @inbounds for k in 1:n
#     coord = A.keys[k]
#     fused = (coord[ax1] - 1) * ddrop + coord[ax2]  # 1-based
#     entries[k] = (ntuple(j -> begin
#       if j == ax1_out
#         fused
#       else
#         coord[old_axis(j)]
#       end
#     end, Val(N - 1)), A.vals[k])
#   end
#   # Use your actual COOTensor constructor order:
#   return COOTensor{T,N-1}(newdims, entries)
# end

@inline function colmajor_linear_index(coord::NTuple{M,Int}, dims::NTuple{M,Int}) where {M}
  idx = 1
  stride = 1
  @inbounds for d in 1:M
    idx += (coord[d] - 1) * stride
    stride *= dims[d]
  end
  return idx
end

function fuse_axes_coo(A::COOTensor{T,N,K}, ax1::Int, ax2::Int) where {T,N,K}
  ax1 == ax2 && throw(ArgumentError("ax1 and ax2 must be distinct"))
  (1 <= ax1 <= N) || throw(BoundsError(A, ax1))
  (1 <= ax2 <= N) || throw(BoundsError(A, ax2))
  @assert abs(ax1 - ax2) == 1 "Currently only adjacent axes can be fused but got ax1=$ax1 and ax2=$ax2"

  dims = A.dims
  dkeep = dims[ax1]
  ddrop = dims[ax2]

  # Output keeps ax1, drops ax2. If ax2 < ax1, ax1 shifts left by 1.
  ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)

  @inline old_axis(j) = (j < ax2) ? j : (j + 1)

  newdims = ntuple(j -> begin
    oa = old_axis(j)
    if j == ax1_out
      dkeep * ddrop
    else
      dims[oa]
    end
  end, Val(N - 1))

  n = length(A.keys)
  newkeys = Vector{NTuple{N-1,K}}(undef, n)
  newvals = Vector{T}(undef, n)

  @inbounds for k in 1:n
    coord = A.keys[k]

    # ax1 is the fast/minor index, ax2 is the slow/major index
    fused = K((coord[ax2] - 1) * dkeep + coord[ax1])

    newcoord = ntuple(j -> begin
      if j == ax1_out
        fused
      else
        coord[old_axis(j)]
      end
    end, Val(N - 1))

    newkeys[k] = newcoord
    newvals[k] = A.vals[k]
  end

  C = COOTensor{T,N-1,K}(newdims, newkeys, newvals, true)
  sort!(C)
  return C
end

# function fuse_axes_coo(A::COOTensor{T,N,K}, ax1::Int, ax2::Int) where {T,N,K}
#   ax1 == ax2 && throw(ArgumentError("ax1 and ax2 must be distinct"))
#   (1 <= ax1 <= N) || throw(BoundsError(A, ax1))
#   (1 <= ax2 <= N) || throw(BoundsError(A, ax2))
#   @assert ax1 - ax2 == 1 || ax2 - ax1 == 1 "Currently only adjacent axes can be fused but got ax1=$ax1 and ax2=$ax2"

#   dims = A.dims
#   dkeep, ddrop = dims[ax1], dims[ax2]

#   ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)
#   @inline old_axis(j) = (j < ax2) ? j : (j + 1)

#   newdims = ntuple(j -> begin
#     oa = old_axis(j)
#     if j == ax1_out
#       dkeep * ddrop
#     else
#       dims[oa]
#     end
#   end, Val(N - 1))

#   n = length(A.keys)
#   entries = Vector{Tuple{NTuple{N-1,Int},T}}(undef, n)
#   @inbounds for k in 1:n
#     coord = A.keys[k]
#     fused = (coord[ax2] - 1) * dkeep + coord[ax1]  # 1-based, with ax2 as the faster index
#     newcoord = ntuple(j -> (j == ax1_out ? fused : coord[old_axis(j)]), Val(N - 1))
#     entries[k] = (newcoord, A.vals[k])
#   end
#   sort!(entries; by = e -> colmajor_linear_index(e[1], newdims))
#   return COOTensor{T,N-1}(newdims, entries)
# end


# # Dense conversion: storage -> Array
# function to_dense(A::COOTensor{T,N,K}) where {T,N,K}
#   out = zeros(T, A.dims...)
#   @inbounds for (coord, v) in A.data
#     out[coord...] = v           # coord is NTuple{N,Int}
#     # alternatively: out[CartesianIndex(coord)] = v
#   end
#   return out
# end

# # Array -> COO
# function coo_from_dense(A::AbstractArray{T,N};
#                         atol::Real = 1e-12,
#                         rtol::Real = 0.0) where {T,N}
#   dims = ntuple(i -> size(A,i), Val(N))
#   d = Dict{NTuple{N,Int},T}()
#   @inbounds for I in CartesianIndices(A)
#     v = A[I]
#     if abs(v) > atol + rtol * abs(v)
#       d[Tuple(I)] = v
#     end
#   end
#   return COOTensor{T,N}(dims, d, NTuple{N,Int}[], true)
# end

# function dense_itensor(A::COOTensor{T,N}, inds::Vararg{Index,N}) where {T,N}
#   return ITensor(to_dense(A), inds...)
# end

# function coo_from_itensor(T::ITensor;
#                           diag_pairs=Tuple{Int,Int}[],
#                           atol=1e-12,
#                           rtol=0.0)
#   A = Array(T, inds(T)...)
#   return coo_from_dense(A, atol=atol, rtol=rtol)
# end
