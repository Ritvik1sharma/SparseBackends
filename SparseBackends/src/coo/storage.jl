mutable struct COOTensor{T,N,K<:Integer} <: SparseTensor{T,N} # AbstractArray{T,N}
  dims::NTuple{N,Int}
  keys::Vector{NTuple{N,K}}   # cached keys for sorted reads (key type K)
  vals::Vector{T}
  dirty::Bool                   # whether keys need refresh/sort
end

# ── Backwards-compatible outer constructors (default K=Int) ──────────────────
# These allow existing code that writes COOTensor{T,N}(...) to keep working.
COOTensor{T,N}(dims::NTuple{N,Int}, keys::Vector{NTuple{N,Int}},
               vals::Vector{T}, dirty::Bool) where {T,N} =
    COOTensor{T,N,Int}(dims, keys, vals, dirty)

COOTensor{T,N}(dims::NTuple{N,Int}) where {T,N} =
    COOTensor{T,N,Int}(dims, Vector{NTuple{N,Int}}(), Vector{T}(), false)

Base.eltype(::Type{COOTensor{T,N,K}}) where {T,N,K} = T
Base.eltype(A::COOTensor) = eltype(typeof(A))

Base.size(A::COOTensor{T,N,K}) where {T,N,K} = A.dims
Base.axes(A::COOTensor{T,N,K}) where {T,N,K} = ntuple(i -> Base.OneTo(A.dims[i]), Val(N))
Base.IndexStyle(::Type{<:COOTensor}) = IndexCartesian()

@inline function _check_coord!(dims::NTuple{N,Int}, I::NTuple{N,Int}) where {N}
  @inbounds for d in 1:N
    1 <= I[d] <= dims[d] || throw(BoundsError())
  end
  return nothing
end

struct ColMajorOrder{N} <: Base.Order.Ordering end

# Accept any Integer element type (covers K=UInt8, K=Int, etc. and mixed comparisons)
@inline function Base.Order.lt(::ColMajorOrder{N},
                               a::NTuple{N,<:Integer},
                               b::NTuple{N,<:Integer}) where {N}
  @inbounds for d in N:-1:1
    ad = a[d]; bd = b[d]
    if ad < bd
      return true
    elseif ad > bd
      return false
    end
  end
  return false
end

# getindex defaults to zero if missing (typical sparse semantics)
function Base.getindex(A::COOTensor{T,N,K}, I::Vararg{Int,N}) where {T,N,K}
  It = ntuple(i -> I[i], Val(N))
  _check_coord!(A.dims, It)
  key = ntuple(i -> K(I[i]), Val(N))   # convert to key type K
  ord = ColMajorOrder{N}()
  idx = searchsortedfirst(A.keys, key, ord)
  @inbounds if idx <= length(A.keys) && A.keys[idx] == key
    return A.vals[idx]
  else
    return zero(T)
  end
end

# Constructors
function COOTensor{T,N}(dims::NTuple{N,Int},
                        entries::Vector{Tuple{NTuple{N,Int},T}}) where {T,N}
  @assert all(dims .>= 1)
  d = Dict{NTuple{N,Int},T}()
  @inbounds for (coord, v) in entries
    _check_coord!(dims, coord)
    @assert !haskey(d, coord) "Duplicate coordinate: $coord"
    if v != zero(T)
      d[coord] = v
    end
  end
  return COOTensor{T,N,Int}(dims, collect(keys(d)), collect(values(d)), false)
end

COOTensor(dims::NTuple{N,Int}, entries::Vector{Tuple{NTuple{N,Int},T}}) where {T,N} =
  COOTensor{T,N}(dims, entries)

function COOTensor{T,N,K}(dims::NTuple{N,Int}) where {T,N,K<:Integer}
  return COOTensor{T,N,K}(dims, Vector{NTuple{N,K}}(), Vector{T}(), false)
end

function Base.setindex!(A::COOTensor{T,N,K}, v, I::Vararg{Int,N}; dirty::Bool=false) where {T,N,K}
  It = ntuple(i -> I[i], Val(N))
  _check_coord!(A.dims, It)
  key = ntuple(i -> K(I[i]), Val(N))   # convert to key type K
  vv = convert(T, v)
  ord = ColMajorOrder{N}()

  if A.dirty
    if vv == zero(T)
      i = 1
      @inbounds while i <= length(A.keys)
        if A.keys[i] == key
          deleteat!(A.keys, i)
          deleteat!(A.vals, i)
        else
          i += 1
        end
      end
      return vv
    else
      push!(A.keys, key)
      push!(A.vals, vv)
      A.dirty = true
      return vv
    end
  end

  # Canonical mode: A.keys must already be sorted by `ord`
  idx = searchsortedfirst(A.keys, key, ord)
  @inbounds if idx <= length(A.keys) && A.keys[idx] == key
    if vv == zero(T)
      deleteat!(A.keys, idx)
      deleteat!(A.vals, idx)
      A.dirty = true
    else
      A.vals[idx] = vv
    end
  else
    if vv != zero(T)
      if !dirty
        insert!(A.keys, idx, key)
        insert!(A.vals, idx, vv)
      else
        push!(A.keys, key)
        push!(A.vals, vv)
        A.dirty = true
      end
    end
  end
  return vv
end

@inline function _check_perm(perm, N::Int)
  @assert length(perm) == N
  seen = falses(N)
  @inbounds for j in 1:N
    p = perm[j]
    (1 <= p <= N) || throw(ArgumentError("perm out of range: perm[$j]=$p for N=$N"))
    seen[p] && throw(ArgumentError("perm has duplicates: value $p appears more than once"))
    seen[p] = true
  end
  return true
end

# Generalized to accept any Integer key type (preserves element type of coord)
@inline function _apply_perm(coord::NTuple{N,<:Integer}, perm::NTuple{N,Int}) where {N}
  return ntuple(j -> coord[perm[j]], Val(N))
end

function Base.permutedims(A::COOTensor{T,N,K}, perm::NTuple{N,Int}) where {T,N,K}
  _check_perm(perm, N)
  dimsB = ntuple(j -> A.dims[perm[j]], Val(N))
  keysB = similar(A.keys)
  valsB = A.vals  # values unchanged, order preserved
  @inbounds for i in eachindex(A.keys)
    keysB[i] = _apply_perm(A.keys[i], perm)
  end
  return COOTensor{T,N,K}(dimsB, keysB, valsB, true)
end

function Base.permutedims(A::COOTensor{T,N,K}, perm::AbstractVector{Int}) where {T,N,K}
  return permutedims(A, ntuple(i -> perm[i], Val(N)))
end

function permutedims!(A::COOTensor{T,N,K}, perm::NTuple{N,Int}) where {T,N,K}
  _check_perm(perm, N)
  A.dims = ntuple(j -> A.dims[perm[j]], Val(N))
  @inbounds for i in eachindex(A.keys)
    A.keys[i] = _apply_perm(A.keys[i], perm)
  end
  A.dirty = true
  return A
end

function permutedims!(A::COOTensor{T,N,K}, perm::AbstractVector{Int}) where {T,N,K}
  return permutedims!(A, ntuple(i -> perm[i], Val(N)))
end

function sort_coo!(A::COOTensor{T,N,K}) where {T,N,K}
  !A.dirty && return A
  p = sortperm(A.keys; by = reverse)
  A.keys = A.keys[p]
  A.vals = A.vals[p]
  A.dirty = false
  return A
end

import Base: sort!

"Default sort!: use the storage's canonical order (column-major)."
function sort!(A::COOTensor{T,N,K}) where {T,N,K}
  return sort_coo!(A)
end

"Optional: allow custom ordering via Base.Order.Ordering."
function sort!(A::COOTensor{T,N,K}, order::Base.Order.Ordering) where {T,N,K}
  p = sortperm(A.keys, order = order)
  A.keys = A.keys[p]
  A.vals = A.vals[p]
  A.dirty = false
  return A
end
