import ITensors

# map over an inds NTuple
_map_inds(inds::NTuple{N,ITensors.Index}, f) where {N} =
  ntuple(i -> f(inds[i]), Val(N))


@inline function _replaceprime_ind(I::ITensors.Index, p::Pair{Int,Int})
    oldp, newp = p.first, p.second
    return ITensors.plev(I) == oldp ? ITensors.setprime(I, newp) : I
end

@inline function _replaceprime_ind(I::ITensors.Index, ps::Vararg{Pair{Int,Int}})
    J = I
    @inbounds for p in ps
        J = _replaceprime_ind(J, p)
    end
    return J
end

@inline function _replaceprime_inds(inds::NTuple{N,ITensors.Index}, ps::Vararg{Pair{Int,Int}}) where {N}
    return ntuple(i -> _replaceprime_ind(inds[i], ps...), Val(N))
end


@inline function _prime_index(I::ITensors.Index, n::Int=1)
  return ITensors.setprime(I, ITensors.plev(I) + n)
end

@inline function _noprime_index(I::ITensors.Index, n::Int=ITensors.plev(I))
  p = ITensors.plev(I)
  return ITensors.setprime(I, max(p - n, 0))
end

@inline function _setprime_index(I::ITensors.Index, p::Int)
  return ITensors.setprime(I, p)
end

@inline function _prime_inds(inds::NTuple{N,ITensors.Index}, n::Int=1) where {N}
  return _map_inds(inds, I -> _prime_index(I, n))
end

@inline function _noprime_inds(inds::NTuple{N,ITensors.Index}) where {N}
  return _map_inds(inds, I -> _noprime_index(I, ITensors.plev(I)))
end

@inline function _noprime_inds(inds::NTuple{N,ITensors.Index}, n::Int) where {N}
  return _map_inds(inds, I -> _noprime_index(I, n))
end

@inline function _setprime_inds(inds::NTuple{N,ITensors.Index}, p::Int) where {N}
  return _map_inds(inds, I -> _setprime_index(I, p))
end

# COO wrapper
function replaceprime(w::SparseBackends.WrappedCOOTensor{T,N}, ps::Pair{Int,Int}...) where {T,N}
    new_inds = _replaceprime_inds(w.inds, ps...)
    return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
end

function prime(w::SparseBackends.WrappedCOOTensor{T,N}, args...) where {T,N}
    new_inds = _prime_inds(w.inds, args...)
    return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
end
function noprime(w::SparseBackends.WrappedCOOTensor{T,N}, args...) where {T,N}
  new_inds = _noprime_inds(w.inds, args...)
  return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
end
function setprime(w::SparseBackends.WrappedCOOTensor{T,N}, args...) where {T,N}
  new_inds = _setprime_inds(w.inds, args...)
  return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
end

# BlockSparse wrapper
function replaceprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, ps::Pair{Int,Int}...) where {T,N,N2,P}
  new_inds = _replaceprime_inds(w.inds, ps...)
  return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
end
function prime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
  new_inds = _prime_inds(w.inds, args...)
  return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
end
function noprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
  new_inds = _noprime_inds(w.inds, args...)
  return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
end
function setprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, args...) where {T,N,N2,P}
  new_inds = _setprime_inds(w.inds, args...)
  return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
end


function ITensors.replaceprime(es::ITensors.ExternalStorage{S}, ps::Pair{Int,Int}...; kwargs...) where {S}
  data = es.data
  data isa WrappedTensorTypes || throw(MethodError(ITensors.replaceprime, (es, ps...)))
  newdata = replaceprime(data, ps...)
  return ITensors._itensor_from_external_storage(newdata)
end

function ITensors.prime(es::ITensors.ExternalStorage{S}, args...; kwargs...) where {S}
  data = es.data
  data isa WrappedTensorTypes || throw(MethodError(ITensors.prime, (es, args...)))
  newdata = prime(data, args...)
  return ITensors._itensor_from_external_storage(newdata)
end

function ITensors.noprime(es::ITensors.ExternalStorage{S}, args...; kwargs...) where {S}
  data = es.data
  data isa WrappedTensorTypes || throw(MethodError(ITensors.noprime, (es, args...)))
  newdata = noprime(data, args...)
  return ITensors._itensor_from_external_storage(newdata)
end

function ITensors.setprime(es::ITensors.ExternalStorage{S}, args...; kwargs...) where {S}
  data = es.data
  data isa WrappedTensorTypes || throw(MethodError(ITensors.setprime, (es, args...)))
  newdata = setprime(data, args...)
  return ITensors._itensor_from_external_storage(newdata)
end

# function ITensors.dag(es::ITensors.ExternalStorage{S}; kwargs...) where {S}
#   data = es.data
#   data isa WrappedTensorTypes || throw(MethodError(ITensors.dag, (es,)))
#   newdata = dag(data)
#   return ITensors._itensor_from_external_storage(newdata)
# end

# function dag(W::WrappedBlockSparse{T,N,Ns,Nd}) where {T,N,Ns,Nd}
#   newdims = map(ITensors.dag, W.dims)
#   newvals = conj.(W.vals)
#   newkeys = W.keys   # sparsity pattern unchanged
#   return WrappedBlockSparse{T,N,Ns,Nd}(newdims, newkeys, newvals)
# end


function ITensors.dag(A::NewBlockSparseSorted{T,N,N2,P}) where {T,N,N2,P}
  newdata = similar(A.data)
  @inbounds @simd for i in eachindex(A.data)
    newdata[i] = conj(A.data[i])
  end
  return NewBlockSparseSorted{T,N,N2,P}(
    A.dims,
    A.blksize,
    copy(A.keys),
    copy(A.ids),
    newdata,
  )
end

function ITensors.dag(W::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
  newA = ITensors.dag(W.blocksparse)
  newinds = ntuple(i -> ITensors.dag(W.inds[i]), N)
  return ITensors._itensor_from_external_storage(WrappedBlockSparse{T,N,N2,P}(newA, newinds))
end

function ITensors.external_dag(es::ITensors.ExternalStorage{S}; kwargs...) where {S}
  data = es.data
  data isa WrappedTensorTypes || throw(MethodError(ITensors.external_dag, (es,)))
  newdata = ITensors.external_dag(data)
  return ITensors._itensor_from_external_storage(newdata)
end

# # COO wrapper
# function ITensors.replaceprime(w::SparseBackends.WrappedCOOTensor{T,N}, ps::Pair{Int,Int}...) where {T,N}
#   new_inds = _replaceprime_inds(w.inds, ps...)
#   return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
# end

# function ITensors.prime(w::SparseBackends.WrappedCOOTensor{T,N}, plevs::Int...) where {T,N}
#   new_inds = _prime_inds(w.inds, plevs...)
#   return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
# end

# function ITensors.noprime(w::SparseBackends.WrappedCOOTensor{T,N}, plevs::Int...) where {T,N}
#   new_inds = _noprime_inds(w.inds, plevs...)
#   return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
# end

# function ITensors.setprime(w::SparseBackends.WrappedCOOTensor{T,N}, p::Int) where {T,N}
#   new_inds = _setprime_inds(w.inds, p)
#   return SparseBackends.WrappedCOOTensor{T,N}(w.coo, new_inds)
# end


# # BlockSparse wrapper
# function ITensors.replaceprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, ps::Pair{Int,Int}...) where {T,N,N2,P}
#   new_inds = _replaceprime_inds(w.inds, ps...)
#   return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
# end

# function ITensors.prime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, plevs::Int...) where {T,N,N2,P}
#   new_inds = _prime_inds(w.inds, plevs...)
#   return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
# end

# function ITensors.noprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, plevs::Int...) where {T,N,N2,P}
#   new_inds = _noprime_inds(w.inds, plevs...)
#   return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
# end

# function ITensors.setprime(w::SparseBackends.WrappedBlockSparse{T,N,N2,P}, p::Int) where {T,N,N2,P}
#   new_inds = _setprime_inds(w.inds, p)
#   return SparseBackends.WrappedBlockSparse{T,N,N2,P}(w.blocksparse, new_inds)
# end
