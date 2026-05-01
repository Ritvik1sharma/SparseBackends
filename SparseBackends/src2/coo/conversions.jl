# Dense conversion: storage -> Array
function to_dense(A::COO{T,N}) where {T,N}
  out = zeros(T, A.dims...) 
  @inbounds for (coord, v) in A.data
    out[coord...] = v           # coord is NTuple{N,Int}
    # alternatively: out[CartesianIndex(coord)] = v
  end
  return out
end

# Array -> COO
function coo_from_dense(A::AbstractArray{T,N};
                        atol::Real = 1e-12,
                        rtol::Real = 0.0) where {T,N}
  dims = ntuple(i -> size(A,i), Val(N))
  d = Dict{NTuple{N,Int},T}()

  @inbounds for I in CartesianIndices(A)
    v = A[I]
    if abs(v) > atol + rtol * abs(v)
      d[Tuple(I)] = v
    end
  end

  return COOMap{T,N}(dims, d, NTuple{N,Int}[], true)
end

function dense_itensor(A::COO{T,N}, inds::Vararg{Index,N}) where {T,N}
  return ITensor(to_dense(A), inds...)
end

function coo_from_itensor(T::ITensor;
                          diag_pairs=Tuple{Int,Int}[],
                          atol=1e-12,
                          rtol=0.0)
  A = Array(T, inds(T)...)
  return coo_from_dense(A, atol=atol, rtol=rtol)
end

Base.Array(A::COO{T,N}) where {T,N} = to_dense(A)
