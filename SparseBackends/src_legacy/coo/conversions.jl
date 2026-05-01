# Dense conversion: storage -> Array
function to_dense(A::RestrictedCOO{T,N}) where {T,N}
  out = zeros(T, A.dims)
  P = N - 2
  nps = _n_prefix_states(A.dims, A.reps)

  if P == 0
    @inbounds for k in eachindex(A.sel)
      (r,c) = A.sel[k]
      out[r,c] = A.val[k]
    end
    return out
  end

  rep_dims = [A.dims[r] for r in A.reps]
  rep_pos  = _rep_pos(A.reps, P)
  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, A.rep_of, rep_pos, rep_vals)
    else
      fill!(prefix, 1)
    end

    lo = A.coo_segment[plin] + 1
    hi = A.coo_segment[plin+1]
    for k in lo:hi
      (r,c) = A.sel[k]
      out[CartesianIndex(prefix..., r, c)] = A.val[k]
    end
  end
  return out
end

# Array -> RestrictedCOO
function coo_from_dense(A::AbstractArray{T,N};
                        diag_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
                        atol::Real = 1e-12,
                        rtol::Real = 0) where {T,N}

  dims = ntuple(i -> size(A, i), Val(N))
  P    = N - 2
  rows = dims[N-1]
  cols = dims[N]

  reps, rep_of = _build_rep_map(dims, diag_pairs)
  nps = _n_prefix_states(dims, reps)

  data = [Vector{Tuple{Tuple{Int,Int},T}}() for _ in 1:nps]

  if P == 0
    @inbounds for r in 1:rows, c in 1:cols
      v = A[r,c]
      if abs(v) > atol
        push!(data[1], ((r,c), v))
      end
    end
    return RestrictedCOO{T}(dims, data; diag_pairs=diag_pairs)
  end

  rep_dims = [dims[r] for r in reps]
  rep_pos  = _rep_pos(reps, P)
  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, rep_of, rep_pos, rep_vals)
    else
      fill!(prefix, 1)
    end

    for r in 1:rows, c in 1:cols
      v = A[CartesianIndex(prefix..., r, c)]
      if abs(v) > atol
        push!(data[plin], ((r,c), v))
      end
    end
  end

  return RestrictedCOO{T}(dims, data; diag_pairs=diag_pairs)
end

function dense_itensor(A::RestrictedCOO{T,N}, inds::Vararg{Index,N}) where {T,N}
  return ITensor(Array(A), inds...)
end

function coo_from_itensor(T::ITensor;
                          diag_pairs=Tuple{Int,Int}[],
                          atol=1e-12,
                          rtol=0.0)
  indsT = inds(T)
  A = Array(T, indsT...)
  return coo_from_dense(A; diag_pairs=diag_pairs, atol=atol, rtol=rtol)
end

Base.Array(A::RestrictedCOO{T,N}) where {T,N} = to_dense(A)
