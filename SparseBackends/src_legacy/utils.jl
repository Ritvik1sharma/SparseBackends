
# --------------------------
# Union-find over prefix axes 1:(N-2)
# --------------------------
function _build_rep_map(dims::NTuple{N,Int}, diag_pairs::Vector{Tuple{Int,Int}}) where {N}
  P = N - 2
  parent = collect(1:P)

  function find(x)
    while parent[x] != x
      parent[x] = parent[parent[x]]
      x = parent[x]
    end
    return x
  end

  function unite(a,b)
    ra, rb = find(a), find(b)
    ra == rb && return
    parent[rb] = ra
  end

  for (a,b) in diag_pairs
    @assert 1 <= a < b <= P
    @assert dims[a] == dims[b] "Diagonal constraint requires dims[$a]==dims[$b]"
    unite(a,b)
  end

  rep_of = [find(i) for i in 1:P]
  reps = sort!(unique(rep_of))
  return reps, rep_of
end

@inline _n_prefix_states(dims::NTuple{N,Int}, reps::Vector{Int}) where {N} =
  isempty(reps) ? 1 : prod(dims[r] for r in reps)

# decode plin -> rep_vals (column-major over reps)
@inline function _rep_decode!(rep_vals::Vector{Int}, plin::Int, rep_dims::Vector{Int})
  x = plin - 1
  @inbounds for m in 1:length(rep_dims)
    d = rep_dims[m]
    rep_vals[m] = (x % d) + 1
    x ÷= d
  end
  return rep_vals
end

# Precompute: rep_pos[r] = position m such that reps[m] == r (r in 1:P), else 0
function _rep_pos(reps::Vector{Int}, P::Int)
  rep_pos = zeros(Int, P)
  @inbounds for (m, r) in enumerate(reps)
    rep_pos[r] = m
  end
  return rep_pos
end

# build full prefix index vector from rep_vals without Dict
@inline function _prefix_from_rep!(prefix::Vector{Int},
                                   rep_of::Vector{Int},
                                   rep_pos::Vector{Int},
                                   rep_vals::Vector{Int})
  @inbounds for a in 1:length(prefix)
    r = rep_of[a]
    m = rep_pos[r]
    prefix[a] = rep_vals[m]
  end
  return prefix
end

# compute plin directly from prefix indices (uses reps only)
@inline function _prefix_to_plin(dims::NTuple{N,Int},
                                 reps::Vector{Int},
                                 prefix::NTuple{P,Int}) where {N,P}
  lin = 1
  stride = 1
  @inbounds for r in reps
    v = prefix[r]
    d = dims[r]
    lin += (v - 1) * stride
    stride *= d
  end
  return lin
end

@inline function _valid_diag_prefix(rep_of::Vector{Int}, I::NTuple{N,Int}) where {N}
  P = N - 2
  if P <= 0
    return true
  end
  @inbounds for a in 1:P
    ra = rep_of[a]                 # representative axis index in 1:P
    if I[a] != I[ra]
      return false
    end
  end
  return true
end

@inline function _plin_from_I(dims::NTuple{N,Int}, reps::Vector{Int}, I::NTuple{N,Int}) where {N}
  # Column-major over reps (same convention as your _rep_decode!)
  lin = 1
  stride = 1
  @inbounds for r in reps
    v = I[r]
    lin += (v - 1) * stride
    stride *= dims[r]
  end
  return lin
end

@inline function _permute_diag_pairs(diag_pairs::Vector{Tuple{Int,Int}}, invperm_prefix::Vector{Int})
  out = Tuple{Int,Int}[]
  for (a,b) in diag_pairs
    a2 = invperm_prefix[a]
    b2 = invperm_prefix[b]
    a2, b2 = (a2 < b2) ? (a2, b2) : (b2, a2)
    push!(out, (a2,b2))
  end
  return out
end

function _prefix_perm_reorder_blocks!(dst_data::Vector{T}, src_data::Vector{T},
                                      blksize::Int,
                                      dims::NTuple{N,Int},
                                      reps_old::Vector{Int}, rep_of_old::Vector{Int},
                                      reps_new::Vector{Int}, rep_of_new::Vector{Int},
                                      prefix_perm::Vector{Int}) where {T,N}
  P = N - 2
  nps = _n_prefix_states(dims, reps_old) # equals new nps if prefix_perm is a permutation of prefix dims
  @assert length(dst_data) == length(src_data) == nps * blksize

  rep_dims_old = [dims[r] for r in reps_old]
  rep_pos_old  = _rep_pos(reps_old, P)
  rep_vals_old = Vector{Int}(undef, length(rep_dims_old))
  prefix_old   = Vector{Int}(undef, P)
  prefix_new   = Vector{Int}(undef, P)

  @inbounds for plin_old in 1:nps
    if !isempty(rep_dims_old)
      _rep_decode!(rep_vals_old, plin_old, rep_dims_old)
      _prefix_from_rep!(prefix_old, rep_of_old, rep_pos_old, rep_vals_old)
    else
      fill!(prefix_old, 1)
    end

    # Apply prefix permutation: new_prefix[k] = old_prefix[prefix_perm[k]]
    @inbounds for k in 1:P
      prefix_new[k] = prefix_old[prefix_perm[k]]
    end

    # Compute new plin using new reps (column-major over reps)
    lin = 1
    stride = 1
    @inbounds for r in reps_new
      v = prefix_new[r]        # r is axis hookup index in 1:P
      d = dims[r]
      lin += (v - 1) * stride
      stride *= d
    end
    plin_new = lin

    # Copy dense block payload
    src_off = (plin_old - 1) * blksize
    dst_off = (plin_new - 1) * blksize
    for t in 1:blksize
      dst_data[dst_off + t] = src_data[src_off + t]
    end
  end

  return dst_data
end

