include("../utils.jl")

# --------------------------
# Storage type
# --------------------------
struct RestrictedCOO{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
  diag_pairs::Vector{Tuple{Int,Int}}
  reps::Vector{Int}
  rep_of::Vector{Int}
  coo_segment::Vector{Int}          # length = nps+1, 0-based offsets
  sel::Vector{Tuple{Int,Int}}       # concatenated (row,col) per prefix-state
  val::Vector{T}
end

Base.eltype(::Type{RestrictedCOO{T}}) where {T} = T
Base.eltype(A::RestrictedCOO) = eltype(typeof(A))

# Convenience alias
const restrictedcoo = RestrictedCOO

# --------------------------
# Invariants checker (debug)
# --------------------------
function _check_invariants(A::RestrictedCOO)
  nps = _n_prefix_states(A.dims, A.reps)
  @assert length(A.coo_segment) == nps + 1
  @assert A.coo_segment[1] == 0
  @assert A.coo_segment[end] == length(A.sel) == length(A.val)
  @assert issorted(A.coo_segment)
  return true
end

# --------------------------
# Main constructor from per-plin data
# data[plin] is Vector{((row,col), val)}
# --------------------------
function RestrictedCOO{T}(dims::NTuple{N,Int},
                         data::Vector{Vector{Tuple{Tuple{Int,Int},T}}};
                         diag_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]) where {T,N}
  @assert N >= 2
  @assert all(dims .>= 1) "All dims must be >= 1"
  reps, rep_of = _build_rep_map(dims, diag_pairs)
  nps = _n_prefix_states(dims, reps)
  @assert length(data) == nps

  for plin in 1:nps
    sort!(data[plin]; by = x -> (x[1][1], x[1][2]))
    for k in 2:length(data[plin])
      @assert data[plin][k][1] != data[plin][k-1][1] "Duplicate (row,col) in segment $plin"
    end
  end

  sel = Tuple{Int,Int}[]
  val = T[]
  seg = Vector{Int}(undef, nps + 1)
  seg[1] = 0
  for plin in 1:nps
    for (rc, v) in data[plin]
      r, c = rc
      @assert 1 <= r <= dims[N-1]
      @assert 1 <= c <= dims[N]
      push!(sel, rc)
      push!(val, v)
    end
    seg[plin + 1] = length(sel)
  end

  A = RestrictedCOO{T,N}(dims, copy(diag_pairs), reps, rep_of, seg, sel, val)
  _check_invariants(A)
  return A
end



# # --------------------------
# # Union-find over prefix axes 1:(N-2)
# # --------------------------
# function _build_rep_map(dims::NTuple{N,Int}, diag_pairs::Vector{Tuple{Int,Int}}) where {N}
#   P = N - 2
#   parent = collect(1:P)

#   function find(x)
#     while parent[x] != x
#       parent[x] = parent[parent[x]]
#       x = parent[x]
#     end
#     return x
#   end

#   function unite(a,b)
#     ra, rb = find(a), find(b)
#     ra == rb && return
#     parent[rb] = ra
#   end

#   for (a,b) in diag_pairs
#     @assert 1 <= a < b <= P
#     @assert dims[a] == dims[b] "Diagonal constraint requires dims[$a]==dims[$b]"
#     unite(a,b)
#   end

#   rep_of = [find(i) for i in 1:P]
#   reps = sort!(unique(rep_of))
#   return reps, rep_of
# end

# @inline _n_prefix_states(dims::NTuple{N,Int}, reps::Vector{Int}) where {N} =
#   isempty(reps) ? 1 : prod(dims[r] for r in reps)

# # decode plin -> rep_vals (column-major over reps)
# @inline function _rep_decode!(rep_vals::Vector{Int}, plin::Int, rep_dims::Vector{Int})
#   x = plin - 1
#   @inbounds for m in 1:length(rep_dims)
#     d = rep_dims[m]
#     rep_vals[m] = (x % d) + 1
#     x ÷= d
#   end
#   return rep_vals
# end

# # Precompute: rep_pos[r] = position m such that reps[m] == r (r in 1:P), else 0
# function _rep_pos(reps::Vector{Int}, P::Int)
#   rep_pos = zeros(Int, P)
#   @inbounds for (m, r) in enumerate(reps)
#     rep_pos[r] = m
#   end
#   return rep_pos
# end
