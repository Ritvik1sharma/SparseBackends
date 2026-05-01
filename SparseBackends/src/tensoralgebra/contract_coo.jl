"""
Ensure `rlab` is the *last* label/axis.
If already last: fast exit (no permute, no sort).
Else: permute tensor axes to move rlab last, update labels+map, then sort!(A).
"""
# function ensure_rlab_last_and_sorted!(
#     A::COOTensor{T,N},
#     labels::Vector,
#     map::Dict,
#     rlab
# ) where {T,N}
#   @inbounds if labels[end] == rlab
#     return nothing  # fast exit
#   end
#   rpos = map[rlab]
#   # Build perm: new axis j = old axis perm[j]
#   # New label order is: all labels except rlab (same relative order), then rlab
#   # So perm is: old positions of those labels in that order.
#   perm_vec = Vector{Int}(undef, N)
#   k = 1
#   @inbounds for i in 1:N
#     lab = labels[i]
#     if lab != rlab
#       perm_vec[k] = i
#       k += 1
#     end
#   end
#   @inbounds perm_vec[N] = rpos
#   perm = ntuple(j -> perm_vec[j], Val(N))
#   # Apply perm to tensor axes (keys only, no sorting inside)
#   permutedims!(A, perm)
#   # Mutate labels in-place to match new order
#   # (stable remove rlab then push to end)
#   deleteat!(labels, rpos)
#   push!(labels, rlab)
#   # Rebuild map in-place
#   empty!(map)
#   @inbounds for i in 1:N
#     map[labels[i]] = i
#   end
#   # Canonicalize order (column-major)
#   sort!(A)
#   return nothing
# end

const Label = NTuple{2,UInt64} 

function ensure_rlab_last_and_sorted(
    A::COOTensor{T,N},
    labels::AbstractVector,
    rlab::Label
) where {T,N}
  # fast exit
  if labels[end] == rlab
    labels2 = collect(labels)
    map2 = Dict(labels2[i] => i for i in 1:N)
    return A, labels2, map2
  end

  # build new label order: keep relative order, move rlab to end
  labels2 = Vector{eltype(labels)}(undef, N)
  k = 1
  @inbounds for i in 1:N
    lab = labels[i]
    if lab != rlab
      labels2[k] = lab
      k += 1
    end
  end
  labels2[N] = rlab

  # perm: new axis j comes from old axis perm[j]
  map = Dict(labels[i] => i for i in 1:N)
  perm = ntuple(j -> map[labels2[j]], Val(N))

  A2 = permutedims(A, perm)  # NON-mutating
  A2.dirty = true
  sort!(A2)

  map2 = Dict(labels2[i] => i for i in 1:N)
  return A2, labels2, map2
end

@inline function _push_term!(C::COOTensor{T,N}, c::NTuple{N,Int}, v) where {T,N}
  vv = convert(T, v)
  vv == zero(T) && return nothing
  push!(C.keys, c)
  push!(C.vals, vv)
  C.dirty = true
  return nothing
end

function contract!(
    C::COOTensor{TC,NC},
    labelsC::Vector{Label},            # make these Vectors if you want in-place label ops
    A::COOTensor{TA,NA},
    labelsA::Vector{Label},
    B::COOTensor{TB,NB},
    labelsB::Vector{Label},
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    rlab::Label
) where {TC,NC,TA,NA,TB,NB}
  if get(ENV, "SB_TRACE", "0") == "1"
    println("[SB_TRACE] contract_coo.contract!  COO(nnz=", length(A.keys),
            ", dims=", A.dims, ") × COO(nnz=", length(B.keys), ", dims=", B.dims, ")",
            "  → COO{NC=", NC, "}")
  end
  # Ensure reduced label is last axis and keys are column-major sorted
  # println("========================= ", A.dirty, B.dirty)
  # sort!(A)
  # sort!(B)
  # println("Sorted A: ", A.keys, " and B: ", B.keys)
  # println("Labels A: ", labelsA, " and B: ", labelsB)
  A, labelsA, mapA = ensure_rlab_last_and_sorted(A, labelsA, rlab)
  B, labelsB, mapB = ensure_rlab_last_and_sorted(B, labelsB, rlab)
  # println("Sorted A: ", A.keys, " and B: ", B.keys)
  # println("Labels A: ", labelsA, " and B: ", labelsB)
  # Build output coordinate assembly plan once:
  src = Vector{UInt8}(undef, NC)
  pos = Vector{Int}(undef, NC)
  @inbounds for t in 1:NC
    lab = labelsC[t]
    if haskey(mapA, lab)
      src[t] = 0x01
      pos[t] = mapA[lab]
    else
      src[t] = 0x02
      pos[t] = mapB[lab]
    end
  end
  # Overwrite C
  empty!(C.keys); empty!(C.vals); C.dirty = true
  # Heuristic reserve (tune if needed)
  sizehint!(C.keys, length(A.keys))
  sizehint!(C.vals, length(A.keys))
  i, j = 1, 1
  nA, nB = length(A.keys), length(B.keys)
  @inbounds while i <= nA && j <= nB
    ka = A.keys[i]
    kb = B.keys[j]
    ra = ka[end]
    rb = kb[end]
    if ra < rb
      i += 1
    elseif rb < ra
      j += 1
    else
      # cross product within matching groups
      j_low = j
      while j <= nB && B.keys[j][end] == rb
        j += 1
      end
      while i <= nA && A.keys[i][end] == ra
        # for ii in i:(i2-1)
        acoord = A.keys[i]
        aval   = A.vals[i]
        for jj in j_low:(j-1)
          bcoord = B.keys[jj]
          bval   = B.vals[jj]
          ccoord = ntuple(t -> (src[t] == 0x01 ? acoord[pos[t]] : bcoord[pos[t]]), Val(NC))
          _push_term!(C, ccoord, aval * bval)
        end
        i += 1
      end
    end
  end
  # println("Contracted COO: ", C.keys, " and inputs ", A.keys, " and ", B.keys, "  ", C.dirty)
  return C
end
