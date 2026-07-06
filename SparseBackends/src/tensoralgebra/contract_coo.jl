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
  if false  # SB_TRACE — flip to true here for debug output
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

# ── COO × COO with MULTIPLE shared labels ────────────────────────────────────
# General sparse contraction for |shared| ≥ 1 (the single-shared case has its own
# faster kernel above). `shared` = labels common to A and B; those NOT in C are
# reduced (summed), those in C are batched (kept once, taken from either operand).
#
# Method: sort both operands by their shared sub-key, merge-join on equal shared
# keys, and cross-product the free legs of each matching group. This exploits
# sparsity fully — only entries whose shared coords coincide ever multiply, and
# non-matching shared groups are skipped by the merge (cost ~ nnz·log·nnz + the
# actual number of coincidences, never dims).
#
# Reduced indices make several (A-entry × B-entry) pairs land on the SAME output
# coordinate, so we ACCUMULATE into a Dict (COO `to_dense` overwrites duplicates
# rather than summing them, so the kernel must combine here). This mirrors the
# accumulate-into-block behavior of the COO×dense→BS kernel.
function contract_shared!(
    C::COOTensor{TC,NC},
    labelsC::Vector{Label},
    A::COOTensor{TA,NA},
    labelsA::Vector{Label},
    B::COOTensor{TB,NB},
    labelsB::Vector{Label},
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    shared::AbstractVector{Label};
    output_inds_hint=nothing,   # COO ignores index-order hints
    allowed_keys_C=nothing,     # not used on the COO path
) where {TC,NC,TA,NA,TB,NB}
  if false  # SB_TRACE — flip to true here for debug output
    println("[SB_TRACE] contract_coo.contract_shared!  COO(nnz=", length(A.keys),
            ") × COO(nnz=", length(B.keys), ")  shared=", shared, "  → COO{NC=", NC, "}")
  end
  ns = length(shared)
  # positions of the shared labels within A and B (same `shared` order for both)
  sposA = ntuple(t -> mapA[shared[t]], ns)
  sposB = ntuple(t -> mapB[shared[t]], ns)
  # output-coordinate assembly plan: each C axis comes from A (0x01) or B (0x02).
  # Batched (shared∩C) labels are present in both; sourcing from A is equivalent.
  src = Vector{UInt8}(undef, NC)
  pos = Vector{Int}(undef, NC)
  @inbounds for t in 1:NC
    lab = labelsC[t]
    if haskey(mapA, lab)
      src[t] = 0x01; pos[t] = mapA[lab]
    else
      src[t] = 0x02; pos[t] = mapB[lab]
    end
  end
  empty!(C.keys); empty!(C.vals); C.dirty = true
  nA, nB = length(A.keys), length(B.keys)
  (nA == 0 || nB == 0) && return C

  # shared sub-key per entry (KA/KB are the operands' coordinate integer types)
  KA = eltype(eltype(A.keys)); KB = eltype(eltype(B.keys))
  skeysA = Vector{NTuple{ns,KA}}(undef, nA)
  skeysB = Vector{NTuple{ns,KB}}(undef, nB)
  @inbounds for i in 1:nA; skeysA[i] = ntuple(t -> A.keys[i][sposA[t]], ns); end
  @inbounds for j in 1:nB; skeysB[j] = ntuple(t -> B.keys[j][sposB[t]], ns); end
  permA = sortperm(skeysA)
  permB = sortperm(skeysB)

  acc = Dict{NTuple{NC,Int},TC}()
  i, j = 1, 1
  @inbounds while i <= nA && j <= nB
    ka = skeysA[permA[i]]
    kb = skeysB[permB[j]]
    if ka < kb
      i += 1
    elseif kb < ka
      j += 1
    else
      # equal shared key: gather the matching runs [i:i_hi) in A, [j:j_hi) in B
      i_hi = i; while i_hi <= nA && skeysA[permA[i_hi]] == ka; i_hi += 1; end
      j_hi = j; while j_hi <= nB && skeysB[permB[j_hi]] == kb; j_hi += 1; end
      for ii in i:(i_hi - 1)
        ai = permA[ii]; acoord = A.keys[ai]; aval = A.vals[ai]
        for jj in j:(j_hi - 1)
          bj = permB[jj]; bcoord = B.keys[bj]; bval = B.vals[bj]
          ccoord = ntuple(t -> Int(src[t] == 0x01 ? acoord[pos[t]] : bcoord[pos[t]]), Val(NC))
          acc[ccoord] = get(acc, ccoord, zero(TC)) + convert(TC, aval * bval)
        end
      end
      i = i_hi; j = j_hi
    end
  end

  sizehint!(C.keys, length(acc)); sizehint!(C.vals, length(acc))
  @inbounds for (coord, v) in acc
    v == zero(TC) && continue
    push!(C.keys, coord)
    push!(C.vals, v)
  end
  C.dirty = true
  return C
end
