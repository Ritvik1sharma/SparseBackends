using LinearAlgebra
using SparseArrays

# # Get dense matrix view for a blocksparse block (no copy)
# @inline function _blkmat(A::blocksparse{T,N}, plin::Int) where {T,N}
#   rows = A.dims[N-1]
#   cols = A.dims[N]
#   off  = (plin - 1) * A.blksize
#   return reshape(@view(A.data[off+1 : off + A.blksize]), rows, cols)
# end

# # Iterate COO entries for a given prefix-state plin (no allocation)
# @inline function _coo_range(A::RestrictedCOO{T,N}, plin::Int) where {T,N}
#   lo = A.coo_segment[plin] + 1
#   hi = A.coo_segment[plin + 1]
#   return lo, hi
# end



function contract!(
    C::blocksparse{TC,NC},
    labelsC::AbstractVector,
    A::blocksparse{TA,NA},
    labelsA::AbstractVector,
    B::blocksparse{TB,NB},
    labelsB::AbstractVector
) where {TC,NC,TA,NA,TB,NB}

  @assert NA >= 2 && NB >= 2 && NC >= 2
  @assert length(labelsA) == NA
  @assert length(labelsB) == NB
  @assert length(labelsC) == NC

  # label -> axis maps
  mapA = Dict(labelsA[i] => i for i in 1:NA)
  mapB = Dict(labelsB[i] => i for i in 1:NB)
  mapC = Dict(labelsC[i] => i for i in 1:NC)

  # find reduced label: shared by A and B but not in C
  sharedAB = intersect(Set(labelsA), Set(labelsB))
  reduced = collect(setdiff(sharedAB, Set(labelsC)))
  @assert length(reduced) == 1 "MVP supports exactly 1 reduced label"
  red = reduced[1]

  a_red = mapA[red]
  b_red = mapB[red]
  
  if a_red > NA - 2 && b_red > N-2
    # Corresponds to lower index multiplication
    contract_bottom!(C, labelsC, A, labelsA, B, labelsB)
  elseif a_red <= NA - 2 && b_red <= N-2
    contract_upper!(C, labelsC, A, labelsA, B, labelsB)
  else
    error("reduction between higher and lower index levels is not supported")
  end  
end

# """
#     Multiply in scenario we have a shared index in the nupper N-2 levels of the 
#     two block sparse tensors. The shared index ios the N-2'th index of the two tensors.
#     For a block of A, B this is an outer prioduct where depending on the index value.
#     i.e. for example A[i, j, k, l] (k, l is dense) * B[a, j, b, c] (bc is dense) = 
#     C[i, a, k, b, l, c] (l, c is dense, the k and b indexes are promoted to upper levels)
# """
# function contract_upper!(
#     C::blocksparse{TC,NC},
#     labelsC::AbstractVector,
#     A::blocksparse{TA,NA},
#     labelsA::AbstractVector,
#     B::blocksparse{TB,NB},
#     labelsB::AbstractVector
# ) where {TC,NC,TA,NA,TB,NB}
#   mapA = Dict(labelsA[i] => i for i in 1:NA)
#   mapB = Dict(labelsB[i] => i for i in 1:NB)
#   mapC = Dict(labelsC[i] => i for i in 1:NC)
#   sharedAB = intersect(Set(labelsA), Set(labelsB))
#   reduced = collect(setdiff(sharedAB, Set(labelsC)))
#   a_red = mapA[reduced[1]]
#   b_red = mapB[reduced[1]]
#   @assert a_red === NA-2
#   @assert b_red === NB-2
#   @assert A.dims[a_red] === A.dims[b_red]


#   dimsC_prefix = [C.dims[i] for i in 1:NC-2]
#   npsC = prod(dimsC_prefix; init=1)


#   @assert length(C.data) == npsC * C.blksize "C.data size mismatch for full-prefix iteration"
#   prefixC_vals = Vector{Int}(undef, PC)
#   prefixA_vals = Vector{Int}(undef, PA)
#   prefixB_vals = Vector{Int}(undef, PB)
#   # Iterate over C blocks
#   @inbounds for plinC in 1:npsC
#     _decode_plin_to_prefix!(prefixC_vals, plinC, dimsC_prefix)
#     for a in 1:PA
#       prefixA_vals[a] = prefixC_vals[projA[a]]
#     end
#     for b in 1:PB
#       prefixB_vals[b] = prefixC_vals[projB[b]]
#     end
#     for red_idx in 1:A.dims[a_red]
#         prefixA_vals[PA] = red_idx
#         prefixB_vals[PB] = red_idx

#         # Diagonal constraint checks for A and B (if violated => treat as zero blocks)
#         if !_check_diag_pairs(prefixA_vals, A.diag_pairs) || !_check_diag_pairs(prefixB_vals, B.diag_pairs)
#             offC = (plinC - 1) * C.blksize
#             fill!(@view(C.data[offC+1 : offC + C.blksize]), zero(TC))
#             continue
#         end

#         plinA = _plin_from_prefix(prefixA_vals, [A.dims[i] for i in 1:PA])
#         plinB = _plin_from_prefix(prefixB_vals, [B.dims[i] for i in 1:PA])

#         @assert 1 <= plinA <= npsA
#         @assert 1 <= plinB <= npsB

#         # Outer multiply blocks
        
#         Ablk = _blkmat(A, plinA)  # m×k
#         Bblk = _blkmat(B, plinB)  # k×n
#         offC = (plinC - 1) * C.blksize
#         Cblk = reshape(@view(C.data[offC+1 : offC + C.blksize]), m, n)

#         mul!(Cblk, Ablk, Bblk)

#   error("Upper index multiplication not implemented yet.")
# end

function contract_bottom!(
    C::blocksparse{TC,NC},
    labelsC::AbstractVector,
    A::blocksparse{TA,NA},
    labelsA::AbstractVector,
    B::blocksparse{TB,NB},
    labelsB::AbstractVector
) where {TC,NC,TA,NA,TB,NB}

  # label -> axis maps
  mapA = Dict(labelsA[i] => i for i in 1:NA)
  mapB = Dict(labelsB[i] => i for i in 1:NB)
  mapC = Dict(labelsC[i] => i for i in 1:NC)

  # find reduced label: shared by A and B but not in C
  sharedAB = intersect(Set(labelsA), Set(labelsB))
  reduced = collect(setdiff(sharedAB, Set(labelsC)))
  red = reduced[1]

  a_red = mapA[red]
  b_red = mapB[red]

  # Enforce reduction is in matrix dims
  @assert a_red == NA "MVP: reduced label must be A axis NA"
  @assert b_red == NB-1 "MVP: reduced label must be B axis NB-1"

  # Output last-2 labels define matrix result axes
  out_row_lab = labelsC[NC-1]
  out_col_lab = labelsC[NC]

  @assert mapA[out_row_lab] == NA-1 "MVP: output row label must be A axis NA-1"
  @assert mapB[out_col_lab] == NB   "MVP: output col label must be B axis NB"

  # Dimension compatibility for the matrix multiply
  m = A.dims[NA-1]
  k = A.dims[NA]
  @assert k == B.dims[NB-1]
  n = B.dims[NB]

  # Output dims must match (prefix dims can differ from A/B!)
  @assert C.dims[NC-1] == m
  @assert C.dims[NC]   == n

  # --- Prefix label projections ---
  prefixA_labs = labelsA[1:NA-2]
  prefixB_labs = labelsB[1:NB-2]
  prefixC_labs = labelsC[1:NC-2]

  # Require that every A/B prefix label appears in C prefix labels (MVP)
  for lab in prefixA_labs
    @assert haskey(mapC, lab) && mapC[lab] <= NC-2 "MVP: A prefix label '$lab' must be in C prefix"
  end
  for lab in prefixB_labs
    @assert haskey(mapC, lab) && mapC[lab] <= NC-2 "MVP: B prefix label '$lab' must be in C prefix"
  end

  # Build projection maps: A prefix axis a -> C prefix axis idx
  # (indices within the prefix vectors, i.e. 1..PA/PC)
  PA = NA - 2
  PB = NB - 2
  PC = NC - 2

  projA = Vector{Int}(undef, PA)
  projB = Vector{Int}(undef, PB)

  @inbounds for a in 1:PA
    lab = prefixA_labs[a]
    projA[a] = mapC[lab]  # axis in full tensor; but since it's prefix, it's also prefix index
  end
  @inbounds for b in 1:PB
    lab = prefixB_labs[b]
    projB[b] = mapC[lab]
  end

  # Prefix dims vectors for linearization (full prefix, not reps-compressed)
  dimsC_prefix = [C.dims[i] for i in 1:PC]
  dimsA_prefix = [A.dims[i] for i in 1:PA]
  dimsB_prefix = [B.dims[i] for i in 1:PB]

  npsC = prod(dimsC_prefix; init=1)
  npsA = prod(dimsA_prefix; init=1)
  npsB = prod(dimsB_prefix; init=1)

  @assert length(C.data) == npsC * C.blksize "C.data size mismatch for full-prefix iteration"

  # Working buffers
  prefixC_vals = Vector{Int}(undef, PC)
  prefixA_vals = Vector{Int}(undef, PA)
  prefixB_vals = Vector{Int}(undef, PB)

  # Iterate over C blocks
  @inbounds for plinC in 1:npsC
    _decode_plin_to_prefix!(prefixC_vals, plinC, dimsC_prefix)

    for a in 1:PA
      prefixA_vals[a] = prefixC_vals[projA[a]]
    end
    for b in 1:PB
      prefixB_vals[b] = prefixC_vals[projB[b]]
    end

    # Diagonal constraint checks for A and B (if violated => treat as zero blocks)
    if !_check_diag_pairs(prefixA_vals, A.diag_pairs) || !_check_diag_pairs(prefixB_vals, B.diag_pairs)
      offC = (plinC - 1) * C.blksize
      fill!(@view(C.data[offC+1 : offC + C.blksize]), zero(TC))
      continue
    end

    plinA = _plin_from_prefix(prefixA_vals, dimsA_prefix)
    plinB = _plin_from_prefix(prefixB_vals, dimsB_prefix)

    @assert 1 <= plinA <= npsA
    @assert 1 <= plinB <= npsB

    # Multiply blocks
    Ablk = _blkmat(A, plinA)  # m×k
    Bblk = _blkmat(B, plinB)  # k×n
    offC = (plinC - 1) * C.blksize
    Cblk = reshape(@view(C.data[offC+1 : offC + C.blksize]), m, n)

    mul!(Cblk, Ablk, Bblk)
  end
  return C
end



# ============================================================
# Update kernels for the 3 requested cases
# ============================================================
# lift (k,l) => C dense is (b,c):   C[b,c] += A[k,l] * B[b,c]
@inline function _update_lift_kl!(Cblk, Ablk, Bblk, k_idx::Int, l_idx::Int)
  α = Ablk[k_idx, l_idx]
  @. Cblk = Cblk + α * Bblk
  return
end
# lift (b,c) => C dense is (k,l):   C[k,l] += B[b,c] * A[k,l]
@inline function _update_lift_bc!(Cblk, Ablk, Bblk, b_idx::Int, c_idx::Int)
  α = Bblk[b_idx, c_idx]
  @. Cblk = Cblk + α * Ablk
  return
end
# lift (k,b) => C dense is (l,c):   C[l,c] += A[k,:] ⊗ B[b,:]
@inline function _update_lift_kb!(Cblk, Ablk, Bblk, k_idx::Int, b_idx::Int)
  u = @view Ablk[k_idx, :]  # length l
  v = @view Bblk[b_idx, :]  # length c
  mul!(Cblk, reshape(u, length(u), 1), reshape(v, 1, length(v)),
       one(eltype(Cblk)), one(eltype(Cblk)))  # Cblk += u * v'
  return
end

"""
contract_upper_outer!(C, labelsC, A, labelsA, B, labelsB;
                      shared_label, lift_case)

Computes sum over `shared_label` which is an upper/prefix index (assumed at NA-2 and NB-2).

Block axes are canonical:
  A dense block dims = (k,l) = (NA-1, NA)
  B dense block dims = (b,c) = (NB-1, NB)

Supported lift_case values (exactly the ones you asked for):
  :lift_kl  => C dense is (b,c); promote k,l into C prefix
  :lift_bc  => C dense is (k,l); promote b,c into C prefix
  :lift_kb  => C dense is (l,c); promote k,b into C prefix

Note: last-2 dims can be swapped via permutedims(C, ...) outside this kernel.
"""
function contract_upper_outer!(
    C::blocksparse{TC,NC},
    labelsC::AbstractVector,
    A::blocksparse{TA,NA},
    labelsA::AbstractVector,
    B::blocksparse{TB,NB},
    labelsB::AbstractVector;
    shared_label,
    lift_case::Symbol
) where {TC,NC,TA,NA,TB,NB}

  @assert NA >= 2 && NB >= 2 && NC >= 2
  @assert length(labelsA) == NA
  @assert length(labelsB) == NB
  @assert length(labelsC) == NC

  mapA = Dict(labelsA[i] => i for i in 1:NA)
  mapB = Dict(labelsB[i] => i for i in 1:NB)
  mapC = Dict(labelsC[i] => i for i in 1:NC)

  @assert haskey(mapA, shared_label) && haskey(mapB, shared_label)
  a_sh = mapA[shared_label]
  b_sh = mapB[shared_label]
  @assert a_sh == NA-2
  @assert b_sh == NB-2
  @assert A.dims[a_sh] == B.dims[b_sh]

  # Canonical block dims
  kdim, ldim = A.dims[NA-1], A.dims[NA]
  bdim, cdim = B.dims[NB-1], B.dims[NB]

  # Identify the four block labels
  k_lab, l_lab = labelsA[NA-1], labelsA[NA]
  b_lab, c_lab = labelsB[NB-1], labelsB[NB]

  # Require: all A/B prefix labels except shared_label are present in C prefix
  for lab in prefixA_labs
    if lab == shared_label; continue; end
    @assert haskey(mapC, lab) && mapC[lab] <= PC
  end
  for lab in prefixB_labs
    if lab == shared_label; continue; end
    @assert haskey(mapC, lab) && mapC[lab] <= PC
  end

  # Determine which are dense in C (must be the last two labels)
  dense1_lab = labelsC[NC-1]
  dense2_lab = labelsC[NC]
  # Validate dense dims for the requested lift_case
  if lift_case == :lift_kl
    @assert dense1_lab == b_lab && dense2_lab == c_lab "lift_kl requires C dense last-2 = (b,c)"
    @assert C.dims[NC-1] == bdim && C.dims[NC] == cdim
    @assert haskey(mapC, k_lab) && mapC[k_lab] <= PC
    @assert haskey(mapC, l_lab) && mapC[l_lab] <= PC
  elseif lift_case == :lift_bc
    @assert dense1_lab == k_lab && dense2_lab == l_lab "lift_bc requires C dense last-2 = (k,l)"
    @assert C.dims[NC-1] == kdim && C.dims[NC] == ldim
    @assert haskey(mapC, b_lab) && mapC[b_lab] <= PC
    @assert haskey(mapC, c_lab) && mapC[c_lab] <= PC
  elseif lift_case == :lift_kb
    @assert dense1_lab == l_lab && dense2_lab == c_lab "lift_kb requires C dense last-2 = (l,c)"
    @assert C.dims[NC-1] == ldim && C.dims[NC] == cdim
    @assert haskey(mapC, k_lab) && mapC[k_lab] <= PC
    @assert haskey(mapC, b_lab) && mapC[b_lab] <= PC
  else
    throw(ArgumentError("Unsupported lift_case=$lift_case. Use :lift_kl, :lift_bc, or :lift_kb."))
  end

  # Prefix label sets
  PA, PB, PC = NA-2, NB-2, NC-2
  prefixA_labs = labelsA[1:PA]
  prefixB_labs = labelsB[1:PB]
  prefixC_labs = labelsC[1:PC]
  
  # Projection maps from C prefix -> A/B prefix (excluding shared_label filled in loop)
  projA = fill(0, PA)
  projB = fill(0, PB)
  @inbounds for a in 1:PA
    lab = prefixA_labs[a]
    projA[a] = (lab == shared_label) ? 0 : mapC[lab]
  end
  @inbounds for b in 1:PB
    lab = prefixB_labs[b]
    projB[b] = (lab == shared_label) ? 0 : mapC[lab]
  end

  dimsC_prefix = [C.dims[i] for i in 1:PC]
  dimsA_prefix = [A.dims[i] for i in 1:PA]
  dimsB_prefix = [B.dims[i] for i in 1:PB]

  npsC = prod(dimsC_prefix; init=1)
  npsA = prod(dimsA_prefix; init=1)
  npsB = prod(dimsB_prefix; init=1)
  @assert length(C.data) == npsC * C.blksize

  prefixC_vals = Vector{Int}(undef, PC)
  prefixA_vals = Vector{Int}(undef, PA)
  prefixB_vals = Vector{Int}(undef, PB)

  @inbounds for plinC in 1:npsC
    _decode_plin_to_prefix!(prefixC_vals, plinC, dimsC_prefix)

    # Build A/B prefix from C (shared_label is filled in inner loop)
    for a in 1:PA
      prefixA_vals[a] = projA[a] == 0 ? 1 : prefixC_vals[projA[a]]
    end
    for b in 1:PB
      prefixB_vals[b] = projB[b] == 0 ? 1 : prefixC_vals[projB[b]]
    end

    # Fetch lifted indices from C prefix (depending on case)
    k_idx = 1; l_idx = 1; b_idx = 1; c_idx = 1
    if lift_case == :lift_kl
      k_idx = prefixC_vals[mapC[k_lab]]
      l_idx = prefixC_vals[mapC[l_lab]]
      @assert 1 <= k_idx <= kdim
      @assert 1 <= l_idx <= ldim
    elseif lift_case == :lift_bc
      b_idx = prefixC_vals[mapC[b_lab]]
      c_idx = prefixC_vals[mapC[c_lab]]
      @assert 1 <= b_idx <= bdim
      @assert 1 <= c_idx <= cdim
    elseif lift_case == :lift_kb
      k_idx = prefixC_vals[mapC[k_lab]]
      b_idx = prefixC_vals[mapC[b_lab]]
      @assert 1 <= k_idx <= kdim
      @assert 1 <= b_idx <= bdim
    end

    # Output block view
    offC = (plinC - 1) * C.blksize
    Cblk = reshape(@view(C.data[offC+1 : offC + C.blksize]), C.dims[NC-1], C.dims[NC])
    fill!(Cblk, zero(TC))

    # Sum over shared upper index
    for jval in 1:A.dims[a_sh]
      prefixA_vals[a_sh] = jval
      prefixB_vals[b_sh] = jval

      # skip invalid diag constraints
      if !_check_diag_pairs(prefixA_vals, A.diag_pairs) || !_check_diag_pairs(prefixB_vals, B.diag_pairs)
        continue
      end

      plinA = _plin_from_prefix(prefixA_vals, dimsA_prefix)
      plinB = _plin_from_prefix(prefixB_vals, dimsB_prefix)
      @assert 1 <= plinA <= npsA
      @assert 1 <= plinB <= npsB

      Ablk = _blkmat(A, plinA)  # k×l
      Bblk = _blkmat(B, plinB)  # b×c

      if lift_case == :lift_kl
        _update_lift_kl!(Cblk, Ablk, Bblk, k_idx, l_idx)
      elseif lift_case == :lift_bc
        _update_lift_bc!(Cblk, Ablk, Bblk, b_idx, c_idx)
      else # :lift_kb
        _update_lift_kb!(Cblk, Ablk, Bblk, k_idx, b_idx)
      end
    end
  end
  return C
end


# function contract!(
#     C::blocksparse{TC,NC},
#     labelsC::AbstractVector,
#     A::restrictedcoo{TA,NA},
#     labelsA::AbstractVector,
#     B::blocksparse{TB,NB},
#     labelsB::AbstractVector
# ) where {TC,NC,TA,NA,TB,NB}

#   @assert NA >= 2 && NB >= 2 && NC >= 2
#   @assert length(labelsA) == NA
#   @assert length(labelsB) == NB
#   @assert length(labelsC) == NC

#   # label -> axis maps
#   mapA = Dict(labelsA[i] => i for i in 1:NA)
#   mapB = Dict(labelsB[i] => i for i in 1:NB)
#   mapC = Dict(labelsC[i] => i for i in 1:NC)

#   # find reduced label: shared by A and B but not in C
#   sharedAB = intersect(Set(labelsA), Set(labelsB))
#   reduced = collect(setdiff(sharedAB, Set(labelsC)))
#   @assert length(reduced) == 1 "MVP supports exactly 1 reduced label"
#   red = reduced[1]

#   a_red = mapA[red]
#   b_red = mapB[red]
  
#   if a_red > NA - 2 && b_red > N-2
#     # Corresponds to lower index multiplication
#     contract_bottom!(C, labelsC, A, labelsA, B, labelsB)
#   elseif a_red <= NA - 2 && b_red <= N-2
#     contract_upper!(C, labelsC, A, labelsA, B, labelsB)
#   else
#     error("reduction between higher and lower index levels is not supported")
#   end  
# end