############################################################################################################
using LinearAlgebra
const BlasFloat = LinearAlgebra.BlasFloat
const BLAS = LinearAlgebra.BLAS


@inline function _outer_add!(Cvec::AbstractVector{TC}, Avec, Bvec) where {TC}
  m = length(Avec)
  n = length(Bvec)
  @assert length(Cvec) == m*n
  @inbounds for j in 1:n
    bj = convert(TC, Bvec[j])
    coloff = (j - 1) * m
    @simd for i in 1:m
      Cvec[coloff + i] += convert(TC, Avec[i]) * bj
    end
  end
  return nothing
end

@inline function _move_dense_axis_to_last_perm(N2::Int, redpos::Int)
  v = Vector{Int}(undef, N2)
  k = 1
  @inbounds for d in 1:N2
    d == redpos && continue
    v[k] = d
    k += 1
  end
  v[N2] = redpos
  return v
end

@inline function _find_permutation_for_tensor(Tns::NewBlockSparseSorted{T,N,N2,P}, map::Dict, rlab) where {T,N,N2,P}
  axisr = map[rlab]
  if axisr <= P
    prefix = collect(1:P)
    perm_prefix = vcat(prefix[prefix .!= axisr], axisr)
    perm = vcat(perm_prefix, collect(P+1:N))
    return perm
  else
    perm = collect(1:P)
    perm = vcat(perm, P .+ _move_dense_axis_to_last_perm(N2, axisr - P))
    return perm
  end
end

@inline function _find_permutation_for_tensor(
    Tns::NewBlockSparseSorted{T,N,N2,P},
    map::Dict,
    rlabs::AbstractVector
) where {T,N,N2,P}
  # Partition rlabs by region (prefix vs dense), and keep order as provided
  pref = Int[]
  dens = Int[]
  @inbounds for lab in rlabs
    ax = map[lab]
    if ax <= P
      push!(pref, ax)
    else
      push!(dens, ax - P)  # dense-local axis 1..N2
    end
  end
  # ----- prefix perm: move those prefix axes to the end, stable
  prefix_axes = collect(1:P)
  keep_pref = [a for a in prefix_axes if !(a in Set(pref))]
  perm_prefix = vcat(keep_pref, pref)
  # ----- dense perm: start with identity, then move specified dense axes to the end, stable
  dense_axes = collect(1:N2)
  keep_dens = [a for a in dense_axes if !(a in Set(dens))]
  perm_dense_local = vcat(keep_dens, dens)  # local 1..N2
  # Combine into full perm (new axis order -> old axis positions)
  perm = vcat(perm_prefix, P .+ perm_dense_local)
  return perm
end


@inline function _permute_r_to_last_prefix(
    Tns::NewBlockSparseSorted{T,N,N2,P},
    labels::AbstractVector,
    map::Dict,
    rlab
) where {T,N,N2,P}
  # prefix length
  axisr = map[rlab]
  @assert axisr <= P "rlab must be in prefix to use prefix kernel"
  if axisr == P
    return Tns, labels, map, axisr
  end
  perm = _find_permutation_for_tensor(Tns, map, rlab)
  Tns = permutedims(Tns, perm)              # your permutedims sorts by prefix lin
  # permutedims!(Tns, perm)              # your permutedims sorts by prefix lin
  labelsp = labels[perm]
  mapp = Dict(lab => i for (i,lab) in enumerate(labelsp))
  return Tns, labelsp, mapp, P
end


# Generic rank-1 update into Cvec interpreted as (m×n) column-major matrix:
# Cmat[i,j] stored at Cvec[i + (j-1)*m]
@inline function _rank1_add_generic_AB!(
    Cvec::AbstractVector{TC}, α::TC,
    x, y   # x length m, y length n
) where {TC}
  m = length(x)
  n = length(y)
  @assert length(Cvec) == m*n
  @inbounds for j in 1:n
    yj = convert(TC, y[j])
    coloff = (j - 1) * m
    @simd for i in 1:m
      Cvec[coloff + i] += α * convert(TC, x[i]) * yj
    end
  end
  return nothing
end

# BLAS GER fast path: Cmat += α * x * y'
@inline function _rank1_add_blas!(
    Cmat::StridedMatrix{T}, α::T,
    x::StridedVector{T}, y::StridedVector{T}
) where {T<:BlasFloat}
  BLAS.ger!(α, x, y, Cmat)
  return nothing
end

function contract!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2B,PB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab
) where {TC,NC,N2C,TA,NA,N2A,TB,NB,N2B,PA,PB,PC}
  axisBr = mapB[rlab]
  if false  # SB_TRACE — flip to true here for debug output
    println("[SB_TRACE] contract_bs_bs.contract!  BS(blocks=", length(A.keys),
            ", PA=", PA, ", N2A=", N2A, ") × BS(blocks=", length(B.keys),
            ", PB=", PB, ", N2B=", N2B, ")  axisBr=", axisBr,
            (axisBr <= PB ? "  → prefix_outer" : "  → dense_bb"))
  end
  if axisBr <= PB
    @timeit TIMER "kbb.dispatch.prefix_outer_bb" begin
      A, labelsA, mapA, _ = _permute_r_to_last_prefix(A, labelsA, mapA, rlab)  # axisAr == PA
      B, labelsB, mapB, _ = _permute_r_to_last_prefix(B, labelsB, mapB, rlab)  # axisBr == PB
      return contract_prefix_outer_bb!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
  else
    error("CHECK IF ENCOUNTERED 22")

    @timeit TIMER "kbb.dispatch.dense_bb" begin
      return contract_dense_bb!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
    # return contract_dense_bb!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
  end
end


"""
    contract_prefix_outer_bb!(
        C, labelsC,
        A, labelsA,
        B, labelsB,
        mapA, mapB, rlab
    )

Assumes:
  * Exactly one shared label between A and B: rlab.
  * rlab is in prefix of both A and B and is reduced (rlab ∉ labelsC).
  * Dense labels are DISJOINT between A and B.
  * C dense labels (last N2C) are either [A_dense..., B_dense...] or [B_dense..., A_dense...]
    (this function supports both orders).
  * C prefix labels (first PC) are some ordering of (A_prefix{rlab}) ∪ (B_prefix{rlab}).

Result:
  Each matching pair of blocks (same r-value) contributes an outer product of dense payloads into C's block.
"""
function contract_prefix_outer_bb!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2B,PB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB,N2B,PB}

  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
  # Dense labels must be disjoint for outer-product kernel
  Adense = labelsA[PA+1:NA]
  Bdense = labelsB[PB+1:NB]
  Cdense = labelsC[PC+1:NC]
  @assert isempty(intersect(Set(Adense), Set(Bdense))) "outer kernel assumes disjoint dense labels"
  if Cdense == vcat(Adense, Bdense)
    dense_order = 1
  elseif Cdense == vcat(Bdense, Adense)
    dense_order = 2
  else
    error("C dense must be [A_dense...,B_dense...] or [B_dense..., A_dense...]")
  end
  @assert C.blksize == A.blksize * B.blksize "C.blksize must equal A.blksize * B.blksize"
  # --- Permute A,B so r is LAST prefix axis (slowest-changing) and keys are sorted accordingly
  axisAr, axisBr = mapA[rlab], mapB[rlab]
  @assert axisAr == PA
  @assert axisBr == PB
  # overwrite semantics
  empty!(C.keys); empty!(C.ids); empty!(C.data)
  src = Vector{Int}(undef, PC)
  for j in 1:PC
    lab = labelsC[j]
    if haskey(mapA, lab)
      src[j] = mapA[lab]
    else
      @assert haskey(mapB, lab) "Label $lab not found in A or B"
      src[j] = -mapB[lab]
    end
  end
  # --- Merge-join A and B on rv runs (rv is last prefix axis => contiguous runs)
  iA = firstindex(A.keys); nA = lastindex(A.keys)
  iB = firstindex(B.keys); nB = lastindex(B.keys)
  while iA <= nA && iB <= nB
    rvA = A.keys[iA][axisAr]
    rvB = B.keys[iB][axisBr]
    if rvA < rvB
      while iA <= nA && A.keys[iA][axisAr] == rvA
        iA += 1
      end
      continue
    elseif rvB < rvA
      while iB <= nB && B.keys[iB][axisBr] == rvB
        iB += 1
      end
      continue
    end
    # Find the matching iB values for all the same rvA values
    b_lo = iB
    while iB <= nB && B.keys[iB][axisBr] == rvA
      iB += 1
    end
    b_hi = iB - 1
    # rv match. Cross product of blocks in the run
    while iA <= nA && A.keys[iA][axisAr] == rvA
      akey = A.keys[iA]
      aid  = A.ids[iA]
      Avec = _block_view(A, aid)
      for iB2 in b_lo:b_hi
        bkey = B.keys[iB2]
        bid  = B.ids[iB2]
        Bvec = _block_view(B, bid)
        ckey = ntuple(Val(PC)) do j
          s = src[j]
          s > 0 ? akey[s] : bkey[-s]
        end
        cid  = _ensure_block!(C, ckey)
        Cvec = _block_view(C, cid)
        if dense_order == 1
          _outer_add!(Cvec, Avec, Bvec)   # C dense = [A_dense, B_dense]
        else
          _outer_add!(Cvec, Bvec, Avec)   # C dense = [B_dense, A_dense]
        end
      end
      iA += 1
    end
  end
  return C
end

@inline function _rank1_add_generic!(Cvec::AbstractVector{TC}, α::TC, x, y) where {TC}
  m = length(x)
  n = length(y)
  @assert length(Cvec) == m*n
  @inbounds for j in 1:n
    yj = convert(TC, y[j])
    coloff = (j - 1) * m
    @simd for i in 1:m
      Cvec[coloff + i] += α * convert(TC, x[i]) * yj
    end
  end
  return nothing
end


@inline function _slice_any_nonzero(vec, off::Int, len::Int)
  @inbounds for i in 1:len
    !iszero(vec[off + i]) && return true
  end
  return false
end


function contract_dense_bb!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2B,PB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab;
) where {TC,NC,N2C,TA,NA,N2A,TB,NB,N2B,PA,PB,PC}

  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
  axisAr = mapA[rlab]; @assert axisAr > PA "rlab must be in A dense tail"
  axisBr = mapB[rlab]; @assert axisBr > PB "rlab must be in B dense tail"
  
  if axisAr != NA
    permA_global = _find_permutation_for_tensor(A, mapA, rlab)
    # permA_global = vcat(collect(1:PA), PA .+ permA_dense)
    A = permutedims(A, permA_global)
    # permutedims!(A, permA_global)
    labelsA = labelsA[permA_global]
  end
  if axisBr != NB
    permB_global = _find_permutation_for_tensor(B, mapB, rlab) # redposB)
    # permB_global = vcat(collect(1:PB), PB .+ permB_dense)
    B = permutedims(B, permB_global)
    # permutedims!(B, permB_global)
    labelsB = labelsB[permB_global]
  end

  @assert labelsA[end] == rlab
  @assert labelsB[end] == rlab
  # ---- dims/chunks
  dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))
  dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2B))
  R = dimsA_dense[end]  # reduction size
  @assert dimsB_dense[end] == R
  chunkA, chunkB = Int(A.blksize / R), Int(B.blksize / R)
  # prod(dimsA_dense[1:end-1]; init=1)
  # chunkB = prod(dimsB_dense[1:end-1]; init=1)
  @assert N2C == (N2A-1) + (N2B-1)
  @assert C.blksize == chunkA * chunkB
  # ---- validate C dense tail order
  Awo = labelsA[PA+1:NA-1]
  Bwo = labelsB[PB+1:NB-1]
  dense_order =
    collect(labelsC[PC+1:NC]) == vcat(Awo, Bwo) ? :AB :
    collect(labelsC[PC+1:NC]) == vcat(Bwo, Awo) ? :BA :
    error("C dense tail must be [A_dense..., B_dense...] or [B_dense..., A_dense...] ", 
          "found ", collect(labelsC[PC+1:NC]), " expected ", vcat(Awo, Bwo), " or ", vcat(Bwo, Awo))
  # @assert collect(labelsC[PC+1:NC]) == expected_dense
  # ---- overwrite semantics
  empty!(C.keys); empty!(C.ids); empty!(C.data)
  # ---- prefix sourcing for ckey
  srcA = zeros(Int, PC)
  srcB = zeros(Int, PC)
  @inbounds for j in 1:PC
    lab = labelsC[j]
    if haskey(mapA, lab)
      ax = mapA[lab]; @assert ax <= PA
      srcA[j] = ax
    else
      ax = mapB[lab]; @assert ax <= PB
      srcB[j] = ax
    end
  end

  can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)
  α1 = one(TC)
  # ---- main
  @inbounds for ai in eachindex(A.keys)
    akey = A.keys[ai]
    aid  = A.ids[ai]
    Avec_full = _block_view(A, aid)   # length chunkA*R
    for bi in eachindex(B.keys)
      bkey = B.keys[bi]
      bid  = B.ids[bi]
      Bvec_full = _block_view(B, bid) # length chunkB*R
      ckey = ntuple(j -> (srcA[j] != 0 ? akey[srcA[j]] : bkey[srcB[j]]), Val(PC))
      created = false
      Cvec = Vector{TC}()  # dummy init; only valid if created==true
      Cmat = nothing

      if dense_order === :AB
        # C interpreted as (chunkA × chunkB) column-major
        for r in 1:R
          aoff = (r - 1) * chunkA
          boff = (r - 1) * chunkB
          _slice_any_nonzero(Avec_full, aoff, chunkA) || continue
          _slice_any_nonzero(Bvec_full, boff, chunkB) || continue

          if !created
            cid  = _ensure_block!(C, ckey)
            Cvec = _block_view(C, cid)
            created = true
            if can_blas
              Cmat = reshape(Cvec, chunkA, chunkB)
            end
          end
          As = @view Avec_full[aoff+1 : aoff+chunkA]
          Bs = @view Bvec_full[boff+1 : boff+chunkB]
          if can_blas
            _rank1_add_blas!(Cmat, α1, As, Bs)
          else
            _rank1_add_generic!(Cvec, α1, As, Bs)
          end
        end
      else
        # :BA : C interpreted as (chunkB × chunkA) column-major
        for r in 1:R
          aoff = (r - 1) * chunkA
          boff = (r - 1) * chunkB
          _slice_any_nonzero(Avec_full, aoff, chunkA) || continue
          _slice_any_nonzero(Bvec_full, boff, chunkB) || continue
          if !created
            cid  = _ensure_block!(C, ckey)
            Cvec = _block_view(C, cid)
            created = true
            if can_blas
              Cmat = reshape(Cvec, chunkB, chunkA)
            end
          end
          As = @view Avec_full[aoff+1 : aoff+chunkA]
          Bs = @view Bvec_full[boff+1 : boff+chunkB]
          if can_blas
            _rank1_add_blas!(Cmat, α1, Bs, As)  # swapped
          else
            _rank1_add_generic!(Cvec, α1, Bs, As)
          end
        end
      end
    end
  end
  return C
end


# -----------------------------------------
# Helpers: join-tuple compare + run advance
# -----------------------------------------

@inline function _cmp_join_tuple(akey, bkey, posA::Vector{Int}, posB::Vector{Int})
  # Keys are sorted column-major in `_prefix_lin`: position 1 is the
  # fastest-changing (least significant) axis, last position is slowest
  # (most significant). For merge-join iteration over sorted keys to work,
  # this comparison must be consistent with that sort order — compare
  # from the LAST shared position down to the first.
  isempty(posA) && return 0
  @inbounds for t in length(posA):-1:1
    av = akey[posA[t]]
    bv = bkey[posB[t]]
    if av < bv
      return -1
    elseif av > bv
      return 1
    end
  end
  return 0
end

@inline function _eq_on_positions(k0, k1, pos::Vector{Int})
  isempty(pos) && return true
  @inbounds for t in 1:length(pos)
    p = pos[t]
    if k0[p] != k1[p]
      return false
    end
  end
  return true
end

@inline function _advance_run(keys, i0::Int, n::Int, pos::Vector{Int})
  isempty(pos) && return n + 1
  k0 = @inbounds keys[i0]
  i  = i0 + 1
  @inbounds while i <= n
    _eq_on_positions(k0, keys[i], pos) || break
    i += 1
  end
  return i
end

# -----------------------------------------
# Determine if Cdense groups as A-then-B or B-then-A (no interleaving).
# Also returns desired keep orderings from Cdense.
# -----------------------------------------
@inline function _cdense_grouping_and_orders(Cdense, keepA::Vector, keepB::Vector)
  keepAset = Set(keepA)
  keepBset = Set(keepB)

  # classify each Cdense label origin; also validate membership
  originA = BitVector(undef, length(Cdense))
  @inbounds for i in 1:length(Cdense)
    lab = Cdense[i]
    if lab in keepAset
      originA[i] = true
    elseif lab in keepBset
      originA[i] = false
    else
      error("Cdense label $lab not found in keepA/keepB")
    end
  end

  # find grouping
  firstB = findfirst(!, originA)         # first B
  lastA  = findlast(identity, originA)   # last A
  firstA = findfirst(identity, originA)  # first A
  lastB  = findlast(!, originA)          # last B
  mode = :AthenB
  if firstB === nothing || firstA === nothing
    mode = :AthenB  # degenerate (only one side)
  elseif lastA < firstB
    mode = :AthenB
  elseif lastB < firstA
    mode = :BthenA
  else
    mode = :interleaved
  end
  desired_keepA = [lab for lab in Cdense if lab in keepAset]
  desired_keepB = [lab for lab in Cdense if lab in keepBset]
  return mode, desired_keepA, desired_keepB
end

# -----------------------------------------
# One-shot perm for A:
#  prefix: [keep_prefix..., shared_prefix...]
#  dense:  [desired_keepA..., red_dense...]
# No cross prefix/dense moves.
# -----------------------------------------
@inline function _find_perm_for_A_join_and_dense_order(
    ::NewBlockSparseSorted{T,NA,N2A,PA},
    labelsA::AbstractVector,
    mapA::Dict,
    shared_prefix::AbstractVector,
    desired_keepA::AbstractVector,
    red_dense::AbstractVector
) where {T,NA,N2A,PA}

  # prefix: move shared_prefix to end
  pref_axes = collect(1:PA)
  shared_pref_axes = Int[mapA[lab] for lab in shared_prefix]
  shared_pref_set  = Set(shared_pref_axes)
  keep_pref_axes   = [ax for ax in pref_axes if !(ax in shared_pref_set)]
  perm_prefix      = vcat(keep_pref_axes, shared_pref_axes)

  # dense local 1..N2A: [desired_keepA..., red_dense...]
  keep_axes = Int[(mapA[lab] - PA) for lab in desired_keepA]
  red_axes  = Int[(mapA[lab] - PA) for lab in red_dense]
  @assert length(keep_axes) + length(red_axes) == N2A

  perm_dense_local = vcat(keep_axes, red_axes)
  return vcat(perm_prefix, PA .+ perm_dense_local)
end

# -----------------------------------------
# One-shot perm for B:
#  prefix: [keep_prefix..., shared_prefix...]
#  dense:  [red_dense..., desired_keepB...]
# No cross prefix/dense moves.
# -----------------------------------------
@inline function _find_perm_for_B_join_and_dense_redfirst_order(
    ::NewBlockSparseSorted{T,NB,N2B,PB},
    labelsB::AbstractVector,
    mapB::Dict,
    shared_prefix::AbstractVector,
    red_dense::AbstractVector,
    desired_keepB::AbstractVector
) where {T,NB,N2B,PB}

  # prefix: move shared_prefix to end
  pref_axes = collect(1:PB)
  shared_pref_axes = Int[mapB[lab] for lab in shared_prefix]
  shared_pref_set  = Set(shared_pref_axes)
  keep_pref_axes   = [ax for ax in pref_axes if !(ax in shared_pref_set)]
  perm_prefix      = vcat(keep_pref_axes, shared_pref_axes)

  # dense local 1..N2B: [red_dense..., desired_keepB...]
  red_axes  = Int[(mapB[lab] - PB) for lab in red_dense]
  keep_axes = Int[(mapB[lab] - PB) for lab in desired_keepB]
  @assert length(red_axes) + length(keep_axes) == N2B

  perm_dense_local = vcat(red_axes, keep_axes)
  return vcat(perm_prefix, PB .+ perm_dense_local)
end

# -----------------------------------------
# Main contract: one permutedims! for A and one for B
# -----------------------------------------
function contract_shared!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2B,PB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    shared_labels::Vector;
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    allowed_keys_C::Union{Nothing,AbstractSet}=nothing,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB,N2B,PB}
  # BS×BS hint path not yet implemented; accept kwargs to keep dispatch consistent.
  # Original assertions still fire if Cdense doesn't fully match keepA0+keepB0.

  @timeit TIMER "kbb.contract_shared.label!" begin
  
    # shared labels are ALWAYS reduced
    @inbounds for lab in shared_labels
      @assert !(lab in labelsC) "shared label $lab must be reduced (must not appear in labelsC)"
    end

    # -----------------------------
    # 0) Classify shared labels
    # -----------------------------
    shared_prefix = eltype(shared_labels)[]
    shared_dense  = eltype(shared_labels)[]
    @inbounds for lab in shared_labels
      @assert haskey(mapA, lab) && haskey(mapB, lab) "shared label $lab must exist in both A and B"
      a_pos = mapA[lab]; b_pos = mapB[lab]
      a_pref = a_pos <= PA; b_pref = b_pos <= PB
      if a_pref && b_pref
        push!(shared_prefix, lab)
      elseif (!a_pref) && (!b_pref)
        push!(shared_dense, lab)
      else
        # Detailed diagnostic on the cross failure.
        println("\n[contract_bs_bs CROSS-FAIL DIAG]")
        println("  shared label that crosses: ", lab, "  A_pos=", a_pos, " (PA=", PA, ")  B_pos=", b_pos, " (PB=", PB, ")")
        println("  A labels (all): ", labelsA)
        println("  B labels (all): ", labelsB)
        println("  C labels (all): ", labelsC)
        println("  shared_labels  : ", shared_labels)
        error("shared label $lab crosses prefix/dense (A pos=$a_pos, B pos=$b_pos); not supported")
      end
    end
  end

  @timeit TIMER "kbb.contract_shared.permute!" begin
    # -----------------------------
    # 1) Compute keep/red sets from CURRENT labels (before permute)
    #    red_dense order = A dense order filtered by shared_dense
    # -----------------------------
    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB[PB+1:NB]
    redset  = Set(shared_dense)

    red_dense = [lab for lab in Adense0 if lab in redset]         # reduction order (from A)
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]      # A kept dense labels
    keepB0    = [lab for lab in Bdense0 if !(lab in redset)]      # B kept dense labels

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved && error("Cdense interleaves A/B kept dims; would require permuting C (unsupported; sort/fallback)")

    # sanity: desired orders must include all kept dims exactly once
    @assert length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0)
    @assert length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)

    # -----------------------------
    # 2) Build ONE perm for A and ONE perm for B that satisfy BOTH:
    #    - join-friendly prefix layout
    #    - GEMM-friendly dense layout
    #    - C dense order within each group (A-keep or B-keep)
    # -----------------------------
    permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)
    permB = _find_perm_for_B_join_and_dense_redfirst_order(B, labelsB, mapB, shared_prefix, red_dense, desired_keepB)
    A = permutedims(A, permA)
    B = permutedims(B, permB)
    # permutedims!(A, permA)
    # permutedims!(B, permB)

    labelsA = labelsA[permA]
    labelsB = labelsB[permB]
    mapA = Dict(l => i for (i,l) in enumerate(labelsA))
    mapB = Dict(l => i for (i,l) in enumerate(labelsB))

    keepA = desired_keepA
    keepB = desired_keepB
    n_keepA = length(keepA)
    n_keepB = length(keepB)
    n_red   = length(red_dense)
  end

  @timeit TIMER "kbb.contract_shared.join_and_contract!" begin
    # -----------------------------
    # 3) Build C prefix key assembly plan
    # -----------------------------
    src = Vector{Int}(undef, PC)  # +i => A key axis i; -i => B key axis i
    @inbounds for j in 1:PC
      lab = labelsC[j]
      if haskey(mapA, lab) && mapA[lab] <= PA
        src[j] = mapA[lab]
      elseif haskey(mapB, lab) && mapB[lab] <= PB
        src[j] = -mapB[lab]
      else
        error("C prefix label $lab must come from A/B prefix")
      end
    end
    join_posA = Int[mapA[lab] for lab in shared_prefix]
    join_posB = Int[mapB[lab] for lab in shared_prefix]
  end

  @timeit TIMER "kbb.contract_shared.dense_contract!" begin
    # -----------------------------
    # 4) Dense contraction shapes
    # -----------------------------
    dimsA_dense = A.dims[PA+1:NA]  # [keepA..., red...]
    dimsB_dense = B.dims[PB+1:NB]  # [red..., keepB...]

    M = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
    K = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+1:end])
    N = (n_keepB == 0) ? 1 : prod(dimsB_dense[n_red+1:end])

    if n_red > 0
      @assert K == prod(dimsB_dense[1:n_red]) "Reduced dense extents mismatch between A and B"
    end
    @assert C.blksize == M * N "C.blksize must equal M*N (got $(C.blksize), expected $(M*N))"

    empty!(C.keys); empty!(C.ids); empty!(C.data)
    iA = firstindex(A.keys); nA = lastindex(A.keys)
    iB = firstindex(B.keys); nB = lastindex(B.keys)

    @inbounds while iA <= nA && iB <= nB
      cmp = _cmp_join_tuple(A.keys[iA], B.keys[iB], join_posA, join_posB)
      if cmp < 0
        iA = _advance_run(A.keys, iA, nA, join_posA)
        continue
      elseif cmp > 0
        iB = _advance_run(B.keys, iB, nB, join_posB)
        continue
      end

      iA2 = _advance_run(A.keys, iA, nA, join_posA)
      iB2 = _advance_run(B.keys, iB, nB, join_posB)

      for ii in iA:(iA2-1)
        akey = A.keys[ii]
        Avec = _block_view(A, A.ids[ii])
        Amat = reshape(Avec, M, K)

        for jj in iB:(iB2-1)
          bkey = B.keys[jj]
          Bvec = _block_view(B, B.ids[jj])
          Bmat = reshape(Bvec, K, N)
          ckey = ntuple(Val(PC)) do j
            s = src[j]
            s > 0 ? akey[s] : bkey[-s]
          end
          cid  = _ensure_block!(C, ckey)
          Cvec = _block_view(C, cid)
          @timeit TIMER "kbb.contract_shared.gemm" begin
            if mode == :AthenB
              # Cdense == [keepA..., keepB...] with your desired within-group orders
              Cmat = reshape(Cvec, M, N)
              mul!(Cmat, Amat, Bmat, one(TC), one(TC))
            else
              # Cdense == [keepB..., keepA...]
              Cmat = reshape(Cvec, N, M)
              mul!(Cmat, transpose(Bmat), transpose(Amat), one(TC), one(TC))
            end
          end
        end
      end
      iA = iA2
      iB = iB2
    end
  end
  return C
end

# Contract two BlockSparse tensors into a fully dense output array (P_C = 0).
# Uses a temporary P=0 NewBlockSparseSorted as the accumulator and delegates
# to the existing contract! (which handles P_C=0 via contract_shared!),
# then copies the single resulting block into the output array C.
function contract_bs_bs_to_dense!(
    C::AbstractArray{TC},
    labelsC::AbstractVector{Label},
    A::NewBlockSparseSorted{TA,NA,NA2,PA},
    labelsA::AbstractVector{Label},
    B::NewBlockSparseSorted{TB,NB,NB2,PB},
    labelsB::AbstractVector{Label},
) where {TC,TA,NA,NA2,PA,TB,NB,NB2,PB}
    NC    = ndims(C)
    dimsC = ntuple(i -> size(C, i), Val(NC))
    # P=0 all-dense sink: N2=NC, P=0. Scalar (NC=0) requires the relaxed assertion (N==0 && N2==0).
    C_bs  = NewBlockSparseSorted{TC, NC, NC}(dimsC)
    contract!(C_bs, labelsC, A, labelsA, B, labelsB)
    if !isempty(C_bs.keys)
        blk = _block_view(C_bs, C_bs.ids[1])
        NC == 0 ? (C[] = blk[1]) : copyto!(C, reshape(blk, dimsC))
    end
    return C
end
