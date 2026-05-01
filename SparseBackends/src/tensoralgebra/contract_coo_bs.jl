using LinearAlgebra
using SparseArrays

@inline function _scale_add_block!(Cvec::AbstractVector{TC}, α::TC, Bvec) where {TC}
  @assert length(Cvec) == length(Bvec)
  @inbounds @simd for i in eachindex(Cvec, Bvec)
    Cvec[i] += α * Bvec[i]
  end
  return nothing
end

const Label = NTuple{2,UInt64}  # (id, plev)

function contract!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::Vector{Label}, # AbstractVector,
    A::COOTensor{TA,NA},
    labelsA::Vector{Label}, # AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2,PB},
    labelsB::Vector{Label}, # AbstractVector,
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    rlab::Label
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2,PB}
  # PB, PC = NB - N2, NC - N2
  # @assert PB >= 0 && PC >= 0
  # r must be in B's PREFIX axes for this kernel
  axisBr = mapB[rlab]
  if get(ENV, "SB_TRACE", "0") == "1"
    println("[SB_TRACE] contract_coo_bs.contract!  COO(nnz=", length(A.keys),
            ", dims=", A.dims, ") × BS(blocks=", length(B.keys),
            ", PB=", PB, ", N2=", N2, ")  axisBr=", axisBr,
            (axisBr <= PB ? "  → prefix" : "  → dense_index"))
  end
  if axisBr <= PB
    # ensure_rlab_last_and_sorted!(A, labelsA, mapA, rlab)
    A, labelsA, mapA = ensure_rlab_last_and_sorted(A, labelsA, rlab)
    B, labelsB, mapB = _permute_r_to_last_prefix(B, labelsB, mapB, rlab)
    # B, labelsB, mapB, _ = _permute_r_to_last_prefix!(B, labelsB, mapB, rlab)
    return contract_prefix!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
  else
    A, labelsA, mapA = ensure_rlab_last_and_sorted(A, labelsA, rlab)
    # ensure_rlab_last_and_sorted!(A, labelsA, mapA, rlab)
    return contract_dense_index!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
  end  
end

"""
  Contract logic if the reduction dimension is in the prefix of the two tensors
"""
function contract_prefix!(
    C::NewBlockSparseSorted{TC,NC,N2,PC},
    labelsC::Vector{Label},
    A::COOTensor{TA,NA},
    labelsA::Vector{Label},
    B::NewBlockSparseSorted{TB,NB,N2,PB},
    labelsB::Vector{Label},
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    rlab
) where {TC,NC,N2,PC,TA,NA,TB,NB,PB}
  @assert PC == NA + PB - 2  # output prefix dims
  # println("In contract_prefix! ", A.keys, "|||", B.keys)
  # overwrite
  empty!(C.keys); empty!(C.ids); empty!(C.data)
  axisAr = mapA[rlab]                # should be NA if you permuted A to make r last
  axisBr = mapB[rlab]                # should be PB if you permuted B to make r last prefix
  # println("axisAr = ", axisAr, ", axisBr = ", axisBr, " map ", mapA, " ", mapB, "\n")
  @assert axisBr <= PB
  iA = firstindex(A.keys)
  iB = firstindex(B.keys)
  nA = lastindex(A.keys)
  nB = lastindex(B.keys)
  while iA <= nA && iB <= nB
    rA = A.keys[iA][axisAr]
    rB = B.keys[iB][axisBr]
    # println("Comparing rA=$rA, rB=$rB, for keya ", A.keys[iA], " and keyb ", B.keys[iB])
    if rA < rB
      # advance A to next r-run
      while iA <= nA && A.keys[iA][axisAr] == rA
        iA += 1
      end
    elseif rB < rA
      # advance B to next r-run
      while iB <= nB && B.keys[iB][axisBr] == rB
        iB += 1
      end
    else
      # matching r: get run bounds
      rv = rA
      b_lo = iB
      while iB <= nB && B.keys[iB][axisBr] == rv
        iB += 1
      end
      b_hi = iB - 1
      # cross product of runs
      while iA <= nA && A.keys[iA][axisAr] == rv
        # @inbounds for ia in a_lo:a_hi
        acoord = A.keys[iA]
        α = convert(TC, A.vals[iA])
        for iB2 in b_lo:b_hi
          bkey = B.keys[iB2]
          bid  = B.ids[iB2]
          ckey = ntuple(j -> begin
            lab = labelsC[j]
            if haskey(mapA, lab)
              acoord[mapA[lab]]
            else
              ax = mapB[lab]
              @assert ax <= PB
              bkey[ax]
            end
          end, Val(PC))
          cid  = _ensure_block!(C, ckey)
          Cvec = _block_view(C, cid)
          Bvec = _block_view(B, bid)
          _scale_add_block!(Cvec, α, Bvec)
        end
        iA += 1
      end
    end
  end
  return C
end


@inline function _scaled_chunk_to_scratch!(
    tmp::AbstractVector{TC}, α::TC, Bvec, chunklen::Int, rv::Int
) where {TC}
  off = (rv - 1) * chunklen
  anynz = false
  @inbounds @simd for t in 1:chunklen
    v = α * Bvec[off + t]
    tmp[t] = v
    anynz |= (v != 0)
  end
  return anynz
end

function contract_dense_index!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::Vector{Label}, # AbstractVector,
    A::COOTensor{TA,NA},
    labelsA::Vector{Label}, # AbstractVector,
    B::NewBlockSparseSorted{TB,NB,N2,PB},
    labelsB::Vector{Label},
    mapA::Dict{Label,Int},
    mapB::Dict{Label,Int},
    rlab
) where {TC,NC,N2C,PC,TA,NA,TB,NB,N2,PB}
  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
  # PB = NB - N2
  # PC = NC - N2C
  @assert N2C == N2 - 1
  # r must be in B dense tail
  axisBr = mapB[rlab]
  @assert axisBr > PB "rlab must be in B dense tail"
  redpos = axisBr - PB
  axisAr = mapA[rlab]
  @assert axisAr == NA "Expected A to have rlab as last axis already (axisAr == NA)"
  # ---- permute B dense tail so r becomes last dense axis (contiguous rv chunk)
  Bp = B
  labelsB_dense = collect(labelsB[PB+1:NB])
  perm_dense = vcat([d for d in 1:N2 if d != redpos], redpos)

  labelsB_dense_p = labelsB_dense
  if redpos != N2
    perm_global = vcat(collect(1:PB), PB .+ perm_dense)
    Bp = permutedims(B, perm_global)
    labelsB_dense_p = labelsB_dense[perm_dense]
  end
  @assert labelsB_dense_p[end] == rlab

  # Dense dims after permute
  dimsB_dense_p = ntuple(i -> Bp.dims[PB+i], Val(N2))
  dim_r    = dimsB_dense_p[end]
  chunklen = prod(dimsB_dense_p[1:end-1]; init=1)

  # Sanity: C dense tail matches B dense tail without r
  expectedC_dense = labelsB_dense_p[1:end-1]
  @assert collect(labelsC[PC+1:NC]) == expectedC_dense
  expectedC_dims = ntuple(i -> dimsB_dense_p[i], Val(N2C))
  @assert ntuple(i -> C.dims[PC+i], Val(N2C)) == expectedC_dims
  @assert C.blksize == chunklen
  @assert Bp.blksize == chunklen * dim_r
  # overwrite semantics
  empty!(C.keys); empty!(C.ids); empty!(C.data)
  # For each C prefix position j: take from A (srcA[j]!=0) or from B prefix (srcB[j]!=0)
  srcA = zeros(Int, PC)
  srcB = zeros(Int, PC)
  @inbounds for j in 1:PC
    lab = labelsC[j]
    if haskey(mapA, lab)
      srcA[j] = mapA[lab]
    else
      ax = mapB[lab]
      @assert ax <= PB "Label $lab in C prefix must come from B prefix (not dense)"
      srcB[j] = ax
    end
  end
  # Scan A once by contiguous r-runs (r is last axis)
  i = firstindex(A.keys)
  n = lastindex(A.keys)
  while i <= n
    rv = A.keys[i][NA]
    # while i <= n && A.keys[i][NA] == rv  
    acoord = A.keys[i]
    α = convert(TC, A.vals[i])
    # println("Processing r-value rv = $rv with accord = $acoord")
    @assert 1 <= rv <= dim_r "Reduction index rv=$rv out of bounds for dim_r=$dim_r" 
      # For this r-run, every B block contributes its rv-slice (cannot filter by rv since r is dense)
    @inbounds for bi in eachindex(Bp.keys)
      bkey = Bp.keys[bi]
      bid  = Bp.ids[bi]
      Bvec = _block_view(Bp, bid)
      # Build C prefix key directly from (acoord, bkey)
      ckey = ntuple(j -> begin
        axA = srcA[j]
        if axA != 0
          acoord[axA]
        else
          axB = srcB[j]
          @assert axB != 0
          bkey[axB]
        end
      end, Val(PC))

      tmp = zeros(TC, chunklen)  # ideally reuse, don’t allocate per-iteration
      anynz = _scaled_chunk_to_scratch!(tmp, α, Bvec, chunklen, rv)
      anynz || continue
      cid  = _ensure_block!(C, ckey)
      Cvec = _block_view(C, cid)
      @inbounds @simd for t in 1:chunklen
        Cvec[t] += tmp[t]
      end
    end
    # println("")      
    i += 1
  end
  return C
end

  # cid  = _ensure_block!(C, ckey)
  # Cvec = _block_view(C, cid)
  # print("bkey is ", bkey, " ckey is ", ckey, "\t")
  # _add_scaled_chunk!(Cvec, α, Bvec, chunklen, rv)

  # while i <= n
  #   rv = A.keys[i][NA]
  #   lo = i
  #   while i <= n && A.keys[i][NA] == rv
  #     i += 1
  #   end
  #   hi = i - 1
  #   if (1 <= rv <= dim_r)
  #     # For this r-run, every B block contributes its rv-slice (cannot filter by rv since r is dense)
  #     @inbounds for bi in eachindex(Bp.keys)
  #       bkey = Bp.keys[bi]
  #       bid  = Bp.ids[bi]
  #       Bvec = _block_view(Bp, bid)
  #       for ia in lo:hi
  #         acoord = A.keys[ia]
  #         α = convert(TC, A.vals[ia])
  #         # Build C prefix key directly from (acoord, bkey)
  #         ckey = ntuple(j -> begin
  #           axA = srcA[j]
  #           if axA != 0
  #             acoord[axA]
  #           else
  #             axB = srcB[j]
  #             @assert axB != 0
  #             bkey[axB]
  #           end
  #         end, Val(PC))
  #         cid  = _ensure_block!(C, ckey)
  #         Cvec = _block_view(C, cid)
  #         _add_scaled_chunk!(Cvec, α, Bvec, chunklen, rv)
  #       end
  #     end
  #   end
  # end


  

# # @inline function _permute_B_make_r_first_prefix!(
# #     B::NewBlockSparseSorted{T,NB,N2,P},
# #     labelsB::AbstractVector,
# #     mapB::Dict,
# #     rlab
# # ) where {T,NB,N2,P}
# #   PB = NB - N2
# #   axisBr = mapB[rlab]
# #   if axisBr > PB
# #     return B, labelsB, mapB, axisBr
# #   end
# #   if axisBr == 1
# #     return B, labelsB, mapB, axisBr
# #   end
# #   perm_prefix = vcat(axisBr, collect(1:PB)[collect(1:PB) .!= axisBr])
# #   perm = vcat(perm_prefix, collect(PB+1:NB))
# #   B = permutedims(B, perm)
# #   labelsB = labelsB[perm]
# #   mapB = Dict(lab => i for (i, lab) in enumerate(labelsB))
# #   return B, labelsB, mapB, 1
# # end

# function _permute_A_make_r_last_then_sort!(A, labelsA::AbstractVector, mapA::Dict, rlab)
#   # println("rpos and ", rlab, " in labelsA is ", mapA[rlab], "\n")
#   NA = length(labelsA)
#   rpos = mapA[rlab]
#   # println("A.keys before permute ", A.keys, "\n")
#   if rpos == NA
#     sort!(A)
#     return A, labelsA, mapA, collect(1:NA)
#   end
#   perm = vcat([d for d in 1:NA if d != rpos], rpos)
#   permutedims!(A, perm)
#   # println("A.keys after permute ", A.keys, "\n")
#   sort!(A)
#   labelsA = labelsA[perm]
#   mapA = Dict(lab => i for (i, lab) in enumerate(labelsA))
#   return A, labelsA, mapA, perm
# end


# # @inline function _permute_B_make_r_last_prefix!(
# #     B::NewBlockSparseSorted{T,NB,N2,P},
# #     labelsB::AbstractVector,
# #     mapB::Dict,
# #     rlab
# # ) where {T,NB,N2,P}
# #   PB = NB - N2
# #   axisBr = mapB[rlab]
# #   if axisBr > PB
# #     return B, labelsB, mapB, axisBr
# #   end
# #   if axisBr == PB
# #     return B, labelsB, mapB, axisBr
# #   end
# #   perm_prefix = vcat(collect(1:PB)[collect(1:PB) .!= axisBr], axisBr)  # move to end
# #   perm = vcat(perm_prefix, collect(PB+1:NB))
# #   permutedims!(B, perm)  # your permutedims already sorts blocks by prefix lin
# #   labelsB = labelsB[perm]
# #   mapB = Dict(lab => i for (i,lab) in enumerate(labelsB))
# #   return B, labelsB, mapB, PB
# # end

# # --------------------------
# # Utilities
# # --------------------------
# # Cvec += α * Bchunk  (Bchunk is contiguous chunklen-long slice)
# @inline function _add_scaled_chunk!(
#     Cvec::AbstractVector{TC},
#     α::TC,
#     Bvec::AbstractVector{TB},
#     chunklen::Int,
#     redval::Int
# ) where {TC,TB}
#   off = (redval - 1) * chunklen
#   @inbounds @simd for i in 1:chunklen
#     Cvec[i] += α * Bvec[off + i]
#   end
#   return nothing
# end

# """
#   Contract logic if the reduction dimension is in the dense part of B
# """
# function contract_dense_optimized!(
#     C::NewBlockSparseSorted{TC,NC,N2C},
#     labelsC::AbstractVector,
#     A::COOTensor{TA,NA},
#     labelsA::AbstractVector,
#     B::NewBlockSparseSorted{TB,NB,N2},
#     labelsB::AbstractVector,
#     mapA::Dict,
#     mapB::Dict,
#     rlab
# ) where {TC,NC,N2C,TA,NA,TB,NB,N2}

#   @assert !(rlab in labelsC)
#   PB = NB - N2
#   PC = NC - N2C
#   @assert N2C == N2 - 1

#   axisBr = mapB[rlab]
#   @assert axisBr > PB
#   redpos = axisBr - PB

#   # ---- permute B dense tail so rlab becomes last dense axis
#   Bp = B
#   labelsB_dense = collect(labelsB[PB+1:NB])
#   perm_dense = vcat([d for d in 1:N2 if d != redpos], redpos)

#   labelsB_dense_p = labelsB_dense
#   if redpos != N2
#     perm_global = vcat(collect(1:PB), PB .+ perm_dense)
#     Bp = permutedims(B, perm_global)
#     labelsB_dense_p = labelsB_dense[perm_dense]
#   end
#   @assert labelsB_dense_p[end] == rlab

#   dimsB_dense_p = ntuple(i -> Bp.dims[PB+i], Val(N2))
#   dim_r    = dimsB_dense_p[end]
#   chunklen = prod(dimsB_dense_p[1:end-1]; init=1)

#   expectedC_dense = labelsB_dense_p[1:end-1]
#   @assert collect(labelsC[PC+1:NC]) == expectedC_dense
#   expectedC_dims = ntuple(i -> dimsB_dense_p[i], Val(N2C))
#   @assert ntuple(i -> C.dims[PC+i], Val(N2C)) == expectedC_dims

#   @assert C.blksize == chunklen
#   @assert Bp.blksize == chunklen * dim_r

#   empty!(C.keys); empty!(C.ids); empty!(C.data)

#   # Precompute prefix source mapping
#   srcA = zeros(Int, PC)
#   srcB = zeros(Int, PC)
#   @inbounds for j in 1:PC
#     lab = labelsC[j]
#     if haskey(mapA, lab)
#       srcA[j] = mapA[lab]
#     else
#       ax = mapB[lab]
#       @assert ax <= PB
#       srcB[j] = ax
#     end
#   end

#   # groups_by_r[rv] : Dict of A-only part of prefix -> accumulated α
#   groups_by_r = [Dict{NTuple{PC,Int},TC}() for _ in 1:dim_r]

#   axisAr = mapA[rlab]
#   @inbounds for i in eachindex(A.keys)
#     acoord = A.keys[i]
#     avalA  = A.vals[i]
#     rv = acoord[axisAr]
#     (1 <= rv <= dim_r) || continue

#     ckeyA = ntuple(j -> begin
#       axA = srcA[j]
#       axA == 0 ? 0 : acoord[axA]
#     end, Val(PC))

#     d = groups_by_r[rv]
#     d[ckeyA] = get(d, ckeyA, zero(TC)) + convert(TC, avalA)
#   end

#   # Precompute only nonempty rv’s + convert dicts to vectors for fast iteration
#   nonempty_rvs = Int[]
#   group_pairs  = Vector{Vector{Pair{NTuple{PC,Int},TC}}}(undef, dim_r)
#   for rv in 1:dim_r
#     d = groups_by_r[rv]
#     if isempty(d)
#       group_pairs[rv] = Pair{NTuple{PC,Int},TC}[]
#     else
#       push!(nonempty_rvs, rv)
#       group_pairs[rv] = collect(pairs(d))
#     end
#   end

#   @inbounds for bi in eachindex(Bp.keys)
#     bkey = Bp.keys[bi]
#     bid  = Bp.ids[bi]
#     Bvec = _block_view(Bp, bid)

#     bpart = ntuple(j -> begin
#       axB = srcB[j]
#       axB == 0 ? 0 : bkey[axB]
#     end, Val(PC))

#     for rv in nonempty_rvs
#       for pr in group_pairs[rv]
#         ckeyA = pr.first
#         α     = pr.second

#         ckey = ntuple(j -> begin
#           va = ckeyA[j]
#           va != 0 ? va : bpart[j]
#         end, Val(PC))

#         cid  = _ensure_block!(C, ckey)
#         Cvec = _block_view(C, cid)
#         _add_scaled_chunk!(Cvec, α, Bvec, chunklen, rv)
#       end
#     end
#   end

#   return C
# end


# function contract_dense_optimized!(
#     C::NewBlockSparseSorted{TC,NC,N2C},
#     labelsC::AbstractVector,
#     A::COOTensor{TA,NA},
#     labelsA::AbstractVector,
#     B::NewBlockSparseSorted{TB,NB,N2},
#     labelsB::AbstractVector,
#     mapA::Dict,
#     mapB::Dict,
#     rlab
# ) where {TC,NC,N2C,TA,NA,TB,NB,N2}

#   @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
#   PB = NB - N2
#   PC = NC - N2C
#   @assert N2C == N2 - 1

#   # rlab must be in B dense tail
#   axisBr = mapB[rlab]
#   @assert axisBr > PB "rlab must be in B dense tail"
#   redpos = axisBr - PB

#   # ---- permute B dense tail so rlab becomes last dense axis
#   Bp = B
#   labelsB_dense = collect(labelsB[PB+1:NB])
#   perm_dense = vcat([d for d in 1:N2 if d != redpos], [redpos])

#   labelsB_dense_p = labelsB_dense
#   if redpos != N2
#     perm_global = vcat(collect(1:PB), PB .+ perm_dense)
#     Bp = permutedims(B, perm_global)
#     labelsB_dense_p = labelsB_dense[perm_dense]
#   end

#   @assert labelsB_dense_p[end] == rlab

#   # Dense dims after permute
#   dimsB_dense_p = ntuple(i -> Bp.dims[PB+i], Val(N2))
#   dim_r   = dimsB_dense_p[end]
#   chunklen = prod(dimsB_dense_p[1:end-1]; init=1)  # slice length

#   # C dense labels must equal B dense labels with rlab removed (after permute)
#   expectedC_dense = labelsB_dense_p[1:end-1]
#   @assert collect(labelsC[PC+1:NC]) == expectedC_dense

#   # C dense dims must match B dense dims with last removed
#   expectedC_dims = ntuple(i -> dimsB_dense_p[i], Val(N2C))
#   @assert ntuple(i -> C.dims[PC+i], Val(N2C)) == expectedC_dims
#   @assert C.blksize == chunklen
#   @assert Bp.blksize == chunklen * dim_r

#   # Clear output
#   empty!(C.keys); empty!(C.ids); empty!(C.data)

#   # Precompute which C prefix positions come from A vs B prefix
#   srcA = zeros(Int, PC)
#   srcB = zeros(Int, PC)
#   @inbounds for j in 1:PC
#     lab = labelsC[j]
#     if haskey(mapA, lab)
#       srcA[j] = mapA[lab]
#     else
#       ax = mapB[lab]
#       @assert ax <= PB "Label $lab in C prefix must come from B prefix"
#       srcB[j] = ax
#     end
#   end

#   # Group A coefficients by rv and A-part of ckey (zeros where B will fill)
#   groups_by_r = [Dict{NTuple{PC,Int},TC}() for _ in 1:dim_r]
#   @inbounds for (acoord, avalA) in A.data
#     rv = acoord[mapA[rlab]]
#     (1 <= rv <= dim_r) || continue
#     ckeyA = ntuple(j -> begin
#       axA = srcA[j]
#       axA == 0 ? 0 : acoord[axA]
#     end, Val(PC))
#     d = groups_by_r[rv]
#     d[ckeyA] = get(d, ckeyA, zero(TC)) + convert(TC, avalA)
#   end

#   # Iterate B blocks once, apply all (rv, ckeyA) groups
#   @inbounds for bi in eachindex(Bp.keys)
#     bkey = Bp.keys[bi]
#     bid  = Bp.ids[bi]
#     Bvec = _block_view(Bp, bid)

#     bpart = ntuple(j -> begin
#       axB = srcB[j]
#       axB == 0 ? 0 : bkey[axB]
#     end, Val(PC))

#     for rv in 1:dim_r
#       d = groups_by_r[rv]
#       isempty(d) && continue
#       for (ckeyA, α) in d
#         ckey = ntuple(j -> begin
#           va = ckeyA[j]
#           va != 0 ? va : bpart[j]
#         end, Val(PC))
#         cid = _ensure_block!(C, ckey)
#         Cvec = _block_view(C, cid)
#         _add_scaled_chunk!(Cvec, α, Bvec, chunklen, rv)
#       end
#     end
#   end
#   return C
# end





# @inline function _block_view(A::NewBlockSparseSorted{T,N,N2}, id::Int) where {T,N,N2}
#   off = (id - 1) * A.blksize
#   return @view(A.data[off+1 : off + A.blksize])
# end

# @inline function _permute_B_make_r_first_prefix(
#     B::NewBlockSparseSorted{T,NB,N2,P},
#     labelsB::AbstractVector,
#     mapB::Dict,
#     rlab
# ) where {T,NB,N2,P}
#   PB = NB - N2
#   axisBr = mapB[rlab]
#   # Only legal/meaningful if r is already in prefix:
#   if axisBr > PB
#     return B, labelsB, mapB, axisBr  # can't fix here
#   end
#   if axisBr == 1
#     return B, labelsB, mapB, axisBr
#   end
#   # Permute within prefix: move axisBr to front, keep others in order; block part unchanged
#   perm_prefix = vcat(axisBr, collect(1:PB)[collect(1:PB) .!= axisBr])
#   perm = vcat(perm_prefix, collect(PB+1:NB))
#   Bp = permutedims(B, perm)
#   labelsB = labelsB[perm]
#   mapB = Dict(lab => i for (i, lab) in enumerate(labelsB))
#   return Bp, labelsB, mapB, 1
# end

# function _permute_A_make_r_first_prefix!(A, labelsA::AbstractVector, mapA::Dict, rlab)
#   NA = length(labelsA)
#   rpos = mapA[rlab]
#   if rpos == NA
#     return labelsA, mapA, collect(1:NA)
#   end
#   # perm maps new axis j -> old axis perm[j]
#   perm = vcat([d for d in 1:NA if d != rpos], rpos)  # new axes -> old axes
#   permutedims!(A, perm)
#   sort!(A)
#   labelsA = labelsA[perm]
#   mapA = Dict(lab => i for (i, lab) in enumerate(labelsA))
#   return A, labelsA, mapA, perm
# end


# function contract_prefix!(
#     C::NewBlockSparseSorted{TC,NC,N2,PC},
#     labelsC::AbstractVector,
#     A::COOTensor{TA,NA},
#     labelsA::AbstractVector,
#     B::NewBlockSparseSorted{TB,NB,N2,PB},
#     labelsB::AbstractVector,
#     mapA::Dict,
#     mapB::Dict,
#     rlab
# ) where {TC,NC,N2,PC,TA,NA,TB,NB,PB}
#   # Block structure constraints: last N2 labels/dims must match between C and B
#   @assert labelsC[PC+1:NC] == labelsB[PB+1:NB]
#   @assert C.dims[PC+1:NC] == B.dims[PB+1:NB]
#   # Overwrite semantics
#   empty!(C.keys); empty!(C.ids); empty!(C.data)

#   dim_r = B.dims[mapB[rlab]]
#   blocks_by_r = [Vector{Tuple{NTuple{PB,Int},Int}}() for _ in 1:dim_r]
#   @inbounds for i in eachindex(B.keys)
#     bkey = B.keys[i]
#     bid  = B.ids[i]
#     rv   = bkey[mapB[rlab]]
#     push!(blocks_by_r[rv], (bkey, bid))
#   end

#   # Output block cache: ckey -> cid
#   key_to_cid = Dict{NTuple{PC,Int},Int}()
#   axisAr = mapA[rlab]

#   # Main multiply: for each A nnz, add α * B_block into the right C block
#   @inbounds for i in eachindex(A.keys)
#     acoord = A.keys[i]
#     avalA  = A.vals[i]

#     rv = acoord[axisAr]
#     (1 <= rv <= dim_r) || continue
#     α = convert(TC, avalA)

#     for (bkey, bid) in blocks_by_r[rv]
#       ckey = ntuple(j -> begin
#         lab = labelsC[j]
#         if haskey(mapA, lab)
#           acoord[mapA[lab]]
#         else
#           ax = mapB[lab]
#           @assert ax <= PB "Label $lab appears in C prefix but is not a B prefix label"
#           bkey[ax]   # IMPORTANT: use ax, not j
#         end
#       end, Val(PC))
#       cid  = ensure_block!(ckey)
#       Cvec = _block_view(C, cid)
#       Bvec = _block_view(B, bid)
#       _scale_add_block!(Cvec, α, Bvec)
#     end
#   end
#   return C
# end



# @inline function _compute_strides(dims::NTuple{N,Int}) where {N}
#   strides = Vector{Int}(undef, N)
#   strides[1] = 1
#   @inbounds for d in 2:N
#     strides[d] = strides[d-1] * dims[d-1]
#   end
#   return strides
# end


# # Cvec (dense rank N2-1) += alpha * slice(Bvec (dense rank N2), fixing axis redpos to redval)
# function _slice_add!(
#     Cvec::AbstractVector{TC},
#     alpha::TC,
#     Bvec::AbstractVector{TB},
#     dimsB::NTuple{N2,Int},
#     redpos::Int,          # 1..N2 in dense dims
#     redval::Int,           # 1..dimsB[redpos]
#     stridesB::Vector{Int}
# ) where {TC,TB,N2}
#   @assert 1 <= redval <= dimsB[redpos]
#   # dims of output dense block (remove redpos)
#   dimsC_vec = Int[]
#   @inbounds for d in 1:N2
#     if d != redpos
#       push!(dimsC_vec, dimsB[d])
#     end
#   end
#   # output size
#   nC = prod(dimsC_vec; init=1)
#   @assert length(Cvec) == nC
#   # iterate all output positions (mixed radix) and map to B linear index
#   # column-major in remaining dims (original order with redpos removed)
#   @inbounds for linC in 1:nC
#     x = linC - 1
#     # base offset from fixed red axis
#     linB = 1 + (redval - 1) * stridesB[redpos]
#     # walk remaining dims in increasing order (skipping redpos)
#     out_dim_idx = 1
#     for d in 1:N2
#       if d != redpos
#         dd = dimsC_vec[out_dim_idx]
#         idx = (x % dd) + 1
#         x ÷= dd
#         linB += (idx - 1) * stridesB[d]
#         out_dim_idx += 1  
#       end
#     end
#     Cvec[linC] += alpha * Bvec[linB]
#   end
#   return nothing
# end

# """
#     contract!(C, labelsC, A, labelsA, B, labelsB)

# Assumptions:
#   * Exactly one shared label r between A and B, and r is reduced (r ∉ labelsC).
#   * r is in B's BLOCK (dense) axes (last N2 labels of B).
#   * C's block labels equal B's block labels with r removed (same order).
#   * C overwrites (clears) its contents.
# """
# function contract_dense!(
#     C::NewBlockSparseSorted{TC,NC,N2C},
#     labelsC::AbstractVector,
#     A::COOTensor{TA,NA},
#     labelsA::AbstractVector,
#     B::NewBlockSparseSorted{TB,NB,N2},
#     labelsB::AbstractVector,
#     mapA::Dict,
#     mapB::Dict,
#     rlab
# ) where {TC,NC,N2C,TA,NA,TB,NB,N2}
#   PB, PC = NB - N2C, NC - N2
#   reductionPos = mapB[rlab] - PB  # local dense position 1..N2
#   @assert N2C == N2 - 1 "Reducing one dense axis: need N2C == N2-1, got N2C=$N2C N2=$N2"
#   # Block label checks: C dense labels = B dense labels with r removed (order preserved)
#   blockLabsB, blockLabsC = collect(labelsB[PB+1:NB]), collect(labelsC[PC+1:NC])
#   expectedC = [lab for lab in blockLabsB if lab != rlab]
#   @assert blockLabsC == expectedC "C block labels must equal B block labels with $rlab removed"
#   # Block dim checks: same removal
#   dimsB_dense = ntuple(i -> B.dims[PB+i], Val(N2))
#   dimsC_dense = ntuple(i -> C.dims[PC+i], Val(N2C))
#   expectedDimsC = Int[]
#   for i in 1:N2
#     if i != reductionPos
#       push!(expectedDimsC, dimsB_dense[i])
#     end
#   end
#   @assert Tuple(expectedDimsC) == dimsC_dense "C block dims must equal B block dims with reduced axis removed"
#   stridesB = _compute_strides(dimsB_dense)
#   # Clear C (overwrite semantics)
#   empty!(C.keys); empty!(C.ids); empty!(C.data)
#   # Iterate A nonzeros and all B blocks: scale-add slices
#   @inbounds for (acoord, avalA) in A.data
#     rv = acoord[mapA[rlab]]
#     # For each B block (prefix key + id), accumulate into the matching C block
#     for bi in eachindex(B.keys)
#       bkey = B.keys[bi]
#       bid  = B.ids[bi]
#       # Build C prefix key from labelsC prefix part
#       ckey = Vector{Int}(undef, PC)
#       for j in 1:PC
#         lab = labelsC[j]
#         if haskey(mapA, lab)
#           ckey[j] = acoord[mapA[lab]]
#         else
#           ax = mapB[lab]
#           @assert ax <= PB "Label $lab appears in C prefix but is not a B prefix label"
#           ckey[j] = bkey[ax]
#         end
#       end
#       cid = _ensure_block!(C, ckey)
#       Cvec = _block_view(C, cid)   # length = prod(dimsC_dense)
#       Bvec = _block_view(B, bid)   # length = prod(dimsB_dense)
#       _slice_add!(Cvec, convert(TC, avalA), Bvec, dimsB_dense, reductionPos, rv, stridesB)
#     end
#   end
#   return C
# end
