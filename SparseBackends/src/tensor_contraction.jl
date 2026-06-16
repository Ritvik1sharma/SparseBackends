import ITensors

_is_link(I::ITensors.Index) = ITensors.hastags(I, "Link")
# _tags_str(I::ITensors.Index) = String(ITensors.tags(I))


function fuse_inds_itensor(I1::ITensors.Index, I2::ITensors.Index)
  d = ITensors.dim(I1) * ITensors.dim(I2)
  p = max(ITensors.plev(I1), ITensors.plev(I2))
  # Keep Link tag if either was a Link
  tagstr = (ITensors.hastags(I1, "Link") || ITensors.hastags(I2, "Link")) ? "Link" : ""
  J = ITensors.Index(d, tagstr)
  # match prime level
  if ITensors.plev(J) < p
    J = ITensors.prime(J, p - ITensors.plev(J))
  elseif ITensors.plev(J) > p
    J = ITensors.noprime(J, ITensors.plev(J) - p)
  end
  return J
end

function _fused_inds_tuple(inds::NTuple{N,ITensors.Index},
                           ax1::Int, ax2::Int;
                           ax1_out::Int = ax1 - (ax2 < ax1 ? 1 : 0)) where {N}
  @assert 1 <= ax1 <= N
  @assert 1 <= ax2 <= N
  @assert ax1 != ax2
  @assert 1 <= ax1_out <= N-1
  # Keep ax1, drop ax2
  J = fuse_inds_itensor(inds[ax1], inds[ax2])
  # Map output axis j -> old axis index, skipping dropped axis ax2
  @inline old_axis(j) = (j < ax2) ? j : (j + 1)
  return ntuple(j -> begin
    if j == ax1_out
      J
    else
      inds[old_axis(j)]
    end
  end, Val(N - 1))
end

# function _fused_inds_tuple(inds::NTuple{N,ITensors.Index},
#                                    ax1::Int, ax2::Int;
#                                    ax1_out::Int) where {N}
#   # Map output axis j -> old axis index (skipping dropped axis ax2).
#   @inline old_axis(j) = (j < ax2) ? j : (j + 1)

#   Ikeep = inds[ax1]
#   Idrop = inds[ax2]
#   Ifused = fuseinds(Ikeep, Idrop)  # <-- replace with your actual "fuse two ITensor.Index" op

#   return ntuple(j -> begin
#     if j == ax1_out
#       Ifused
#     else
#       inds[old_axis(j)]
#     end
#   end, Val(N - 1))
# end


function _new_canonical_bond(raw::ITensors.Index, level::Integer)
  c = ITensors.Index(ITensors.dim(raw), "Link,l=$level")
  p = ITensors.plev(raw)
  if ITensors.plev(c) < p
    c = ITensors.prime(c, p - ITensors.plev(c))
  elseif ITensors.plev(c) > p
    c = ITensors.noprime(c, ITensors.plev(c) - p)
  end
  return c
end

# ---------------- relabel helpers ----------------
function relabel_axis!(inds::NTuple{N,ITensors.Index}, ax::Int, newI::ITensors.Index) where {N}
  v = collect(inds); v[ax] = newI; return Tuple(v)
end

function relabel_ind!(W::WrappedTensorTypes, oldI::ITensors.Index, newI::ITensors.Index)
  ax = findfirst(==(oldI), W.inds)
  ax === nothing && error("Index not found to relabel.")
  W.inds = relabel_axis!(W.inds, ax, newI)
  return W
end

function fuse_axes!(W::WrappedCOOTensor{T,N}, ax1::Int, ax2::Int) where {T,N}
  ax1 == ax2 && return W
  (1 ≤ ax1 ≤ N) || error("ax1 out of range")
  (1 ≤ ax2 ≤ N) || error("ax2 out of range")

  newcoo = fuse_axes_coo(W.coo, ax1, ax2)

  # Keep ax1, drop ax2; ax1 shifts left if ax2 < ax1
  ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)
  newinds = _fused_inds_tuple(W.inds, ax1, ax2; ax1_out=ax1_out)
  return WrappedCOOTensor{T,N-1}(newcoo, newinds)
end

function fuse_axes!(W::WrappedBlockSparse{T,N,N2,P}, ax1::Int, ax2::Int) where {T,N,N2,P}
  ax1 == ax2 && return W
  # a1, a2 = min(ax1, ax2), max(ax1, ax2)
  newbs = fuse_two_axes!(W.blocksparse, ax1, ax2)  # check: does it return NewBlockSparseSorted{T,N-1,...}?
  ax1_out = ax1 - (ax2 < ax1 ? 1 : 0)
  newinds = _fused_inds_tuple(W.inds, ax1, ax2; ax1_out=ax1_out)    # must match the same order as fuse_two_axes!
  return WrappedBlockSparse(newbs, newinds)
end

function _extract_link_level(I::ITensors.Index)
  s = string(I)  # e.g. "(dim=2|id=123|\"Link,l=4\")"
  m = match(r"l=(\d+)", s)
  return m === nothing ? nothing : parse(Int, m.captures[1])
end


const ITI = ITensors.Index{Int64}
const BondKey = Tuple{ITI, ITI, Symbol}
const BondMap = Dict{BondKey, ITI}
# const BondKey = Tuple{ITensors.Index, ITensors.Index, Symbol}
# const BondMap = Dict{BondKey, ITensors.Index}

function get_or_create_cbond!(
  bondmap::BondMap,
  aBond::ITensors.Index,
  bBond::ITensors.Index,
  region::Symbol,
  raw::ITensors.Index
)
  key = (aBond, bBond, region)
  if haskey(bondmap, key)
    c = bondmap[key]
    ITensors.dim(c) == ITensors.dim(raw) || error("bondmap dim mismatch for key.")
    return c
  end

  la = _extract_link_level(aBond)
  lb = _extract_link_level(bBond)
  level = (la !== nothing && lb !== nothing && la == lb) ? la : nothing

  c = _new_canonical_bond(raw, level)
  bondmap[key] = c
  return c
end


function _fuse_axes_list!(W::WrappedTensorTypes, axes_in)
  axes = Int[axes_in...]
  Base.sort!(axes)
  while length(axes) > 1
    ax1, ax2 = axes[1], axes[2]
    W = fuse_axes!(W, ax1, ax2)   # <-- REBIND, because rank/type changed
    axes = Int[(a < ax2 ? a : a - 1) for a in axes if a != ax2]
    Base.sort!(axes)
  end
  return W
end


# after fusion, find the (single) axis corresponding to this bond in region by allowing product dim
function _find_axis_after_fuse(W::WrappedTensorTypes, region_axes::Vector{Int}, d1::Int, d2::Int)
  want = Set((d1, d2, d1*d2))
  axs = [ax for ax in region_axes if _is_link(W.inds[ax]) && ITensors.dim(W.inds[ax]) in want]
  isempty(axs) && return nothing
  return axs[1]
end


_same_index(I::ITensors.Index, J::ITensors.Index) =
  ITensors.id(I) == ITensors.id(J) && ITensors.plev(I) == ITensors.plev(J)

function _find_axis_of_ind(inds::NTuple{N,ITensors.Index}, target::ITensors.Index) where {N}
  @inbounds for ax in 1:N
    _same_index(inds[ax], target) && return ax
  end
  return nothing
end


function head_tail_regions(W::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
  head = Int[]; tail = Int[]
  @inbounds for ax in 1:length(W.inds)
    if _is_link(W.inds[ax])
      if ax <= P
        push!(head, ax)
      else
        push!(tail, ax)
      end
    end
  end
  return head, tail
end


# @inline function _link_region(W::WrappedBlockSparse, ax::Int)
#   return ax <= W.blocksparse ? :head : :tail
# end
@inline function _link_region(W::WrappedBlockSparse{T,N,N2,P}, ax::Int) where {T,N,N2,P}
    return ax <= P ? :head : :tail
end


function _canonicalize_bond_by_region!(
  Cw::WrappedBlockSparse,
  bondmap::BondMap,
  aBond::ITensors.Index,
  bBond::ITensors.Index,
  aLegs::AbstractVector{<:ITensors.Index},
  bLegs::AbstractVector{<:ITensors.Index},
)
  # Map region -> raw index for each side (region computed in Cw!)
  amap = Dict{Any,ITensors.Index}()
  for a in aLegs
    ax = _find_axis_of_ind(Cw.inds, a)
    ax === nothing && continue
    amap[_link_region(Cw, ax)] = Cw.inds[ax]
  end

  bmap = Dict{Any,ITensors.Index}()
  for b in bLegs
    ax = _find_axis_of_ind(Cw.inds, b)
    ax === nothing && continue
    bmap[_link_region(Cw, ax)] = Cw.inds[ax]
  end

  # canonicalize per-region
  for reg in union(keys(amap), keys(bmap))
    rawA = get(amap, reg, nothing)
    rawB = get(bmap, reg, nothing)

    if rawA !== nothing
      can = get_or_create_cbond!(bondmap, aBond, bBond, reg, rawA)
      relabel_ind!(Cw, rawA, can)
    end
    if rawB !== nothing
      can = get_or_create_cbond!(bondmap, aBond, bBond, reg, rawB)
      relabel_ind!(Cw, rawB, can)
    end
  end

  return Cw, bondmap
end

@inline function _find_axes_of_ind(inds_tuple, ind::ITensors.Index)
  axes = Int[]
  for (ax, I) in pairs(inds_tuple)
    if I == ind
      push!(axes, ax)
    end
  end
  return axes
end

function _canon_fuse_one_bond!(
  Cw,
  bondmap::BondMap,
  aBondLogical::ITensors.Index,
  bBondLogical::ITensors.Index,
  region::Symbol,
  aLegs::AbstractVector{<:ITensors.Index},
  bLegs::AbstractVector{<:ITensors.Index},
)
  # Collect all axes in Cw that correspond to any of the provided legs.
  axes = Int[]
  for ind in aLegs
    append!(axes, _find_axes_of_ind(Cw.inds, ind))
  end
  for ind in bLegs
    append!(axes, _find_axes_of_ind(Cw.inds, ind))
  end
  if isempty(axes)
    if ITensors.dim(aBondLogical) == 1 && ITensors.dim(bBondLogical) == 1
      return Cw, bondmap
    end
    error("Neither bond leg found in Cw.inds for region=$region (aBond=$aBondLogical, bBond=$bBondLogical)")
  end
  # Fuse all of them into the first axis.
  base = first(axes)
  for ax in reverse(axes[2:end])
    Cw = fuse_axes!(Cw, base, ax)
  end
  raw = Cw.inds[base]
  can = get_or_create_cbond!(bondmap, aBondLogical, bBondLogical, region, raw)
  relabel_ind!(Cw, raw, can)
  return Cw, bondmap
end



function _canonicalize_bond_blocksparse!(
  Cw::WrappedBlockSparse{T,N,N2,P},
  bondmap::BondMap,
  aBondLogical::ITensors.Index,
  bBondLogical::ITensors.Index,
  aLegs::AbstractVector{<:ITensors.Index},
  bLegs::AbstractVector{<:ITensors.Index},
) where {T,N,N2,P}
  # Collect axes for a set of legs, split into head/tail
  function collect_head_tail_axes(legs)
    head = Int[]
    tail = Int[]
    for ind in legs
      for ax in _find_axes_of_ind(Cw.inds, ind)
        if ax <= Int(P)
          push!(head, ax)
        else
          push!(tail, ax)
        end
      end
    end
    unique!(head); sort!(head)
    unique!(tail); sort!(tail)
    return head, tail
  end

  # We collect using current Cw (pre-fusion)
  headA, tailA = collect_head_tail_axes(aLegs)
  headB, tailB = collect_head_tail_axes(bLegs)
  head_axes = sort!(unique!(vcat(headA, headB)))
  tail_axes = sort!(unique!(vcat(tailA, tailB)))

  # Only fuse+relabel if there are >= 2 axes (you requested this behavior)
  function fuse_bucket!(axes::Vector{Int}, region::Symbol)
    length(axes) >= 2 || return Cw, bondmap
    # Fuse into the smallest axis; do it from right-to-left so indices remain valid
    base = first(axes)
    for ax in reverse(axes[2:end])
      Cw = fuse_axes!(Cw, base, ax)
    end
    raw = Cw.inds[base]
    can = get_or_create_cbond!(bondmap, aBondLogical, bBondLogical, region, raw)
    relabel_ind!(Cw, raw, can)
    return Cw, bondmap
  end

  Cw, bondmap = fuse_bucket!(tail_axes, :dense)
  Cw, bondmap = fuse_bucket!(head_axes, :sparse)
  return Cw, bondmap
end

function _canonicalize_bond_single!(
  Cw,
  bondmap::BondMap,
  aBondLogical::ITensors.Index,
  bBondLogical::ITensors.Index,
  aLegs::AbstractVector{<:ITensors.Index},
  bLegs::AbstractVector{<:ITensors.Index},
)
  # For COO/Dense semantics, treat as a single region
  return _canon_fuse_one_bond!(Cw, bondmap, aBondLogical, bBondLogical, :all, aLegs, bLegs)
end



function contract_and_fuse_links(
  A::ITensors.ITensor, B::ITensors.ITensor,
  Abackend::Symbol, Bbackend::Symbol,
  bondmap::BondMap;
  denseLinksA::Union{Nothing,Int}=nothing,
  denseLinksB::Union{Nothing,Int}=nothing,
  aLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  aRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing
)
  if Abackend === :dense && Bbackend === :dense
    return ITensors.contract(A, B), bondmap
  end
  # When either side requests :aliased storage, route through the aliased
  # per-site kernel. This is the only entry point that handles the :aliased
  # backend; other paths remain unchanged.
  if Abackend === :aliased || Bbackend === :aliased
    return contract_and_fuse_links_aliased(A, B, Abackend, Bbackend, bondmap;
      denseLinksA=denseLinksA, denseLinksB=denseLinksB,
      aLeft=aLeft, aRight=aRight, bLeft=bLeft, bRight=bRight)
  end
  Aw = wrap_itensor(A; backend=Abackend, denseLinks=denseLinksA)
  Bw = wrap_itensor(B; backend=Bbackend, denseLinks=denseLinksB)
  return contract_and_fuse_links(Aw, Bw, bondmap; aLeft=aLeft, aRight=aRight, bLeft=bLeft, bRight=bRight)
end

@inline _bs_head_len(::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = P

function contract_and_fuse_links(
  Aw::WrappedTensorTypes, Bw::WrappedTensorTypes,
  bondmap::BondMap;
  aLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  aRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing
)
  Cw = contract(Aw, Bw)  
  # println("    dirty? ", Cw.coo.dirty)
  _nz(v) = (v !== nothing && !isempty(v))
  if _nz(aLeft) && _nz(bLeft)
    aBondLogical = first(aLeft)  # stable logical id for bondmap key
    bBondLogical = first(bLeft)
    if Cw isa WrappedBlockSparse
      Cw, bondmap = _canonicalize_bond_blocksparse!(Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
    else
      Cw, bondmap = _canonicalize_bond_single!(Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
    end
  end
  # RIGHT bond canonicalization
  if _nz(aRight) && _nz(bRight)
    aBondLogical = first(aRight)
    bBondLogical = first(bRight)
    if Cw isa WrappedBlockSparse
      Cw, bondmap = _canonicalize_bond_blocksparse!(Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
    else
      Cw, bondmap = _canonicalize_bond_single!(Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
    end
  end
  Cw isa ITensors.ITensor && return Cw, bondmap
  if Cw isa WrappedBlockSparse && _bs_head_len(Cw) == 0
    dense_data = to_dense(Cw.blocksparse)
    return ITensors.ITensor(dense_data, Cw.inds...), bondmap
  end
  # println("------------- ", Cw.inds, " with bondmap keys ", keys(bondmap))
  # println(Cw.coo.keys, " with info ", Cw.coo.dirty)
  # error("err")
  return ITensors._itensor_from_external_storage(Cw), bondmap
end




# function _fuse_and_canonicalize_links!(
#   Cw::WrappedCOOTensor,
#   bondmap::BondMap;
#   #   bondmap::Dict{Tuple{ITensors.Index,ITensors.Index}, ITensors.Index};
#   aLeft::Union{Nothing,ITensors.Index}=nothing,
#   aRight::Union{Nothing,ITensors.Index}=nothing,
#   bLeft::Union{Nothing,ITensors.Index}=nothing,
#   bRight::Union{Nothing,ITensors.Index}=nothing
# )
#   # helper for one bond
#   function canon_one_bond(Cw_local::WrappedCOOTensor, aBond::ITensors.Index, bBond::ITensors.Index)
#     axA = _find_axis_of_ind(Cw_local.inds, aBond)
#     axB = _find_axis_of_ind(Cw_local.inds, bBond)


#     if axA === nothing && axB === nothing
#         if ITensors.dim(aBond) == 1 && ITensors.dim(bBond) == 1
#             return Cw_local  # scalar bond vanished; fine
#         else
#             error("COO: neither bond leg found in Cw.inds (aBond=$(aBond), bBond=$(bBond))")
#         end
#     elseif axA === nothing
#       # only bBond survived (often because dim(aBond)==1 and got dropped)
#       raw = Cw_local.inds[axB]
#       can = get_or_create_cbond!(bondmap, aBond, bBond, :all, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     elseif axB === nothing
#       # only aBond survived (often because dim(bBond)==1 and got dropped)
#       raw = Cw_local.inds[axA]
#       can = get_or_create_cbond!(bondmap, aBond, bBond, :all, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     elseif axA == axB
#       # already same axis (rare)
#       raw = Cw_local.inds[axA]
#       can = get_or_create_cbond!(bondmap, aBond, bBond, :all, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     else
#       # fuse the two legs
#       a1, a2 = min(axA, axB), max(axA, axB)
#       Cw_local = fuse_axes!(Cw_local, a1, a2)
#       raw = Cw_local.inds[a1]  # fused lives at a1
#       can = get_or_create_cbond!(bondmap, aBond, bBond, :all, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     end
#   end

#   if aLeft !== nothing && bLeft !== nothing
#     Cw = canon_one_bond(Cw, aLeft, bLeft)
#   end
#   if aRight !== nothing && bRight !== nothing
#     Cw = canon_one_bond(Cw, aRight, bRight)
#   end

#   return Cw, bondmap
# end


# function _fuse_and_canonicalize_links!(
#   Cw::WrappedBlockSparse,
#   bondmap::BondMap;
#   aLeft::Union{Nothing,ITensors.Index}=nothing,
#   aRight::Union{Nothing,ITensors.Index}=nothing,
#   bLeft::Union{Nothing,ITensors.Index}=nothing,
#   bRight::Union{Nothing,ITensors.Index}=nothing
# )
#   # Canonicalize a single bond (aBond,bBond) possibly represented by 0/1/2 surviving axes.
#   function fuse_one_bond(Cw_local::WrappedBlockSparse, aBond::ITensors.Index, bBond::ITensors.Index)
#     axA = _find_axis_of_ind(Cw_local.inds, aBond)
#     axB = _find_axis_of_ind(Cw_local.inds, bBond)
#     # Case 0: both legs vanished (typically dim-1 axes got dropped upstream)
#     if axA === nothing && axB === nothing
#       if ITensors.dim(aBond) == 1 && ITensors.dim(bBond) == 1
#         return Cw_local
#       else
#         error("BS: neither bond leg found in Cw.inds (aBond=$(aBond), bBond=$(bBond))")
#       end
#     end
#     # Case 1: only one leg survived -> just canonicalize that survivor (region from its axis)
#     if axA === nothing
#       raw = Cw_local.inds[axB]
#       reg = _link_region(Cw_local, axB)
#       can = get_or_create_cbond!(bondmap, aBond, bBond, reg, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     end
#     if axB === nothing
#       raw = Cw_local.inds[axA]
#       reg = _link_region(Cw_local, axA)
#       can = get_or_create_cbond!(bondmap, aBond, bBond, reg, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     end
#     if axA == axB
#       raw = Cw_local.inds[axA]
#       reg = _link_region(Cw_local, axA)
#       can = get_or_create_cbond!(bondmap, aBond, bBond, reg, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     end

#     regA = _link_region(Cw_local, axA)
#     regB = _link_region(Cw_local, axB)

#     if regA == regB
#       # Same region: fuse in that region, then canonicalize fused leg
#       a1, a2 = min(axA, axB), max(axA, axB)
#       Cw_local = fuse_axes!(Cw_local, a1, a2)
#       raw = Cw_local.inds[a1]                 # fused index lives at a1
#       can = get_or_create_cbond!(bondmap, aBond, bBond, regA, raw)
#       relabel_ind!(Cw_local, raw, can)
#       return Cw_local
#     else
#       # Cross head/tail: CANNOT fuse. Canonicalize each leg separately under distinct keys.
#       rawA = Cw_local.inds[axA]
#       canA = get_or_create_cbond!(bondmap, aBond, bBond, regA, rawA)
#       relabel_ind!(Cw_local, rawA, canA)
#       rawB = Cw_local.inds[axB]
#       canB = get_or_create_cbond!(bondmap, aBond, bBond, regB, rawB)
#       relabel_ind!(Cw_local, rawB, canB)
#       return Cw_local
#     end
#   end
#   if aLeft !== nothing && bLeft !== nothing
#     Cw = fuse_one_bond(Cw, aLeft, bLeft)
#   end
#   if aRight !== nothing && bRight !== nothing
#     Cw = fuse_one_bond(Cw, aRight, bRight)
#   end
#   return Cw, bondmap
# end