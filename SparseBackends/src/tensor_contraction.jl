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



function top_level_contract(
  A::ITensors.ITensor, B::ITensors.ITensor,
  Abackend::Symbol, Bbackend::Symbol,
  Cbackend::Union{Nothing,Symbol},
  bondmap::BondMap;
  denseLinksA::Union{Nothing,Int}=nothing,
  denseLinksB::Union{Nothing,Int}=nothing,
  aLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  aRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing
)
  # Parse the public Symbol API to the canonical Backend enum (strict: a
  # non-canonical name like the old :aliasedblocksparse throws here).
  Ab = to_backend(Abackend)
  Bb = to_backend(Bbackend)
  Cb = Cbackend === nothing ? nothing : to_backend(Cbackend)
  # An ALIASED input backend signals "produce an aliased output" (the old
  # contract_and_fuse_links_aliased semantics): a dense input with ALIASED
  # backend is wrapped DENSE below, and the COO/Dense→Aliased kernel makes the
  # output aliased — but only if Cb===ALIASED reaches contract_and_fuse_links.
  # Infer it here when the caller left Cbackend unset (e.g. build_setup's
  # contract(...,:coo,:aliased; denseLinksB=0)).
  if Cb === nothing && (Ab === ALIASED || Bb === ALIASED)
    Cb = ALIASED
  end
  if Ab === DENSE && Bb === DENSE
    if Cb === DENSE
      return ITensors.contract(A, B), bondmap
    end
    error("Not optimal backend combination: both inputs are dense but Cbackend is not :dense.")
  end
  # Wrap inputs if needed
  Aw = if Ab === ALIASED
      if ITensors.has_external_storage(A)
          wrap_itensor_aliased(A; denseLinks=denseLinksA)
      else
          wrap_itensor(A; backend=DENSE, denseLinks=denseLinksA)
      end
    else
        wrap_itensor(A; backend=Ab, denseLinks=denseLinksA)
  end
  Bw = if Bb === ALIASED
      if ITensors.has_external_storage(B)
          wrap_itensor_aliased(B; denseLinks=denseLinksB)
      else
          wrap_itensor(B; backend=DENSE, denseLinks=denseLinksB)
      end
    else
      wrap_itensor(B; backend=Bb, denseLinks=denseLinksB)
  end
  return contract_and_fuse_links(Aw, Bw, Cb, bondmap;
                                 aLeft=aLeft, aRight=aRight, bLeft=bLeft, bRight=bRight)
end

@inline _bs_head_len(::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = P

function contract_and_fuse_links(
  Aw::WrappedTensorTypes, Bw::WrappedTensorTypes, Cbackend::Union{Nothing,Symbol,Backend},
  bondmap::BondMap;
  aLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  aRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bLeft::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing,
  bRight::Union{Nothing,AbstractVector{<:ITensors.Index}}=nothing
)
  Cb = Cbackend === nothing ? nothing : to_backend(Cbackend)
  if Aw isa WrappedAliasedBlockSparse || Bw isa WrappedAliasedBlockSparse || Cb === ALIASED
    if Cb === ALIASED
      Cw = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)  # <-- aliased output
    elseif Cb === DENSE
      Cw = wrapped_contract_aliased(Aw, Bw)  # <-- dense output
    else
      error("Unsupported Cbackend=$Cb for contract_and_fuse_links with aliased inputs.")
    end
  else
    Cw = contract(Aw, Bw)
  end
  _nz(v) = (v !== nothing && !isempty(v))
  if _nz(aLeft) && _nz(bLeft)
    aBondLogical = first(aLeft)  # stable logical id for bondmap key
    bBondLogical = first(bLeft)
    if Cw isa WrappedAliasedBlockSparse
      # Native canonicalization preserves aliasing (fuse_axes! is now
      # native for prefix-only / tail-only cases). If a mixed fuse is
      # forced, fuse_axes! demotes locally and the result becomes BS.
      Cw, bondmap = _canonicalize_bond_aliased!(
          Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
    elseif Cw isa WrappedBlockSparse
      Cw, bondmap = _canonicalize_bond_blocksparse!(Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
    else
      Cw, bondmap = _canonicalize_bond_single!(Cw, bondmap, aBondLogical, bBondLogical, aLeft, bLeft)
    end
  end
  # RIGHT bond canonicalization
  if _nz(aRight) && _nz(bRight)
    aBondLogical = first(aRight)
    bBondLogical = first(bRight)
    if Cw isa WrappedAliasedBlockSparse
            Cw, bondmap = _canonicalize_bond_aliased!(
                Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)  
    elseif Cw isa WrappedBlockSparse
      Cw, bondmap = _canonicalize_bond_blocksparse!(Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
    else
      Cw, bondmap = _canonicalize_bond_single!(Cw, bondmap, aBondLogical, bBondLogical, aRight, bRight)
    end
  end
  Cw isa ITensors.ITensor && return Cw, bondmap
  # Collapse P=0 results to plain ITensor
  if Cw isa WrappedAliasedBlockSparse && _abs_head_len(Cw) == 0
      dense_data = to_dense(Cw.aliased)
      return ITensors.ITensor(dense_data, Cw.inds...), bondmap
  end
  if Cw isa WrappedBlockSparse && _bs_head_len(Cw) == 0
      dense_data = to_dense(Cw.blocksparse)
      return ITensors.ITensor(dense_data, Cw.inds...), bondmap
  end
  return ITensors._itensor_from_external_storage(Cw), bondmap
end