# module SparseBackendsITensorsExt

import ITensors
# import ITensorMPS
using SparseBackends

abstract type WrappedTensorTypes{T,N} end  

mutable struct WrappedCOOTensor{T,N} <: WrappedTensorTypes{T,N}
  coo::COOTensor{T,N}
  inds::NTuple{N,ITensors.Index}
end

mutable struct WrappedBlockSparse{T,N,N2,P} <: WrappedTensorTypes{T,N}
  blocksparse::NewBlockSparseSorted{T,N,N2,P}
  inds::NTuple{N,ITensors.Index}
end

mutable struct WrappedTensor{T,N} <: WrappedTensorTypes{T,N}
  data::AbstractArray{T,N}
  inds::NTuple{N,ITensors.Index}
end

import Base: show, summary

# Small helper: dims tuple for each wrapper
_dims(w::WrappedTensor) = size(w.data)
_dims(w::WrappedCOOTensor) = w.coo.dims
_dims(w::WrappedBlockSparse) = w.blocksparse.dims
_backend(::WrappedTensor) = :dense
_backend(::WrappedCOOTensor) = :coo
_backend(::WrappedBlockSparse) = :blocksparse





# concise single-line
function summary(io::IO, w::WrappedTensorTypes{T,N}) where {T,N}
    print(io, "Wrapped", _backend(w), "{", T, ",", N, "}",
          " dims=", w.inds, " nnz = ")
    if w isa WrappedCOOTensor
        nnz = length(w.coo.vals)  # adapt to your COO storage fields
        total = prod(_dims(w))
        print(io, " ", nnz, " dense size ", total, "\n")
    elseif w isa WrappedBlockSparse
        # adapt to your blocksparse fields; example:
        nnz = length(w.blocksparse.data)  # change if different
        total = prod(_dims(w))
        print(io, " ", nnz, " dense size ", total, "\n")
    elseif w isa WrappedTensor
        print(io, " dense size ", prod(size(w.data)), "\n")
    else
        throw(ArgumentError("Unknown wrapper type: $(typeof(w))"))
    end
end

function show(io::IO, w::WrappedTensorTypes)
    summary(io, w)
end

# richer REPL display
function show(io::IO, ::MIME"text/plain", w::WrappedTensorTypes{T,N}) where {T,N}
  summary(io, w)
  println(io)
  println(io, "  labels: ", w.labels)
  println(io, "  inds:   ", map(I -> (dim=ITensors.dim(I), tags=String(ITensors.tags(I)), plev=ITensors.plev(I)), w.inds))
  if w isa WrappedCOOTensor
    println(io, "  nnz:    ", length(w.coo.vals))  # adapt to your COO storage fields
  elseif w isa WrappedBlockSparse
    # adapt to your blocksparse fields; example:
    println(io, "  blocks: ", length(w.blocksparse.data))  # change if different
  elseif w isa WrappedTensor
    println(io, "  eltype: ", eltype(w.data))
  end
end

# Make tags string safe for a Julia Symbol (remove quotes, commas, spaces, etc.)
@inline function _sanitize_tagstr(s::AbstractString)
  return replace(s, r"[^A-Za-z0-9_]+" => "_")
end

@inline function split_bra_ket(
  site_inds::AbstractVector{<:ITensors.Index};
  bra_policy::Symbol = :all_nonmin
)
  bra = ITensors.Index[]
  ket = ITensors.Index[]
  isempty(site_inds) && return bra, ket
  groups = Dict{ITensors.Index, Vector{ITensors.Index}}()
  @inbounds for I in site_inds
    base = ITensors.noprime(I)
    push!(get!(groups, base, ITensors.Index[]), I)
  end

  for (_, g) in groups
    if length(g) == 1
      # default: bra
      push!(bra, g[1])
      continue
    end
    # sort by prime level (stable) so we can pick min/max cleanly
    sort!(g; by = ITensors.plev)
    # minimal plev -> ket
    pmin = ITensors.plev(g[1])
    i = 1
    while i <= length(g) && ITensors.plev(g[i]) == pmin
      push!(ket, g[i])
      i += 1
    end
    if bra_policy === :all_nonmin
      while i <= length(g)
        push!(bra, g[i])
        i += 1
      end
    elseif bra_policy === :only_max
      push!(bra, g[end])
    else
      error("Unknown bra_policy=$bra_policy (use :all_nonmin or :only_max)")
    end
  end
  return bra, ket
end



function link_level(I::ITensors.Index)
  for t in ITensors.tags(I)
    ts = String(t)  # convert SmallString -> String
    if startswith(ts, "l=")
      return parse(Int, ts[3:end])
    end
  end
  return nothing
end

"Sort link indices by their l=<n> tag (ascending)."
function sort_link_inds(link_inds::Vector{<:ITensors.Index}; missing=:error)
  levels = map(link_level, link_inds)
  if any(==(nothing), levels)
    if missing == :error
      bad = link_inds[findall(==(nothing), levels)]
      error("Some link inds are missing an l=<n> tag: $(bad)")
    elseif missing == :last
      # Put missing at the end, preserving relative order of missings.
      return sort(link_inds; by=I -> something(link_level(I), typemax(Int)))
    elseif missing == :first
      return sort(link_inds; by=I -> something(link_level(I), typemin(Int)))
    else
      error("Unknown missing policy: $missing")
    end
  end

  return sort(link_inds; by=I -> link_level(I))
end



function mpo_axes_itensor(T::ITensors.ITensor; max_scan_plev::Int=6,
                          bra_plev::Union{Nothing,Int}=nothing, ket_plev::Union{Nothing,Int}=nothing)
  is = collect(ITensors.inds(T))
  links = sort_link_inds([I for I in is if ITensors.hastags(I, "Link")])
  sites = [I for I in is if ITensors.hastags(I, "Site")]
  isempty(sites) && (sites = [I for I in is if !ITensors.hastags(I, "Link")])
  plev_of(I) = (for p in 0:max_scan_plev
                  ITensors.hasplev(I,p) && return p
                end; -1)
  sitepos(I) = begin
    s = repr(ITensors.tags(I))   # <-- FIX
    k = findfirst("n=", s)
    k === nothing && return typemax(Int)
    i = first(k) + 2
    v = 0
    while i <= lastindex(s)
      c = s[i]
      ('0' <= c <= '9') || break
      v = 10v + (Int(c) - Int('0'))
      i += 1
    end
    v
  end

  plevs = sort!(unique(map(plev_of, sites)))
  any(<(0), plevs) && error("mpo_axes_itensor: site plev outside 0:$max_scan_plev")
  bra = ITensors.Index[]
  ket = ITensors.Index[]
  if length(plevs) == 0
    # no sites
  elseif length(plevs) == 1
    (plevs[1] == 0 ? (bra = sites) : (ket = sites))
  elseif length(plevs) == 2
    p0, p1 = plevs
    bra = ITensors.Index[I for I in sites if ITensors.hasplev(I,p0)]
    ket = ITensors.Index[I for I in sites if ITensors.hasplev(I,p1)]
  else
    error("mpo_axes_itensor: >2 site prime levels found: $plevs")
  end
  length(bra) > 1 && sort!(bra, by = sitepos)
  length(ket) > 1 && sort!(ket, by = sitepos)
  return bra, ket, links
end

@inline _dim1(I::ITensors.Index) = (ITensors.dim(I) == 1)

@inline function sort_links(links::AbstractVector{<:ITensors.Index})
    return sort(links; by=ITensors.id)
end

# function _drop_unit_axes(array_full,
#                          inds_full::NTuple{N,ITensors.Index}) where {N}
#   drop = Int[]
#   @inbounds for i in 1:N
#     ITensors.dim(inds_full[i]) == 1 && push!(drop, i)
#   end
#   if isempty(drop)
#     return array_full, inds_full
#   end
#   array_kept = dropdims(array_full; dims=Tuple(drop))
#   keep = [i for i in 1:N if !(i in drop)]
#   return array_kept, Tuple(inds_full[keep])
# end


function WrappedTensor(T::ITensors.ITensor;
                       bra_plev::Union{Nothing,Int}=nothing,
                       ket_plev::Union{Nothing,Int}=nothing)
  bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
  links = sort_links(links)
  if bra !== nothing && ket !== nothing
    inds_full = Tuple(vcat(bra, ket, links))
  elseif bra !== nothing
    inds_full = Tuple(vcat(bra, links))
  elseif ket !== nothing
    inds_full = Tuple(vcat(ket, links))
  else
    inds_full = Tuple(ITensors.Index[])
  end
  array_full = Array(T, inds_full...)
  return WrappedTensor(array_full, inds_full)
end

function WrappedCOOTensor(T::ITensors.ITensor;
                          bra_plev::Union{Nothing,Int}=nothing,
                          ket_plev::Union{Nothing,Int}=nothing)

  bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
  links = sort_links(links)
  if bra !== nothing && ket !== nothing
    inds_full = Tuple(vcat(bra, ket, links))
  elseif bra !== nothing
    inds_full = Tuple(vcat(bra, links))
  elseif ket !== nothing
    inds_full = Tuple(vcat(ket, links))
  else
    inds_full = Tuple(ITensors.Index[])
  end
  array_full = Array(T, inds_full...)
  coo = SparseBackends.coo_from_dense(array_full; atol=1e-12, rtol=0.0)
  return WrappedCOOTensor(coo, inds_full)
end

function WrappedBlockSparse(T::ITensors.ITensor, denseLinks::Int;
                            bra_plev::Union{Nothing,Int}=nothing,
                            ket_plev::Union{Nothing,Int}=nothing)
  bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
  links = sort_links(links)
  if bra !== nothing && ket !== nothing
    inds_full = Tuple(vcat(bra, ket, links))
    # inds_full = Tuple(vcat(ITensors.Index[bra, ket], links))
  elseif bra !== nothing
    inds_full = Tuple(vcat(bra, links))
    # inds_full = Tuple(vcat(ITensors.Index[bra], links))
  elseif ket !== nothing
    inds_full = Tuple(vcat(ket, links))
    # inds_full = Tuple(vcat(ITensors.Index[ket], links))
  else
    inds_full = Tuple(ITensors.Index[])
  end  
  array_full = Array(T, inds_full...)
  # array, inds = _drop_unit_axes(array_full, inds_full)
  denseLinks_kept = count(I -> ITensors.hastags(I, "Link"), inds_full)
  @assert denseLinks == denseLinks_kept "denseLinks=$denseLinks, but after dropping dim-1 axes there are $denseLinks_kept Link axes"
  bs = SparseBackends.blocksparse_from_dense(array_full, Val(denseLinks_kept))
  return WrappedBlockSparse(bs, inds_full)
end


function WrappedCOOTensor(::Type{T},
                          dims::NTuple{N,Int},
                          inds::NTuple{N,ITensors.Index}) where {T,N}
  entries = Vector{Tuple{NTuple{N,Int},T}}()           # empty entries
  coo = SparseBackends.COOTensor{T,N}(dims, entries)     # NOTE: {T} not {T,N}
  return WrappedCOOTensor(coo, inds)
end

function WrappedBlockSparse(::Type{T},
                            dims::NTuple{N,Int},
                            denseLinks::Int,
                            inds::NTuple{N,ITensors.Index}) where {T,N}
  bs = SparseBackends.NewBlockSparseSorted{T,N,denseLinks}(dims)
  return WrappedBlockSparse(bs, inds)
end

function permute_inds!(w::WrappedTensorTypes, perm::AbstractVector{Int})
  @assert length(perm) == length(w.inds)
  w.inds = ntuple(i -> w.inds[perm[i]], Val(length(w.inds)))
  return w
end

backend_hint(::ITensors.ITensor) = :dense  # user can overload in their own code if desired

function wrap_itensor(T::ITensors.ITensor; backend::Symbol=backend_hint(T), denseLinks::Union{Nothing,Int}=nothing)
  if ITensors.has_external_storage(T)
    if backend === :coo
      tensor = ITensors.get_external_storage(T)
      if tensor isa WrappedCOOTensor
        return tensor
      else
        throw(ArgumentError("External storage is not a WrappedCOOTensor; cannot wrap as COO"))
      end
    elseif backend === :blocksparse
      tensor = ITensors.get_external_storage(T)
      if tensor isa WrappedBlockSparse
        return tensor
      else
        throw(ArgumentError("External storage is not a WrappedBlockSparse; cannot wrap as BlockSparse"))
      end
    else
      throw(ArgumentError("Unsupported backend=$backend for ITensor with external storage"))
    end
  end

  if backend === :dense
    return WrappedTensor(T)
  elseif backend === :coo
    return WrappedCOOTensor(T)
  elseif backend === :blocksparse
    @assert denseLinks !== nothing "must pass denseLinks for :blocksparse"
    return WrappedBlockSparse(T, denseLinks)
  else
    throw(ArgumentError("Unknown backend=$backend; use :dense, :coo, or :blocksparse"))
  end
end

dense_inds(w::WrappedCOOTensor) = Set{ITensors.Index}()  # none
dense_inds(w::WrappedTensor)    = Set(w.inds)            # all
dense_inds(w::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P} =
    Set(ntuple(i -> w.inds[N - N2 + i], N2))
# dense_inds(w::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P} =
#     Set(@view(w.inds[end-N2+1:end]))     
    
function infer_denseLinksC(Aw::WrappedTensorTypes,
                        Bw::WrappedTensorTypes,
                        indsC::NTuple{NC,ITensors.Index}) where {NC}
    dA = dense_inds(Aw)
    dB = dense_inds(Bw)
    dC = Set(indsC) ∩ (dA ∪ dB)
    return length(dC)
end

@inline infer_C_backend(::WrappedCOOTensor, ::WrappedCOOTensor) = :coo
@inline infer_C_backend(::WrappedBlockSparse, ::WrappedBlockSparse) = :blocksparse
@inline infer_C_backend(::WrappedCOOTensor, ::WrappedTensor) = :blocksparse
@inline infer_C_backend(::WrappedCOOTensor, ::WrappedBlockSparse) = :blocksparse
@inline infer_C_backend(::WrappedBlockSparse, ::WrappedTensor) = :blocksparse

function _dense_last_order(indsC_vec::Vector{ITensors.Index},
                           dense_set::Set{ITensors.Index})
    sparse = ITensors.Index[I for I in indsC_vec if !(I in dense_set)]
    dense  = ITensors.Index[I for I in indsC_vec if  (I in dense_set)]
    return vcat(sparse, dense)
end

function contract(A::ITensors.ITensor, B::ITensors.ITensor,
                  Abackend::Symbol,
                  Bbackend::Symbol;
                  denseLinksA::Union{Nothing,Int}=nothing,
                  denseLinksB::Union{Nothing,Int}=nothing)
  if Abackend === :dense && Bbackend === :dense
    return ITensors.contract(A, B)
  end
  Aw = wrap_itensor(A; backend=Abackend, denseLinks=denseLinksA)
  Bw = wrap_itensor(B; backend=Bbackend, denseLinks=denseLinksB)
  Cw = contract(Aw, Bw)
  Cw isa ITensors.ITensor && return Cw  # P_C=0: already a plain Dense ITensor
  return ITensors._itensor_from_external_storage(Cw)
end

function contract(A::ITensors.ITensor, B::WrappedTensorTypes{TB,NB}; kwargs...) where {TB,NB}
  Aw = wrap_itensor(A; backend=:dense)
  time = @elapsed begin
    Cw = contract(Aw, B; kwargs...)
  end
  if Cw isa WrappedBlockSparse
    if is_dense(Cw.blocksparse)
      time = @elapsed begin
        data = to_dense(Cw.blocksparse)
        inds = Cw.inds
        result = ITensors.ITensor(data, inds...)    
      end
      # println("Time to convert to dense ITensor 1 : $time")
      return result
    end
  end
  return ITensors._itensor_from_external_storage(Cw)
end

function contract(A::WrappedTensorTypes{TA,NA}, B::ITensors.ITensor; kwargs...) where {TA,NA}
  Bw = wrap_itensor(B; backend=:dense)
  time = @elapsed begin
    Cw = contract(A, Bw; kwargs...)
  end

  if Cw isa WrappedBlockSparse
    if is_dense(Cw.blocksparse)
      time = @elapsed begin
        data = to_dense(Cw.blocksparse)
        inds = Cw.inds
        # println(" @@@@@@@@@@@ TRIGGER CONVERSION @@@@@@@@@@@@")
        result = ITensors.ITensor(data, inds...)
      end
      # println("Time to convert to dense ITensor 2 : $time")
      return result
    end
  end
  return ITensors._itensor_from_external_storage(Cw)
end



@inline is_link(I) = ITensors.hastags(I, "Link")


# Sort link indices by their l=<n> tag (ascending) with cached levels
function sort_link_inds_cached!(link_inds::Vector{<:ITensors.Index}; missing::Symbol = :error)
    n = length(link_inds)
    n == 0 && return link_inds
    levels = Vector{Int}(undef, n)
    has_missing = false
    @inbounds for i in 1:n
        lvl = link_level(link_inds[i])
        if lvl === nothing
            has_missing = true
            levels[i] = 0  # placeholder; handled below
        else
            levels[i] = lvl
        end
    end
    if has_missing
        if missing == :error
            bad = ITensors.Index[]
            @inbounds for i in 1:n
                link_level(link_inds[i]) === nothing && push!(bad, link_inds[i])
            end
            error("Some link inds are missing an l=<n> tag: $(bad)")
        elseif missing == :last
            @inbounds for i in 1:n
                if link_level(link_inds[i]) === nothing
                    levels[i] = typemax(Int)
                end
            end
        elseif missing == :first
            @inbounds for i in 1:n
                if link_level(link_inds[i]) === nothing
                    levels[i] = typemin(Int)
                end
            end
        else
            error("Unknown missing policy: $missing")
        end
    end
    p = sortperm(levels)                # O(n log n) on Ints
    link_inds .= link_inds[p]           # permute in-place
    return link_inds
end

function output_inds(
    indsA::NTuple{NA,ITensors.Index{T}},
    indsB::NTuple{NB,ITensors.Index{T}},
    denseA::Set{ITensors.Index{T}},
    denseB::Set{ITensors.Index{T}},
) where {NA,NB,T}
    # Build membership set for intersection test with minimal allocation
    setA = Set{ITensors.Index{T}}()
    sizehint!(setA, NA)
    @inbounds for I in indsA
        push!(setA, I)
    end
    # Preallocate buckets with decent upper bounds
    sparse_nonlink = Vector{ITensors.Index}(undef, 0); sizehint!(sparse_nonlink, NA + NB)
    sparse_link    = Vector{ITensors.Index}(undef, 0); sizehint!(sparse_link,    NA + NB)
    dense_tail     = Vector{ITensors.Index}(undef, 0); sizehint!(dense_tail,     NA + NB)

    # helper: classify a single index
    @inline function push_classified!(I::ITensors.Index)
        if (I in denseA) || (I in denseB)
            push!(dense_tail, I)
        elseif is_link(I)
            push!(sparse_link, I)
        else
            push!(sparse_nonlink, I)
        end
        return nothing
    end

    # Add unique-from-A (i.e., not in B)
    setB = Set{ITensors.Index{T}}()
    sizehint!(setB, NB)
    @inbounds for I in indsB
        push!(setB, I)
    end
    @inbounds for I in indsA
        (I in setB) && continue
        push_classified!(I)
    end

    # Add unique-from-B (i.e., not in A)
    @inbounds for I in indsB
        (I in setA) && continue
        push_classified!(I)
    end
    # Sort sparse_link by l=<n> tag once, without repeated parsing in comparator
    sort_link_inds_cached!(sparse_link)
    indsC_vec = vcat(sparse_link, sparse_nonlink, dense_tail)
    return Tuple(indsC_vec), dense_tail
end

function output_inds(
    indsA::NTuple{NA,ITensors.Index{T}},
    indsB::NTuple{NB,ITensors.Index{T}},
    denseA_in::AbstractSet{<:ITensors.Index},
    denseB_in::AbstractSet{<:ITensors.Index},
) where {NA,NB,T}
    denseA = Set{ITensors.Index{T}}(denseA_in)  # OK: iterable ctor
    denseB = Set{ITensors.Index{T}}(denseB_in)
    return output_inds(indsA, indsB, denseA, denseB)
end


function perm_from_label_reorder(old_labels::AbstractVector{Symbol},
                                 new_labels::AbstractVector{Symbol})
  @assert length(old_labels) == length(new_labels)
  pos = Dict{Symbol,Int}()
  @inbounds for i in eachindex(old_labels)
    pos[old_labels[i]] = i
  end
  perm = Vector{Int}(undef, length(new_labels))
  @inbounds for i in eachindex(new_labels)
    p = get(pos, new_labels[i], 0)
    p == 0 && error("label $(new_labels[i]) not found in old_labels")
    perm[i] = p
  end
  return perm
end

@inline function permute_inds(inds::NTuple{N,ITensors.Index}, perm::AbstractVector{Int}) where {N}
  @assert length(perm) == N
  return ntuple(i -> inds[perm[i]], Val(N))
end

function contract(A::WrappedTensorTypes{TA,NA}, B::WrappedTensorTypes{TB,NB}; kwargs...) where {TA,TB,NA,NB}
  if A isa WrappedTensor && B isa WrappedCOOTensor
    return wrapped_contract(B, A; kwargs...)
  elseif A isa WrappedTensor && B isa WrappedBlockSparse
    result = wrapped_contract(B, A; kwargs...)
    return result
  elseif A isa WrappedBlockSparse && B isa WrappedCOOTensor
    return wrapped_contract(B, A; kwargs...)
  else
    result = wrapped_contract(A, B; kwargs...)
    return result
  end
end


# --- fast rep dispatch ---
const Label = NTuple{2,UInt64}

@inline label_key_for_ind(I::ITensors.Index)::Label =
    (ITensors.id(I), UInt64(ITensors.plev(I)))

mutable struct ContractScratch
    labelsA::Vector{Label}
    labelsB::Vector{Label}
    labelsC::Vector{Label}
end

const _scratch = Vector{ContractScratch}()

function __init__()
    empty!(_scratch)
    resize!(_scratch, Threads.nthreads())
    for t in 1:Threads.nthreads()
        _scratch[t] = ContractScratch(Label[], Label[], Label[])
    end
end

@inline scratch() = _scratch[Threads.threadid()]

@inline function fill_labels!(buf::Vector{Label}, inds::NTuple{N,ITensors.Index}) where {N}
    resize!(buf, N)
    @inbounds for i in 1:N
        buf[i] = label_key_for_ind(inds[i])
    end
    return buf
end

@inline rep(A::WrappedTensor) = A.data
@inline rep(A::WrappedCOOTensor) = A.coo
@inline rep(A::WrappedBlockSparse) = A.blocksparse  # adjust to your actual type name



function wrapped_contract(A::WrappedTensorTypes{TA,NA},
                          B::WrappedTensorTypes{TB,NB}; kwargs...) where {TA,TB,NA,NB}
    # time = @elapsed begin
      Arep = rep(A)
      Brep = rep(B)
      indsA = A.inds
      indsB = B.inds
      denseA = dense_inds(A)
      denseB = dense_inds(B)
    # end
    if get(ENV, "SB_TRACE", "0") == "1"
      println("[SB_TRACE] wrapped_contract  A=", _backend(A), "{T=", TA, ",N=", NA, "}",
              "  B=", _backend(B), "{T=", TB, ",N=", NB, "}",
              "  → C=", infer_C_backend(A, B))
    end
    # time2 = @elapsed begin
      indsC, denseC = output_inds(indsA, indsB, denseA, denseB)
    # end
    # time3 = @elapsed begin
      dimsC = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
      TC = promote_type(eltype(Arep), eltype(Brep))
      Cbackend = infer_C_backend(A, B)
      sc = scratch()
      labelsA_vec = fill_labels!(sc.labelsA, indsA)
      labelsB_vec = fill_labels!(sc.labelsB, indsB)
      labelsC_vec = fill_labels!(sc.labelsC, indsC)
    # end    
    if Cbackend === :blocksparse
        # println("inds of a and b: ", indsA, " ", indsB)
        # println("number of sparse and dense of A and B ", length(indsA) - length(denseA), " ", length(denseA), " ", length(indsB) - length(denseB), " ", length(denseB))
        # println("Contracting into BlockSparse. dimsC: ", dimsC, " denseC: ", denseC)
        denseLinksC = length(denseC)
        if denseLinksC == length(indsC)
            # P_C = 0: all output indices are dense → return plain ITensor
            C_data = zeros(TC, dimsC...)
            SparseBackends.contract_bs_bs_to_dense!(
                C_data, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec)
            return length(indsC) == 0 ? ITensors.ITensor(C_data[]) : ITensors.ITensor(C_data, indsC...)
        end
        # time = @elapsed begin
          C = WrappedBlockSparse(TC, dimsC, denseLinksC, indsC)
        # end
        # println("Time to construct empty blocksparse: $time")
        # time = @elapsed begin
        C.blocksparse = SparseBackends.contract!(
            C.blocksparse, labelsC_vec,
            Arep, labelsA_vec,
            Brep, labelsB_vec
        )
        # println("Time to contract into blocksparse: $time")
        return C
    elseif Cbackend === :coo
        C = WrappedCOOTensor(TC, dimsC, indsC)
        C.coo = SparseBackends.contract!(
            C.coo, labelsC_vec,
            Arep, labelsA_vec,
            Brep, labelsB_vec
        )
        return C
    else
        error("Unsupported backend: $Cbackend")
    end
end


ITensors.inds(w::SparseBackends.WrappedTensorTypes) = w.inds

link_label(I::ITensors.Index) = ITensors.tags(ITensors.noprime(I))   # returns a TagSet


# function find_link_merge_groups(inds::Tuple)
#   bylabel = Dict{typeof(link_label(first(inds))), Vector{Int}}()
#   for (ax, I) in pairs(inds)
#     if ITensors.hastags(I, "Link")
#       lbl = link_label(I)
#       push!(get!(bylabel, lbl, Int[]), ax)
#     end
#   end
#   groups_axes = [axs for axs in values(bylabel) if length(axs) == 2]
#   for axs in groups_axes
#     sort!(axs)
#   end
#   sort!(groups_axes; by = first)
#   groups_inds = [ITensors.Index[inds[ax] for ax in axs] for axs in groups_axes]
#   return groups_inds, groups_axes
# end

function find_link_merge_groups(inds::Tuple)
    bylabel = Dict{Tuple, Vector{Int}}()
    for (ax, I) in pairs(inds)
        if ITensors.hastags(I, "Link")
            key = (link_label(I), ITensors.plev(I))
            push!(get!(bylabel, key, Int[]), ax)
        end
    end
    groups_axes = [axs for axs in values(bylabel) if length(axs) == 2]
    for axs in groups_axes
        sort!(axs)
    end
    sort!(groups_axes; by=first)
    groups_inds = [ITensors.Index[inds[ax] for ax in axs] for axs in groups_axes]
    return groups_inds, groups_axes
end

function _to_dense(w::SparseBackends.WrappedTensorTypes)
  if w isa WrappedTensor
      return w.data
  elseif w isa WrappedCOOTensor
      arr = to_dense(w.coo)
      return arr
  elseif w isa WrappedBlockSparse
      # print("\tConverting BlockSparse to dense. dims: ", w.inds, " ", w.blocksparse.dims)
      ind_groups, merged_axes = find_link_merge_groups(w.inds)
      # println("  merging axes: ", merged_axes, " with groups: ", ind_groups)
      arr = to_dense(w.blocksparse; merged_axes=merged_axes)
      return arr
  else
      throw(ArgumentError("Unknown wrapper type: $(typeof(w))"))
  end
end

function to_dense(T::ITensors.ITensor)
  # println("to_dense called on ITensor. Has external storage? ", ITensors.has_external_storage(T))
  if T.tensor isa ITensors.ExternalStorage
    es = T.tensor::ITensors.ExternalStorage
    data = es.data
    data isa SparseBackends.WrappedTensorTypes || error("ExternalStorage payload not a SparseBackends wrapper")
    A = _to_dense(data)
    return A
  else
    arr = Array(T, ITensors.inds(T)...)
    return arr
    # # for normal ITensors, use ITensors' own Array conversion:
    # bra, ket, links = mpo_axes_itensor(T)
    # if bra !== nothing && ket !== nothing
    #   inds = Tuple(vcat(bra, ket, links))
    # elseif bra !== nothing
    #   inds = Tuple(vcat(bra, links))
    # elseif ket !== nothing
    #   inds = Tuple(vcat(ket, links))
    # else
    #   inds = Tuple(ITensors.Index[])
    # end
    # # println("\t\t\t   see uuif we did links correctly: ", inds)
    # arr = Array(T, inds...)
    # # println("\tConverting ITensor to dense. inds: ", inds, " dims: ", ITensors.dim.(inds))
    # return arr
  end
end

function to_dense_itensors(T::ITensors.ITensor)::ITensors.ITensor
  data = to_dense(T)               # Array (links may be merged)
  orig_inds = ITensors.inds(T)
  # Compute post-merge index list to match the array rank
  if T.tensor isa ITensors.ExternalStorage && T.tensor.data isa WrappedBlockSparse
    ind_groups, _ = find_link_merge_groups(orig_inds)
    # Build fused indices: for each merge group, combine into one index with product dim
    merged_inds = collect(orig_inds)
    for grp_inds in ind_groups
      fused_dim = prod(ITensors.dim(i) for i in grp_inds)
      fused_idx = ITensors.Index(fused_dim, ITensors.tags(ITensors.noprime(grp_inds[1])))
      # Replace first index of group with fused, remove the rest
      first_pos = findfirst(==(grp_inds[1]), merged_inds)
      merged_inds[first_pos] = fused_idx
      filter!(i -> i ∉ grp_inds[2:end], merged_inds)
    end
    return ITensors.ITensor(data, merged_inds...)
  else
    return ITensors.ITensor(data, orig_inds...)
  end
end

import Base: ndims, size, eltype
import LinearAlgebra: norm

Base.ndims(es::ITensors.ExternalStorage) = length(ITensors.inds(es))
Base.size(es::ITensors.ExternalStorage) = Tuple(ITensors.dim.(ITensors.inds(es)))
Base.eltype(es::ITensors.ExternalStorage{S}) where {S} = eltype(es.data)

LinearAlgebra.norm(w::WrappedBlockSparse) = norm(w.blocksparse.data)
LinearAlgebra.norm(w::WrappedCOOTensor)   = norm(w.coo.vals)
LinearAlgebra.norm(w::WrappedTensor)      = norm(w.data)
LinearAlgebra.norm(es::ITensors.ExternalStorage{<:WrappedTensorTypes}) = norm(es.data)

# Element-wise map! for ExternalStorage ITensors (used by scale!, axpy!, normalize!)
function _apply_elementwise!(f, R::WrappedBlockSparse, A::WrappedBlockSparse)
    length(R.blocksparse.data) == length(A.blocksparse.data) ||
        error("Mismatched sparsity patterns in _external_map! (WrappedBlockSparse)")
    @inbounds @simd for i in eachindex(R.blocksparse.data)
        R.blocksparse.data[i] = f(R.blocksparse.data[i], A.blocksparse.data[i])
    end
end

function _apply_elementwise!(f, R::WrappedCOOTensor, A::WrappedCOOTensor)
    length(R.coo.vals) == length(A.coo.vals) ||
        error("Mismatched sparsity patterns in _external_map! (WrappedCOOTensor)")
    @inbounds @simd for i in eachindex(R.coo.vals)
        R.coo.vals[i] = f(R.coo.vals[i], A.coo.vals[i])
    end
end

function _apply_elementwise!(f, R::WrappedTensor, A::WrappedTensor)
    R.data .= f.(R.data, A.data)
end

function ITensors._external_map_storage!(
    f::Function,
    sR::ITensors.ExternalStorage,
    R::ITensors.ITensor,
    A::ITensors.ITensor,
)
    sA = A.tensor
    sA isa ITensors.ExternalStorage ||
        error("Expected external storage for A, got $(typeof(sA))")

    _apply_elementwise!(f, sR.data, sA.data)
    return R
end

# function ITensors._external_map_storage!(f::Function,
#                                          R::ITensors.ITensor,
#                                          A::ITensors.ITensor)
#     es_R = R.tensor::ITensors.ExternalStorage
#     es_A = A.tensor::ITensors.ExternalStorage
#     _apply_elementwise!(f, es_R.data, es_A.data)
#     return R
# end

# similar for ExternalStorage ITensors — creates a zero tensor with the same sparsity structure
function _zero_similar(w::WrappedBlockSparse{_T,N,N2,P}, ::Type{ElT}) where {_T,N,N2,P,ElT}
    bs = w.blocksparse
    new_bs = NewBlockSparseSorted{ElT,N,N2,P}(
        bs.dims, bs.blksize, copy(bs.keys), copy(bs.ids), zeros(ElT, length(bs.data))
    )
    return WrappedBlockSparse{ElT,N,N2,P}(new_bs, w.inds)
end

function _zero_similar(w::WrappedCOOTensor{_T,N}, ::Type{ElT}) where {_T,N,ElT}
    new_coo = COOTensor{ElT,N}(w.coo.dims, copy(w.coo.keys), zeros(ElT, length(w.coo.vals)), false)
    return WrappedCOOTensor{ElT,N}(new_coo, w.inds)
end

function _zero_similar(w::WrappedTensor{_T,N}, ::Type{ElT}) where {_T,N,ElT}
    return WrappedTensor{ElT,N}(zeros(ElT, size(w.data)), w.inds)
end

function ITensors._external_similar(T::ITensors.ITensor, ElT::Type{<:Number})
    data = ITensors.get_external_storage(T)
    return ITensors._itensor_from_external_storage(_zero_similar(data, ElT))
end

function ITensors._external_similar(T::ITensors.ITensor)
    data = ITensors.get_external_storage(T)
    return ITensors._itensor_from_external_storage(_zero_similar(data, eltype(T)))
end

function ITensors._contract_external_storage(A::ITensors.ITensor,
                                             Bw::WrappedTensorTypes; kwargs...)
  # println("A")
  result = contract(A, Bw; kwargs...)
  return result
end


# external-storage × ITensor
function ITensors._contract_external_storage(Aw::WrappedTensorTypes,
                                             B::ITensors.ITensor; kwargs...)
  # println("B")
  result = contract(Aw, B; kwargs...)
  return result
end

function ITensors._contract_external_storage(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes; kwargs...)
  # println("C")
  result = contract(Aw, Bw; kwargs...)
  result isa ITensors.ITensor && return result  # P_C=0: already a plain Dense ITensor, use directly
  return ITensors._itensor_from_external_storage(result)
end

function ITensors._dims(w::WrappedTensorTypes)
    return _dims(w)
end

# Factorize WrappedBlockSparse for MPS left-orthogonalization.
# Returns (Q::WrappedBlockSparse, R::ITensors.ITensor)
function left_factorize(w::WrappedBlockSparse{T,3,1,2},
                         l_ind, s_ind, r_ind) where T
    Q_bs, R_mat, l_vals, r_vals = blocksparse_left_qr(w.blocksparse)
    new_bond_dim = size(Q_bs.dims[3])
    new_bond_ind = ITensors.Index(new_bond_dim; tags=ITensors.tags(r_ind))
    # scatter R_mat (k × n_r) back to full k × d_r dense ITensor
    d_r     = w.blocksparse.dims[2]
    R_full  = zeros(T, new_bond_dim, d_r)
    for (ri, r_val) in enumerate(r_vals)
        R_full[:, r_val] .= R_mat[:, ri]
    end
    R = ITensors.ITensor(R_full, new_bond_ind, r_ind)

    Q_inds = (l_ind, s_ind, new_bond_ind)
    Q = WrappedBlockSparse(Q_bs, Q_inds)
    return Q, R
end

# Sparse-psi binding: block-diagonal SVD for WrappedBlockSparse phi.
# Returns (L_it, R_it, spec) — same contract as stable_factorize's return.
# w.inds layout must be (l_link, r_link, site1, site2): prefix first, dense last.
function itensor_blocksparse_svd(
    phi::ITensors.ITensor,
    indsMb;
    ortho::String="left",
    maxdim::Int=typemax(Int),
    mindim::Int=1,
    cutoff::Float64=0.0,
    tags=ITensors.ts"Link,l",
)
    w  = ITensors.get_external_storage(phi)::WrappedBlockSparse
    bs = w.blocksparse   # NewBlockSparseSorted{T,4,2,2}
    # w.inds = (l_link, r_link, site1, site2) = (prefix1, prefix2, dense1, dense2)
    l_ind, r_ind, s1_ind, s2_ind = w.inds

    U_bs, SV_bs, svs_kept, spec = blocksparse_svd(
        bs; ortho, maxdim, mindim, cutoff
    )

    new_bond_dim = length(svs_kept)
    new_bond_ind = ITensors.Index(new_bond_dim; tags)

    # U: prefix=(l, new_bond), dense=(s1)
    L_it = ITensors._itensor_from_external_storage(
        WrappedBlockSparse(U_bs, (l_ind, new_bond_ind, s1_ind))
    )
    # SV: prefix=(new_bond, r), dense=(s2)
    R_it = ITensors._itensor_from_external_storage(
        WrappedBlockSparse(SV_bs, (new_bond_ind, r_ind, s2_ind))
    )
    return L_it, R_it, spec
end

# _external_factorize hook: dispatches to sparse block-diagonal SVD.
function ITensors._external_factorize(A::ITensors.ITensor, Linds...; kwargs...)
    storage = ITensors.get_external_storage(A)
    if storage isa WrappedBlockSparse
        ortho  = String(get(kwargs, :ortho,  "left"))
        maxdim = Int(something(get(kwargs, :maxdim,  nothing), typemax(Int)))
        mindim = Int(something(get(kwargs, :mindim,  nothing), 1))
        cutoff = Float64(something(get(kwargs, :cutoff, nothing), 0.0))
        tags   = get(kwargs, :tags, ITensors.ts"Link,fact")
        return itensor_blocksparse_svd(A, Linds; ortho, maxdim, mindim, cutoff, tags)
    end
    error("factorize not implemented for external storage type $(typeof(storage))")
end

# permute override for external-storage ITensors.
# Identity permutations are no-ops; non-identity permutations just relabel inds
# (data layout is unchanged — callers in DMRG that use block-sparse psi must avoid
# non-identity permutations, e.g. by skipping reorder_split_tensors for sparse L/R).
function ITensors.permute(T::ITensors.ITensor, new_inds::ITensors.Index...; kwargs...)
    if !ITensors.has_external_storage(T)
        return invoke(ITensors.permute,
                      Tuple{ITensors.ITensor, Vararg{ITensors.Index}},
                      T, new_inds...; kwargs...)
    end
    w = ITensors.get_external_storage(T)
    N = length(w.inds)
    length(new_inds) == N || error("permute: expected $N indices, got $(length(new_inds))")
    perm = [findfirst(==(new_inds[i]), w.inds) for i in 1:N]
    any(isnothing, perm) && error("permute: index not found in $(w.inds)")
    perm == collect(1:N) && return T   # identity: no-op
    permute_inds!(w, perm)             # relabel only (data layout unchanged)
    return T
end