# module SparseBackendsITensorsExt

import ITensors
# import ITensorMPS
using SparseBackends
import Base

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

dim(w::WrappedTensorTypes) = _dims(w)
dim(w::WrappedCOOTensor) = _dims(w)
dim(w::WrappedBlockSparse) = _dims(w)



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
function sort_link_inds(link_inds::Vector{<:ITensors.Index}; missing=:last)
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


const _wrap_cache = Dict{UInt, Any}()

function clear_wrap_cache!()
    empty!(_wrap_cache)
end

# function _wrap_itensor_uncached(T::ITensors.ITensor;
#                                 bra_plev::Union{Nothing,Int}=nothing,
#                                 ket_plev::Union{Nothing,Int}=nothing)
#   bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
#   links      = sort_links(links)
#   if bra !== nothing && ket !== nothing
#     inds_full = Tuple(vcat(bra, ket, links))
#   elseif bra !== nothing
#     inds_full = Tuple(vcat(bra, links))
#   elseif ket !== nothing
#     inds_full = Tuple(vcat(ket, links))
#   else
#     inds_full = Tuple(ITensors.Index[])
#   end
#   inds_full  = Tuple(vcat(bra, ket, links))
#   array_full = Array(T, inds_full...)
#   return WrappedTensor(array_full, inds_full)
# end


function _wrap_itensor_uncached(T::ITensors.ITensor;
                                bra_plev::Union{Nothing,Int}=nothing,
                                ket_plev::Union{Nothing,Int}=nothing)
    bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
    links = sort_links(links)
    # Keep original conditional logic — bra/ket may both be non-empty
    # even when the tensor has only one site leg, so unconditional vcat
    # would over-count indices.
    inds_full = if !isempty(bra) && !isempty(ket)
        Tuple(vcat(bra, ket, links))
    elseif !isempty(bra)
        Tuple(vcat(bra, links))
    elseif !isempty(ket)
        Tuple(vcat(ket, links))
    else
        Tuple(links)
    end
    dims_full = map(ITensors.dim, inds_full)
    array_full = if ITensors.inds(T) == inds_full
        # ITensors.data(T) returns the flat underlying Vector directly.
        raw = ITensors.data(T)
        reshape(raw, dims_full)
    else
        Array(T, inds_full...)
    end    
    return WrappedTensor(array_full, inds_full)
end

function WrappedTensor(T::ITensors.ITensor;
                      backend::Symbol=:dense,   # kept for call-site compat
                      bra_plev::Union{Nothing,Int}=nothing,
                      ket_plev::Union{Nothing,Int}=nothing)
    # key    = objectid(T)
    # cached = get(_wrap_cache, key, nothing)
    # if cached !== nothing
    #     return cached::WrappedTensor
    # end
    w = _wrap_itensor_uncached(T; bra_plev=bra_plev, ket_plev=ket_plev)
    # _wrap_cache[key] = w
    return w
end


# function WrappedTensor(T::ITensors.ITensor;
#                        bra_plev::Union{Nothing,Int}=nothing,
#                        ket_plev::Union{Nothing,Int}=nothing)
#   bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
#   links = sort_links(links)
#   if bra !== nothing && ket !== nothing
#     inds_full = Tuple(vcat(bra, ket, links))
#   elseif bra !== nothing
#     inds_full = Tuple(vcat(bra, links))
#   elseif ket !== nothing
#     inds_full = Tuple(vcat(ket, links))
#   else
#     inds_full = Tuple(ITensors.Index[])
#   end
#   array_full = Array(T, inds_full...)
#   return WrappedTensor(array_full, inds_full)
# end


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

function wrap_itensor(T::ITensors.ITensor; backend::Union{Symbol,Backend}=backend_hint(T), denseLinks::Union{Nothing,Int}=nothing)
  b = to_backend(backend)
  if ITensors.has_external_storage(T)
    if b === COO
      tensor = ITensors.get_external_storage(T)
      if tensor isa WrappedCOOTensor
        return tensor
      else
        throw(ArgumentError("External storage is not a WrappedCOOTensor; cannot wrap as COO"))
      end
    elseif b === BLOCKSPARSE
      tensor = ITensors.get_external_storage(T)
      if tensor isa WrappedBlockSparse
        return tensor
      else
        throw(ArgumentError("External storage is not a WrappedBlockSparse; cannot wrap as BlockSparse"))
      end
    elseif b === ALIASED
      return wrap_itensor_aliased(T; denseLinks=denseLinks)
    else
      throw(ArgumentError("Unsupported backend=$b for ITensor with external storage"))
    end
  end

  if b === DENSE
    # Fast path for plain dense ITensors: just reshape the raw data buffer
    # to its inds-order layout (which already matches ITensors.data(T)).
    # Skips mpo_axes_itensor's index classification + sort_links, since
    # downstream code only needs (data, inds) in a self-consistent order.
    inds_full = Tuple(ITensors.inds(T))
    dims_full = map(ITensors.dim, inds_full)
    raw       = ITensors.data(T)
    array_full = reshape(raw, dims_full)
    return WrappedTensor(array_full, inds_full)
  elseif b === COO
    return WrappedCOOTensor(T)
  elseif b === BLOCKSPARSE
    @assert denseLinks !== nothing "must pass denseLinks for :blocksparse"
    return WrappedBlockSparse(T, denseLinks)
  elseif b === ALIASED
    return wrap_itensor_aliased(T; denseLinks=denseLinks)
  else
    throw(ArgumentError("Unknown backend=$b"))
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
                  Abackend::Union{Symbol,Backend},
                  Bbackend::Union{Symbol,Backend};
                  denseLinksA::Union{Nothing,Int}=nothing,
                  denseLinksB::Union{Nothing,Int}=nothing,
                  preserve_bs_output::Bool=false)
  Ab = to_backend(Abackend); Bb = to_backend(Bbackend)
  if Ab === DENSE && Bb === DENSE
    return ITensors.contract(A, B)
  end
  Aw = wrap_itensor(A; backend=Ab, denseLinks=denseLinksA)
  Bw = wrap_itensor(B; backend=Bb, denseLinks=denseLinksB)
  Cw = contract(Aw, Bw; preserve_bs_output=preserve_bs_output)
  Cw isa ITensors.ITensor && return Cw  # P_C=0: already a plain Dense ITensor
  return ITensors._itensor_from_external_storage(Cw)
end

# Three-backend public form: explicit OUTPUT backend `Cbackend`. Prefer this over
# calling the internal `contract_aliased_itensor` directly — it is the consistent
# public entry point when the output storage must be forced independently of the
# input backends (e.g. `:coo × :dense → :aliased`, which neither input backend
# would yield on its own). Delegates to the aliased kernel for `Cbackend=:aliased`
# and to the two-backend method otherwise.
function contract(A::ITensors.ITensor, B::ITensors.ITensor,
                  Abackend::Union{Symbol,Backend},
                  Bbackend::Union{Symbol,Backend},
                  Cbackend::Union{Symbol,Backend};
                  denseLinksA::Union{Nothing,Int}=nothing,
                  denseLinksB::Union{Nothing,Int}=nothing,
                  preserve_bs_output::Bool=true,
                  preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                  next_op=nothing,
                  output_perm::Union{Nothing,Vector{Int}}=nothing,
                  emit_window_map::Bool=false)
  Cb = to_backend(Cbackend)
  if Cb === ALIASED
    return contract_aliased_itensor(A, B, Abackend, Bbackend;
                                    denseLinksA=denseLinksA, denseLinksB=denseLinksB,
                                    preserve_bs_output=preserve_bs_output,
                                    preferred_output_labels=preferred_output_labels,
                                    next_op=next_op, output_perm=output_perm,
                                    emit_window_map=emit_window_map)
  end
  return contract(A, B, Abackend, Bbackend;
                  denseLinksA=denseLinksA, denseLinksB=denseLinksB,
                  preserve_bs_output=preserve_bs_output)
end

# Convenience: contract two ITensors (each carrying any storage) while forcing
# the output to keep its WrappedBlockSparse storage even when the natural
# output would be a plain dense ITensor. Used by Path-B sparse DMRG where
# KrylovKit's geneigsolve needs storage stability across Krylov iterations.
#
# If `template` is given (an ITensor with WrappedBlockSparse external storage
# whose Indices match the natural output's), the result is re-cast so that
# its (dense-axes, sparse-axes) classification AND inds-ordering match the
# template's. This makes the result type-identical to the template, which is
# what KrylovKit's `scale!`/`axpy!` require across operator applications.
function contract_preserve_bs(A::ITensors.ITensor, B::ITensors.ITensor;
                              template::Union{Nothing,ITensors.ITensor}=nothing,
                              output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                              preferred_output_labels::Union{Nothing,AbstractVector}=nothing,
                              next_op=nothing,
                              remaining_ops=nothing,
                              output_perm::Union{Nothing,Vector{Int}}=nothing,
                              in_position::Bool=false)
 @timeit TIMER "contract_preserve_bs" begin
  # Both inputs plain dense → no BS storage to preserve; standard contract.
  if !ITensors.has_external_storage(A) && !ITensors.has_external_storage(B)
    output_perm !== nothing && error("contract_preserve_bs: output_perm given but both operands are dense — the static output-permutation table only applies to the aliased/BS-preserving path")
    return ITensors.contract(A, B)
  end
  if ITensors.has_external_storage(A)
    Aw = ITensors.get_external_storage(A)
  else
    Aw = wrap_itensor(A; backend=:dense)
  end
  if ITensors.has_external_storage(B)
    Bw = ITensors.get_external_storage(B)
  else
    Bw = wrap_itensor(B; backend=:dense)
  end
  # Auto-derive hint from template if caller didn't supply one (template's
  # dense_inds is exactly the set of axes that should be dense in the output).
  # UNSAFE, kept hardcoded off: the in-kernel hint path is not yet implemented
  # (without kernel support, hint causes correctness errors). Do not flip this
  # to true (was BMF_USE_HINT, default-off knob, never safe to enable).
  if false &&
     output_inds_hint === nothing && template !== nothing &&
     ITensors.has_external_storage(template) &&
     ITensors.get_external_storage(template) isa WrappedBlockSparse
    output_inds_hint = dense_inds(ITensors.get_external_storage(template))
  end
  # Aliased Path-B fix (route 2): drive the output axis classification from an
  # ALIASED template's dense_inds, so the contract produces output with φ's
  # EXACT {N2,P} split via the native aliased fission kernel rather than letting
  # a dense Minv reclassify a channel axis into the dense tail (the Path-B
  # alias-collapse bug). Requires SB_ALIASED_NATIVE_FISSION=1 for the native
  # (dedup-preserving) fission path; without it the kernel BS-delegates (trivial
  # dedup). Hardened 2026-06 — always on (was SB_ALIASED_MINV_HINT, default-on knob).
  if output_inds_hint === nothing && template !== nothing &&
     ITensors.has_external_storage(template) &&
     ITensors.get_external_storage(template) isa WrappedAliasedBlockSparse
    output_inds_hint = dense_inds(ITensors.get_external_storage(template))
  end
  template_for_filter = nothing
  # UNSAFE, kept hardcoded off: see BMF_USE_HINT note above.
  if false && template !== nothing &&
     ITensors.has_external_storage(template) &&
     ITensors.get_external_storage(template) isa WrappedBlockSparse
    template_for_filter = ITensors.get_external_storage(template)::WrappedBlockSparse
  end
  # #recast-kill (gated SB_ALIASED_ALIGN_OUTPUT): ask the aliased contract to emit
  # its output already in the aliased template's axis order, so the downstream
  # align_aliased_axes becomes a no-op. Only set for an aliased template
  # (routes to the aliased contract overload that accepts preferred_output_labels).
  # Caller-provided ordered output labels (e.g. the dense-chain next-step hint)
  # take precedence; otherwise derive from the aliased template (recast-align).
  _pref_order = preferred_output_labels
  # Hardened 2026-06 — always try to align to the template's own axis order
  # when one is available (was SB_ALIASED_ALIGN_OUTPUT, default-off knob); the
  # kernel already falls back gracefully (_ALIGN_FALLBACK) when it can't honor
  # the request, so this is a pure win when it succeeds and a no-op otherwise.
  if _pref_order === nothing &&
     template !== nothing && ITensors.has_external_storage(template) &&
     ITensors.get_external_storage(template) isa WrappedAliasedBlockSparse
    _pref_order = collect(ITensors.inds(template))
  end
  Cw = if _pref_order !== nothing
    @timeit TIMER "cpb.contract" contract(Aw, Bw; preserve_bs_output=true,
        output_inds_hint=output_inds_hint, template_for_filter=template_for_filter,
        preferred_output_labels=_pref_order, next_op=next_op, remaining_ops=remaining_ops,
        output_perm=output_perm, in_position=in_position)
  else
    @timeit TIMER "cpb.contract" contract(Aw, Bw; preserve_bs_output=true,
        output_inds_hint=output_inds_hint, template_for_filter=template_for_filter,
        next_op=next_op, remaining_ops=remaining_ops,
        output_perm=output_perm, in_position=in_position)
  end
  Cw isa ITensors.ITensor && return Cw

  # Optionally recast to match template's axis classification + ordering.
  if template !== nothing && ITensors.has_external_storage(template) &&
     Cw isa WrappedBlockSparse
    Tw = ITensors.get_external_storage(template)
    if Tw isa WrappedBlockSparse
      Cw = @timeit TIMER "cpb.recast" recast_bs_to_template(Cw, Tw)
    end
  end
  return ITensors._itensor_from_external_storage(Cw)
 end
end

# Re-cast a WrappedBlockSparse result so its inds-ordering, axis classification
# (N2 = #dense axes), AND block-key list+ordering all match the template.
# Requires the result's Index identities to form the same set as the template's.
#
# The block-key match is critical: KrylovKit / VectorInterface in-place ops
# (`scale!`, `axpy!`, etc.) iterate the raw `.data` buffer position-by-position
# and assume both operands have the same .keys list. A mismatch silently
# scrambles values. By copying the template's key list verbatim, we make the
# result drop-in compatible with Krylov subspace bookkeeping.
#
# Data is taken from Cw via densify → permute → gather. Cost is O(nnz_dense)
# per call; for Krylov inner loops this is negligible next to the contraction.
# Layer 1 of in-kernel filter: when Cw and Tw have IDENTICAL axis split
# (same inds order, same blksize), recasting reduces to dropping non-Tw keys.
# No densify, no permute. O(nblocks) instead of O(prod(dims)).
function filter_bs_keys_to_template(Cw::WrappedBlockSparse{TC,N,N2c,Pc},
                                    Tw::WrappedBlockSparse{TT,N,N2t,Pt}) where {TC,TT,N,N2c,Pc,N2t,Pt}
  # Strict prereq: both BS storages already share order, split, and inds.
  if Cw.inds !== Tw.inds && Cw.inds != Tw.inds
    return nothing  # signal: structure mismatch, fall back to full recast
  end
  N2c == N2t || return nothing
  Pc == Pt || return nothing
  Cw.blocksparse.blksize == Tw.blocksparse.blksize || return nothing

  bs_c = Cw.blocksparse
  bs_t = Tw.blocksparse
  blksize = bs_c.blksize
  c_keymap = Dict{NTuple{Pc,Int}, Int}()
  sizehint!(c_keymap, length(bs_c.keys))
  @inbounds for i in eachindex(bs_c.keys)
    c_keymap[bs_c.keys[i]] = bs_c.ids[i]
  end
  new_keys = copy(bs_t.keys)
  new_ids  = copy(bs_t.ids)
  new_data = Vector{TC}(undef, length(bs_t.data))
  @inbounds for i in eachindex(bs_t.keys)
    k = bs_t.keys[i]
    out_id = bs_t.ids[i]
    out_base = (out_id - 1) * blksize
    src_id = get(c_keymap, k, 0)
    if src_id == 0
      # Key in template but not in Cw → zero block.
      for j in 1:blksize
        new_data[out_base + j] = zero(TC)
      end
    else
      src_base = (src_id - 1) * blksize
      for j in 1:blksize
        new_data[out_base + j] = bs_c.data[src_base + j]
      end
    end
  end
  bs_new = NewBlockSparseSorted{TC,N,N2t,Pt,Int}(
      bs_t.dims, blksize, new_keys, new_ids, new_data)
  return WrappedBlockSparse(bs_new, Tw.inds)
end

# Fission a BS storage: take a BS with some axes in its dense suffix, move
# specified axes into the sparse prefix. Each natural block splits into multiple
# smaller blocks indexed by the moved axes' values. No densification — one
# pass over `.data` writing into the new key list.
#
# Inputs:
#   Cw_natural : the BS storage with natural axis split
#   target_inds : Tuple of Index that defines the desired axis order in output
#   target_n2   : number of dense axes in output (last `target_n2` of target_inds)
function fission_bs(Cw_natural::WrappedBlockSparse{TC,N,N2nat,Pnat},
                    target_inds::NTuple{N,ITensors.Index},
                    target_n2::Int) where {TC,N,N2nat,Pnat}
  target_P = N - target_n2
  bs_nat = Cw_natural.blocksparse
  nat_inds = Cw_natural.inds
  # 1. Locate each target_inds[k] in nat_inds (by Index identity).
  nat_pos_of_target = ntuple(k -> findfirst(==(target_inds[k]), nat_inds), Val(N))
  any(==(nothing), nat_pos_of_target) && error("fission_bs: target inds not subset of natural inds")
  # 2. Classify each natural axis by whether it ends up in target's prefix or dense.
  target_pos_of_nat = ntuple(j -> findfirst(==(nat_inds[j]), target_inds), Val(N))
  is_target_prefix = ntuple(j -> target_pos_of_nat[j] <= target_P, Val(N))
  # 3. For each natural block, iterate over its dense slab using nat-axes order.
  #    For each dense element, compute the target block key (from prefix-positioned axes)
  #    and the target dense offset (from dense-positioned axes).
  nat_blksize = bs_nat.blksize
  nat_dims = bs_nat.dims
  nat_suffix_dims = ntuple(j -> nat_dims[Pnat + j], Val(N2nat))
  nat_suffix_CI = CartesianIndices(nat_suffix_dims)
  nat_suffix_LI = LinearIndices(nat_suffix_dims)
  # Target dimensions
  target_dims = ntuple(k -> ITensors.dim(target_inds[k]), Val(N))
  target_suffix_dims = ntuple(k -> target_dims[target_P + k], Val(target_n2))
  target_suffix_LI = LinearIndices(target_suffix_dims)
  target_blksize = prod(target_suffix_dims; init=1)
  # Build target keys via Dict-dedup
  new_keys = NTuple{target_P,Int}[]
  new_data_dict = Dict{NTuple{target_P,Int}, Vector{TC}}()
  @inbounds for bi in eachindex(bs_nat.keys)
    nat_prefix = bs_nat.keys[bi]
    bid = bs_nat.ids[bi]
    base = (bid - 1) * nat_blksize
    for sCI in nat_suffix_CI
      suffix = Tuple(sCI)::NTuple{N2nat,Int}
      lin = nat_suffix_LI[sCI]
      val = bs_nat.data[base + lin]
      # Build full N-tuple of natural indices for this element
      nat_full = _full_index(nat_prefix, suffix, Val(N))
      # Build target prefix tuple and target dense linear index
      tprefix = ntuple(k -> nat_full[nat_pos_of_target[k]], Val(target_P))
      tsuffix = ntuple(k -> nat_full[nat_pos_of_target[target_P + k]], Val(target_n2))
      tlin = target_suffix_LI[CartesianIndex(tsuffix)]
      buf = get(new_data_dict, tprefix, nothing)
      if buf === nothing
        buf = zeros(TC, target_blksize)
        new_data_dict[tprefix] = buf
        push!(new_keys, tprefix)
      end
      buf[tlin] = val
    end
  end
  # 4. Sort keys lex (BS invariant) and assemble final .data buffer
  target_prefix_dims = ntuple(k -> target_dims[k], Val(target_P))
  sort!(new_keys; by = k -> _prefix_lin(k, target_prefix_dims))
  new_ids = collect(1:length(new_keys))
  new_data = Vector{TC}(undef, length(new_keys) * target_blksize)
  @inbounds for (i, k) in enumerate(new_keys)
    buf = new_data_dict[k]
    copyto!(new_data, (i-1)*target_blksize + 1, buf, 1, target_blksize)
  end
  bs_new = NewBlockSparseSorted{TC, N, target_n2, target_P, Int}(
      target_dims, target_blksize, new_keys, new_ids, new_data)
  return WrappedBlockSparse(bs_new, target_inds)
end

const RECAST_SCRATCH = Dict{Tuple{DataType, NTuple, NTuple}, NamedTuple}()

@inline function _get_recast_scratch(::Type{TC}, dims_C::NTuple{N,Int}, dims_T::NTuple{N,Int}) where {TC, N}
  key = (TC, dims_C, dims_T)
  buf = get(RECAST_SCRATCH, key, nothing)
  if buf === nothing
    new_buf = (data_dense = Array{TC,N}(undef, dims_C),
               data_perm  = Array{TC,N}(undef, dims_T))
    RECAST_SCRATCH[key] = new_buf
    return new_buf
  end
  return buf::NamedTuple{(:data_dense, :data_perm), Tuple{Array{TC,N}, Array{TC,N}}}
end

# Clear scratch pool. Called between eigsolves / bonds if buffer dims churn.
recast_scratch_clear!() = empty!(RECAST_SCRATCH)

function _recast_per_block(Cw::WrappedBlockSparse{TC,N,N2c,Pc},
                            Tw::WrappedBlockSparse{TT,N,N2t,Pt},
                            c_inds, t_inds) where {TC,TT,N,N2c,Pc,N2t,Pt}
  # Build full-axis permutation: c_axis_for_t[k] = position in C.inds of t_inds[k]
  c_axis_for_t = ntuple(k -> findfirst(==(t_inds[k]), c_inds), Val(N))
  any(==(nothing), c_axis_for_t) && return nothing  # shouldn't happen given Set check
  # Every T-prefix axis must map to a C-prefix axis, AND every T-dense axis to
  # a C-dense axis. Otherwise classification crosses; bail to slow path.
  @inbounds for k in 1:Pt
    c_axis_for_t[k] <= Pc || return nothing
  end
  @inbounds for k in (Pt+1):N
    c_axis_for_t[k] > Pc || return nothing
  end
  # σ_prefix[j] (j ∈ 1..Pc) = position in T.prefix of C.prefix axis j
  # Inverse view: for each C prefix axis j, find which T prefix axis maps there.
  σ_prefix = Vector{Int}(undef, Pc)
  @inbounds for k in 1:Pt
    σ_prefix[c_axis_for_t[k]] = k
  end
  # perm_block[k] = C-dense axis (1..N2c) that becomes T-dense axis k
  perm_block = ntuple(k -> c_axis_for_t[Pt + k] - Pc, Val(N2t))

  bs_c = Cw.blocksparse
  bs_t = Tw.blocksparse
  blksize_c = bs_c.blksize
  blksize_t = bs_t.blksize
  blksize_c == blksize_t || return nothing  # shouldn't differ; safety

  c_dense_dims = ntuple(i -> bs_c.dims[Pc + i], Val(N2c))
  t_dense_dims = ntuple(i -> bs_t.dims[Pt + i], Val(N2t))

  c_keymap = Dict{NTuple{Pc,Int}, Int}()
  sizehint!(c_keymap, length(bs_c.keys))
  @inbounds for i in eachindex(bs_c.keys)
    c_keymap[bs_c.keys[i]] = bs_c.ids[i]
  end

  new_keys = copy(bs_t.keys)
  new_ids  = copy(bs_t.ids)
  new_data = Vector{TC}(undef, length(bs_t.data))

  is_identity_block = all(perm_block[i] == i for i in 1:N2t)

  @inbounds for i in eachindex(bs_t.keys)
    tk = bs_t.keys[i]
    tid = bs_t.ids[i]
    t_base = (tid - 1) * blksize_t
    # ck[j] = tk[σ_prefix[j]]  for j ∈ 1..Pc
    ck = ntuple(j -> tk[σ_prefix[j]], Val(Pc))
    c_id = get(c_keymap, ck, 0)
    if c_id == 0
      for j in 1:blksize_t
        new_data[t_base + j] = zero(TC)
      end
    else
      c_base = (c_id - 1) * blksize_c
      if is_identity_block
        copyto!(new_data, t_base + 1, bs_c.data, c_base + 1, blksize_t)
      else
        c_block = reshape(view(bs_c.data, c_base+1:c_base+blksize_c), c_dense_dims)
        t_block = reshape(view(new_data, t_base+1:t_base+blksize_t), t_dense_dims)
        Base.permutedims!(t_block, c_block, perm_block)
      end
    end
  end

  bs_new = NewBlockSparseSorted{TC,N,N2t,Pt,Int}(
      bs_t.dims, blksize_t, new_keys, new_ids, new_data)
  return WrappedBlockSparse(bs_new, Tw.inds)
end

function recast_bs_to_template(Cw::WrappedBlockSparse{TC,N,N2c,Pc},
                                Tw::WrappedBlockSparse{TT,N,N2t,Pt}) where {TC,TT,N,N2c,Pc,N2t,Pt}
  c_inds = collect(Cw.inds)
  t_inds = collect(Tw.inds)
  if Set(c_inds) != Set(t_inds)
    return Cw  # Cannot recast — Index identities differ.
  end
  # Layer-1 fast path: if structures already match, just key-filter (no densify).
  fast = @timeit TIMER "recast.layer1_try" filter_bs_keys_to_template(Cw, Tw)
  if fast !== nothing
    @timeit TIMER "recast.layer1_hit" begin end
    return fast
  end

  # Layer-2 fast path: when same dense/sparse classification (N2c == N2t and
  # every T-dense axis is also C-dense), skip the full densify. For each T key,
  # find C's block (via permuted key lookup), permute that small block's dense
  # axes into T's order, write to new_data. Cost O(num_keys × blksize) vs slow
  # path's O(prod(dims)) densify+permute. Triggers automatically whenever the
  # allowed_keys_C filter has made C.keys a subset of T.keys (after axis perm).
  if N2c == N2t
    fast2 = @timeit TIMER "recast.layer2_try" _recast_per_block(Cw, Tw, c_inds, t_inds)
    if fast2 !== nothing
      @timeit TIMER "recast.layer2_hit" begin end
      return fast2
    end
  end
  @timeit TIMER "recast.slow_densify" begin end
  # Permute Cw's dense form into the template's inds order.
  perm = ntuple(i -> findfirst(==(t_inds[i]), c_inds), Val(N))
  # Use cached scratch buffers for the two big intermediates. Output `new_data`
  # (held by the returned WrappedBlockSparse) is still freshly allocated.
  scratch = _get_recast_scratch(TC, Cw.blocksparse.dims, Tw.blocksparse.dims)
  data_dense = scratch.data_dense
  data_perm  = scratch.data_perm
  to_dense!(data_dense, Cw.blocksparse)
  if N === 0
    # 0-D edge case — scratch already-aligned.
    data_perm = data_dense
  else
    Base.permutedims!(data_perm, data_dense, perm)
  end

  # Build new storage with the template's exact (keys, ids) list. Gather
  # block data from data_perm using the template's key prefixes.
  bs_t = Tw.blocksparse
  dims_t = bs_t.dims
  blksize_t = bs_t.blksize
  P_t = Pt
  suffix_dims = ntuple(i -> dims_t[P_t + i], Val(N2t))
  suffix_CI = CartesianIndices(suffix_dims)
  suffix_LI = LinearIndices(suffix_dims)

  new_keys = copy(bs_t.keys)
  new_ids  = copy(bs_t.ids)
  new_data = Vector{TC}(undef, length(bs_t.data))

  @inbounds for i in eachindex(new_keys)
    prefix = new_keys[i]            # NTuple{P_t,Int}
    bid    = new_ids[i]
    base   = (bid - 1) * blksize_t
    for sCI in suffix_CI
      suffix  = Tuple(sCI)::NTuple{N2t,Int}
      full    = _full_index(prefix, suffix, Val(N))
      lin     = suffix_LI[sCI]
      new_data[base + lin] = data_perm[full...]
    end
  end

  bs_new = NewBlockSparseSorted{TC,N,N2t,P_t,Int}(
    dims_t, blksize_t, new_keys, new_ids, new_data
  )
  return WrappedBlockSparse(bs_new, Tw.inds)
end

function contract(A::ITensors.ITensor, B::WrappedTensorTypes{TB,NB}; kwargs...) where {TB,NB}
  @timeit TIMER "wrap_itensor(A,dense)" begin
    Aw = wrap_itensor(A; backend=:dense)
  end
  Cw = contract(Aw, B; kwargs...)
  # Cw = contract(Aw, B; kwargs...)
  Cw isa ITensors.ITensor && return Cw
  if Cw isa WrappedBlockSparse && is_dense(Cw.blocksparse)
    shared = Set(ITensors.inds(A)) ∩ Set(B.inds)
    println("Dense output D*S: ", ITensors.inds(A), " x ", B.inds, " = ", shared)
    @timeit TIMER "to_dense(C)" begin
      data = to_dense(Cw.blocksparse)
      out = ITensors.ITensor(data, Cw.inds...)
    end
    error("Output is dense but currently wrapped in a WrappedBlockSparse; consider returning a plain ITensor instead for better performance")
    return out
  end
  return ITensors._itensor_from_external_storage(Cw)
end

function contract(A::WrappedTensorTypes{TA,NA}, B::ITensors.ITensor; kwargs...) where {TA,NA}
  @timeit TIMER "wrap_itensor(B,dense)" begin
    Bw = wrap_itensor(B; backend=:dense)
  end
  Cw = contract(A, Bw; kwargs...)
  Cw isa ITensors.ITensor && return Cw
  if Cw isa WrappedBlockSparse && is_dense(Cw.blocksparse)
    shared = Set(A.inds) ∩ Set(ITensors.inds(B))
    println("Dense output S*D: ", A.inds, " x ", ITensors.inds(B), " = ", shared)
    @timeit TIMER "to_dense(C)" begin
      data = to_dense(Cw.blocksparse)
      out = ITensors.ITensor(data, Cw.inds...)
    end
    return out
  end
  return ITensors._itensor_from_external_storage(Cw)
end



@inline is_link(I) = ITensors.hastags(I, "Link")


# Sort link indices by their l=<n> tag (ascending) with cached levels
function sort_link_inds_cached!(link_inds::Vector{<:ITensors.Index}; missing::Symbol = :last)
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
    denseB::Set{ITensors.Index{T}};
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
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
    # moved_dense_links: when hint is provided, axes that were dense in operands
    # but classified sparse in output (kernel will FISSION them). Placed LAST
    # in the sparse section so lex-sort of C keys puts A_pref axes FIRST →
    # consecutive sorted C blocks for one A_pref differ only in moved axes →
    # consecutive in C.data → single mul! writes directly (free fission).
    moved_dense_links = Vector{ITensors.Index}(undef, 0); sizehint!(moved_dense_links, NA + NB)
    dense_tail        = Vector{ITensors.Index}(undef, 0); sizehint!(dense_tail,        NA + NB)

    # helper: classify a single index
    # Match hint by Index id (ignoring plev) — the contract output may have
    # axes at plev=1 that get replaceprime'd to plev=0 downstream, but they
    # have the SAME id as the hint entries (plev=0). So id-matching is correct.
    hint_ids = output_inds_hint === nothing ?
               nothing :
               Set(ITensors.id(I) for I in output_inds_hint)
    @inline function push_classified!(I::ITensors.Index)
        if hint_ids !== nothing
            if ITensors.id(I) in hint_ids
                push!(dense_tail, I)
            else
                # Sparse in output. If from an operand's DENSE set, it's a
                # "moved" axis (will be fissioned) → go LAST in sparse.
                from_dense = (I in denseA) || (I in denseB)
                if from_dense && is_link(I)
                    push!(moved_dense_links, I)
                elseif is_link(I)
                    push!(sparse_link, I)
                else
                    push!(sparse_nonlink, I)
                end
            end
            return nothing
        end
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
    sort_link_inds_cached!(moved_dense_links)
    # sparse_link LAST (simpler heuristic). Empirically plev-based subsort
    # gave only ~14% permA_identity hits and the per-call sort overhead
    # negated savings. The current ordering is a no-op cost-wise. Larger gain
    # would require eliminating the permutedims COPY (in-place / data-share).
    indsC_vec = vcat(sparse_nonlink, moved_dense_links, sparse_link, dense_tail)
    return Tuple(indsC_vec), dense_tail
end

function output_inds(
    indsA::NTuple{NA,ITensors.Index{T}},
    indsB::NTuple{NB,ITensors.Index{T}},
    denseA_in::AbstractSet{<:ITensors.Index},
    denseB_in::AbstractSet{<:ITensors.Index};
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
) where {NA,NB,T}
    denseA = Set{ITensors.Index{T}}(denseA_in)  # OK: iterable ctor
    denseB = Set{ITensors.Index{T}}(denseB_in)
    return output_inds(indsA, indsB, denseA, denseB; output_inds_hint=output_inds_hint)
end

# Reorder indsC so it matches the canonical layout that `contract_bs_dense_to_dense!`
# writes into Ctgt: [keepA (BS dense-tail kept), keepB (dense-side kept),
# c_prefix (BS sparse-prefix kept)]. When labelsC is in canonical order, the
# kernel takes its "write directly into C" branch and skips the trailing
# permute_back copy. Order within each group follows the BS side's / dense
# side's original index order, matching the kernel's classify pass.
function canonical_indsC_for_bd(
    indsBS::NTuple{NA,ITensors.Index{T}},
    denseBS::Set{ITensors.Index{T}},
    indsDense::NTuple{NB,ITensors.Index{T}},
    indsC::Tuple,
) where {NA,NB,T}
    setC = Set{ITensors.Index{T}}()
    @inbounds for I in indsC; push!(setC, I); end
    setBS = Set{ITensors.Index{T}}()
    @inbounds for I in indsBS; push!(setBS, I); end
    keepA    = ITensors.Index{T}[]; sizehint!(keepA, NA)
    c_prefix = ITensors.Index{T}[]; sizehint!(c_prefix, NA)
    @inbounds for I in indsBS
        (I in setC) || continue
        if I in denseBS
            push!(keepA, I)
        else
            push!(c_prefix, I)
        end
    end
    keepB = ITensors.Index{T}[]; sizehint!(keepB, NB)
    @inbounds for I in indsDense
        if (I in setC) && !(I in setBS)
            push!(keepB, I)
        end
    end
    return Tuple(vcat(keepA, keepB, c_prefix))
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
    # Install the aliased inner ⟨y|x⟩ fast-path into ITensors.inner so KrylovKit's
    # Lanczos dots reduce position-wise over shared template buffers instead of a
    # full aliased×aliased contraction (the `wrapped×wrapped` dots). Mutating the
    # Ref's contents (not a method def) ⇒ no precompile conflict with the
    # ITensorsVectorInterfaceExt extension. Gated; default ON.
    ITensors._INNER_FASTPATH[] = _sparse_inner_fastpath
end

# ⟨y|x⟩ = Σ conj(y)·x for key-aligned aliased operands; nothing ⇒ fall through to
# the standard ITensors contraction. _alias_inner conjugates the first arg, exactly
# matching ITensors.inner(y,x) = (dag(y)*x)[].
function _sparse_inner_fastpath(y::ITensors.ITensor, x::ITensors.ITensor)
    # HARDENED: always taken for key-aligned aliased operands (bit-identical,
    # faster); returns nothing → standard contraction only when not applicable.
    yw = _alias_storage(y); xw = _alias_storage(x)
    (yw !== nothing && xw !== nothing) || return nothing
    return @timeit TIMER "vecop.inner" _alias_inner(yw, xw)
end

# Task-local contract-label scratch. Previously `_scratch[Threads.threadid()]`,
# which is UNSAFE under Threads.@spawn: tasks can migrate threads mid-call, so two
# concurrent contractions (e.g. KrylovKit's threaded orthogonalization calling
# inner() on aliased vectors, or our own threaded matvec kernel) could collide on
# the same scratch slot and corrupt the label buffers. task_local_storage is keyed
# by TASK, not thread, so it is migration-safe. Bit-identical for the serial main
# task (one task ⇒ one ContractScratch, reused across all calls exactly as before).
# (`_scratch`/`__init__` population below is now vestigial — kept to avoid churn.)
@inline scratch() = get!(() -> ContractScratch(Label[], Label[], Label[]),
                         task_local_storage(), :sb_contract_scratch)::ContractScratch

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



# Build the set of allowed C-prefix keys, in C's prefix order, from a template
# WrappedBlockSparse. The template's keys are in its own prefix order; we permute
# them so each j-th entry of the C-tuple is the value from the template axis
# whose Index id matches indsC[j]. Returns nothing if template/indsC are
# structurally incompatible (different PC, or some indsC[j] not found in
# template) — in that case the caller skips filtering.
function _allowed_keys_from_template(Tw::WrappedBlockSparse{TT,NT,N2T,PT},
                                     indsC::NTuple{N,ITensors.Index},
                                     denseLinksC::Int) where {TT,NT,N2T,PT,N}
  PC = N - denseLinksC
  PC == PT || return nothing  # classification mismatch
  N == NT || return nothing
  # σ[j] = position (in 1..PT) of indsC[j] within Tw.inds, matching by Index id.
  σ = Vector{Int}(undef, PC)
  t_ids = ntuple(i -> ITensors.id(Tw.inds[i]), Val(length(Tw.inds)))
  @inbounds for j in 1:PC
    target_id = ITensors.id(indsC[j])
    pos = 0
    for i in 1:PT
      if t_ids[i] == target_id
        pos = i; break
      end
    end
    pos == 0 && return nothing  # indsC[j] not in template prefix
    σ[j] = pos
  end
  PCv = PC
  allowed = Set{NTuple{PCv, Int}}()
  sizehint!(allowed, length(Tw.blocksparse.keys))
  @inbounds for tk in Tw.blocksparse.keys
    push!(allowed, ntuple(j -> tk[σ[j]], Val(PCv)))
  end
  return allowed
end

function wrapped_contract(A::WrappedTensorTypes{TA,NA},
                          B::WrappedTensorTypes{TB,NB};
                          output_backend::Symbol=:blocksparse,
                          preserve_bs_output::Bool=false,
                          output_inds_hint::Union{Nothing,AbstractSet}=nothing,
                          template_for_filter::Union{Nothing,WrappedBlockSparse}=nothing,
                          in_position::Bool=false) where {TA,TB,NA,NB}
  # `preserve_bs_output=true` is an opt-in pathway that forces the result of
  # a BlockSparse-involving contraction to remain WrappedBlockSparse, even
  # when the natural output has no sparse axes (P_C = 0) or shares only
  # dense indices. The default (false) preserves the existing behavior of
  # returning a plain dense ITensor in those cases. Used by Path-B sparse
  # DMRG where the Krylov subspace requires storage-type stability across
  # operator applications.
  @timeit TIMER "wrapped_contract" begin
    @timeit TIMER "wc.setup" begin
      Arep = rep(A)
      Brep = rep(B)
      indsA = A.inds
      indsB = B.inds
      denseA = dense_inds(A)
      denseB = dense_inds(B)
    end
    if false  # SB_TRACE — flip to true here for debug output
      println("[SB_TRACE] wrapped_contract  A=", _backend(A), "{T=", TA, ",N=", NA, "}",
              "  B=", _backend(B), "{T=", TB, ",N=", NB, "}",
              "  → C=", infer_C_backend(A, B))
    end
    @timeit TIMER "wc.output_inds" begin
      # Gate: hint path only supported by the BS×Dense kernel. For BS×BS or
      # other paths, suppress the hint so output_inds + kernel run the
      # natural ordering (no regression on those paths).
      hint_for_kernel = (Arep isa NewBlockSparseSorted && Brep isa NewBlockSparseSorted) ?
                       nothing : output_inds_hint
      indsC, denseC = output_inds(indsA, indsB, denseA, denseB; output_inds_hint=hint_for_kernel)
    end
    @timeit TIMER "wc.labels" begin
      dimsC = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
      TC = promote_type(eltype(Arep), eltype(Brep))
      Cbackend = infer_C_backend(A, B)
      sc = scratch()
      labelsA_vec = fill_labels!(sc.labelsA, indsA)
      labelsB_vec = fill_labels!(sc.labelsB, indsB)
      labelsC_vec = fill_labels!(sc.labelsC, indsC)
    end
    if Cbackend === :blocksparse
        denseLinksC = length(denseC)
        if denseLinksC == length(indsC) && !preserve_bs_output
            # P_C = 0: all output indices are dense → return plain ITensor
            @timeit TIMER "wc.alloc_dense" begin
              C_data = zeros(TC, dimsC...)
            end
            if Arep isa NewBlockSparseSorted && Brep isa NewBlockSparseSorted
              @timeit TIMER "kern.bs_bs_to_dense" begin
                SparseBackends.contract_bs_bs_to_dense!(
                    C_data, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec)
              end
            elseif Arep isa NewBlockSparseSorted && Brep isa AbstractArray
              @timeit TIMER "kern.bs_dense_to_dense" begin
                SparseBackends.contract_bs_dense_to_dense!(
                    C_data, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec; in_position=in_position)
              end
            elseif Arep isa AbstractArray && Brep isa NewBlockSparseSorted
              @timeit TIMER "kern.bs_dense_to_dense" begin
                SparseBackends.contract_bs_dense_to_dense!(
                    C_data, labelsC_vec, Brep, labelsB_vec, Arep, labelsA_vec; in_position=in_position)
              end
            else
              @timeit TIMER "kern.dense_dense_einsum" begin
                C_bs = NewBlockSparseSorted{TC, length(dimsC), length(dimsC)}(dimsC)
                contract!(C_bs, labelsC_vec, NewBlockSparseSorted(Arep), labelsA_vec, Brep, labelsB_vec)
              end
            end
            @timeit TIMER "wc.wrap_output" begin
              if get(ENV, "SB_NO_ALIAS_OUTPUT", "0") == "1"
                out = length(indsC) == 0 ? ITensors.ITensor(C_data[]) : ITensors.ITensor(C_data, indsC...)
              else
                out = length(indsC) == 0 ? ITensors.ITensor(C_data[]) :
                      ITensors.ITensor(ITensors.AllowAlias(), C_data, indsC...)
              end
            end
            return out
        end

        # if output_backend === :dense
        shared = indsA ∩ indsB
        # println("Shared indices: ", shared, " with dense ", denseA, " and ", denseB)
        if (A isa WrappedTensor && B isa WrappedBlockSparse) ||
          (A isa WrappedBlockSparse && B isa WrappedTensor)
          if preserve_bs_output
            output_backend = :blocksparse
          else
            output_backend = :dense
            for I in shared
              if !(I in denseA) && !(I in denseB)
                output_backend = :blocksparse
                break
              end
            end
          end
          # if output_backend === :dense
          #   println("Output will be dense because shared index ", shared, " is dense")
          # else
          #   println("Output will be block-sparse because shared index ", shared, " is sparse")
          # end
        end

        # SB_TRACE_ONE removed 2026-06 — was a fire-once debug trace; hardcoded off.
        _trace = false
        if output_backend === :dense
          if _trace
            println("\n[TRACE] wrapped_contract bs×dense path")
            println("  indsA (BS):   ", [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in indsA])
            println("  indsB (dense):", [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in indsB])
            println("  indsC (pre-canonical):", [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in indsC])
            println("  denseA (BS dense axes):", [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in denseA])
          end
          # Reorder indsC to canonical [keepA, keepB, c_prefix] for the bd
          # kernel so it skips its trailing permute_back copy. Only applies
          # to BS×Dense (and Dense×BS); BS×BS keeps the original output_inds
          # order. ITensor identifies by Index identity, so the reorder is
          # invisible to callers.
          if Arep isa NewBlockSparseSorted && !(Brep isa NewBlockSparseSorted)
              @timeit TIMER "wc.canonicalize" begin
                indsC = canonical_indsC_for_bd(indsA, denseA, indsB, indsC)
                dimsC = ntuple(i -> ITensors.dim(indsC[i]), length(indsC))
                labelsC_vec = fill_labels!(sc.labelsC, indsC)
              end
              if _trace
                println("  indsC (post-canonical [keepA,keepB,c_prefix]):",
                        [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in indsC])
              end
          end
          @timeit TIMER "wc.alloc_dense" begin
            C_data = zeros(TC, dimsC...)
          end
          if Arep isa NewBlockSparseSorted
            if Brep isa NewBlockSparseSorted
              @timeit TIMER "kern.bs_bs_to_dense" begin
                SparseBackends.contract_bs_bs_to_dense!(
                    C_data, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec)
              end
            else Brep isa AbstractArray
              @timeit TIMER "kern.bs_dense_to_dense" begin
                SparseBackends.contract_bs_dense_to_dense!(
                    C_data, labelsC_vec, Arep, labelsA_vec, Brep, labelsB_vec; in_position=in_position)
              end
            end
          end
          @timeit TIMER "wc.wrap_output" begin
            # C_data is freshly allocated in wc.alloc_dense and not referenced
            # elsewhere — safe to alias into the returned ITensor instead of
            # copying. Saved ~3.8 s + 4.4 GiB at md=80 single-core.
            # Gate-able via SB_NO_ALIAS_OUTPUT=1 to compare against the
            # copying path for regression checks.
            if get(ENV, "SB_NO_ALIAS_OUTPUT", "0") == "1"
              out = length(indsC) == 0 ? ITensors.ITensor(C_data[]) : ITensors.ITensor(C_data, indsC...)
            else
              out = length(indsC) == 0 ? ITensors.ITensor(C_data[]) :
                    ITensors.ITensor(ITensors.AllowAlias(), C_data, indsC...)
              if _trace
                println("  inds(out) wrapped: ", [(ITensors.dim(i), ITensors.tags(i), ITensors.plev(i)) for i in inds(out)])
                println("  size(C_data): ", size(C_data))
                println("  dimsC: ", dimsC)
                ENV["SB_TRACE_ONE"] = "0"  # only trace once
              end
            end
          end
          return out
        else
          @timeit TIMER "wc.alloc_bs" begin
            C = WrappedBlockSparse(TC, dimsC, denseLinksC, indsC)
          end
          # Build allowed_keys_C from template by permuting template's keys
          # into C's prefix coordinate system. Only when (a) template provided,
          # (b) hint path is active (BS×Dense only), (c) prefix sizes match.
          allowed_keys_C = nothing
          if template_for_filter !== nothing && hint_for_kernel !== nothing &&
             Arep isa NewBlockSparseSorted && !(Brep isa NewBlockSparseSorted)
            @timeit TIMER "wc.build_allowed_keys" begin
              allowed_keys_C = _allowed_keys_from_template(
                  template_for_filter, indsC, denseLinksC)
            end
          end
          @timeit TIMER "kern.contract!_bs_out" begin
            C.blocksparse = SparseBackends.contract!(
                C.blocksparse, labelsC_vec,
                Arep, labelsA_vec,
                Brep, labelsB_vec;
                output_inds_hint=hint_for_kernel,
                allowed_keys_C=allowed_keys_C,
            )
          end
        end
        return C
    elseif Cbackend === :coo
        @timeit TIMER "wc.alloc_coo" begin
          C = WrappedCOOTensor(TC, dimsC, indsC)
        end
        @timeit TIMER "kern.contract!_coo_out" begin
          C.coo = SparseBackends.contract!(
              C.coo, labelsC_vec,
              Arep, labelsA_vec,
              Brep, labelsB_vec
          )
        end
        return C
    else
        error("Unsupported backend: $Cbackend")
    end
  end  # @timeit "wrapped_contract"
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
  elseif w isa WrappedAliasedBlockSparse
      return to_dense(w.aliased)
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

"""
    to_dense_itensors_unfused(T) -> ITensor

Same as `to_dense_itensors` but does NOT fuse doubled-link indices. Returns a
dense ITensor with EXACTLY the same Index objects as `T`, just dense storage.
Required when downstream code (e.g. `factorize`/`uniqueinds`) needs to keep
Index identity stable across the conversion.
"""
function to_dense_itensors_unfused(T::ITensors.ITensor)::ITensors.ITensor
  if T.tensor isa ITensors.ExternalStorage
    es = T.tensor::ITensors.ExternalStorage
    data = es.data
    if data isa WrappedBlockSparse
      arr = to_dense(data.blocksparse)         # no merged_axes — keep all axes
      return ITensors.ITensor(arr, ITensors.inds(T)...)
    end
  end
  return ITensors.ITensor(to_dense(T), ITensors.inds(T)...)
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

import Base: ndims, size, eltype, *, +, -, copy
import LinearAlgebra: norm

Base.ndims(es::ITensors.ExternalStorage) = length(ITensors.inds(es))
Base.size(es::ITensors.ExternalStorage) = Tuple(ITensors.dim.(ITensors.inds(es)))
Base.eltype(es::ITensors.ExternalStorage{S}) where {S} = eltype(es.data)

LinearAlgebra.norm(w::WrappedBlockSparse) = norm(w.blocksparse.data)
LinearAlgebra.norm(w::WrappedCOOTensor)   = norm(w.coo.vals)
LinearAlgebra.norm(w::WrappedTensor)      = norm(w.data)
LinearAlgebra.norm(es::ITensors.ExternalStorage{<:WrappedTensorTypes}) = norm(es.data)

Base.eltype(::WrappedBlockSparse{T}) where T = T

# ── Scalar multiply for BS tensors (needed by KrylovKit/VectorInterface) ─────
function Base.:*(α::Number, wbs::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    bs = wbs.blocksparse
    new_bs = NewBlockSparseSorted{T,N,N2,P}(
        bs.dims, bs.blksize, copy(bs.keys), copy(bs.ids), T(α) .* bs.data
    )
    WrappedBlockSparse(new_bs, wbs.inds)
end

# Returns an ITensor so that `itensor(α * tensor(T))` = `itensor(ITensor)` = identity
Base.:*(α::Number, es::ITensors.ExternalStorage{<:WrappedBlockSparse}) =
    ITensors._itensor_from_external_storage(α * es.data)

Base.:-(wbs::WrappedBlockSparse) = (-1) * wbs
Base.:/(wbs::WrappedBlockSparse, α::Number) = (one(eltype(wbs))/α) * wbs

# ── Deep copy ─────────────────────────────────────────────────────────────────
function Base.copy(wbs::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    bs = wbs.blocksparse
    new_bs = NewBlockSparseSorted{T,N,N2,P}(
        bs.dims, bs.blksize, copy(bs.keys), copy(bs.ids), copy(bs.data)
    )
    WrappedBlockSparse(new_bs, wbs.inds)
end

# ── Block-wise addition (union of block keys) ─────────────────────────────────
function Base.:+(wA::WrappedBlockSparse{T,N,N2,P}, wB::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P}
    C = copy(wA)
    bs_C = C.blocksparse
    bs_B = wB.blocksparse
    for (key, id_b) in blocks_sorted(bs_B)
        id_c = _ensure_block!(bs_C, key)
        bv_c = _block_view(bs_C, id_c)
        bv_b = _block_view(bs_B, id_b)
        @inbounds for i in eachindex(bv_c)
            bv_c[i] += bv_b[i]
        end
    end
    C
end

# Hook for ITensors addition: a + b calls _add(tensor(a), tensor(b))
function ITensors._add(
    es_A::ITensors.ExternalStorage{<:WrappedBlockSparse},
    es_B::ITensors.ExternalStorage{<:WrappedBlockSparse},
)
    ITensors._itensor_from_external_storage(es_A.data + es_B.data)
end

# Hook for fill!(T, x) with external storage (needed by zerovector!/broadcast scalar fill)
function ITensors._external_fill!(T::ITensors.ITensor, storage::WrappedBlockSparse, x::Number)
    fill!(storage.blocksparse.data, x)
    return T
end

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

# ITensors hook: `_external_map_storage!(f, storage, R, A)` where storage = get_external_storage(R)
function ITensors._external_map_storage!(
    f::Function,
    storage::WrappedTensorTypes,
    R::ITensors.ITensor,
    A::ITensors.ITensor,
)
    sA = ITensors.get_external_storage(A)
    _apply_elementwise!(f, storage, sA)
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
  @timeit TIMER "ext_dispatch[dense×wrapped]" begin
    result = contract(A, Bw; kwargs...)
  end
  return result
end


# external-storage × ITensor
function ITensors._contract_external_storage(Aw::WrappedTensorTypes,
                                             B::ITensors.ITensor; kwargs...)
  @timeit TIMER "ext_dispatch[wrapped×dense]" begin
    result = contract(Aw, B; kwargs...)
  end
  return result
end

function ITensors._contract_external_storage(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes; kwargs...)
  @timeit TIMER "ext_dispatch[wrapped×wrapped]" begin
    result = contract(Aw, Bw; kwargs...)
    result isa ITensors.ITensor && return result  # P_C=0: already a plain Dense ITensor, use directly
    return ITensors._itensor_from_external_storage(result)
  end
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

# # Sparse-psi binding: block-diagonal SVD for WrappedBlockSparse phi.
# # Returns (L_it, R_it, spec) — same contract as stable_factorize's return.
# # w.inds layout must be (l_link, r_link, site1, site2): prefix first, dense last.
# function itensor_blocksparse_svd(
#     phi::ITensors.ITensor,
#     indsMb;
#     ortho::String="left",
#     maxdim::Int=typemax(Int),
#     mindim::Int=1,
#     cutoff::Float64=0.0,
#     tags=ITensors.ts"Link,l",
# )
#     w  = ITensors.get_external_storage(phi)::WrappedBlockSparse
#     println("Performing block-diagonal SVD on WrappedBlockSparse with dims ", w.inds, " and block dims ", w.blocksparse.dims)

#     bs = w.blocksparse   # NewBlockSparseSorted{T,4,2,2}
#     # w.inds = (l_link, r_link, site1, site2) = (prefix1, prefix2, dense1, dense2)
#     l_ind, r_ind, s1_ind, s2_ind = w.inds

#     U_bs, SV_bs, svs_kept, spec = blocksparse_svd(
#         bs; ortho, maxdim, mindim, cutoff
#     )

#     new_bond_dim = length(svs_kept)
#     new_bond_ind = ITensors.Index(new_bond_dim; tags)

#     # U: prefix=(l, new_bond), dense=(s1)
#     L_it = ITensors._itensor_from_external_storage(
#         WrappedBlockSparse(U_bs, (l_ind, new_bond_ind, s1_ind))
#     )
#     # SV: prefix=(new_bond, r), dense=(s2)
#     R_it = ITensors._itensor_from_external_storage(
#         WrappedBlockSparse(SV_bs, (new_bond_ind, r_ind, s2_ind))
#     )
#     return L_it, R_it, spec
# end

function permute(
    phi::WrappedBlockSparse,
    desired::Vararg{ITensors.Index};
    allow_alias::Bool = true,
)
  N = length(phi.inds)
  # 1. Validate inputs
  length(desired) == N || error("Wrong number of indices in permute")
  inds_old = phi.inds
  inds_new = Tuple(desired)
  # Build permutation: where each new index came from
  perm = ntuple(i -> begin
      idx = findfirst(==(inds_new[i]), inds_old)
      idx === nothing && error("Index $(inds_new[i]) not found in tensor")
      idx
  end, N)
  # Check it's a valid permutation
  if sort(collect(perm)) != collect(1:N)
      error("Invalid permutation")
  end
  # 2. Fast path: identity perm
  if perm == ntuple(identity, N)
      return allow_alias ? phi : WrappedBlockSparse(
          copy(phi.blocksparse),
          inds_new
      )
  end
  # 3. Apply permutation to storage
  # We assume this returns a new object
  bs_new = permutedims(phi.blocksparse, perm)
  # 4. Construct result
  return WrappedBlockSparse(bs_new, inds_new)
end

function permute(phi::ITensors.ITensor, desired::Vararg{ITensors.Index}; allow_alias::Bool = true)
  w = ITensors.get_external_storage(phi)::WrappedBlockSparse
  return ITensors._itensor_from_external_storage(permute(w, desired...; allow_alias))
end


function reorder_invariant(legs, dense_set)
  sp_link    = ITensors.Index[]
  sp_nonlink = ITensors.Index[]
  dense_tail = ITensors.Index[]
  for I in legs
    if I in dense_set
      push!(dense_tail, I)
    elseif is_link(I)
      push!(sp_link, I)
    else
      push!(sp_nonlink, I)
    end
  end
  sort_link_inds_cached!(sp_link)
  return sp_link, sp_nonlink, dense_tail
end


# Channel-aware SVD: factorizes phi back into L, R where L and R inherit
# the EXACT block-key structure of the OLD M[b] and M[b+1] respectively.
# The new bond reuses the old bond's sparse Index, so each channel value k
# (= old bond sparse-axis value) labels the corresponding new-bond sector.
# Phi blocks are partitioned by channel via lookups built from M[b]/M[b+1]'s
# block keys: for each phi block (lk_tuple, rk_tuple) the unique channel k
# satisfying (lk_tuple, k) ∈ M[b]_blocks AND (k, rk_tuple) ∈ M[b+1]_blocks is
# located, and phi is SVD'd per channel. Cross-channel rows are disjoint
# (each channel pairs with disjoint (lk, k) sets), so the isometric factor
# is globally isometric.
function itensor_blocksparse_svd_channel_aware(
    phi::ITensors.ITensor,
    M_b::ITensors.ITensor,
    M_b1::ITensors.ITensor;
    ortho::String   = "left",
    maxdim::Int     = typemax(Int),
    mindim::Int     = 1,
    cutoff::Float64 = 0.0,
)
    @assert ITensors.has_external_storage(phi) "phi must be block-sparse"
    @assert ITensors.has_external_storage(M_b) "M_b must be block-sparse"
    @assert ITensors.has_external_storage(M_b1) "M_b1 must be block-sparse"
    w_phi = ITensors.get_external_storage(phi)::WrappedBlockSparse
    w_b   = ITensors.get_external_storage(M_b)::WrappedBlockSparse
    w_b1  = ITensors.get_external_storage(M_b1)::WrappedBlockSparse

    # Identify OLD shared bond between M_b and M_b1.
    shared      = collect(ITensors.commoninds(M_b, M_b1))
    dense_b     = dense_inds(w_b)
    bond_sparse = first(I for I in shared if !(I in dense_b))
    bond_mult   = first(I for I in shared if  (I in dense_b))
    bond_sp_dim = ITensors.dim(bond_sparse)

    # indsMb = legs of M_b that survive into phi's L-side (everything except shared bond).
    indsMb = [I for I in ITensors.inds(M_b) if !(I in shared)]

    # ----- bipartition phi (mirror itensor_blocksparse_svd) -----
    phi_inds_all = collect(w_phi.inds)
    dense_set    = Set(dense_inds(w_phi))
    in_U   = Set(filter(i -> i ∈ phi_inds_all, indsMb))
    U_legs = filter(i ->  i ∈ in_U, phi_inds_all)
    V_legs = filter(i -> !(i ∈ in_U), phi_inds_all)
    U_spL, U_spN, U_d = reorder_invariant(U_legs, dense_set)
    V_spL, V_spN, V_d = reorder_invariant(V_legs, dense_set)
    nls = length(U_spL) + length(U_spN)
    nrs = length(V_spL) + length(V_spN)
    nld = length(U_d)
    nrd = length(V_d)

    desired = (U_spL..., U_spN..., V_spL..., V_spN..., U_d..., V_d...)
    phi_p   = permute(phi, desired...; allow_alias = true)
    bs_p    = ITensors.get_external_storage(phi_p).blocksparse

    L_phi_sparse = (U_spL..., U_spN...)
    R_phi_sparse = (V_spL..., V_spN...)

    # ---- Build templates from M_b, M_b1 blocks -------------------------------
    M_b_inds        = collect(w_b.inds)
    M_b1_inds       = collect(w_b1.inds)
    dense_b1        = dense_inds(w_b1)
    M_b_sparse_pos  = [i for i in 1:length(M_b_inds)  if !(M_b_inds[i]  in dense_b)]
    M_b1_sparse_pos = [i for i in 1:length(M_b1_inds) if !(M_b1_inds[i] in dense_b1)]

    bond_pos_in_b  = findfirst(p -> M_b_inds[p]  == bond_sparse, M_b_sparse_pos)
    bond_pos_in_b1 = findfirst(p -> M_b1_inds[p] == bond_sparse, M_b1_sparse_pos)

    # Map each non-bond sparse axis of M_b to its slot in L_phi_sparse storage order.
    non_bond_b_pos  = [p for (i, p) in enumerate(M_b_sparse_pos)  if i != bond_pos_in_b]
    non_bond_b1_pos = [p for (i, p) in enumerate(M_b1_sparse_pos) if i != bond_pos_in_b1]
    perm_b  = [findfirst(I -> I == M_b_inds[p],  L_phi_sparse) for p in non_bond_b_pos]
    perm_b1 = [findfirst(I -> I == M_b1_inds[p], R_phi_sparse) for p in non_bond_b1_pos]
    @assert all(!isnothing, perm_b)  "M_b non-bond sparse axes don't all map into L_phi_sparse"
    @assert all(!isnothing, perm_b1) "M_b1 non-bond sparse axes don't all map into R_phi_sparse"

    Kt = eltype(eltype(w_b.blocksparse.keys))
    LTupT = NTuple{nls, Kt}
    RTupT = NTuple{nrs, Kt}

    left_template = Tuple{LTupT, Kt}[]
    for key in w_b.blocksparse.keys
        vals     = [key[p] for p in M_b_sparse_pos]
        ch       = Kt(vals[bond_pos_in_b])
        non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b]
        lk       = Vector{Kt}(undef, nls)
        for (s, d) in enumerate(perm_b); lk[d] = Kt(non_bond[s]); end
        push!(left_template, (NTuple{nls,Kt}(lk), ch))
    end
    right_template = Tuple{Kt, RTupT}[]
    for key in w_b1.blocksparse.keys
        vals     = [key[p] for p in M_b1_sparse_pos]
        ch       = Kt(vals[bond_pos_in_b1])
        non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b1]
        rk       = Vector{Kt}(undef, nrs)
        for (s, d) in enumerate(perm_b1); rk[d] = Kt(non_bond[s]); end
        push!(right_template, (ch, NTuple{nrs,Kt}(rk)))
    end

    # Auto-detect bond factor structure from M_b's and M_b1's other LINK sparse
    # axes. If their dims multiply to bond_sp_dim, the bond is at the overlap of
    # two (I+C) factors → factor_dims = [left_link_dim, right_link_dim]. Else
    # treat as monolithic (single factor).
    other_link_dim_b  = 1
    for p in M_b_sparse_pos
        I = M_b_inds[p]
        if I != bond_sparse && ITensors.hastags(I, "Link")
            other_link_dim_b = max(other_link_dim_b, ITensors.dim(I))
        end
    end
    other_link_dim_b1 = 1
    for p in M_b1_sparse_pos
        I = M_b1_inds[p]
        if I != bond_sparse && ITensors.hastags(I, "Link")
            other_link_dim_b1 = max(other_link_dim_b1, ITensors.dim(I))
        end
    end
    bond_factor_dims = if other_link_dim_b * other_link_dim_b1 == bond_sp_dim &&
                          other_link_dim_b > 1 && other_link_dim_b1 > 1
        Int[other_link_dim_b, other_link_dim_b1]
    else
        Int[bond_sp_dim]
    end

    # SB_USE_GROUPED_SVD deprecated
    U_bs, SV_bs, svs_kept, spec = blocksparse_svd_channel_aware(bs_p;
        n_left_sparse = nls,
        n_left_dense  = nld,
        left_template, right_template,
        bond_sparse_dim = bond_sp_dim,
        bond_factor_dims = bond_factor_dims,
        ortho, maxdim, mindim, cutoff)

    # Reuse old bond_sparse Index as new bond's sparse axis. Fresh Index for mult.
    new_sp  = bond_sparse
    n_new_d = U_bs.dims[nls + 1 + nld + 1]
    new_d   = ITensors.Index(n_new_d; tags = ITensors.tags(bond_mult))

    U_inds_storage  = (U_spL..., U_spN..., new_sp, U_d..., new_d)
    SV_inds_storage = (new_sp, V_spL..., V_spN..., new_d, V_d...)

    @assert ntuple(i -> ITensors.dim(U_inds_storage[i]),  length(U_inds_storage)) ==
            U_bs.dims  "U inds/storage dim mismatch"
    @assert ntuple(i -> ITensors.dim(SV_inds_storage[i]), length(SV_inds_storage)) ==
            SV_bs.dims "SV inds/storage dim mismatch"

    L_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(U_bs,  U_inds_storage))
    R_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(SV_bs, SV_inds_storage))

    U_dense_set  = Set([U_d..., new_d])
    SV_dense_set = Set([V_d..., new_d])
    L_spL, L_spN, L_d = reorder_invariant(collect(U_inds_storage),  U_dense_set)
    R_spL, R_spN, R_d = reorder_invariant(collect(SV_inds_storage), SV_dense_set)
    L_it = permute(L_it, L_spL..., L_spN..., L_d...; allow_alias = true)
    R_it = permute(R_it, R_spL..., R_spN..., R_d...; allow_alias = true)

    return L_it, R_it, spec
end


# Sparse-axis dim of the bond shared between two adjacent block-sparse tensors.
# Returns -1 if either tensor lacks BlockSparse external storage (i.e. no
# structured target to preserve — caller will use the SVD's natural sparse dim).
function shared_bond_sparse_dim(A::ITensors.ITensor, B::ITensors.ITensor)::Int
    (ITensors.has_external_storage(A) && ITensors.has_external_storage(B)) || return -1
    wA = ITensors.get_external_storage(A)
    wA isa WrappedBlockSparse || return -1
    dA = dense_inds(wA)
    shared = ITensors.commoninds(A, B)
    sparse_shared = [I for I in shared if !(I in dA)]
    isempty(sparse_shared) && return -1
    return prod(ITensors.dim(I) for I in sparse_shared; init = 1)
end


function itensor_blocksparse_svd(
    phi::ITensors.ITensor,
    indsMb;
    ortho::String   = "left",
    maxdim::Int     = typemax(Int),
    mindim::Int     = 1,
    cutoff::Float64 = 0.0,
    tags            = ITensors.ts"Link,l",   # caller MUST encode the level, see §3
    bin_by_right::Bool = false,              # when true, bin per-right-sparse-key
                                              # (use this in orthogonalize! to preserve
                                              # the right-bond's sparse structure)
    target_n_new_sp::Int = -1,               # if positive, pad new bond's sparse dim
                                              # to at least this size (boundary case
                                              # where natural binning would give 1).
)
  w        = ITensors.get_external_storage(phi)::WrappedBlockSparse
  phi_inds = collect(w.inds)

  # Authoritative dense set — replaces `_P(bs)` + positional slicing of w.inds,
  # which was the original source of the partition error.
  dense_set = Set(dense_inds(w))

  # ----- bipartition by membership in indsMb -----
  in_U   = Set(filter(i -> i ∈ phi_inds, indsMb))
  U_legs = filter(i ->  i ∈ in_U, phi_inds)
  V_legs = filter(i -> !(i ∈ in_U), phi_inds)

  # Apply invariant ordering per side (links sorted, non-links, dense).
  U_spL, U_spN, U_d = reorder_invariant(U_legs, dense_set)
  V_spL, V_spN, V_d = reorder_invariant(V_legs, dense_set)

  nls = length(U_spL) + length(U_spN)
  nrs = length(V_spL) + length(V_spN)
  nld = length(U_d)
  nrd = length(V_d)

  # ---------- KEY FIX ----------
  # blocksparse_svd partitions positionally: dims[1:nls] is U-side sparse,
  # dims[nls+1 : nls+nrs] is V-side sparse, then U dense, then V dense.
  # Permute phi to match that layout so storage and the legs we picked agree.
  desired = (U_spL..., U_spN..., V_spL..., V_spN..., U_d..., V_d...)
  phi_p   = permute(phi, desired...; allow_alias = true)
  bs_p    = ITensors.get_external_storage(phi_p).blocksparse

  # println("itensor_blocksparse_svd: phi sp=$(nls+nrs) d=$(nld+nrd), ",
  #         "U side nls=$nls nld=$nld")

  U_bs, SV_bs, svs_kept, spec = if bin_by_right
      # Bin by only the right-side LINK-typed sparse axes (V_spL count).
      # This prevents the new bond's sparse dim from blowing up over right-side
      # site (non-link) indices — those fold into sub-matrix cols instead and
      # contribute to n_new_d (multiplicity dim). R is globally isometric.
      blocksparse_svd_right_binned(bs_p;
          n_left_sparse        = nls,
          n_left_dense         = nld,
          n_right_bin_sparse   = length(V_spL),
          target_n_new_sp,
          ortho, maxdim, mindim, cutoff)
  else
      # Mirror of the above: bin by left-side LINK-typed sparse axes (U_spL).
      # Left-side site (non-link) sparse axes fold into matrix rows. L is
      # globally isometric and the new bond's sparse dim matches the
      # left-link structure rather than collapsing to total-QN-sector count.
      blocksparse_svd_left_binned(bs_p;
          n_left_sparse       = nls,
          n_left_dense        = nld,
          n_left_bin_sparse   = length(U_spL),
          target_n_new_sp,
          ortho, maxdim, mindim, cutoff)
  end

  # ----- new bond legs (one sparse + one dense). Tag carries the l-level. -----
  n_new_sp = U_bs.dims[nls + 1]
  n_new_d  = U_bs.dims[nls + 1 + nld + 1]
  new_sp   = ITensors.Index(n_new_sp; tags = tags)
  new_d    = ITensors.Index(n_new_d;  tags = tags)

  # ----- inds in storage order produced by blocksparse_svd -----
  #   U  storage: (U_spL..., U_spN..., new_sp, U_d..., new_d)
  #   SV storage: (new_sp, V_spL..., V_spN..., new_d, V_d...)
  U_inds_storage  = (U_spL..., U_spN..., new_sp, U_d..., new_d)
  SV_inds_storage = (new_sp, V_spL..., V_spN..., new_d, V_d...)

  # Sanity: storage and inds must agree leg-by-leg. Catches the original bug
  # at its source if anything regresses.
  @assert ntuple(i -> ITensors.dim(U_inds_storage[i]),  length(U_inds_storage)) ==
          U_bs.dims  "U inds/storage dim mismatch: $(map(ITensors.dim, U_inds_storage)) vs $(U_bs.dims)"
  @assert ntuple(i -> ITensors.dim(SV_inds_storage[i]), length(SV_inds_storage)) ==
          SV_bs.dims "SV inds/storage dim mismatch: $(map(ITensors.dim, SV_inds_storage)) vs $(SV_bs.dims)"

  L_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(U_bs,  U_inds_storage))
  R_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(SV_bs, SV_inds_storage))

  # ----- final permute to invariant order -----
  # Storage order ≠ invariant order in general (e.g. a sparse non-link sits
  # before new_sp in U's storage, but the invariant wants links first). Route
  # through reorder_invariant — the same helper that owns the rule for
  # contractions — so this function bakes in zero link logic of its own.
  U_dense_set  = Set([U_d..., new_d])
  SV_dense_set = Set([V_d..., new_d])

  L_spL, L_spN, L_d = reorder_invariant(collect(U_inds_storage),  U_dense_set)
  R_spL, R_spN, R_d = reorder_invariant(collect(SV_inds_storage), SV_dense_set)

  L_it = permute(L_it, L_spL..., L_spN..., L_d...; allow_alias = true)
  R_it = permute(R_it, R_spL..., R_spN..., R_d...; allow_alias = true)

  return L_it, R_it, spec
end


# Channel-aware QR wrapper. Mirrors `itensor_blocksparse_svd_channel_aware` but
# calls `blocksparse_qr_channel_aware` to get lossless reconstruction in the
# "clean" bond cases (max_cMs_per_cL == 1 for ortho="left"). For (2,4,2)-
# pattern bonds (max_cMs_per_cL > 1) the QR kernel errors out and this
# wrapper falls back to the existing channel-aware SVD path.
#
# Diagnostic (SB_QR_DIAG=1): prints sparse-key structure of phi and the
# templates derived from M_b, M_b1.
function itensor_blocksparse_qr_channel_aware(
    phi::ITensors.ITensor,
    M_b::ITensors.ITensor,
    M_b1::ITensors.ITensor;
    ortho::String   = "left",
    maxdim::Int     = typemax(Int),
    mindim::Int     = 1,
    cutoff::Float64 = 0.0,
)
    @assert ITensors.has_external_storage(phi) "phi must be block-sparse"
    @assert ITensors.has_external_storage(M_b) "M_b must be block-sparse"
    @assert ITensors.has_external_storage(M_b1) "M_b1 must be block-sparse"
    w_phi = ITensors.get_external_storage(phi)::WrappedBlockSparse
    w_b   = ITensors.get_external_storage(M_b)::WrappedBlockSparse
    w_b1  = ITensors.get_external_storage(M_b1)::WrappedBlockSparse

    shared      = collect(ITensors.commoninds(M_b, M_b1))
    dense_b     = dense_inds(w_b)
    bond_sparse = first(I for I in shared if !(I in dense_b))
    bond_mult   = first(I for I in shared if  (I in dense_b))
    bond_sp_dim = ITensors.dim(bond_sparse)

    indsMb = [I for I in ITensors.inds(M_b) if !(I in shared)]

    phi_inds_all = collect(w_phi.inds)
    dense_set    = Set(dense_inds(w_phi))
    in_U   = Set(filter(i -> i ∈ phi_inds_all, indsMb))
    U_legs = filter(i ->  i ∈ in_U, phi_inds_all)
    V_legs = filter(i -> !(i ∈ in_U), phi_inds_all)
    U_spL, U_spN, U_d = reorder_invariant(U_legs, dense_set)
    V_spL, V_spN, V_d = reorder_invariant(V_legs, dense_set)
    nls = length(U_spL) + length(U_spN)
    nrs = length(V_spL) + length(V_spN)
    nld = length(U_d)
    nrd = length(V_d)

    desired = (U_spL..., U_spN..., V_spL..., V_spN..., U_d..., V_d...)
    phi_p   = permute(phi, desired...; allow_alias = true)
    bs_p    = ITensors.get_external_storage(phi_p).blocksparse

    L_phi_sparse = (U_spL..., U_spN...)
    R_phi_sparse = (V_spL..., V_spN...)

    # Templates from M_b, M_b1.
    M_b_inds        = collect(w_b.inds)
    M_b1_inds       = collect(w_b1.inds)
    dense_b1        = dense_inds(w_b1)
    M_b_sparse_pos  = [i for i in 1:length(M_b_inds)  if !(M_b_inds[i]  in dense_b)]
    M_b1_sparse_pos = [i for i in 1:length(M_b1_inds) if !(M_b1_inds[i] in dense_b1)]

    bond_pos_in_b  = findfirst(p -> M_b_inds[p]  == bond_sparse, M_b_sparse_pos)
    bond_pos_in_b1 = findfirst(p -> M_b1_inds[p] == bond_sparse, M_b1_sparse_pos)

    non_bond_b_pos  = [p for (i, p) in enumerate(M_b_sparse_pos)  if i != bond_pos_in_b]
    non_bond_b1_pos = [p for (i, p) in enumerate(M_b1_sparse_pos) if i != bond_pos_in_b1]
    perm_b  = [findfirst(I -> I == M_b_inds[p],  L_phi_sparse) for p in non_bond_b_pos]
    perm_b1 = [findfirst(I -> I == M_b1_inds[p], R_phi_sparse) for p in non_bond_b1_pos]
    @assert all(!isnothing, perm_b)
    @assert all(!isnothing, perm_b1)

    Kt = eltype(eltype(w_b.blocksparse.keys))
    LTupT = NTuple{nls, Kt}
    RTupT = NTuple{nrs, Kt}

    left_template = Tuple{LTupT, Kt}[]
    for key in w_b.blocksparse.keys
        vals     = [key[p] for p in M_b_sparse_pos]
        ch       = Kt(vals[bond_pos_in_b])
        non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b]
        lk       = Vector{Kt}(undef, nls)
        for (s, d) in enumerate(perm_b); lk[d] = Kt(non_bond[s]); end
        push!(left_template, (NTuple{nls,Kt}(lk), ch))
    end
    right_template = Tuple{Kt, RTupT}[]
    for key in w_b1.blocksparse.keys
        vals     = [key[p] for p in M_b1_sparse_pos]
        ch       = Kt(vals[bond_pos_in_b1])
        non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b1]
        rk       = Vector{Kt}(undef, nrs)
        for (s, d) in enumerate(perm_b1); rk[d] = Kt(non_bond[s]); end
        push!(right_template, (ch, NTuple{nrs,Kt}(rk)))
    end

    # SB_QR_DIAG removed 2026-06 — flip to true here for debug output.
    if false
        phi_keys = collect(bs_p.keys)
        println(stdout, "[QR_WRAP] phi sparse-keys (n=$(length(phi_keys))): ", phi_keys)
        println(stdout, "[QR_WRAP] left_template (M_b, n=$(length(left_template))): ", left_template)
        println(stdout, "[QR_WRAP] right_template (M_b1, n=$(length(right_template))): ", right_template)
        println(stdout, "[QR_WRAP] bond_sp_dim=$bond_sp_dim ortho=$ortho")
    end

    # QR + Gram-Schmidt for all cases. No SVD fallback — kernel handles
    # both clean (max_cMs_per_cL==1) and (2,4,2) cases internally.
    U_bs, SV_bs, svs_kept, spec = blocksparse_qr_channel_aware(bs_p;
        n_left_sparse = nls,
        n_left_dense  = nld,
        left_template, right_template,
        bond_sparse_dim = bond_sp_dim,
        ortho, maxdim, mindim, cutoff,
        verbose = false)

    new_sp  = bond_sparse
    n_new_d = U_bs.dims[nls + 1 + nld + 1]
    new_d   = ITensors.Index(n_new_d; tags = ITensors.tags(bond_mult))

    U_inds_storage  = (U_spL..., U_spN..., new_sp, U_d..., new_d)
    SV_inds_storage = (new_sp, V_spL..., V_spN..., new_d, V_d...)

    @assert ntuple(i -> ITensors.dim(U_inds_storage[i]),  length(U_inds_storage)) ==
            U_bs.dims  "U inds/storage dim mismatch"
    @assert ntuple(i -> ITensors.dim(SV_inds_storage[i]), length(SV_inds_storage)) ==
            SV_bs.dims "SV inds/storage dim mismatch"

    L_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(U_bs,  U_inds_storage))
    R_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(SV_bs, SV_inds_storage))

    U_dense_set  = Set([U_d..., new_d])
    SV_dense_set = Set([V_d..., new_d])
    L_spL, L_spN, L_d = reorder_invariant(collect(U_inds_storage),  U_dense_set)
    R_spL, R_spN, R_d = reorder_invariant(collect(SV_inds_storage), SV_dense_set)
    L_it = permute(L_it, L_spL..., L_spN..., L_d...; allow_alias = true)
    R_it = permute(R_it, R_spL..., R_spN..., R_d...; allow_alias = true)

    return L_it, R_it, spec
end
# DISABLED 2026-06: itensor_blocksparse_svd_owned_channel_aware is no longer called
# from anywhere (its only dispatch site, mps.jl:1577, is commented out; SB_USE_OWNED_SVD
# was never exercised by any script). Commented out rather than deleted; uncomment this
# block + the dispatch branch in mps.jl (and verify_iso.jl) to re-enable.
# # Channel-aware SVD with primary-ownership, NO Gram-Schmidt. Designed for
# # Path B (BMF_ISO_PATH=0). Lossless reconstruction via primary-ownership,
# # relaxed iso across channels (M-correction in geneigsolve absorbs slack).
# function itensor_blocksparse_svd_owned_channel_aware(
#     phi::ITensors.ITensor,
#     M_b::ITensors.ITensor,
#     M_b1::ITensors.ITensor;
#     ortho::String   = "left",
#     maxdim::Int     = typemax(Int),
#     mindim::Int     = 1,
#     cutoff::Float64 = 0.0,
# )
#     @assert ITensors.has_external_storage(phi) "phi must be block-sparse"
#     @assert ITensors.has_external_storage(M_b) "M_b must be block-sparse"
#     @assert ITensors.has_external_storage(M_b1) "M_b1 must be block-sparse"
#     w_phi = ITensors.get_external_storage(phi)::WrappedBlockSparse
#     w_b   = ITensors.get_external_storage(M_b)::WrappedBlockSparse
#     w_b1  = ITensors.get_external_storage(M_b1)::WrappedBlockSparse

#     shared      = collect(ITensors.commoninds(M_b, M_b1))
#     dense_b     = dense_inds(w_b)
#     bond_sparse = first(I for I in shared if !(I in dense_b))
#     bond_mult   = first(I for I in shared if  (I in dense_b))
#     bond_sp_dim = ITensors.dim(bond_sparse)

#     indsMb = [I for I in ITensors.inds(M_b) if !(I in shared)]

#     phi_inds_all = collect(w_phi.inds)
#     dense_set    = Set(dense_inds(w_phi))
#     in_U   = Set(filter(i -> i ∈ phi_inds_all, indsMb))
#     U_legs = filter(i ->  i ∈ in_U, phi_inds_all)
#     V_legs = filter(i -> !(i ∈ in_U), phi_inds_all)
#     U_spL, U_spN, U_d = reorder_invariant(U_legs, dense_set)
#     V_spL, V_spN, V_d = reorder_invariant(V_legs, dense_set)
#     nls = length(U_spL) + length(U_spN)
#     nrs = length(V_spL) + length(V_spN)
#     nld = length(U_d)
#     nrd = length(V_d)

#     desired = (U_spL..., U_spN..., V_spL..., V_spN..., U_d..., V_d...)
#     phi_p   = permute(phi, desired...; allow_alias = true)
#     bs_p    = ITensors.get_external_storage(phi_p).blocksparse

#     L_phi_sparse = (U_spL..., U_spN...)
#     R_phi_sparse = (V_spL..., V_spN...)

#     M_b_inds        = collect(w_b.inds)
#     M_b1_inds       = collect(w_b1.inds)
#     dense_b1        = dense_inds(w_b1)
#     M_b_sparse_pos  = [i for i in 1:length(M_b_inds)  if !(M_b_inds[i]  in dense_b)]
#     M_b1_sparse_pos = [i for i in 1:length(M_b1_inds) if !(M_b1_inds[i] in dense_b1)]

#     bond_pos_in_b  = findfirst(p -> M_b_inds[p]  == bond_sparse, M_b_sparse_pos)
#     bond_pos_in_b1 = findfirst(p -> M_b1_inds[p] == bond_sparse, M_b1_sparse_pos)

#     non_bond_b_pos  = [p for (i, p) in enumerate(M_b_sparse_pos)  if i != bond_pos_in_b]
#     non_bond_b1_pos = [p for (i, p) in enumerate(M_b1_sparse_pos) if i != bond_pos_in_b1]
#     perm_b  = [findfirst(I -> I == M_b_inds[p],  L_phi_sparse) for p in non_bond_b_pos]
#     perm_b1 = [findfirst(I -> I == M_b1_inds[p], R_phi_sparse) for p in non_bond_b1_pos]
#     @assert all(!isnothing, perm_b)
#     @assert all(!isnothing, perm_b1)

#     Kt = eltype(eltype(w_b.blocksparse.keys))
#     LTupT = NTuple{nls, Kt}
#     RTupT = NTuple{nrs, Kt}

#     left_template = Tuple{LTupT, Kt}[]
#     for key in w_b.blocksparse.keys
#         vals     = [key[p] for p in M_b_sparse_pos]
#         ch       = Kt(vals[bond_pos_in_b])
#         non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b]
#         lk       = Vector{Kt}(undef, nls)
#         for (s, d) in enumerate(perm_b); lk[d] = Kt(non_bond[s]); end
#         push!(left_template, (NTuple{nls,Kt}(lk), ch))
#     end
#     right_template = Tuple{Kt, RTupT}[]
#     for key in w_b1.blocksparse.keys
#         vals     = [key[p] for p in M_b1_sparse_pos]
#         ch       = Kt(vals[bond_pos_in_b1])
#         non_bond = [vals[i] for i in 1:length(vals) if i != bond_pos_in_b1]
#         rk       = Vector{Kt}(undef, nrs)
#         for (s, d) in enumerate(perm_b1); rk[d] = Kt(non_bond[s]); end
#         push!(right_template, (ch, NTuple{nrs,Kt}(rk)))
#     end

#     U_bs, SV_bs, svs_kept, spec = blocksparse_svd_owned_channel_aware(bs_p;
#         n_left_sparse = nls,
#         n_left_dense  = nld,
#         left_template, right_template,
#         bond_sparse_dim = bond_sp_dim,
#         ortho, maxdim, mindim, cutoff,
#         verbose = (get(ENV, "SB_QR_DIAG", "0") == "1"))

#     new_sp  = bond_sparse
#     n_new_d = U_bs.dims[nls + 1 + nld + 1]
#     new_d   = ITensors.Index(n_new_d; tags = ITensors.tags(bond_mult))

#     U_inds_storage  = (U_spL..., U_spN..., new_sp, U_d..., new_d)
#     SV_inds_storage = (new_sp, V_spL..., V_spN..., new_d, V_d...)

#     @assert ntuple(i -> ITensors.dim(U_inds_storage[i]),  length(U_inds_storage)) ==
#             U_bs.dims  "U inds/storage dim mismatch"
#     @assert ntuple(i -> ITensors.dim(SV_inds_storage[i]), length(SV_inds_storage)) ==
#             SV_bs.dims "SV inds/storage dim mismatch"

#     L_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(U_bs,  U_inds_storage))
#     R_it = ITensors._itensor_from_external_storage(WrappedBlockSparse(SV_bs, SV_inds_storage))

#     U_dense_set  = Set([U_d..., new_d])
#     SV_dense_set = Set([V_d..., new_d])
#     L_spL, L_spN, L_d = reorder_invariant(collect(U_inds_storage),  U_dense_set)
#     R_spL, R_spN, R_d = reorder_invariant(collect(SV_inds_storage), SV_dense_set)
#     L_it = permute(L_it, L_spL..., L_spN..., L_d...; allow_alias = true)
#     R_it = permute(R_it, R_spL..., R_spN..., R_d...; allow_alias = true)

#     return L_it, R_it, spec
# end
