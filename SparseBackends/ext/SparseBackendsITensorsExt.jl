module SparseBackendsITensorsExt

import ITensors
import ITensorMPS
using SparseBackends

abstract type WrappedTensorTypes{T,N} end  


mutable struct WrappedCOOTensor{T,N} <: WrappedTensorTypes{T,N}
  coo::COOTensor{T,N}
  labels::NTuple{N,Symbol}
  map::Dict{Symbol,Int}
end

mutable struct WrappedBlockSparse{T,N,N2,P} <: WrappedTensorTypes{T,N}
  blocksparse::NewBlockSparseSorted{T,N,N2,P}
  labels::NTuple{N,Symbol}
  map::Dict{Symbol,Int}
end

mutable struct WrappedTensor{T,N} <: WrappedTensorTypes{T,N}
  data::AbstractArray{T,N}
  labels::NTuple{N,Symbol}
  map::Dict{Symbol,Int}
end


@inline function _labels_map(labels::NTuple{N,Symbol}) where {N}
  d = Dict{Symbol,Int}()
  @inbounds for i in 1:N
      d[labels[i]] = i
  end
  return d
end

# function mpo_axes_itensor(T::ITensors.ITensor, bra_plev::Int, ket_plev::Int)
#   is = collect(ITensors.inds(T))
#   println("All indices: ", is)
#   s_bra = only([I for I in is if ITensors.hastags(I, "Site") && ITensors.hasplev(I, bra_plev)])
#   s_ket = only([I for I in is if ITensors.hastags(I, "Site") && ITensors.hasplev(I, ket_plev)])
#   links = [I for I in is if ITensors.hastags(I, "Link")]
#   return s_bra, s_ket, links
# end

# @inline function _mpo_labels(bra_plev::Int, ket_plev::Int, nlinks::Int)
#   return ntuple(i -> begin
#       if i == 1
#           Symbol("s_p$(bra_plev)")
#       elseif i == 2
#           Symbol("s_p$(ket_plev)")
#       else
#           Symbol("pL$(i-2)")
#       end
#   end, 2 + nlinks)
# end


# function WrappedTensor(T::ITensors.ITensor)
#     bra_plev, ket_plev = 1, 0
#     bra, ket, links = mpo_axes_itensor(T, bra_plev, ket_plev)
#     array = Array(T, bra, ket, links...)
#     labels = _mpo_labels(bra_plev, ket_plev, length(links))
#     return WrappedTensor(array, labels, _labels_map(labels))
# end

# function WrappedCOOTensor(T::ITensors.ITensor)
#     bra_plev, ket_plev = 1, 0
#     bra, ket, links = mpo_axes_itensor(T, bra_plev, ket_plev)
#     array = Array(T, bra, ket, links...)
#     coo = SparseBackends.coo_from_dense(array; atol=1e-12, rtol=0.0)
#     labels = _mpo_labels(bra_plev, ket_plev, length(links))
#     return WrappedCOOTensor(coo, labels, _labels_map(labels))
# end

# function WrappedBlockSparse(T::ITensors.ITensor, denseLinks::Int)
#     bra_plev, ket_plev = 1, 0
#     bra, ket, links = mpo_axes_itensor(T, bra_plev, ket_plev)
#     @assert denseLinks == length(links) "denseLinks must match the number of Link indices"
#     array = Array(T, bra, ket, links...)
#     bs = SparseBackends.blocksparse_from_dense(array, Val(denseLinks))
#     labels = _mpo_labels(bra_plev, ket_plev, length(links))
#     return WrappedBlockSparse(bs, labels, _labels_map(labels))
# end

# function WrappedCOOTensor(::Type{T},
#                           dims::NTuple{N,Int},
#                           labels::NTuple{N,Symbol}) where {T,N}
#     coo = SparseBackends.COOTensor{T,N}(dims, Tuple{NTuple{N,Int},T}[])
#     return WrappedCOOTensor(coo, labels, _labels_map(labels))
# end

# function WrappedBlockSparse(::Type{T},
#                             dims::NTuple{N,Int},
#                             denseLinks::Int,
#                             labels::NTuple{N,Symbol}) where {T,N}
#     bs = SparseBackends.NewBlockSparseSorted{T,N,denseLinks}(dims)
#     return WrappedBlockSparse(bs, labels, _labels_map(labels))
# end


# Make tags string safe for a Julia Symbol (remove quotes, commas, spaces, etc.)
@inline function _sanitize_tagstr(s::AbstractString)
    # keep alnum + '_' only
    return replace(s, r"[^A-Za-z0-9_]+" => "_")
end

@inline function label_for_index(I::ITensors.Index)
    # Prefer a readable tag component, but ensure uniqueness via id + primelevel.
    tagstr = _sanitize_tagstr(string(ITensors.tags(I)))
    p      = ITensors.plev(I)
    id     = ITensors.id(I)
    # Example: Site_n_1__p1_id12345
    return Symbol("$(tagstr)_p$(p)_id$(id)")
end

function mpo_axes_itensor(T::ITensors.ITensor;
                          bra_plev::Union{Nothing,Int}=nothing,
                          ket_plev::Union{Nothing,Int}=nothing)
    is = collect(ITensors.inds(T))
    site_inds = [I for I in is if ITensors.hastags(I, "Site")]
    link_inds = [I for I in is if ITensors.hastags(I, "Link")]
    if length(site_inds) == 2 && bra_plev === nothing && ket_plev === nothing
      # Assign by primelevel ordering
      plevs = ITensors.plev.(site_inds)
      hi = argmax(plevs)
      lo = argmin(plevs)
      s_bra = site_inds[hi]
      s_ket = site_inds[lo]
    else
      @assert bra_plev !== nothing && ket_plev !== nothing """
      Ambiguous Site indices: found $(length(site_inds)) Site inds.
      Pass bra_plev=... and ket_plev=... to disambiguate.
      """
      s_bra = only([I for I in site_inds if ITensors.hasplev(I, bra_plev)])
      s_ket = only([I for I in site_inds if ITensors.hasplev(I, ket_plev)])
    end
    println("Identified bra index: ", s_bra)
    println("Identified ket index: ", s_ket)
    println("Identified link indices: ", link_inds)
    return s_bra, s_ket, link_inds
end

# Deterministic ordering for Link indices when converting to Array:
# preserve MPO left-to-right order by sorting by Index id (stable across copies)
@inline function sort_links(links::AbstractVector{<:ITensors.Index})
    return sort(links; by=ITensors.id)
end

function WrappedTensor(T::ITensors.ITensor; bra_plev::Union{Nothing,Int}=nothing, ket_plev::Union{Nothing,Int}=nothing)
    bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
    links = sort_links(links)

    array  = Array(T, bra, ket, links...)  # axis order matches label order
    labels = (label_for_index(bra), label_for_index(ket), map(label_for_index, links)...)

    return WrappedTensor(array, labels, _labels_map(labels))
end

function WrappedCOOTensor(T::ITensors.ITensor; bra_plev::Union{Nothing,Int}=nothing, ket_plev::Union{Nothing,Int}=nothing)
    bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
    links = sort_links(links)

    array  = Array(T, bra, ket, links...)
    coo    = SparseBackends.coo_from_dense(array; atol=1e-12, rtol=0.0)
    labels = (label_for_index(bra), label_for_index(ket), map(label_for_index, links)...)

    return WrappedCOOTensor(coo, labels, _labels_map(labels))
end

function WrappedBlockSparse(T::ITensors.ITensor, denseLinks::Int;
                            bra_plev::Union{Nothing,Int}=nothing, ket_plev::Union{Nothing,Int}=nothing)
    bra, ket, links = mpo_axes_itensor(T; bra_plev=bra_plev, ket_plev=ket_plev)
    links = sort_links(links)

    @assert denseLinks == length(links) "denseLinks must match the number of Link indices"

    array  = Array(T, bra, ket, links...)
    bs     = SparseBackends.blocksparse_from_dense(array, Val(denseLinks))
    labels = (label_for_index(bra), label_for_index(ket), map(label_for_index, links)...)

    return WrappedBlockSparse(bs, labels, _labels_map(labels))
end

# # Constructors for "empty" wrappers unchanged (they already take explicit labels)
# function WrappedCOOTensor(::Type{T},
#                           dims::NTuple{N,Int},
#                           labels::NTuple{N,Symbol}) where {T,N}
#     coo = SparseBackends.COOTensor{T,N}(dims, Tuple{NTuple{N,Int},T}[])
#     return WrappedCOOTensor(coo, labels, _labels_map(labels))
# end
function WrappedCOOTensor(::Type{T},
                          dims::NTuple{N,Int},
                          labels::NTuple{N,Symbol}) where {T,N}
    entries = Vector{Tuple{NTuple{N,Int},T}}()           # empty entries
    coo = SparseBackends.COOTensor{T}(dims, entries)     # NOTE: {T} not {T,N}
    return WrappedCOOTensor(coo, labels, _labels_map(labels))
end

function WrappedBlockSparse(::Type{T},
                            dims::NTuple{N,Int},
                            denseLinks::Int,
                            labels::NTuple{N,Symbol}) where {T,N}
    bs = SparseBackends.NewBlockSparseSorted{T,N,denseLinks}(dims)
    return WrappedBlockSparse(bs, labels, _labels_map(labels))
end

backend_hint(::ITensors.ITensor) = :dense  # user can overload in their own code if desired

function wrap_itensor(T::ITensors.ITensor; backend::Symbol=backend_hint(T), denseLinks::Union{Nothing,Int}=nothing)
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

# what labels are "dense dims" for each wrapper?
dense_labels(w::WrappedCOOTensor) = Set{Symbol}()  # none
dense_labels(w::WrappedTensor)    = Set(w.labels)  # treat all dims dense
dense_labels(w::WrappedBlockSparse{T,N,N2,P}) where {T,N,N2,P} =
    Set(@view(w.labels[end-N2+1:end]))  # last N2 labels are dense

function output_labels(labelsA::AbstractVector{Symbol}, labelsB::AbstractVector{Symbol})
    common = Set(labelsA) ∩ Set(labelsB)
    out = Symbol[]
    for l in labelsA
        l in common || push!(out, l)
    end
    for l in labelsB
        l in common || push!(out, l)
    end
    return out
end

@inline infer_C_backend(::WrappedCOOTensor, ::WrappedCOOTensor) = :coo
@inline infer_C_backend(::WrappedBlockSparse, ::WrappedBlockSparse) = :blocksparse
@inline infer_C_backend(::WrappedCOOTensor, ::WrappedTensor) = :blocksparse
@inline infer_C_backend(::WrappedCOOTensor, ::WrappedBlockSparse) = :blocksparse
@inline infer_C_backend(::WrappedBlockSparse, ::WrappedTensor) = :blocksparse

# infer denseLinksC = number of "dense" labels that survive to output
function infer_denseLinksC(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes, labelsC::Vector{Symbol})
    dA = dense_labels(Aw)
    dB = dense_labels(Bw)
    dC = Set(labelsC) ∩ (dA ∪ dB)
    return length(dC)
end

function output_dims(Arep, labelsA::NTuple{NA,Symbol},
                     Brep, labelsB::NTuple{NB,Symbol},
                     labelsC) where {NA,NB}
    # dims of reps (works for Array, COO, BlockSparse as long as size(...) is defined)
    dimsA = size(Arep)
    dimsB = size(Brep)
    # build label -> axis maps (you already store maps in wrappers, but here we do it locally)
    mapA = Dict{Symbol,Int}(@inbounds (labelsA[i] => i for i in 1:NA))
    mapB = Dict{Symbol,Int}(@inbounds (labelsB[i] => i for i in 1:NB))
    labelsC_vec = labelsC isa NTuple ? collect(labelsC) : Vector{Symbol}(labelsC)
    NC = length(labelsC_vec)
    out = Vector{Int}(undef, NC)
    @inbounds for i in 1:NC
        l = labelsC_vec[i]
        ia = get(mapA, l, 0)
        ib = get(mapB, l, 0)

        if ia != 0 && ib == 0
            out[i] = dimsA[ia]
        elseif ia == 0 && ib != 0
            out[i] = dimsB[ib]
        elseif ia != 0 && ib != 0
            da = dimsA[ia]
            db = dimsB[ib]
            da == db || throw(DimensionMismatch("Label $l appears in both A and B but dims differ: $da vs $db"))
            out[i] = da
        else
            throw(ArgumentError("Label $l in labelsC not found in A or B"))
        end
    end
    return Tuple(out)  # NTuple{NC,Int}
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
  return contract(Aw, Bw)
end

# function contract(A::WrappedTensorTypes{TA,NA}, B::WrappedTensorTypes{TB,NB}; kwargs...) where {TA,TB,NA,NB}
#   # Get underlying arrays/sparse tensors + labels
#   Arep = A isa WrappedTensor ? A.data : A isa WrappedCOOTensor ? A.coo : A.blocksparse
#   Brep = B isa WrappedTensor ? B.data : B isa WrappedCOOTensor ? B.coo : B.blocksparse

#   labelsA = A.labels
#   labelsB = B.labels
#   labelsC = collect(output_labels(collect(labelsA), collect(labelsB)))

#   dimsC = output_dims(Arep, labelsA, Brep, labelsB, labelsC)
#   TC = promote_type(eltype(Arep), eltype(Brep))
#   Cbackend = infer_C_backend(A, B)
#   if Cbackend === :blocksparse
#     C = WrappedBlockSparse(TC, dimsC, infer_denseLinksC(A, B, labelsC), labelsC)
#     C.blocksparse = SparseBackends.contract!(C.blocksparse, labelsC, Arep, labelsA, Brep, labelsB; debug=false)
#     return C
#   elseif Cbackend === :coo
#     C = WrappedCOOTensor(TC, collect(dimsC), collect(labelsC))
#     C.coo = SparseBackends.contract!(C.coo, labelsC, Arep, labelsA, Brep, labelsB; debug=false)
#     return C
#   else
#     error("Unsupported backend: $Cbackend")
#   end
# end

function contract(A::WrappedTensorTypes{TA,NA}, B::WrappedTensorTypes{TB,NB}; kwargs...) where {TA,TB,NA,NB}
  Arep = A isa WrappedTensor ? A.data : A isa WrappedCOOTensor ? A.coo : A.blocksparse
  Brep = B isa WrappedTensor ? B.data : B isa WrappedCOOTensor ? B.coo : B.blocksparse

  labelsA = A.labels
  labelsB = B.labels

  labelsC_vec = output_labels(collect(labelsA), collect(labelsB))  # Vector{Symbol}
  dimsC_tup    = output_dims(Arep, labelsA, Brep, labelsB, labelsC_vec) # NTuple{NC,Int}
  labelsC_tup  = Tuple(labelsC_vec)  # NTuple{NC,Symbol}

  TC = promote_type(eltype(Arep), eltype(Brep))
  Cbackend = infer_C_backend(A, B)

  if Cbackend === :blocksparse
    C = WrappedBlockSparse(TC, dimsC_tup, infer_denseLinksC(A, B, labelsC_vec), labelsC_tup)
    C.blocksparse = SparseBackends.contract!(C.blocksparse, labelsC_vec, Arep, collect(labelsA), Brep, collect(labelsB); debug=false)
    return C

  elseif Cbackend === :coo
    C = WrappedCOOTensor(TC, dimsC_tup, labelsC_tup)
    C.coo = SparseBackends.contract!(C.coo, labelsC_vec, Arep, collect(labelsA), Brep, collect(labelsB))
    return C

  else
    error("Unsupported backend: $Cbackend")
  end
end

function contract(A::ITensorMPS.MPO, B::ITensorMPS.MPO,
                  Abackend::Symbol, Bbackend::Symbol;
                  denseLinksA::Union{Nothing,Int}=nothing,
                  denseLinksB::Union{Nothing,Int}=nothing,
                  cutoff=1e-14,
                  maxdim=ITensorMPS.maxlinkdim(A) * ITensorMPS.maxlinkdim(B),
                  debug::Bool=false,
                  kwargs...)
  N = length(A)
  N == length(B) || throw(DimensionMismatch("MPO lengths don't match"))

  # You still need to make sure they don't share both site indices per site.
  # Mimic ITensors behavior: if same siteinds, tell user to prime one side.
  sA = collect(ITensors.siteinds(A))
  sB = collect(ITensors.siteinds(B))

  if ITensors.hassameinds(sA, sB)
    error("A and B have the same site indices; prime one MPO (e.g. Bp = prime(B, \"Site\")) then contract and replaceprime.")
  end
  C = ITensorMPS.MPO(N)
  for i in 1:N
    println("Contracting site $i / $N")
    # Your existing ITensor-level glue does the backend conversion:
    C[i] = contract(A[i], B[i], Abackend, Bbackend;
                    denseLinksA=denseLinksA, denseLinksB=denseLinksB)
  end
  return C
end


end  # module