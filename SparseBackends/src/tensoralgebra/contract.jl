include("contract_coo.jl")
include("contract_bs_bs.jl")
include("contract_coo_bs.jl")
include("contract_coo_dense.jl")
include("contract_bs_dense.jl")

# @inline function _dim_of(X, lab, mapX::Dict)
#     dims = _dims(X)
#     ax = mapX[lab]
#     return dims[ax]
# end

_dims(x::AbstractArray) = size(x)
_dims(x::COOTensor) = x.dims
_dims(x::NewBlockSparseSorted) = x.dims

# Label = (id, plev). id may exceed Int64 -> use UInt64.
const Label = NTuple{2,UInt64}

# Find position of a label in a small label vector.
# Returns 0 if not found.
@inline function findpos(labels::AbstractVector{Label}, lab::Label)::Int
  @inbounds for i in eachindex(labels)
    labels[i] == lab && return i
  end
  return 0
end

# Fast dim lookup using a precomputed map (recommended when you already build mapA/mapB)
@inline function _dim_of(X, lab::Label, mapX::AbstractDict{Label,Int})::Int
  i = get(mapX, lab, 0)
  i == 0 && error("Label $lab not found in tensor labels")
  return _dims(X)[i]
end


function contract!(
    C,
    labelsC::AbstractVector{Label},
    A,
    labelsA::AbstractVector{Label},
    B,
    labelsB::AbstractVector{Label};
    output_inds_hint::Union{Nothing,AbstractSet}=nothing,
    allowed_keys_C::Union{Nothing,AbstractSet}=nothing,
)
 @timeit TIMER "disp.body_full" begin
  @timeit TIMER "disp.preamble" begin
    @assert length(labelsA) == length(_dims(A)) "labelsA length must match A rank"
    @assert length(labelsB) == length(_dims(B)) "labelsB length must match B rank"
    @assert length(labelsC) == length(_dims(C)) "labelsC length must match C rank"
  end

  @timeit TIMER "disp.build_maps" begin
    mapA = Dict{Label,Int}()
    mapB = Dict{Label,Int}()
    sizehint!(mapA, length(labelsA))
    sizehint!(mapB, length(labelsB))

    @inbounds for i in eachindex(labelsA)
      mapA[labelsA[i]] = i
    end
    @inbounds for i in eachindex(labelsB)
      mapB[labelsB[i]] = i
    end
  end

  @timeit TIMER "disp.find_shared" begin
    shared = Label[]
    sizehint!(shared, min(length(labelsA), length(labelsB)))

    @inbounds for i in eachindex(labelsA)
      lab = labelsA[i]
      haskey(mapB, lab) && push!(shared, lab)
    end
  end

  @timeit TIMER "disp.find_reduced" begin
    reduced = Label[]
    sizehint!(reduced, length(shared))

    @inbounds for lab in shared
      haskey(mapA, lab) || continue
      if !any(lc -> lc == lab, labelsC)
        push!(reduced, lab)
      end
    end
  end

  if length(reduced) == 1
    # Keep your assumption if you want:
    @assert length(shared) == 1 "Expected exactly 1 shared label; got $shared"

    rlab = reduced[1]

    da = _dim_of(A, rlab, mapA)
    db = _dim_of(B, rlab, mapB)
    if da != db
      println("labelsA: ", labelsA, " dimsA: ", _dims(A),
              " labelsB: ", labelsB, " dimsB: ", _dims(B),
              " shared label: ", rlab)
    end
    @assert da == db "Dim mismatch on shared label $rlab"

    @assert !any(lc -> lc == rlab, labelsC) "Reduction label $rlab must NOT appear in output labels"

    @timeit TIMER "kbd.single_contract" begin
      # Hint path not yet implemented for single-shared-label kernel.
      # Fall through to original; hint will be a no-op here.
      return contract!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
  else
    @timeit TIMER "kbd.general_contract" begin
      return contract_shared!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared;
                              output_inds_hint=output_inds_hint,
                              allowed_keys_C=allowed_keys_C)
    end
  end
 end  # disp.body_full
end