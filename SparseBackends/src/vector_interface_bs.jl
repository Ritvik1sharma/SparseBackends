# Specializations of VectorInterface methods (add!, scale!, inner, norm, etc.)
# for ITensors backed by WrappedBlockSparse external storage. Goal: preserve
# BS classification through KrylovKit's internal Krylov basis manipulations
# (which otherwise densify or collapse to flat-BS P=0).
#
# These methods assume KrylovKit basis vectors share the SAME key list
# (enforced by all B_op outputs being recast to the same template). Under that
# assumption, add!/scale!/inner/norm reduce to position-wise operations on the
# underlying .data buffers — O(nnz·blksize) instead of O(prod(dims)) with
# densification.

import VectorInterface

@inline function _bs_storage(t::ITensors.ITensor)
  ITensors.has_external_storage(t) || return nothing
  w = ITensors.get_external_storage(t)
  return w isa WrappedBlockSparse ? w : nothing
end

@inline function _bs_same_layout(aw::WrappedBlockSparse, bw::WrappedBlockSparse)
  bs_a = aw.blocksparse
  bs_b = bw.blocksparse
  bs_a.dims == bs_b.dims || return false
  bs_a.blksize == bs_b.blksize || return false
  length(bs_a.keys) == length(bs_b.keys) || return false
  length(bs_a.data) == length(bs_b.data) || return false
  # Verify same key sequence (Arnoldi basis-vector invariant)
  @inbounds for i in eachindex(bs_a.keys)
    bs_a.keys[i] == bs_b.keys[i] || return false
  end
  return true
end

# --- add!(a, b, α): a += α * b ---
function VectorInterface.add!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number)
  aw = _bs_storage(a)
  bw = _bs_storage(b)
  if aw !== nothing && bw !== nothing && _bs_same_layout(aw, bw)
    da = aw.blocksparse.data
    db = bw.blocksparse.data
    if α == one(α)
      @inbounds @simd for i in eachindex(da)
        da[i] += db[i]
      end
    else
      @inbounds @simd for i in eachindex(da)
        da[i] += α * db[i]
      end
    end
    return a
  end
  # Fallback: existing ITensors implementation
  if ITensors.has_external_storage(a)
    result = a + b * α
    a.tensor = result.tensor
    return a
  end
  a .= a .+ b .* α
  return a
end

function VectorInterface.add!(a::ITensors.ITensor, b::ITensors.ITensor)
  return VectorInterface.add!(a, b, one(eltype(a)))
end

# --- add!(a, b, α, β): a = α*b + β*a ---
function VectorInterface.add!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number, β::Number)
  aw = _bs_storage(a)
  bw = _bs_storage(b)
  if aw !== nothing && bw !== nothing && _bs_same_layout(aw, bw)
    da = aw.blocksparse.data
    db = bw.blocksparse.data
    @inbounds @simd for i in eachindex(da)
      da[i] = β * da[i] + α * db[i]
    end
    return a
  end
  if ITensors.has_external_storage(a)
    result = a * β + b * α
    a.tensor = result.tensor
    return a
  end
  a .= a .* β .+ b .* α
  return a
end

# --- add!!: in-place if type-compat, else allocate ---
function VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number)
  if promote_type(eltype(a), eltype(b), typeof(α)) <: eltype(a)
    return VectorInterface.add!(a, b, α)
  end
  # Type-promotion needed: fall back to allocation
  return a + b * α
end
function VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number, β::Number)
  if promote_type(eltype(a), eltype(b), typeof(α), typeof(β)) <: eltype(a)
    return VectorInterface.add!(a, b, α, β)
  end
  return a * β + b * α
end
function VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor)
  return VectorInterface.add!!(a, b, one(eltype(a)))
end

# --- scale!(a, α): a *= α ---
function VectorInterface.scale!(a::ITensors.ITensor, α::Number)
  aw = _bs_storage(a)
  if aw !== nothing
    da = aw.blocksparse.data
    @inbounds @simd for i in eachindex(da)
      da[i] *= α
    end
    return a
  end
  a .= a .* α
  return a
end

# --- scale!(a_dest, a_src, α): a_dest = a_src * α ---
function VectorInterface.scale!(a_dest::ITensors.ITensor, a_src::ITensors.ITensor, α::Number)
  dw = _bs_storage(a_dest)
  sw = _bs_storage(a_src)
  if dw !== nothing && sw !== nothing && _bs_same_layout(dw, sw)
    dd = dw.blocksparse.data
    sd = sw.blocksparse.data
    @inbounds @simd for i in eachindex(dd)
      dd[i] = α * sd[i]
    end
    return a_dest
  end
  a_dest .= a_src .* α
  return a_dest
end

function VectorInterface.scale!!(a::ITensors.ITensor, α::Number)
  if promote_type(eltype(a), typeof(α)) <: eltype(a)
    return VectorInterface.scale!(a, α)
  end
  return a * α
end
function VectorInterface.scale!!(a_dest::ITensors.ITensor, a_src::ITensors.ITensor, α::Number)
  if promote_type(eltype(a_dest), eltype(a_src), typeof(α)) <: eltype(a_dest)
    return VectorInterface.scale!(a_dest, a_src, α)
  end
  return a_src * α
end

# --- inner(a, b): ⟨a | b⟩ ---
function VectorInterface.inner(a::ITensors.ITensor, b::ITensors.ITensor)
  aw = _bs_storage(a)
  bw = _bs_storage(b)
  if aw !== nothing && bw !== nothing && _bs_same_layout(aw, bw)
    da = aw.blocksparse.data
    db = bw.blocksparse.data
    s = zero(promote_type(eltype(da), eltype(db)))
    @inbounds @simd for i in eachindex(da)
      s += conj(da[i]) * db[i]
    end
    return s
  end
  return ITensors.inner(a, b)
end

# --- zerovector! ---
function VectorInterface.zerovector!(a::ITensors.ITensor)
  aw = _bs_storage(a)
  if aw !== nothing
    da = aw.blocksparse.data
    @inbounds @simd for i in eachindex(da)
      da[i] = zero(eltype(da))
    end
    return a
  end
  a .= zero(eltype(a))
  return a
end
