# aliased/factor_core.jl — factor-core primitives for ψ = P·core (diagonal P).
#
# The aliased ψ stores each template as one core physical-slice `core[:, s, :]`
# (verified: contract_aliased_coo_dense makes template = α·B[:,rv]). These let a
# DMRG/TDVP driver treat the *templates* as the variational `core` while P (the
# keys/alias_ids/scalars) stays fixed:
#   read_core(w)        → the dense core tensor at this site (physical + core links)
#   write_core!(w, core)→ scatter a core tensor back into the templates (P fixed)
#   slice_to_template(w)→ core-slice s → template id (uses the cached hint field if
#                         populated, else derives it from `keys`).
# Only valid where each template is a single clean core slice (diagonal P); errors
# on scramble (a template shared across two physical slices).

# Position of the physical (Site-tagged) axis among the P prefix axes.
function _phys_prefix_pos(w::WrappedAliasedBlockSparse)
    a = w.aliased
    Pn = length(a.dims) - _n2(w)
    @inbounds for i in 1:Pn
        ITensors.hastags(w.inds[i], "Site") && return i
    end
    return 0
end
_n2(w::WrappedAliasedBlockSparse{T,N,N2,P}) where {T,N,N2,P} = N2

# core-slice → template id. Uses the cached hint if present, else derives it by
# grouping keys on the physical axis (and asserts no scramble).
function slice_to_template(w::WrappedAliasedBlockSparse)
    a = w.aliased
    isempty(a.keys) && return Int[]
    !isempty(a.slice_to_template) && return a.slice_to_template
    phys = _phys_prefix_pos(w)
    phys == 0 && error("factor-core: no Site axis in prefix; cannot derive slice_to_template")
    s2t = zeros(Int, a.dims[phys])
    @inbounds for (i, k) in enumerate(a.keys)
        s = k[phys]; tid = Int(a.alias_ids[i])
        if s2t[s] == 0
            s2t[s] = tid
        elseif s2t[s] != tid
            error("factor-core: scramble at physical slice $s (templates $(s2t[s]) vs $tid) — not a clean P·core")
        end
    end
    return s2t
end

# output_P: re-impose P's clean deduped schema on a de-duped aliased tensor (the
# matvec output H·φ). Groups blocks by physical coord (the Site prefix axes) and
# collapses their per-channel templates to ONE template per coord. Keeps keys &
# scalars; only re-dedups templates + remaps alias_ids. EXACT iff every channel of
# a physical coord carries the same template (block/scalar) — i.e. H·φ ∈ image(P)
# (guaranteed by [H,P]=0); returns the max deviation as the leakage/consistency
# check. This is the "apply P on the output" step (in template space — the channel
# bonds are already correct from the env, so no P contraction / FSM-bond doubling).
function output_P(w::WrappedAliasedBlockSparse{T,N,N2,P}; atol::Real=1e-10) where {T,N,N2,P}
    a = w.aliased
    Pn = length(a.dims) - N2
    phys_pos = [i for i in 1:Pn if ITensors.hastags(w.inds[i], "Site")]
    isempty(phys_pos) && error("output_P: no Site axis in prefix")
    bs = a.blksize
    Kt = eltype(eltype(a.keys)); AIt = eltype(a.alias_ids)
    coord_tid = Dict{NTuple{length(phys_pos),Int}, Int}()
    new_templates = T[]
    new_alias = Vector{AIt}(undef, length(a.keys))
    maxdev = 0.0
    @inbounds for (i, k) in enumerate(a.keys)
        coord = ntuple(j -> Int(k[phys_pos[j]]), length(phys_pos))
        off = (Int(a.alias_ids[i]) - 1) * bs
        if haskey(coord_tid, coord)
            tid = coord_tid[coord]; eoff = (tid - 1) * bs
            for j in 1:bs
                maxdev = max(maxdev, abs(a.templates[off+j] - new_templates[eoff+j]))
            end
        else
            tid = length(coord_tid) + 1; coord_tid[coord] = tid
            for j in 1:bs; push!(new_templates, a.templates[off+j]); end
        end
        new_alias[i] = AIt(tid)
    end
    n_new = length(coord_tid)
    new_ali = AliasedBlockSparse{T,N,N2,P,Kt}(a.dims, bs, new_templates, n_new,
                                              copy(a.keys), new_alias, copy(a.scalars))
    return WrappedAliasedBlockSparse{T,N,N2,P}(new_ali, w.inds), maxdev
end

# Eagerly build + cache the slice_to_template hint in the storage field (the
# driver calls this once on the fresh ψ = P·core so later read_core/write_core!
# skip re-derivation). Reassigns a fresh vector (never mutates the shared empty
# default). No-op if already populated or if the tensor isn't a clean P·core.
function populate_slice_map!(w::WrappedAliasedBlockSparse)
    a = w.aliased
    (isempty(a.keys) || !isempty(a.slice_to_template)) && return w
    _phys_prefix_pos(w) == 0 && return w            # not a per-site P·core tensor
    a.slice_to_template = slice_to_template(w)       # fresh vector; hint now cached
    return w
end

# Recover the dense core tensor at this site: core[:, s, :] = template[s2t[s]].
# Returned on (physical index, dense/core-link indices) — the aliased tensor's own.
function read_core(w::WrappedAliasedBlockSparse)
    a = w.aliased
    phys = _phys_prefix_pos(w)
    phys == 0 && error("factor-core: no Site axis")
    Pn = length(a.dims) - _n2(w)
    phys_ind   = w.inds[phys]
    dense_inds = w.inds[Pn+1:end]
    dense_dims = a.dims[Pn+1:end]
    bs = a.blksize
    s2t = slice_to_template(w)
    core_arr = zeros(eltype(a.templates), a.dims[phys], dense_dims...)
    tail_ci = CartesianIndices(Tuple(dense_dims))
    @inbounds for s in 1:a.dims[phys]
        tid = s2t[s]; tid == 0 && continue           # physical s absent (P forbids)
        off = (tid - 1) * bs
        for (lin, ci) in enumerate(tail_ci)
            core_arr[s, Tuple(ci)...] = a.templates[off + lin]
        end
    end
    return ITensors.ITensor(core_arr, phys_ind, dense_inds...)
end

# Scatter a dense core tensor back into the templates in place (keys/alias_ids/
# scalars untouched). `core` must live on the same physical + dense indices.
function write_core!(w::WrappedAliasedBlockSparse, core::ITensors.ITensor)
    a = w.aliased
    phys = _phys_prefix_pos(w)
    phys == 0 && error("factor-core: no Site axis")
    Pn = length(a.dims) - _n2(w)
    phys_ind   = w.inds[phys]
    dense_inds = w.inds[Pn+1:end]
    dense_dims = a.dims[Pn+1:end]
    bs = a.blksize
    s2t = slice_to_template(w)
    core_arr = Array(core, phys_ind, dense_inds...)   # [physical, dense…] in ψ's order
    tail_ci = CartesianIndices(Tuple(dense_dims))
    seen = falses(a.n_templates)
    @inbounds for s in 1:a.dims[phys]
        tid = s2t[s]; tid == 0 && continue
        seen[tid] && error("factor-core: template $tid shared by 2 physical slices — cannot write_core! (value-dedup)")
        seen[tid] = true
        off = (tid - 1) * bs
        for (lin, ci) in enumerate(tail_ci)
            a.templates[off + lin] = core_arr[s, Tuple(ci)...]
        end
    end
    return w
end
