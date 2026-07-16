# core_helpers/factor_core.jl — factor-core primitives for ψ = P·core (diagonal AND flip P).
#
# The aliased ψ stores each template as one pre-P core physical-slice `core[:, rv, :]`
# (verified: contract_aliased_coo_dense makes template = α·B[:,rv]). These let a
# DMRG/TDVP driver treat the *templates* as the variational `core` while P (the
# keys/alias_ids/scalars) stays fixed:
#   read_core(w)             → the dense core tensor at this site (physical + core links)
#   slice_to_template(w)     → rv-slice → template id (cached hint field, else derived)
#   window_write_map(φ,…)    → read the emitted (rv_b,rv_{b+1})→tid map off a 2-site φ
#   write_core_window!(φ,…)  → scatter a merged 2-site core back into φ's templates
# Round-trip is exact where each template is a single clean core slice; errors on
# scramble (a template shared across two physical slices).

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

# Consume the (rv_b, rv_{b+1}) → template-id map that the AliasedBS×AliasedBS
# contract_shared! EMITTED into φ.window_slice_map when the window φ = ψ[b]·ψ[b+1] was
# built with emit_window_map=true. The map cannot be reconstructed here post-hoc for a
# flip P (φ sums over the internal FSM bond), so it is produced during that contraction
# and only READ back now. Returns (rvmap, [axis_b, axis_bp1]) with φ's Site axes ordered
# (b, b+1) — matching the emitted key order (rv of ψ[b], rv of ψ[b+1]) — so
# core_arr[rv_b, rv_{b+1}] lines up. Identity for diagonal P (rv == output), the flip
# permutation for KL. Errors if φ carries no emitted map.
function window_write_map(wphi::WrappedAliasedBlockSparse,
                          wb::WrappedAliasedBlockSparse, wbp1::WrappedAliasedBlockSparse)
    a = wphi.aliased
    isempty(a.window_slice_map) &&
        error("window_write_map: φ carries no emitted window map — build φ with emit_window_map=true")
    Pn = length(a.dims) - _n2(wphi)
    sp = [i for i in 1:Pn if ITensors.hastags(wphi.inds[i], "Site")]
    length(sp) == 2 || error("window_write_map: expected exactly 2 Site axes (got $(length(sp)))")
    sib   = wb.inds[_phys_prefix_pos(wb)]
    sibp1 = wbp1.inds[_phys_prefix_pos(wbp1)]
    jb   = findfirst(p -> ITensors.id(wphi.inds[p]) == ITensors.id(sib),   sp)
    jbp1 = findfirst(p -> ITensors.id(wphi.inds[p]) == ITensors.id(sibp1), sp)
    (jb === nothing || jbp1 === nothing) && error("window_write_map: φ Site axes don't match ψ[b]/ψ[b+1] ids")
    return a.window_slice_map, [sp[jb], sp[jbp1]]
end

# 2-site (window) counterpart of write_core!: scatter a dense MERGED 2-site core into
# the templates of an aliased window φ = ψ[b]·ψ[b+1], IN PLACE (keys/alias_ids/scalars
# untouched), using a PRECOMPUTED map from window_write_map (write does NOT re-derive).
# Used by the factor-core matvec to re-attach the ket-P to the evolving Krylov core each
# step: `write_core_window!(φ, core, wmap); product(...)`. `core` lives on φ's Site +
# dense (core-link) indices.
function write_core_window!(w::WrappedAliasedBlockSparse, core::ITensors.ITensor,
                            wmap::Tuple{<:AbstractDict,<:AbstractVector})
    tup2tid, sp = wmap
    a = w.aliased
    Pn = length(a.dims) - _n2(w)
    dense_inds = w.inds[Pn+1:end]
    dense_dims = a.dims[Pn+1:end]
    bs = a.blksize
    site_inds = [w.inds[p] for p in sp]
    core_arr = Array(core, site_inds..., dense_inds...)   # [phys-tuple…, dense…] in φ's order
    tail_ci = CartesianIndices(Tuple(dense_dims))
    @inbounds for (key, tid) in tup2tid
        off = (tid - 1) * bs
        for (lin, ci) in enumerate(tail_ci)
            a.templates[off + lin] = core_arr[key..., Tuple(ci)...]
        end
    end
    return w
end
