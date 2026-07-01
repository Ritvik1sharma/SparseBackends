# Debug/diagnostic helpers for Path-B (path_b_helpers.jl). Kept separate so the
# main apply_minv function isn't cluttered with print-only code.

import ITensors

# Trace the prefix/dense split of a tensor at one stage of the M⁻¹-apply
# pipeline, for ALIASED tensors — used to localize where φ's canonical split
# is lost (channel moved into the dense tail). Prints P / prefix / dense.
# Debug-only; called from apply_minv guarded by `_minv_diag` (hardcoded false).
function minv_diag_dump(lbl, T)
    if ITensors.has_external_storage(T) && T.tensor.data isa WrappedAliasedBlockSparse
        w = T.tensor.data; P = _abs_head_len(w); N = ndims(w.aliased)
        _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
        println("   [MINV_DIAG ", lbl, "] P=$P  prefix=", [_tg(w.inds[i]) for i in 1:P],
                "  dense=", [_tg(w.inds[i]) for i in P+1:N])
    else
        println("   [MINV_DIAG ", lbl, "] storage=", ITensors.has_external_storage(T) ? string(typeof(T.tensor.data)) : "dense")
    end
end
