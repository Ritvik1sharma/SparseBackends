module SparseBackends

using TimerOutputs: TimerOutputs, TimerOutput, @timeit, reset_timer!, print_timer

# Always-on instrumentation timer for the SparseBackends contract paths.
# Reset between phases via `reset_timer!(SparseBackends.TIMER)` and print with
# `print_timer(SparseBackends.TIMER)` from user code.
const TIMER = TimerOutput()

export TIMER

# ── DMRG contraction FLOP counter (gated by SB_FLOP_COUNT=1) ──────────────────
# Honest, symmetric multiply–accumulate (MAC) accounting for ALL the tensor
# contractions DMRG performs to APPLY the Hamiltonian — both the local matvec
# (eigsolve) AND the position! environment rebuilds — so the aliased PHP path
# can be compared FLOP-for-FLOP against the dense PHP path on the SAME workload.
#
# Two phases are tracked separately (via the SB_IN_POSITION flag dmrg.jl sets):
#   matvec   — contractions inside product() / eigsolve.
#   position — contractions inside _makeL!/_makeR! env rebuilds.
# and within each phase, by which backend did the work:
#   dense_macs   — Σ over DENSE contractions of ∏(dims of all distinct indices)
#                  (one BLAS GEMM ⇒ MACs = output·contracted). Counted for the
#                  dense-PHP run AND for the dense pieces of the aliased run
#                  (env L/R steps, ψ·ψ† env folds).
#   ali_macs     — Σ over aliased-kernel contractions of nA·M·K·N (the ACTUAL
#                  per-block GEMM work the aliased×dense kernel performs).
#   ali_de_macs  — for those same aliased contractions, the MACs a FULLY DENSE
#                  contraction WOULD cost (sparse channel un-factored). The ratio
#                  ali_de/ali = in-kernel block-sparsity saving.
# A dense contraction is NEVER also counted as aliased (the matvec/env hooks gate
# on "operand not aliased"; aliased operands route through the kernel, which
# counts itself), so there is no double-counting.
#
# A "MAC" is one multiply-add. Real FLOPs ≈ 2·MAC; for ComplexF64 a complex
# multiply-add ≈ 8 real FLOPs (report_flops prints both conventions).
mutable struct _FlopCounter
    mv_dense_steps :: Int;  mv_dense_macs :: Int
    mv_ali_steps   :: Int;  mv_ali_macs   :: Int;  mv_ali_de_macs :: Int
    po_dense_steps :: Int;  po_dense_macs :: Int
    po_ali_steps   :: Int;  po_ali_macs   :: Int;  po_ali_de_macs :: Int
    # reshuffle fire-counts (aliased kernel): how often permute_B / permute_back
    # ACTUALLY permute (non-identity) vs are no-ops, split matvec/position.
    mv_pB_fire :: Int;  mv_pBack_fire :: Int
    po_pB_fire :: Int;  po_pBack_fire :: Int
end
const FLOP_COUNTER = _FlopCounter(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
@inline _flop_count_enabled() = get(ENV, "SB_FLOP_COUNT", "0") == "1"
@inline _flop_in_position()   = get(ENV, "SB_IN_POSITION", "0") == "1"
function reset_flops!()
    f = FLOP_COUNTER
    f.mv_dense_steps = 0; f.mv_dense_macs = 0
    f.mv_ali_steps = 0; f.mv_ali_macs = 0; f.mv_ali_de_macs = 0
    f.po_dense_steps = 0; f.po_dense_macs = 0
    f.po_ali_steps = 0; f.po_ali_macs = 0; f.po_ali_de_macs = 0
    f.mv_pB_fire = 0; f.mv_pBack_fire = 0; f.po_pB_fire = 0; f.po_pBack_fire = 0
    return f
end
# Record whether the aliased kernel's permute_B / permute_back actually fired
# (non-identity) on this call. Split matvec vs position via SB_IN_POSITION.
@inline function add_reshuffle!(pB_fired::Bool, pBack_fired::Bool)
    f = FLOP_COUNTER
    if _flop_in_position()
        f.po_pB_fire += pB_fired; f.po_pBack_fire += pBack_fired
    else
        f.mv_pB_fire += pB_fired; f.mv_pBack_fire += pBack_fired
    end
    return nothing
end
@inline function add_dense_macs!(macs::Integer)
    f = FLOP_COUNTER
    if _flop_in_position()
        f.po_dense_steps += 1; f.po_dense_macs += macs
    else
        f.mv_dense_steps += 1; f.mv_dense_macs += macs
    end
    return nothing
end
@inline function add_aliased_macs!(actual_macs::Integer, denseequiv_macs::Integer)
    f = FLOP_COUNTER
    if _flop_in_position()
        f.po_ali_steps += 1; f.po_ali_macs += actual_macs; f.po_ali_de_macs += denseequiv_macs
    else
        f.mv_ali_steps += 1; f.mv_ali_macs += actual_macs; f.mv_ali_de_macs += denseequiv_macs
    end
    return nothing
end
function report_flops(label::AbstractString="")
    f = FLOP_COUNTER
    _saving(de, ac) = ac > 0 ? string(round(de / ac; digits=2), "x") : "—"
    mv_total = f.mv_dense_macs + f.mv_ali_macs
    po_total = f.po_dense_macs + f.po_ali_macs
    tot      = mv_total + po_total
    println("\n========== DMRG contraction FLOP count ",
            isempty(label) ? "" : "($label) ", "==========")
    println("  MATVEC (eigsolve):")
    println("    dense   : steps=", f.mv_dense_steps, "  MACs=", f.mv_dense_macs)
    println("    aliased : steps=", f.mv_ali_steps, "  MACs(actual)=", f.mv_ali_macs,
            "  MACs(dense-equiv)=", f.mv_ali_de_macs,
            "  in-kernel saving=", _saving(f.mv_ali_de_macs, f.mv_ali_macs))
    println("    matvec total MACs = ", mv_total)
    println("  POSITION! (env rebuild):")
    println("    dense   : steps=", f.po_dense_steps, "  MACs=", f.po_dense_macs)
    println("    aliased : steps=", f.po_ali_steps, "  MACs(actual)=", f.po_ali_macs,
            "  MACs(dense-equiv)=", f.po_ali_de_macs,
            "  in-kernel saving=", _saving(f.po_ali_de_macs, f.po_ali_macs))
    println("    position total MACs = ", po_total)
    println("  ── OVERALL total MACs = ", tot)
    println("     ≈ real FLOPs (×2)    = ", 2 * tot)
    println("     ≈ complex FLOPs (×8) = ", 8 * tot)
    println("  RESHUFFLE fires (aliased kernel, non-identity permutes):")
    println("    matvec  : permute_B=", f.mv_pB_fire, "  permute_back=", f.mv_pBack_fire,
            "   (of ", f.mv_ali_steps, " aliased steps → ",
            f.mv_ali_steps > 0 ? round((f.mv_pB_fire + f.mv_pBack_fire) / f.mv_ali_steps; digits=2) : 0.0,
            " reshuffles/step)")
    println("    position: permute_B=", f.po_pB_fire, "  permute_back=", f.po_pBack_fire,
            "   (of ", f.po_ali_steps, " aliased steps)")
    return tot
end
export FLOP_COUNTER, reset_flops!, report_flops, add_dense_macs!, add_aliased_macs!, add_reshuffle!

# export whatever should be public:
export NewBlockSparseSorted, blocksparse_from_dense, to_dense
export COOTensor, coo_from_dense, to_dense
export AliasedBlockSparse, to_blocksparse, to_dense, contract_aliased!, compression_ratio
export WrappedAliasedBlockSparse, contract_aliased_itensor,
       contract_coo_dense_aliased

# ── Backend tag (enum) ───────────────────────────────────────────────────────
# The single source of truth for a tensor-storage backend. Using an enum (not a
# bare Symbol) means there is exactly ONE name per backend — a second alias for
# the same datatype (e.g. the old :aliasedblocksparse for :aliased) is impossible.
# The public contract API still accepts Symbols for ergonomics; `to_backend`
# converts at the boundary and ERRORS on any non-canonical name. Internal
# dispatch compares the enum.
@enum Backend DENSE COO BLOCKSPARSE ALIASED
export Backend, DENSE, COO, BLOCKSPARSE, ALIASED, to_backend

@inline to_backend(b::Backend) = b
@inline function to_backend(s::Symbol)::Backend
    s === :dense       ? DENSE :
    s === :coo         ? COO :
    s === :blocksparse ? BLOCKSPARSE :
    s === :aliased     ? ALIASED :
    throw(ArgumentError("Unknown backend $(repr(s)); valid: :dense, :coo, :blocksparse, :aliased"))
end
# Reverse map for the few internal helpers still keyed on a Symbol.
@inline Base.Symbol(b::Backend) =
    b === DENSE ? :dense : b === COO ? :coo : b === BLOCKSPARSE ? :blocksparse : :aliased

include("base.jl")

# ── BlockSparse ──────────────────────────────────────────────────────────────
include("blocksparse/storage.jl")
include("blocksparse/conversions.jl")

# ── COO ──────────────────────────────────────────────────────────────────────
include("coo/storage.jl")
include("coo/conversions.jl")

# ── Contraction dispatch + BlockSparse/COO kernels ───────────────────────────
# contract.jl also defines: Label, _dims, _prefix_lin, aligned_A_to_Cprefix,
# _cdense_grouping_and_orders, _advance_run, _cmp_join_tuple (via sub-includes)
include("tensoralgebra/contract.jl")

# ── AliasedBlockSparse ───────────────────────────────────────────────────────
# Must come after blocksparse/storage.jl (needs _check_no_cross_perm, _prefix_lin,
# NewBlockSparseSorted) and after contract.jl (needs Label, _dims).
include("aliased/storage.jl")
include("aliased/conversions.jl")

# ── AliasedBlockSparse contraction kernels ───────────────────────────────────
# Must come after aliased/storage.jl and after contract.jl helpers.
include("tensoralgebra/contract_aliased_coo_dense.jl")   # COOTensor × Dense → AliasedBS
include("tensoralgebra/contract_aliased.jl")              # AliasedBS × Dense / AliasedBS × AliasedBS (single label)
include("tensoralgebra/contract_aliased_shared.jl")       # helpers + AliasedBS × AliasedBS (multi-label)
include("tensoralgebra/contract_aliased_dense_shared.jl") # AliasedBS × Dense (multi-label, threaded + serial)
include("tensoralgebra/contract_aliased_dense_legacy.jl") # legacy reduction-stationary serial kernel (A/B only, SB_ALIASED_LEGACY=1)
include("tensoralgebra/contract_coo_aliased.jl")          # COOTensor × AliasedBS → AliasedBS (single label, r in B prefix)
include("tensoralgebra/contract_aliased_dense_to_dense.jl") # AliasedBS × Dense → Dense (alias-amplified, for matvec)

include("ops_factorize.jl")
include("ops_factorize_qr.jl")
include("ops_factorize_svd_owned.jl")
include("tensor_wrappers.jl")
include("tensor_index.jl")
include("tensor_contraction.jl")
include("tensor_wrappers_aliased.jl")   # WrappedAliasedBlockSparse + top-level aliased contract functions
include("aliased/factorize.jl")         # Aliased-aware factorize (Tier 1: dense SVD + re-aliasify)

# Path-B (generalized eigsolve) helpers for sparse DMRG. Must be after
# tensor_wrappers.jl so contract_preserve_bs / recast_bs_to_template are in scope.
include("path_b_utils.jl")
include("path_b_helpers.jl")
end
