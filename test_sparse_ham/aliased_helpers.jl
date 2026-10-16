# aliased_helpers.jl
#
# Shared utilities for aliased BlockSparse PHP tests.
# Include from a test driver with:  include("aliased_helpers.jl")
#
# Provides:
#   run_dmrg_ground             — warmup + timed ground-state DMRG + timer reports
#   run_dmrg_excited            — warmup + timed excited-state DMRG + timer reports

using SparseBackends
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer

# ─────────────────────────────────────────────────────────────────────────────
# DMRG runners — ground state and excited state are separate functions.
#
# Both follow the same pattern:
#   1. Warmup JIT sweep (discarded, error caught silently).
#   2. Reset SparseBackends.TIMER and ITensorMPS.PROJMPO_TIMER.
#   3. GC twice to reduce allocation noise.
#   4. Timed production run.
#   5. Print ProjMPO and SparseBackends timer reports.
#   6. Optionally print SB_PERM_PROFILE / SB_ALIAS_STATS diagnostics.
#
# Return: (dmrg_result_tuple, wall_seconds_excluding_jit)
# ─────────────────────────────────────────────────────────────────────────────
function run_dmrg_ground(label, H, psi0; nsweeps, maxdim, mindim, cutoff, roofline::Bool=false, kwargs...)
    println("[warmup pass: 1 JIT sweep, results discarded]")
    try
        dmrg(H, deepcopy(psi0); nsweeps=1, maxdim=10, mindim=10, cutoff=1e-12,
             outputlevel=0, use_early_exit=false)
    catch e
        println("  warmup failed: ", sprint(showerror, e))
    end
    reset_timer!(SparseBackends.TIMER)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    # Zero the roofline/flop accumulators once before the (single, non-warmup)
    # timed dmrg call below — was gated on SB_FLOP_COUNT env var.
    SparseBackends.reset_roofline!(roofline); SparseBackends.reset_flops!(roofline)
    GC.gc(); GC.gc()
    wall = @elapsed begin
        result = dmrg(H, psi0; nsweeps, maxdim, mindim, cutoff, use_early_exit=false, roofline=roofline, kwargs...)
    end
    if roofline
        SparseBackends.report_flops(label)
        SparseBackends.show_roofline()
        SparseBackends.show_cas_stats()
        ITensorMPS.print_env_footprint()
        SparseBackends.show_gemm_dims_hist()
    end
    println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
    println("\n--- ProjMPO matvec breakdown ---")
    print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
    println("\n--- SparseBackends contract dispatch breakdown ---")
    print_timer(SparseBackends.TIMER; sortby=:firstexec)
    get(ENV, "SB_PERM_PROFILE", "0") == "1" && (SparseBackends._report_perm_profile(); SparseBackends._reset_perm_profile!())
    get(ENV, "SB_ALIAS_STATS",  "0") == "1" && (SparseBackends._report_alias_stats();  SparseBackends._reset_alias_stats!())
    println("==========\n")
    return result, wall
end

function run_dmrg_excited(label, H, Ms::Vector{MPS}, psi0; nsweeps, maxdim, mindim,
                          cutoff, weight, kwargs...)
    println("[warmup pass: 1 JIT sweep, results discarded]")
    try
        dmrg(H, Ms, deepcopy(psi0); nsweeps=1, maxdim=10, mindim=10, cutoff=1e-12,
             weight=weight, outputlevel=0, use_early_exit=false)
    catch e
        println("  warmup failed: ", sprint(showerror, e))
    end
    reset_timer!(SparseBackends.TIMER)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    GC.gc(); GC.gc()
    wall = @elapsed begin
        result = dmrg(H, Ms, psi0; nsweeps, maxdim, mindim, cutoff,
                      weight, use_early_exit=false, kwargs...)
    end
    println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
    println("\n--- ProjMPO matvec breakdown ---")
    print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
    println("\n--- SparseBackends contract dispatch breakdown ---")
    print_timer(SparseBackends.TIMER; sortby=:firstexec)
    get(ENV, "SB_PERM_PROFILE", "0") == "1" && (SparseBackends._report_perm_profile(); SparseBackends._reset_perm_profile!())
    get(ENV, "SB_ALIAS_STATS",  "0") == "1" && (SparseBackends._report_alias_stats();  SparseBackends._reset_alias_stats!())
    println("==========\n")
    return result, wall
end
