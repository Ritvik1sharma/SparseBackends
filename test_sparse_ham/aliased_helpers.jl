# aliased_helpers.jl
#
# Shared utilities for aliased BlockSparse PHP tests.
# Include from a test driver with:  include("aliased_helpers.jl")
#
# Provides:
#   fuse_sparse_links!          — fuse each bond's key link strands (ITensorMPS fork)
#   prepermute_aliased_mpo!     — dense blocks as [right, left] link (ITensorMPS fork)
#   report_aliased_footprint    — per-site memory table (dense / BS / aliased) for an aliased MPO
#   run_dmrg_ground             — warmup + timed ground-state DMRG + timer reports
#   run_dmrg_excited            — warmup + timed excited-state DMRG + timer reports

using SparseBackends
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer

# ─────────────────────────────────────────────────────────────────────────────
# Bond layout of an aliased MPO. The implementation lives in the ITensorMPS.jl fork
# (src/aliased_mpo_bonds.jl), and contract(A::MPO, B::MPO, backends...) already applies
# both steps to an aliased product; calling these again is a no-op on such an MPO.
# ─────────────────────────────────────────────────────────────────────────────
"""Fuse each bond's same-tag key link strands into one key index."""
fuse_sparse_links!(H::MPO) = ITensorMPS.fuse_aliased_bond_strands!(H)

"""Store every aliased site's dense block as [right link, left link]."""
function prepermute_aliased_mpo!(H::MPO)
    n = ITensorMPS.order_aliased_dense_blocks!(H)
    println("  prepermute_aliased_mpo!: permuted $n of $(length(H)) sites")
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-site memory footprint reporter for an aliased MPO.
# Prints dense / BS / aliased element counts and compression ratios.
# ─────────────────────────────────────────────────────────────────────────────
function report_aliased_footprint(H::MPO, label::String)
    total_bytes = sum(Base.summarysize(W) for W in H)
    println("$label total MPO memory: $(round(total_bytes/1e6; digits=4)) MB")
    println("  ", rpad("site", 5), rpad("nb", 6), rpad("ntmpl", 7),
            rpad("blksize", 9), rpad("dense", 10), rpad("BS", 10),
            rpad("aliased", 10), rpad("vs-dense", 10), "vs-BS")
    tot_d = 0; tot_b = 0; tot_a = 0
    for i in 1:length(H)
        T = H[i]
        ITensors.has_external_storage(T) &&
            T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse || continue
        ali  = T.tensor.data.aliased
        nb   = length(ali.keys); nt = ali.n_templates; bksz = ali.blksize
        ds   = prod(ali.dims);   bs = nb * bksz;       as   = nt * bksz + nb
        tot_d += ds; tot_b += bs; tot_a += as
        println("  ", rpad(string(i), 5), rpad(string(nb), 6), rpad(string(nt), 7),
                       rpad(string(bksz), 9), rpad(string(ds), 10),
                       rpad(string(bs), 10), rpad(string(as), 10),
                       rpad(string(round(ds/max(as,1); digits=2)), 10),
                       round(bs/max(as,1); digits=2))
    end
    println("  ", rpad("TOT", 5), rpad("", 6), rpad("", 7), rpad("", 9),
                  rpad(string(tot_d), 10), rpad(string(tot_b), 10),
                  rpad(string(tot_a), 10),
                  rpad(string(round(tot_d/max(tot_a,1); digits=2)), 10),
                  round(tot_b/max(tot_a,1); digits=2))
end

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
