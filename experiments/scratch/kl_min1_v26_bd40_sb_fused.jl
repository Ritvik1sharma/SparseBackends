# kl_min1_v26_bd40_sb_fused.jl
#
# Aliased P†HP, dense psi, run_mode=:fused (partial-fused matvec).
# Standalone entry point. To run all SparseBackends variants of this config in
# one process (paying codegen once), use run_group.jl instead:
#   julia --project=. experiments/run_group.jl kl_min1_v26_bd40
#
#   julia --project=. experiments/kl_min1_v26_bd40_sb_fused.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("kl_min1_v26_bd40", :sb_fused;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
