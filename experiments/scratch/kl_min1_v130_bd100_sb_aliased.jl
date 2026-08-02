# kl_min1_v130_bd100_sb_aliased.jl
#
# Aliased P†HP (AliasedBlockSparse), dense psi, run_mode=:standard.
# Standalone entry point. To run all SparseBackends variants of this config in
# one process (paying codegen once), use run_group.jl instead:
#   julia --project=. experiments/run_group.jl kl_min1_v130_bd100
#
#   julia --project=. experiments/kl_min1_v130_bd100_sb_aliased.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("kl_min1_v130_bd100", :sb_aliased;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
