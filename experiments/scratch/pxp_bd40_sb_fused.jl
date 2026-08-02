# pxp_bd40_sb_fused.jl
#   julia --project=. experiments/pxp_bd40_sb_fused.jl [seed]
include(joinpath(@__DIR__, "bench_common.jl"))
run_config("pxp_bd40", :sb_fused;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
