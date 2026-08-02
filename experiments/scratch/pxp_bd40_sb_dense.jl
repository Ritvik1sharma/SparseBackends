# pxp_bd40_sb_dense.jl
#   julia --project=. experiments/pxp_bd40_sb_dense.jl [seed]
include(joinpath(@__DIR__, "bench_common.jl"))
run_config("pxp_bd40", :sb_dense;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
