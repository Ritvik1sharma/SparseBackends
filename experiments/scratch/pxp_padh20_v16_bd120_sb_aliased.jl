# pxp_padh20_v16_bd120_sb_aliased.jl — padded-H cost experiment. PHYSICS IS ARTIFICIAL (zero-padded
# bare-H bonds); only per-sweep timing and PHP memory are meaningful.
#   julia --project=. experiments/pxp_padh20_v16_bd120_sb_aliased.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("pxp_padh20_v16_bd120", :sb_aliased;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
