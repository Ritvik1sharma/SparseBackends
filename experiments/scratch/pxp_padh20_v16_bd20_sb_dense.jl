# pxp_padh20_v16_bd20_sb_dense.jl — padded-H cost experiment. PHYSICS IS ARTIFICIAL (zero-padded
# bare-H bonds); only per-sweep timing and PHP memory are meaningful.
#   julia --project=. experiments/pxp_padh20_v16_bd20_sb_dense.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("pxp_padh20_v16_bd20", :sb_dense;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
