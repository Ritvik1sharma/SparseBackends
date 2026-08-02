# kl_padh20_v18_bd20_sb_fused.jl — padded-H cost experiment. PHYSICS IS ARTIFICIAL (zero-padded
# bare-H bonds); only per-sweep timing and PHP memory are meaningful.
#   julia --project=. experiments/kl_padh20_v18_bd20_sb_fused.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("kl_padh20_v18_bd20", :sb_fused;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
