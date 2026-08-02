# pxp_v12_bd40_sb_aliased.jl
#
# Historical-reproduction config (1-BLAS-thread comparison). See configs.jl.
#   BENCH_BLAS_THREADS=1 julia --project=. experiments/pxp_v12_bd40_sb_aliased.jl [seed]

include(joinpath(@__DIR__, "bench_common.jl"))

run_config("pxp_v12_bd40", :sb_aliased;
           seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0")))
nothing
