# run_group.jl — run several SparseBackends variants of one config in ONE julia
# process, so the SparseBackends codegen cost is paid once instead of per variant.
#
#   julia --project=. experiments/run_group.jl <config_id> [variants] [seed]
#
#   variants  comma-separated, default "sb_aliased,sb_fused,sb_dense"
#             (that order is deliberate: cheapest / most important first, so an
#              OOM in sb_dense cannot lose the aliased results — each variant
#              writes its JSON the moment it finishes)
#   seed      default $BENCH_SEED, else 0
#
# Example:
#   BENCH_OUTDIR=$SCRATCH/sparse_bench/results \
#     julia --project=. experiments/run_group.jl kl_min1_v26_bd40 sb_aliased,sb_fused,sb_dense 0
#
# A variant that throws is recorded as a failure JSON and does NOT stop the rest.

include(joinpath(@__DIR__, "bench_common.jl"))

length(ARGS) < 1 && error("usage: julia --project=. experiments/run_group.jl <config_id> [variants] [seed]")

const CONFIG_ID = ARGS[1]
const VARIANTS  = length(ARGS) >= 2 && !isempty(ARGS[2]) ?
                  Symbol.(split(ARGS[2], ",")) : SB_VARIANT_ORDER
const SEED      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) :
                  parse(Int, get(ENV, "BENCH_SEED", "0"))
const OUTDIR    = bench_outdir()

println("[run_group] config=$CONFIG_ID seed=$SEED variants=" * join(VARIANTS, ","))
println("[run_group] outdir=$OUTDIR")

failures = String[]
for v in VARIANTS
    try
        run_config(CONFIG_ID, v; seed = SEED, outdir = OUTDIR)
    catch e
        push!(failures, String(v))
        println("\n[run_group] VARIANT FAILED: $v")
        showerror(stdout, e, catch_backtrace())
        println()
        # Record the failure so the collector can distinguish "did not run"
        # from "ran and died" (e.g. dense OOM at the largest configs).
        try
            write_result_json(OUTDIR, Dict{String,Any}(
                "config"  => CONFIG_ID,
                "variant" => String(v),
                "seed"    => SEED,
                "status"  => "failed",
                "error"   => sprint(showerror, e),
                "env"     => env_metadata(),
            ))
        catch
        end
    end
end

if isempty(failures)
    println("\n[run_group] all variants completed: " * join(VARIANTS, ", "))
else
    println("\n[run_group] completed with failures: " * join(failures, ", "))
end
nothing
