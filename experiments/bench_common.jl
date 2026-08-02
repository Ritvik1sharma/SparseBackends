# bench_common.jl — SparseBackends side of the benchmark.
#
# Defines run_config(config_id, variant; seed) for the three SparseBackends
# variants. The baseline variant :orig_dense lives in the sibling tree
# experiments/manual_tests/ because it must run under a different Julia project
# (tensornetworks/, i.e. packages/ITensors.jl on branch min-edits).
#
#   :sb_dense    dense PHP,   dense psi,  run_mode = :standard
#   :sb_aliased  aliased PHP, dense psi,  run_mode = :standard
#   :sb_fused    aliased PHP, dense psi,  run_mode = :fused
#
# :sb_aliased and :sb_fused share one operator; only the matvec differs.

using LinearAlgebra: BLAS

# BLAS thread count is the single core-budget knob and is recorded in every
# result. SB_ALIASED_NTHREADS is deliberately left at its default of 1: at >1
# the aliased kernel switches to a task-parallel path that pins BLAS to 1
# internally and is documented as FP-close but NOT bit-identical to serial.
BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", string(Sys.CPU_THREADS))))

include(joinpath(@__DIR__, "..", "sparse_runner_utils.jl"))
include(joinpath(@__DIR__, "configs.jl"))
include(joinpath(@__DIR__, "models.jl"))

const SB_VARIANTS = Dict(
    :sb_dense   => (php = :dense,   run_mode = :standard),
    :sb_aliased => (php = :aliased, run_mode = :standard),
    :sb_fused   => (php = :aliased, run_mode = :fused),
)

"""
Per-site aliased footprint (block count, template count, element counts). Only
meaningful for :sb_aliased / :sb_fused; returns `nothing` otherwise. This is the
compression evidence that Base.summarysize alone does not show.
"""
function aliased_footprint(H::MPO)
    tot = Dict{String,Any}("n_aliased_sites" => 0, "dense_elems" => 0,
                           "bs_elems" => 0, "aliased_elems" => 0, "n_blocks" => 0,
                           "n_templates" => 0)
    for i in 1:length(H)
        T = H[i]
        (ITensors.has_external_storage(T) &&
         T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse) || continue
        a = T.tensor.data.aliased
        nb, nt, bk = length(a.keys), a.n_templates, a.blksize
        tot["n_aliased_sites"] += 1
        tot["n_blocks"]        += nb
        tot["n_templates"]     += nt
        tot["dense_elems"]     += prod(a.dims)
        tot["bs_elems"]        += nb * bk
        tot["aliased_elems"]   += nt * bk + nb
    end
    tot["n_aliased_sites"] == 0 && return nothing
    tot["compression_vs_dense"] = tot["dense_elems"] / max(tot["aliased_elems"], 1)
    tot["compression_vs_bs"]    = tot["bs_elems"]    / max(tot["aliased_elems"], 1)
    return tot
end

"""
Build the PHP operator, the raw Hamiltonian and the initial state for one
config + variant. Returns `(H_raw, H_php, psi0, sites, build_seconds, extras)`.
"""
function build_problem(cfg::NamedTuple, variant::Symbol, seed::Int)
    spec = SB_VARIANTS[variant]
    extras = Dict{String,Any}()

    if cfg.model === :kl
        sites, H_raw, ConsOps1, ConsOps2 = kl_operators(cfg.nplaq, cfg.spin, cfg.psign; pad_h_chi=get(cfg, :pad_h_chi, 0))
        build_seconds = @elapsed begin
            H_php = spec.php === :dense ? kl_php_dense(ConsOps1, H_raw) :
                                          kl_php_aliased(ConsOps1, H_raw)
        end
        psi0 = kl_psi0(sites, ConsOps2, seed)
    elseif cfg.model === :pxp
        sites, H_raw, P = pxp_operators(cfg.nsites; pad_h_chi=get(cfg, :pad_h_chi, 0))
        build_seconds = @elapsed begin
            H_php = spec.php === :dense ? pxp_php_dense(P, H_raw) :
                                          pxp_php_aliased(P, H_raw)
        end
        psi0 = pxp_psi0(sites, P, seed)
        extras["P"] = P
    else
        error("build_problem: unknown model $(cfg.model)")
    end

    extras["sites"] = sites
    return H_raw, H_php, psi0, sites, build_seconds, extras
end

"""
Run one (config, variant, seed) and write its JSON. Raises on failure so the
caller (run_group.jl) can record the failure and continue with the next variant.
"""
function run_config(config_id::AbstractString, variant::Symbol;
                    seed::Int = parse(Int, get(ENV, "BENCH_SEED", "0")),
                    outdir::AbstractString = bench_outdir())
    haskey(SB_VARIANTS, variant) ||
        error("run_config: variant $variant is not a SparseBackends variant; " *
              ":orig_dense lives in experiments/manual_tests/")
    cfg  = config_or_die(config_id)
    spec = SB_VARIANTS[variant]

    println("\n", "="^78)
    println("[bench] config=$config_id  variant=$variant  seed=$seed")
    println("[bench] model=$(cfg.model)  maxdim=$(cfg.maxdim)  nsweeps=$(cfg.nsweeps)")
    println("[bench] BLAS threads=$(BLAS.get_num_threads())  julia threads=$(Threads.nthreads())")
    println("="^78)

    H_raw, H_php, psi0, sites, build_seconds, extras = build_problem(cfg, variant, seed)

    result = Dict{String,Any}(
        "config"          => config_id,
        "variant"         => String(variant),
        "seed"            => seed,
        "model"           => String(cfg.model),
        "maxdim"          => cfg.maxdim,
        "nsweeps"         => cfg.nsweeps,
        "cutoff"          => cfg.cutoff,
        "run_mode"        => String(spec.run_mode),
        "php_backend"     => String(spec.php),

        "dense_sandwich"  => "exact_is_ctn",   # no SVD; see pxp_php_dense docstring
        "nsites"          => length(sites),

        "pad_h_chi"       => get(cfg, :pad_h_chi, 0),
        "build_seconds"   => build_seconds,
        "php"             => mpo_stats(H_php),
        "h_raw"           => mpo_stats(H_raw),
        "psi0_bytes"      => mps_bytes(psi0),
        "maxrss_after_build_bytes" => Int(Sys.maxrss()),
        "env"             => env_metadata(),
    )
    cfg.model === :kl && (result["nplaq"] = cfg.nplaq; result["psign"] = cfg.psign;
                          result["spin"] = cfg.spin)
    fp = aliased_footprint(H_php)
    fp !== nothing && (result["php"]["aliased_footprint"] = fp)

    @printf("[bench] PHP memory: %.4f MB   build %.3f s\n",
            result["php"]["summarysize_bytes"] / 1e6, build_seconds)

    md = [cfg.maxdim]
    # roofline=true arms SparseBackends' MAC / permute-fire counters for this run
    # (dmrg calls set_roofline! with it). Tied to the same BENCH_TIMERS knob as the
    # TimerOutputs trees so one env var turns on the whole profiling picture.
    # SparseBackends-only kwarg — never added on the packages/ baseline side.
    profiling = get(ENV, "BENCH_TIMERS", "0") == "1"
    common = (nsweeps = cfg.nsweeps, maxdim = md, mindim = md, cutoff = cfg.cutoff,
              run_mode = spec.run_mode, roofline = profiling)

    # ── Ground state ─────────────────────────────────────────────────────────
    E0, psi_g, _, rec_g = timed_dmrg_ground("$config_id/$variant/ground",
                                            H_php, deepcopy(psi0); common...)
    rec_g["energy_raw_H"] = real(inner(psi_g', H_raw, psi_g))
    rec_g["psi_bytes"]    = mps_bytes(psi_g)
    result["ground"] = rec_g

    # ── First excited (PXP only) ─────────────────────────────────────────────
    if cfg.model === :pxp
        # Same P as the ground-state init; fresh random MPS from seed+1, matching
        # test_pxp_aliased.jl's excited-state initialisation.
        psi0_e = pxp_psi0(sites, extras["P"], seed + 1)
        E1, psi_e, _, rec_e = timed_dmrg_excited("$config_id/$variant/excited",
                                                 H_php, MPS[psi_g], psi0_e;
                                                 weight = cfg.weight, common...)
        rec_e["energy_raw_H"] = real(inner(psi_e', H_raw, psi_e))
        rec_e["psi_bytes"]    = mps_bytes(psi_e)
        result["excited"] = rec_e
        result["gap_php"] = rec_e["energy_php"] - rec_g["energy_php"]
    end

    # BENCH_TIMERS=1 → dump the ProjMPO / SparseBackends timer trees. Used to
    # diagnose the sb_dense-vs-orig_dense gap: the two produce an operator of
    # identical chi and memory, yet sb_dense is 7-15% slower on KL and 21-52% on
    # PXP, so the cost is per-call overhead somewhere in this fork's matvec
    # (6 @timeit sites and 29 has_external_storage checks that packages/ lacks).
    # packages/ has no PROJMPO_TIMER, so this shows the SB side's internal split
    # only — it localises the cost within this fork, it does not diff the forks.
    # BENCH_TIMERS=1 → dump ALL FOUR profilers. Each variant is covered by a
    # different subset, so the full set is printed unconditionally and the ones
    # that do not apply print an empty table:
    #
    #   sb_dense    PROJMPO_TIMER only (never enters the aliased kernel; the SB
    #               TIMER prints "0.0% measured" with no rows)
    #   sb_aliased  PROJMPO_TIMER + SB TIMER (add.* sections) + FLOP counts
    #   sb_fused    PROJMPO_TIMER + FLOP counts + the fused 4-slot profiler.
    #               NOTE its SB TIMER add.* rows cover only the ~2 EDGE bonds:
    #               use_fused is gated on both envs being non-OneITensor
    #               (dmrg.jl), so bulk bonds bypass the instrumented kernel
    #               entirely. Reading those rows as "the fused profile" is wrong.
    if profiling
        println("\n===== PROJMPO_TIMER  ($config_id / $variant) =====")
        show(stdout, ITensorMPS.PROJMPO_TIMER; sortby = :time); println()
        println("\n===== SparseBackends.TIMER =====")
        show(stdout, SparseBackends.TIMER; sortby = :time); println()
        println("\n===== fused partial-matvec profile ($variant) =====")
        if isdefined(ITensorMPS, :report_partial_prof!)
            # reps=1 ⇒ the µs column is the RUN TOTAL, not a per-call average
            # (the call count is not tracked here); the % column is unaffected.
            getfield(ITensorMPS, :report_partial_prof!)(1)
        else
            println("    (fused profiler not available in this build)")
        end
        report_flops("$config_id / $variant")
    end

    result["maxrss_bytes"] = Int(Sys.maxrss())
    write_result_json(outdir, result)
    return result
end
