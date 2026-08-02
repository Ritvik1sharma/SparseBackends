# sparse_runner_utils.jl — measurement harness shared by the dense-vs-aliased benchmark.
#
# Included by BOTH benchmark trees, which run under different Julia projects and
# different (incompatible) ITensors forks:
#   experiments/manual_tests/        (baseline, packages/ ITensors — min-edits)
#   SparseBackends/experiments/      (sb_dense / sb_aliased / sb_fused)
#
# ONE shared copy, not one per tree, on purpose. The quantity being measured
# includes the fork penalty (sb_dense vs orig_dense at identical chi and memory),
# so if each tree carried its own harness any drift between the two copies —
# a different warmup rule, a different byte accounting — would show up as a fake
# fork penalty and make that number uninterpretable.
#
# It lives at the SparseBackends root rather than inside either experiments/ dir
# so neither tree owns it. NOTE this file must stay FREE of any SparseBackends
# dependency: it is `include`d by the baseline tree, which runs under
# --project=tensornetworks and cannot load the SparseBackends package. Only
# ITensors, ITensorMPS, JSON, Printf, LinearAlgebra and TimerOutputs are used,
# and all SparseBackends lookups are done defensively via Base.loaded_modules.
#
# Provides:
#   SweepTimer            — AbstractObserver capturing per-sweep wall time,
#                           energy and peak RSS (works in both ITensorMPS forks;
#                           dmrg calls checkdone! once per sweep)
#   timed_dmrg_ground     — warmup + GC + timed ground-state run
#   timed_dmrg_excited    — warmup + GC + timed excited-state run
#   mpo_bytes / mpo_stats — Hamiltonian memory accounting
#   env_metadata          — host / SLURM / thread / version provenance
#   write_result_json     — one JSON per (config, variant, seed)
#
# Timing convention (matches test_sparse_ham/aliased_helpers.jl): one warmup
# sweep at maxdim=10 is run and discarded so codegen/JIT is excluded, then two
# GC passes, then the timed run. Codegen is therefore never inside a reported
# number — it only costs queue time.

using ITensors, ITensorMPS
using JSON
using Printf
using LinearAlgebra
using TimerOutputs

# ─────────────────────────────────────────────────────────────────────────────
# Per-sweep instrumentation
# ─────────────────────────────────────────────────────────────────────────────

mutable struct SweepTimer <: ITensorMPS.AbstractObserver
    t_prev::Float64
    seconds::Vector{Float64}
    energies::Vector{Float64}
    maxrss::Vector{Int}
end
SweepTimer() = SweepTimer(time(), Float64[], Float64[], Int[])

"""Reset the clock immediately before the timed dmrg call."""
start!(o::SweepTimer) = (o.t_prev = time(); o)

# dmrg calls this once per sweep, after the sweep's own bookkeeping. Returning
# false never stops the run early — nsweeps is fixed and use_early_exit=false.
function ITensorMPS.checkdone!(o::SweepTimer; energy = nothing, kwargs...)
    now = time()
    push!(o.seconds, now - o.t_prev)
    o.t_prev = now
    push!(o.energies, energy === nothing ? NaN : real(energy))
    push!(o.maxrss, Int(Sys.maxrss()))
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# Memory accounting
# ─────────────────────────────────────────────────────────────────────────────

mpo_bytes(H::MPO) = sum(Base.summarysize(W) for W in H)
mps_bytes(psi::MPS) = sum(Base.summarysize(W) for W in psi)

"""
    true_bond_dims(H)

Bond dimension of each MPO bond, as the PRODUCT of every index shared by the two
adjacent site tensors.

Do NOT use `maxlinkdim` for this. `maxlinkdim` goes through `linkind`, which
returns ONE common index; the aliased PHP carries each bond on TWO of them (a
projector factor and a Hamiltonian factor, e.g. `Link,l=9`=16 alongside
`Link,PadH,l=9`=20 for a bond that is really 320). It therefore under-reports the
aliased bond by exactly chi_H and makes the aliased operator look like it has a
smaller bond than the dense one it is bit-for-bit equivalent to. Both operators
have the same bond; only the factorization differs.
"""
function true_bond_dims(H::MPO)
    dims = Int[]
    for j in 1:(length(H) - 1)
        ci = commoninds(H[j], H[j + 1])
        push!(dims, isempty(ci) ? 1 : prod(dim.(ci)))
    end
    return dims
end

"""
Structural + memory summary of a Hamiltonian MPO. `summarysize_bytes` is the
cross-variant comparable number. `bond_max` / `bond_dims` are the TRUE bonds (see
`true_bond_dims`); `maxlinkdim` is retained only for continuity with earlier
result files and is NOT comparable between dense and aliased operators.
"""
function mpo_stats(H::MPO)
    d = Dict{String,Any}(
        "nsites"            => length(H),
        "summarysize_bytes" => mpo_bytes(H),
    )
    try
        d["maxlinkdim"] = maxlinkdim(H)
    catch
        d["maxlinkdim"] = nothing
    end
    try
        bd = true_bond_dims(H)
        d["bond_max"]  = isempty(bd) ? nothing : maximum(bd)
        d["bond_dims"] = bd
    catch e
        d["bond_max"] = nothing; d["bond_dims"] = nothing
        d["bond_error"] = sprint(showerror, e)
    end
    return d
end

# ─────────────────────────────────────────────────────────────────────────────
# Timed DMRG runners
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Profiler reset (BENCH_TIMERS=1)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _reset_profilers!()

Zero every profiling accumulator, so a `BENCH_TIMERS=1` report covers only the
TIMED run and not the warmup sweep. Called after warmup, before the timed dmrg.

Four independent accumulators, because the three code paths are instrumented by
three different mechanisms and no single one covers the variants:

  ITensorMPS.PROJMPO_TIMER   TimerOutputs — dmrg-level split (all variants)
  SparseBackends.TIMER       TimerOutputs — the aliased kernel's add.* sections;
                             populated ONLY by :standard (`sb_aliased`), since
                             `contract_aliased_dense_to_dense` is the thing it
                             instruments and both `sb_dense` and the fused bulk
                             path bypass it
  ITensorMPS._PF_TIMES       4-slot manual profiler for `matvec_partial_fused_full`
                             — the ONLY instrumentation the :fused bulk path has
  SparseBackends.FLOP_COUNTER MAC counts + permute fire-counts; gated by dmrg's
                             `roofline` kwarg, which bench_common.jl sets from
                             this same env var

Everything is looked up defensively: `packages/` (the orig_dense baseline) has
no PROJMPO_TIMER, no SparseBackends, and no fused kernel, and this file is shared
with that tree.
"""
function _reset_profilers!()
    get(ENV, "BENCH_TIMERS", "0") == "1" || return nothing
    isdefined(ITensorMPS, :PROJMPO_TIMER) &&
        TimerOutputs.reset_timer!(ITensorMPS.PROJMPO_TIMER)
    # Fused matvec's own 4-slot profiler. Arm it here too: it is a plain Ref, so
    # if it is never set to true the fused path records nothing and the report
    # prints "(no profiling data)".
    if isdefined(ITensorMPS, :_PF_PROF)
        getfield(ITensorMPS, :_PF_PROF)[] = true
        isdefined(ITensorMPS, :reset_partial_prof!) &&
            getfield(ITensorMPS, :reset_partial_prof!)()
    end
    SB = get(Base.loaded_modules,
             Base.PkgId(Base.UUID("60b00394-95c5-4a10-8e1c-b93543744110"), "SparseBackends"),
             nothing)
    if SB !== nothing
        isdefined(SB, :TIMER) && TimerOutputs.reset_timer!(getfield(SB, :TIMER))
        # reset_flops!(true) both zeroes the counter and arms it. dmrg's own
        # `set_roofline!(roofline)` call runs later and would otherwise be the
        # only thing arming it — this makes the state explicit per timed run.
        isdefined(SB, :reset_flops!) && getfield(SB, :reset_flops!)(true)
    end
    return nothing
end

function _warmup_ground(H, psi0; kwargs...)
    try
        dmrg(H, deepcopy(psi0); nsweeps = 1, maxdim = 10, mindim = 10,
             cutoff = 1e-12, outputlevel = 0, use_early_exit = false, kwargs...)
    catch e
        println("  [warmup] discarded, failed: ", sprint(showerror, e))
    end
    return nothing
end

function _warmup_excited(H, Ms, psi0; weight, kwargs...)
    try
        dmrg(H, Ms, deepcopy(psi0); nsweeps = 1, maxdim = 10, mindim = 10,
             cutoff = 1e-12, weight = weight, outputlevel = 0,
             use_early_exit = false, kwargs...)
    catch e
        println("  [warmup] discarded, failed: ", sprint(showerror, e))
    end
    return nothing
end

"""
Run ground-state DMRG with warmup excluded. Returns
`(energy, psi, truncerr, record::Dict)` where `record` holds the per-sweep and
whole-run timing/memory numbers.
"""
function timed_dmrg_ground(label, H, psi0; nsweeps, maxdim, mindim, cutoff, kwargs...)
    println("\n[$label] warmup sweep (discarded, excludes codegen from timing)")
    _warmup_ground(H, psi0; kwargs...)

    # Zero every profiler AFTER the warmup so BENCH_TIMERS reports only the
    # timed run. See _reset_profilers! for what is covered and why it is guarded.
    _reset_profilers!()
    GC.gc(); GC.gc()
    obs = start!(SweepTimer())
    gc0 = Base.gc_num()
    wall = @elapsed begin
        energy, psi, _nsw, truncerr = dmrg(H, psi0; nsweeps, maxdim, mindim, cutoff,
                                          observer = obs, use_early_exit = false,
                                          outputlevel = 1, kwargs...)
    end
    gcd = Base.GC_Diff(Base.gc_num(), gc0)

    record = Dict{String,Any}(
        "total_seconds"        => wall,
        "sweep_seconds"        => obs.seconds,
        "sweep_energies"       => obs.energies,
        "sweep_maxrss_bytes"   => obs.maxrss,
        "sweeps_completed"     => length(obs.seconds),
        "sweeps_requested"     => nsweeps,
        "mean_sweep_seconds"   => isempty(obs.seconds) ? nothing : sum(obs.seconds) / length(obs.seconds),
        "energy_php"           => real(energy),
        "truncerr"             => truncerr,
        "allocated_bytes"      => gcd.allocd,
        "gc_seconds"           => gcd.total_time / 1e9,
        "maxrss_bytes"         => Int(Sys.maxrss()),
    )
    @printf("[%s] total %.3f s over %d sweeps (mean %.3f s/sweep)\n",
            label, wall, length(obs.seconds),
            isempty(obs.seconds) ? NaN : wall / length(obs.seconds))
    return energy, psi, truncerr, record
end

"""Excited-state counterpart; `Ms` are the states to project out."""
function timed_dmrg_excited(label, H, Ms::Vector{MPS}, psi0; nsweeps, maxdim, mindim,
                            cutoff, weight, kwargs...)
    println("\n[$label] warmup sweep (discarded, excludes codegen from timing)")
    _warmup_excited(H, Ms, psi0; weight = weight, kwargs...)

    # Zero every profiler AFTER the warmup so BENCH_TIMERS reports only the
    # timed run. See _reset_profilers! for what is covered and why it is guarded.
    _reset_profilers!()
    GC.gc(); GC.gc()
    obs = start!(SweepTimer())
    gc0 = Base.gc_num()
    wall = @elapsed begin
        energy, psi, _nsw, truncerr = dmrg(H, Ms, psi0; nsweeps, maxdim, mindim, cutoff,
                                          weight, observer = obs, use_early_exit = false,
                                          outputlevel = 1, kwargs...)
    end
    gcd = Base.GC_Diff(Base.gc_num(), gc0)

    record = Dict{String,Any}(
        "total_seconds"        => wall,
        "sweep_seconds"        => obs.seconds,
        "sweep_energies"       => obs.energies,
        "sweep_maxrss_bytes"   => obs.maxrss,
        "sweeps_completed"     => length(obs.seconds),
        "sweeps_requested"     => nsweeps,
        "mean_sweep_seconds"   => isempty(obs.seconds) ? nothing : sum(obs.seconds) / length(obs.seconds),
        "energy_php"           => real(energy),
        "truncerr"             => truncerr,
        "weight"               => weight,
        "allocated_bytes"      => gcd.allocd,
        "gc_seconds"           => gcd.total_time / 1e9,
        "maxrss_bytes"         => Int(Sys.maxrss()),
    )
    @printf("[%s] total %.3f s over %d sweeps (mean %.3f s/sweep)\n",
            label, wall, length(obs.seconds),
            isempty(obs.seconds) ? NaN : wall / length(obs.seconds))
    return energy, psi, truncerr, record
end

# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────

function _cpu_model()
    try
        return string(Sys.cpu_info()[1].model)
    catch
        return nothing
    end
end

"""
Everything needed to know whether two result rows are comparable. Thread counts
in particular: a runtime is only meaningful alongside the core budget it had.
"""
function env_metadata()
    return Dict{String,Any}(
        "hostname"             => gethostname(),
        "cpu_model"            => _cpu_model(),
        "ncores_detected"      => Sys.CPU_THREADS,
        "julia_version"        => string(VERSION),
        "julia_threads"        => Threads.nthreads(),
        "blas_threads"         => LinearAlgebra.BLAS.get_num_threads(),
        "blas_vendor"          => (try string(LinearAlgebra.BLAS.get_config()) catch; nothing end),
        "sb_aliased_nthreads"  => get(ENV, "SB_ALIASED_NTHREADS", "1"),
        "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", nothing),
        "project"              => Base.active_project(),
        "slurm_job_id"         => get(ENV, "SLURM_JOB_ID", nothing),
        "slurm_array_task_id"  => get(ENV, "SLURM_ARRAY_TASK_ID", nothing),
        "slurm_cpus_per_task"  => get(ENV, "SLURM_CPUS_PER_TASK", nothing),
        "slurm_nodelist"       => get(ENV, "SLURM_JOB_NODELIST", nothing),
        "git_commit"           => get(ENV, "BENCH_GIT_COMMIT", nothing),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

"""
Write one JSON per (config, variant, seed). Written the moment a variant
finishes, so a later variant dying in the same task cannot lose it.
"""
function write_result_json(outdir::AbstractString, result::Dict)
    mkpath(outdir)
    fname = @sprintf("%s__%s__seed%d.json",
                     result["config"], result["variant"], result["seed"])
    path = joinpath(outdir, fname)
    tmp  = path * ".partial"
    open(tmp, "w") do io
        JSON.print(io, result, 2)
    end
    mv(tmp, path; force = true)
    println("\n[bench] wrote ", path)
    return path
end

"""Default output directory: the BENCH_OUTDIR env var, else ./bench_results."""
bench_outdir() = get(ENV, "BENCH_OUTDIR", joinpath(pwd(), "bench_results"))
