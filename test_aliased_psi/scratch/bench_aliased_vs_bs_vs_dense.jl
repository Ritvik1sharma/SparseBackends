# Head-to-head benchmark: dense / BlockSparse / Aliased psi DMRG.
# Same H, same maxdim schedule, same number of sweeps. Reports per-sweep
# wall time (excluding sweep 1 for JIT) and final MPS memory footprint.
# With aliased psi, L is not exactly iso (templates shared across bond-channel
# values create off-diagonal coupling in L'L), so this needs Path-B (the
# M-corrected eigsolve via the gram cache) — dmrg()'s default run_mode=:bop_aliased.

using SparseBackends, ITensors, ITensorMPS
using Random
using Printf

include("../test_sparse_psi/utils.jl")

const N_PLAQ        = parse(Int, get(ENV, "BENCH_N_PLAQ", "2"))
const MAXDIM_SCHED  = [parse(Int, s) for s in split(get(ENV, "BENCH_MD", "4,8"), ",")]
const N_SWEEPS_AT_M = parse(Int, get(ENV, "BENCH_NSWEEPS", "3"))

function build_setup(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5,          "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
        t += 0.5 * psign,  "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    return H, P_sparse, psi0
end

mpsize(psi) = Base.summarysize(psi)

function run_dmrg_timed(label, H, psi)
    sweep_times = Float64[]
    cum_excl1 = 0.0
    E = NaN
    for ms in MAXDIM_SCHED
        for k in 1:N_SWEEPS_AT_M
            sw = Sweeps(1); setmaxdim!(sw, ms); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
            t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
            push!(sweep_times, t)
            if length(sweep_times) > 1; cum_excl1 += t; end
            @printf("  [%s sweep %d / md=%d] %7.3fs  E=%.10f\n",
                    label, length(sweep_times), ms, t, E)
        end
    end
    n_excl1 = length(sweep_times) - 1
    avg_excl1 = n_excl1 > 0 ? cum_excl1 / n_excl1 : NaN
    return E, psi, sweep_times, cum_excl1, avg_excl1
end

println("=== Benchmark: dense vs BS vs Aliased DMRG ===")
println("N_plaq=$N_PLAQ, maxdim schedule = $MAXDIM_SCHED, $N_SWEEPS_AT_M sweeps each")
println()

H, P_sparse, psi0 = build_setup(N_PLAQ, +1)
println("Built H, P_sparse, psi0.  System: $(length(psi0)) sites.")
println()

# ── 1) Dense (bare-H, dense psi) — matches test_profile_bareh.jl baseline. ──
println("--- Dense DMRG (psi0 dense, H dense) ---")
psi_d   = copy(psi0)
E_dense, psi_dense_final, t_dense, cum_d, avg_d = run_dmrg_timed("dense", H, psi_d)
mem_d = mpsize(psi_dense_final)

println()
println("--- BS DMRG (psi BS via contract(P, psi0, :coo, :dense)) ---")
psi_bs0 = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
E_bs, psi_bs_final, t_bs, cum_bs, avg_bs = run_dmrg_timed("BS", H, psi_bs0)
mem_bs = mpsize(psi_bs_final)

println()
println("--- Aliased DMRG (psi aliased via contract(P, psi0, :coo, :aliased)) ---")
psi_ali0 = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
E_ali, psi_ali_final, t_ali, cum_ali, avg_ali = run_dmrg_timed("ALI", H, psi_ali0)
mem_ali = mpsize(psi_ali_final)

println()
println("================== SUMMARY ==================")
@printf("Final energies:\n")
@printf("  dense:   E = %.10f\n", E_dense)
@printf("  BS:      E = %.10f\n", E_bs)
@printf("  ALI:     E = %.10f\n", E_ali)
println()
@printf("Per-sweep times (sweep #1 = JIT, excluded from averages):\n")
@printf("  dense sweeps: %s\n", join([@sprintf("%.3fs", t) for t in t_dense], "  "))
@printf("  BS    sweeps: %s\n", join([@sprintf("%.3fs", t) for t in t_bs],    "  "))
@printf("  ALI   sweeps: %s\n", join([@sprintf("%.3fs", t) for t in t_ali],   "  "))
println()
@printf("Averages (excl sweep 1):\n")
@printf("  dense  avg/sweep = %.3fs   total excl1 = %.3fs\n", avg_d,  cum_d)
@printf("  BS     avg/sweep = %.3fs   total excl1 = %.3fs   speedup_vs_dense = %.2fx\n",
        avg_bs, cum_bs, avg_d / max(avg_bs, 1e-9))
@printf("  ALI    avg/sweep = %.3fs   total excl1 = %.3fs   speedup_vs_dense = %.2fx   speedup_vs_BS = %.2fx\n",
        avg_ali, cum_ali, avg_d / max(avg_ali, 1e-9), avg_bs / max(avg_ali, 1e-9))
println()
@printf("Final MPS footprint:\n")
@printf("  dense: %.3f MiB\n", mem_d   / 2^20)
@printf("  BS:    %.3f MiB   compression vs dense = %.2fx\n", mem_bs / 2^20, mem_d  / max(mem_bs,  1))
@printf("  ALI:   %.3f MiB   compression vs dense = %.2fx   vs BS = %.2fx\n",
        mem_ali / 2^20, mem_d / max(mem_ali, 1), mem_bs / max(mem_ali, 1))
nothing
