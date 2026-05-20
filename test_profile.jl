# Profile sparse DMRG in production regime for a given N.
# Usage:  julia test_profile.jl <N_plaq> [n_prof_sweeps]
# Warmup: 2 sweeps ramping to maxdim=40 (JIT compile + bond growth).
# Profile: n_prof_sweeps sweeps at maxdim=40 with reset timers, per-sweep GC stats.
# Reports ITensorMPS.PROJMPO_TIMER and SparseBackends.TIMER.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using Printf
include("utils.jl")

function build_setup(N::Int)
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
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

let
    N_plaq     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    n_prof     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6

    println("Building setup for N=$N_plaq plaquettes ...")
    H_sp, psi_sp = build_setup(N_plaq)
    println("System: $(length(psi_sp)) sites")
    println("Initial linkdims: ", [ITensors.dim(commonind(psi_sp[i], psi_sp[i+1])) for i in 1:length(psi_sp)-1])

    # === Phase 1: WARMUP. Run 2 sweeps with ramp to maxdim=40 so bond dims fully grow. ===
    println("\n=== WARMUP (2 sweeps, ramp to maxdim=40) ===")
    sweeps_warm = Sweeps(2)
    setmaxdim!(sweeps_warm, 20, 40)
    setmindim!(sweeps_warm, 1)
    setcutoff!(sweeps_warm, 1e-10)
    t_warm = @elapsed (E_warm, psi_warm) = dmrg(H_sp, psi_sp, sweeps_warm; outputlevel=1, use_early_exit=false)
    println("Warmup done in $(round(t_warm, digits=1))s. E_after_warm = $E_warm")
    println("Warm linkdims: ", [ITensors.dim(commonind(psi_warm[i], psi_warm[i+1])) for i in 1:length(psi_warm)-1])

    # === Phase 2: PROFILE n_prof sweeps with reset timers, per-sweep GC stats. ===
    println("\n=== PROFILE ($n_prof sweeps at maxdim=40, per-sweep GC stats) ===")
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)

    psi_cur = psi_warm
    E_prof  = 0.0
    for k in 1:n_prof
        sw = Sweeps(1)
        setmaxdim!(sw, 40)
        setmindim!(sw, 1)
        setcutoff!(sw, 1e-10)
        gcs_before = Base.gc_num()
        t_one = @elapsed (E_prof, psi_cur) = dmrg(H_sp, psi_cur, sw; outputlevel=0, use_early_exit=false)
        gcs_after  = Base.gc_num()
        gc_time_s  = (gcs_after.total_time - gcs_before.total_time) / 1e9
        alloc_gb   = (gcs_after.allocd     - gcs_before.allocd)     / 1e9
        n_allocs   = gcs_after.poolalloc   - gcs_before.poolalloc
        @printf("  Sweep %d:  t=%6.2fs  GC=%5.2fs (%4.1f%%)  alloc=%5.2fGB  n_allocs=%.2e  E=%.10f\n",
                k, t_one, gc_time_s, 100*gc_time_s/t_one, alloc_gb, n_allocs, real(E_prof))
    end

    println("\n========== ITensorMPS.PROJMPO_TIMER (cumulative over $n_prof sweeps) ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER (cumulative over $n_prof sweeps) ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
