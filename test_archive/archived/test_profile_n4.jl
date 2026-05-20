# Profile N=4 maxdim=40 in production regime: warmup sweep first (JIT compile),
# then reset timers and profile a single subsequent sweep where mult is fully
# grown to 40. Reports both ITensorMPS.PROJMPO_TIMER (DMRG-level breakdown)
# and SparseBackends.TIMER (kernel-level breakdown).
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
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
    N_plaq = 4
    println("Building setup for N=$N_plaq plaquettes ...")
    H_sp, psi_sp = build_setup(N_plaq)
    println("System: $(length(psi_sp)) sites")
    println("Initial linkdims: ", [ITensors.dim(commonind(psi_sp[i], psi_sp[i+1])) for i in 1:length(psi_sp)-1])

    # === Phase 1: WARMUP. Run 2 sweeps with ramp to maxdim=40 so mult fully grows. ===
    println("\n=== WARMUP (2 sweeps, ramp to maxdim=40) ===")
    sweeps_warm = Sweeps(2)
    setmaxdim!(sweeps_warm, 20, 40)
    setmindim!(sweeps_warm, 1)
    setcutoff!(sweeps_warm, 1e-10)
    t_warm = @elapsed (E_warm, psi_warm) = dmrg(H_sp, psi_sp, sweeps_warm; outputlevel=1, use_early_exit=false)
    println("Warmup done in $(round(t_warm, digits=1))s. E_after_warm = $E_warm")
    println("Warm linkdims: ", [ITensors.dim(commonind(psi_warm[i], psi_warm[i+1])) for i in 1:length(psi_warm)-1])

    # === Phase 2: PROFILE 1 production-regime sweep with timers reset. ===
    println("\n=== PROFILE (1 sweep at maxdim=40 with reset timers) ===")
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)

    sweeps_prof = Sweeps(1)
    setmaxdim!(sweeps_prof, 40)
    setmindim!(sweeps_prof, 1)
    setcutoff!(sweeps_prof, 1e-10)
    t_prof = @elapsed (E_prof, psi_prof) = dmrg(H_sp, psi_warm, sweeps_prof; outputlevel=1, use_early_exit=false)
    println("Profile sweep done in $(round(t_prof, digits=1))s. E = $E_prof")

    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
