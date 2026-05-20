# Time breakdown of sparse DMRG: position! vs eigsolve vs replacebond! vs gram_envs.
# Uses the codebase's pre-existing PROJMPO_TIMER (TimerOutput).
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs
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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_sparse, psi_sp, H_dense, psi_dense
end

let
    N = 3
    println("=== Profile breakdown: sparse vs dense DMRG (N=$N) ===")
    H_sp, psi_sp, H_d, psi_d = build_setup(N)
    sweeps = Sweeps(2); setmaxdim!(sweeps, 10, 20); setmindim!(sweeps, 1); setcutoff!(sweeps, 1e-10)

    # Warm up Julia JIT and any timer scaffolding by running 1 sweep before measuring.
    println("\nJIT warmup (sparse, 1 sweep)...")
    sw1 = Sweeps(1); setmaxdim!(sw1, 10); setcutoff!(sw1, 1e-10)
    _ = dmrg(H_sp, psi_sp, sw1; outputlevel=0)

    # SPARSE timed.  Rebuild fresh psi since sparse ψ has no copy() method.
    H_sp2, psi_sp2, _, _ = build_setup(N)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    println("\n--- Sparse run ---")
    t_sp = @elapsed (E_sp, _) = dmrg(H_sp2, psi_sp2, sweeps; outputlevel=0)
    println("Sparse total: $(round(t_sp; digits=2))s  E=$E_sp")
    println("\nSparse timer breakdown:")
    show(stdout, ITensorMPS.PROJMPO_TIMER; allocations=false, compact=false)
    println()

    # DENSE timed.  Plain MPS / MPO have copy method.
    _, _, H_d2, psi_d2 = build_setup(N)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    println("\n--- Dense run ---")
    t_d = @elapsed (E_d, _) = dmrg(H_d2, psi_d2, sweeps; outputlevel=0)
    println("Dense total: $(round(t_d; digits=2))s  E=$E_d")
    println("\nDense timer breakdown:")
    show(stdout, ITensorMPS.PROJMPO_TIMER; allocations=false, compact=false)
    println()

    println("\nOverall sparse/dense wallclock ratio: $(round(t_sp/t_d; digits=1))x")
end
nothing
