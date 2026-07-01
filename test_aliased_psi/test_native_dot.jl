# test_native_dot.jl — correctness + fire-count for #1 (native aliased dot) and
# #2 (Ffull scratch pool). Runs the SAME problem with the gates OFF then ON in
# one process (gates are read per-call via ENV) and checks per-sweep energy is
# bit-identical (FP-tol). Also reports how often the both-aliased contraction
# took the native kernel vs the dense-materialise fallback — directly confirming
# the fallback IS exercised and the native path replaces it.
#
# Run (single job, pinned, no contention):
#   cd /home/ritvik/temp/temp/edited_packages
#   taskset -c 0,2,4,6,8,10 env OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 \
#     VN=12 VMD=40 VSW=4 julia --project=. --threads=1 test_aliased_psi/test_native_dot.jl
# standard aliased gates (match the benchmarked regime)
for (k, v) in ("SB_ALIASED_PERCM_CAP"=>"0", "SB_USE_QR"=>"1", "SB_BALANCED_OWNERSHIP"=>"1",
               "SB_ADAPTIVE_RANK"=>"1",
               "SB_ALIASED_NATIVE_FISSION"=>"1", "BMF_BOP_PROJECT"=>"1", "BMF_MINV_RTOL"=>"1e-2")
    ENV[k] = v
end

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra, Random, Printf
include("../test_sparse_psi/utils.jl")

function build_setup(N::Int, psign::Int, spin::Int)
    Random.seed!(42)
    sites = spin == 2 ? siteinds("S=1/2", 2*N+2) : siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N; os += "Sx", 2*j-1, "Sx", 2*j+2; os += "Sy", 2*j, "Sy", 2*j+1; end
    cs = 0.5 * psign; os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1,2*j,2*j+1,2*j+2])) for j in 1:N]
    mulMPO(A,B) = replaceprime(contract(A, prime(B,"Site"), :coo,:coo), 2=>1)
    P = ConsOps1[1]; for j in 2:length(ConsOps1); P = mulMPO(P, ConsOps1[j]); end
    H = MPO(os, sites); psi0 = random_mps(sites)
    psi = replaceprime(contract(P, copy(psi0), :coo,:aliased; denseLinksB=0), 1=>0)
    return H, psi
end

const N  = parse(Int, get(ENV, "VN", "12"))
const MD = parse(Int, get(ENV, "VMD", "40"))
const NSW = parse(Int, get(ENV, "VSW", "4"))

function runE(native_dot::Bool, kernel_pool::Bool, align::Bool=native_dot)
    ENV["SB_ALIASED_NATIVE_DOT"]   = native_dot ? "1" : "0"
    ENV["SB_ALIASED_KERNEL_POOL"]  = kernel_pool ? "1" : "0"
    ENV["SB_ALIASED_ALIGN_OUTPUT"] = align ? "1" : "0"
    H, psi = build_setup(N, -1, 3)               # deterministic (seed 42)
    SparseBackends.reset_dot_hits!()
    Es = Float64[]
    stats = @timed for i in 1:NSW
        sw = Sweeps(1); setmaxdim!(sw, MD); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
        (E, psi, _, _) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        push!(Es, E)
    end
    return Es, SparseBackends._NATIVE_DOT_HITS[], SparseBackends._DENSIFY_DOT_HITS[],
           stats.bytes/2^30, stats.gctime, stats.time,
           SparseBackends._RECAST_PERMUTE_HITS[], SparseBackends._ALIGN_OK[], SparseBackends._ALIGN_FALLBACK[]
end

println("=== test_native_dot  N=$N md=$MD sweeps=$NSW  BLAS=", BLAS.get_num_threads(), " ===")
println("ISOLATION A/B: both arms run lean-dot + Ffull-pool; only ALIGN_OUTPUT differs.")
println("--- run 1/3: warmup ---"); runE(true, true, false)
println("--- run 2/3: baseline (lean+pool, ALIGN OFF) ---")
Eoff, no, do_, gib_off, gc_off, t_off, rc_off, aok_off, afb_off = runE(true, true, false)
println("--- run 3/3: treatment (lean+pool, ALIGN ON) ---")
Eon, n1, d1, gib_on, gc_on, t_on, rc_on, aok_on, afb_on = runE(true, true, true)

@printf("\nsweep        E_off                 E_on               |ΔE|\n")
for i in 1:NSW
    @printf("  %2d   %.12f   %.12f   %.2e\n", i, Eoff[i], Eon[i], abs(Eoff[i]-Eon[i]))
end
maxd = maximum(abs.(Eoff .- Eon))
@printf("\nmax |ΔE| = %.3e  →  %s\n", maxd, maxd < 1e-9 ? "BIT-IDENTICAL (FP-tol PASS)" : "MISMATCH (FAIL)")
if get(ENV, "SB_ALIASED_DOT_CHECK", "0") == "1"
    @printf("\nper-dot max ABSOLUTE error |native−dense| = %.3e   (|dense| there = %.3e)\n",
            SparseBackends._DOT_CHECK_MAXABS[], SparseBackends._DOT_CHECK_MAXMAG[])
    @printf("  → if abs error ~1e-13 or below, the native dot value is correct to FP;\n")
    @printf("    the per-sweep energy drift is iterative amplification of FP reassociation.\n")
end
@printf("\nboth-aliased contraction hits:\n")
@printf("  OFF: native=%-8d densify=%-8d\n", no, do_)
@printf("  ON : native=%-8d densify=%-8d   (native should be >0; densify should drop)\n", n1, d1)
@printf("\nrecast permutedims calls (the #recast-kill target):\n")
@printf("  OFF: recast=%-8d   ON: recast=%-8d  (ALIGN_OK=%d, fallback=%d) — ON should be ~0 if alignment works\n",
        rc_off, rc_on, aok_on, afb_on)
@printf("\nalloc/GC over %d sweeps (run includes no JIT — warmup done):\n", NSW)
@printf("  OFF: alloc=%.2f GiB  GC=%.2fs  wall=%.2fs\n", gib_off, gc_off, t_off)
@printf("  ON : alloc=%.2f GiB  GC=%.2fs  wall=%.2fs   (Δalloc=%.2f GiB, %.1f%%)\n",
        gib_on, gc_on, t_on, gib_off-gib_on, 100*(gib_off-gib_on)/max(gib_off,1e-9))
