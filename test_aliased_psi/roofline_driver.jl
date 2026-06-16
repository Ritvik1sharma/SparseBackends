# ROOFLINE driver (no HW counters — WSL2 has no uncore IMC PMU).
#   Ceilings : peak ZGEMM GFLOP/s + STREAM-triad GB/s (Julia microbenchmarks).
#   Kernel   : SB_ROOFLINE=1 accumulates exact Σ8·M·N·K FLOPs + compulsory bytes
#              in contract_shared! (covers H-matvec AND M⁻¹ apply). Kernel time
#              comes from SparseBackends.TIMER (printed). achieved GFLOP/s =
#              FLOPs / kernel_contract_time;  AI = FLOPs / bytes.
ENV["SB_ALIASED_ENABLE"] = "1"; ENV["BMF_ISO_PATH"] = "0"; ENV["BMF_APPLY_MINV"] = "1"
ENV["SB_ROOFLINE"]  = "1"; ENV["SB_CAS_STATS"] = "1"; ENV["SB_FUSION_DIAG"] = "1"

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra, Random, Printf
using TimerOutputs: reset_timer!, print_timer
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

# ---- ceilings ----
function peak_zgemm_gflops(n=2048, reps=5)
    A=rand(ComplexF64,n,n); B=rand(ComplexF64,n,n); C=zeros(ComplexF64,n,n)
    mul!(C,A,B)  # warmup
    t = @elapsed (for _ in 1:reps; mul!(C,A,B); end)
    return 8.0*n^3*reps / t / 1e9
end
function stream_triad_gbs(N=50_000_000, reps=5)
    a=rand(N); b=rand(N); c=rand(N); s=3.0
    @. a = b + s*c  # warmup
    t = @elapsed (for _ in 1:reps; @. a = b + s*c; end)
    return 3.0*N*8*reps / t / 1e9   # 2 read + 1 write, 8 B/Float64
end

let
    N=parse(Int,get(ENV,"VN","12")); md=parse(Int,get(ENV,"VMD","80")); nsw=parse(Int,get(ENV,"VSW","4"))
    ps = get(ENV,"VPS","-1")=="-1" ? -1 : 1
    println("=== ROOFLINE  N=$N md=$md sweeps=$nsw  BLAS_threads=", BLAS.get_num_threads(), " ===")
    println("SB_ALIASED_PERCM_CAP=", get(ENV,"SB_ALIASED_PERCM_CAP","0"),
            "  (0=UNCAPPED hbd=channel×maxdim [default/benchmarked]; 1=capped hbd≤maxdim)")
    println("--- ceilings (microbenchmarks) ---")
    pk = peak_zgemm_gflops(); bw = stream_triad_gbs()
    @printf("peak ZGEMM         = %.1f GFLOP/s\n", pk)
    @printf("STREAM-triad BW    = %.1f GB/s\n", bw)
    @printf("roofline ridge (AI*)= %.3f FLOP/byte  (peak/BW)\n", pk/bw)

    H, psi = build_setup(N, ps, 3)
    SparseBackends.reset_roofline!(); SparseBackends.reset_cas_stats!()
    reset_timer!(SparseBackends.TIMER)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    E=NaN
    dmrg_stats = @timed begin
        for i in 1:nsw
            sw=Sweeps(1); setmaxdim!(sw,md); setmindim!(sw,1); setcutoff!(sw,1e-10)
            (E,psi,_,_)=dmrg(H,psi,sw; outputlevel=0, use_early_exit=false)
            @printf("  [sweep %d] E=%.10f\n", i, E)
        end
    end
    @printf("\n--- run summary: wall=%.3f s  GC=%.3f s (%.1f%%)  alloc=%.3f GiB ---\n",
            dmrg_stats.time, dmrg_stats.gctime,
            100.0 * dmrg_stats.gctime / max(dmrg_stats.time, 1e-9),
            dmrg_stats.bytes / 2^30)
    println("\n--- kernel FLOP/byte (SB_ROOFLINE) ---")
    SparseBackends.show_roofline()
    SparseBackends.show_cas_stats()
    println("\n--- SparseBackends.TIMER (for kernel contract time) ---")
    print_timer(SparseBackends.TIMER; sortby=:time)
    println("\n--- ITensorMPS.PROJMPO_TIMER ---")
    print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:time)
    println()
end
