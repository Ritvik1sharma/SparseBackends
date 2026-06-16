# diag_alloc_provenance.jl
# VERIFY where the DMRG allocation churn (≈72 GiB/sweep at N=12/md80) actually
# comes from, BEFORE designing a no-churn pathway. Uses Julia's allocation
# profiler (Profile.Allocs) to attribute sampled allocations by (a) type and
# (b) leaf stack site and (c) owning module. The question to settle:
#   Is the churn reassigned temporary/output TENSOR DATA (Memory/Vector{ComplexF64}
#   templates allocated fresh per matvec/M⁻¹ call), or is it bookkeeping
#   (Dicts / key-tuples / wrapper structs) — which would need a different fix?
#
# Run (single job, pinned, no contention), e.g.:
#   cd /home/ritvik/temp/temp/edited_packages
#   taskset -c 0,2,4,6,8,10 env OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 \
#     SB_ALIASED_PERCM_CAP=0 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
#     SB_ALIASED_MINV_HINT=1 SB_ALIASED_NATIVE_FISSION=1 BMF_BOP_PROJECT=1 BMF_MINV_RTOL=1e-2 \
#     VN=12 VMD=40 julia --project=. --threads=1 test_aliased_psi/diag_alloc_provenance.jl
ENV["SB_ALIASED_ENABLE"] = "1"; ENV["BMF_ISO_PATH"] = "0"; ENV["BMF_APPLY_MINV"] = "1"

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra, Random, Printf
using Profile
include("../test_sparse_psi/utils.jl")

# --- identical setup to roofline_driver.jl ---
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
const SR = parse(Float64, get(ENV, "ALLOC_SAMPLE_RATE", "0.01"))

println("=== ALLOC PROVENANCE  N=$N md=$MD  sample_rate=$SR  BLAS=", BLAS.get_num_threads(), " ===")
H, psi = build_setup(N, -1, 3)

onesweep() = begin
    sw = Sweeps(1); setmaxdim!(sw, MD); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
    (E, p, _, _) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
    global psi = p
    E
end

# 1) JIT warmup (do NOT profile compilation allocations)
E0 = onesweep(); @printf("warmup  E=%.10f\n", E0)

# 2) Gross per-sweep allocation (cross-check vs roofline's ~72 GiB/sweep)
gross = @allocated (global E1 = onesweep())
@printf("one sweep @allocated = %.3f GiB   E=%.10f\n", gross/2^30, E1)

# 3) Allocation profile over one sweep
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=SR (global E2 = onesweep())
res = Profile.Allocs.fetch()
allocs = res.allocs
nsamp = length(allocs)
sampled = sum(a.size for a in allocs; init=0)
est = sampled / SR
@printf("\nprofiled sweep E=%.10f\n", E2)
@printf("sampled: n=%d  bytes=%.3f GiB  →  est total ≈ %.1f GiB/sweep (÷ sample_rate)\n",
        nsamp, sampled/2^30, est/2^30)

# ---- helpers ----
typename(a) = begin
    t = a.type
    t === nothing ? "nothing" : (t isa Type ? string(t) : string(typeof(t)))  # UnknownType etc.
end
modof(file::AbstractString) =
    occursin("SparseBackends", file) ? "SparseBackends" :
    occursin("ITensorMPS",     file) ? "ITensorMPS"     :
    occursin("NDTensors",      file) ? "NDTensors"      :
    occursin(r"ITensors(?!MPS)", file) || occursin("/ITensors", file) ? "ITensors" :
    occursin("KrylovKit",      file) ? "KrylovKit"      :
    (occursin("LinearAlgebra", file) || occursin("/libblas", file) || occursin("openblas", file)) ? "BLAS/LinAlg" :
    occursin("julia/stdlib",   file) || occursin("/base/", file) || file == "" ? "Base/stdlib" :
    "other"
leaflabel(a) = isempty(a.stacktrace) ? "<no stack>" :
    let f = a.stacktrace[1]; string(f.func, "  @ ", basename(String(f.file)), ":", f.line); end
# first user (SparseBackends/ITensorMPS) frame in the stack, to localize churn
function userframe(a)
    for f in a.stacktrace
        m = modof(String(f.file))
        if m == "SparseBackends" || m == "ITensorMPS"
            return string(m, ":", f.func, " @ ", basename(String(f.file)), ":", f.line)
        end
    end
    return "<no user frame>"
end

function topagg(keyfn, label; n=18)
    d = Dict{String,Int}(); c = Dict{String,Int}()
    for a in allocs
        k = keyfn(a)::String
        d[k] = get(d, k, 0) + a.size
        c[k] = get(c, k, 0) + 1
    end
    println("\n── top $label by sampled bytes ──")
    for (k, v) in first(sort(collect(d); by = x -> -x[2]), n)
        @printf("  %6.1f GiB/sw  (%4.1f%%)  n=%-6d  %s\n",
                (v/SR)/2^30, 100*v/sampled, c[k], k)
    end
end

# 4) attribution: by type, by owning module, by leaf site, by first user frame
topagg(typename,  "TYPE")
topagg(a -> (isempty(a.stacktrace) ? "<none>" : modof(String(a.stacktrace[1].file))), "owning MODULE (leaf)"; n=10)
topagg(leaflabel,  "LEAF allocation site")
topagg(userframe,  "first SparseBackends/ITensorMPS frame")

println("\n=== VERDICT HINTS ===")
println("• If TYPE is dominated by Memory{ComplexF64}/Vector{ComplexF64} and the leaf")
println("  sites are tensor-data allocations (zeros/similar/undef) inside the matvec /")
println("  apply_minv path → churn IS reassigned temporary/output tensor data (poolable).")
println("• If Dict/Tuple/Pair/struct types dominate → churn is bookkeeping (different fix).")
